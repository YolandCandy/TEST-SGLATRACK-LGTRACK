from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.lgtrack.utils import combine_tokens, recover_tokens
enabled_layer_num = 1
start_layer = 8 # 8+1

class SelectionModule(nn.Module):
    """
    Selection Module (3-layer MLP) for Similarity Guided Layer Adaptation (SGLA).
    Proposed in LGTrack architecture to dynamically select downstream transformer layers.
    """
    def __init__(self, input_dim=320, output_dim=3, hidden_dim=160):
        super(SelectionModule, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Standard 3-layer MLP matching LGTrack architecture diagram
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self):
        # Initialize fc2 as identity-like for seamless pass-through
        nn.init.eye_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.trunc_normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.trunc_normal_(self.fc3.weight, std=0.02)
        nn.init.zeros_(self.fc3.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """
        Support backward compatibility with legacy 2-layer MLP checkpoints:
        If checkpoint contains 'fc2.weight' matching output_dim,
        route it to fc3, and set fc2 to identity.
        """
        fc2_w_key = prefix + 'fc2.weight'
        fc2_b_key = prefix + 'fc2.bias'
        fc3_w_key = prefix + 'fc3.weight'
        fc3_b_key = prefix + 'fc3.bias'

        if fc2_w_key in state_dict and fc3_w_key not in state_dict:
            fc2_weight = state_dict[fc2_w_key]
            if fc2_weight.shape == self.fc3.weight.shape:
                state_dict[fc3_w_key] = state_dict.pop(fc2_w_key)
                if fc2_b_key in state_dict:
                    state_dict[fc3_b_key] = state_dict.pop(fc2_b_key)
                state_dict[fc2_w_key] = torch.eye(self.fc2.out_features, self.fc2.in_features)
                state_dict[fc2_b_key] = torch.zeros(self.fc2.out_features)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # Standardize 1-D features: support [B, L, C] tokens or [B, L] vector
        if x.dim() == 3:
            if x.shape[1] == self.input_dim:
                x = x[:, :, 0]
            else:
                x = x.mean(dim=-1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        pro = self.sigmoid(x)
        return pro


ThreeLayerMLP = SelectionModule  # Alias for backward compatibility

class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.return_stage = cfg.MODEL.RETURN_STAGES
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

        # resize patch embedding
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        # for cls token (keep it but not used)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        # self.cls_token = None
        # self.pos_embed = None

        if self.return_inter:
            for i_layer in self.return_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

        self.MLP = ThreeLayerMLP(input_dim=320, output_dim=12-1-start_layer)

    def forward_features(self, z, x):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)

        cos_list = []

        for i, blk in enumerate(self.blocks):
            if i < start_layer:
                x = blk(x)
            elif i == start_layer:
                x = blk(x)
                mid = x.detach()
                pro = self.MLP(x[:,:,0].clone())
                topk_values, topk_indices = torch.topk(pro, enabled_layer_num, dim=1)
                sorted_topk_indices = torch.sort(topk_indices, dim=1).values + start_layer + 1
                # sorted_topk_values = torch.sort(topk_values, dim=1).values
            else:
                idx = torch.where(sorted_topk_indices[:,:]==i)[0]
                if len(idx) > 0:
                    x[idx] = blk(x[idx])
        
        with torch.no_grad():
            cos_tensor = torch.ones(B,12-1-start_layer, device=x.device)
            for i, blk in enumerate(self.blocks):
                if i > start_layer:
                    temp = blk(mid)
                    cos = F.cosine_similarity(mid,temp)
                    cos_tensor[:, i-(start_layer + 1)] = cos.mean(dim=1)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]



        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None, "cos_tensor": cos_tensor.detach(), "pro":pro}
        return self.norm(x), aux_dict

    def forward(self, z, x, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        if self.training:
            x, aux_dict = self.forward_features(z, x)
        else:
            x, aux_dict = self.forward_test(z, x)

        return x, aux_dict

    def forward_test(self, z, x):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z
        x += self.pos_embed_x

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x)
        
        x = combine_tokens(z, x, mode=self.cat_mode)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks): 
            if i < start_layer:
                x = blk(x)
            elif i == start_layer:
                x = blk(x)
                pro = self.MLP(x[:,:,0].clone())
                topk_values, topk_indices = torch.topk(pro, enabled_layer_num, dim=1)
                sorted_topk_indices = torch.sort(topk_indices, dim=1).values + start_layer + 1
                # sorted_topk_values = torch.sort(topk_values, dim=1).values
            else:
                idx = torch.where(sorted_topk_indices[:,:]==i)[0]
                if len(idx) > 0:
                    x[idx] = blk(x[idx])
                    break
                
        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        x = recover_tokens(x, lens_z, lens_x, mode=self.cat_mode)

        aux_dict = {"attn": None}

        return self.norm(x), aux_dict