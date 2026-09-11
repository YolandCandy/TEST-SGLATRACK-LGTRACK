import os
from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))

    # Set your local paths here.
    settings.davis_dir = ''
    settings.dtb70_path = os.path.join(root_dir, 'data/DTB70')
    settings.got10k_lmdb_path = os.path.join(root_dir, 'data/got10k_lmdb')
    settings.got10k_path = os.path.join(root_dir, 'data/got10k')
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = os.path.join(root_dir, 'data/itb')
    settings.lasot_extension_subset_path_path = os.path.join(root_dir, 'data/lasot_extension_subset')
    settings.lasot_lmdb_path = os.path.join(root_dir, 'data/lasot_lmdb')
    settings.lasot_path = os.path.join(root_dir, 'data/lasot')
    settings.network_path = os.path.join(root_dir, 'output/test/networks')
    settings.nfs_path = os.path.join(root_dir, 'data/nfs')
    settings.otb_path = os.path.join(root_dir, 'data/otb')
    settings.prj_dir = root_dir
    settings.result_plot_path = os.path.join(root_dir, 'output/test/result_plots')
    settings.results_path = os.path.join(root_dir, 'output/test/tracking_results')
    settings.save_dir = os.path.join(root_dir, 'output')
    settings.segmentation_path = os.path.join(root_dir, 'output/test/segmentation_results')
    settings.tc128_path = os.path.join(root_dir, 'data/TC128')
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = os.path.join(root_dir, 'data/tnl2k')
    settings.tpl_path = ''
    settings.trackingnet_path = os.path.join(root_dir, 'data/trackingnet')
    settings.uav123_10fps_path = os.path.join(root_dir, 'data/UAV123_10fps')
    settings.uav123_path = os.path.join(root_dir, 'data/UAV123')
    settings.uav_path = os.path.join(root_dir, 'data/UAV123')
    settings.uavdt_path = os.path.join(root_dir, 'data/uavdt')
    settings.uavtrack_path = os.path.join(root_dir, 'data/V4RFlight112')
    settings.visdrone_path = os.path.join(root_dir, 'data/VisDrone2018-SOT-test-dev')
    settings.vot18_path = os.path.join(root_dir, 'data/vot2018')
    settings.vot22_path = os.path.join(root_dir, 'data/vot2022')
    settings.vot_path = os.path.join(root_dir, 'data/VOT2019')
    settings.youtubevos_dir = ''

    return settings

