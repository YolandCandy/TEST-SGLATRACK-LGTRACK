from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.biodrone_path = '/home/nvidia/minh/LGTrack/data/biodrone'
    settings.davis_dir = ''
    settings.dtb70_path = '/media/xiaoyang/new_ssd_volume/ampa_migra/F/data/DTB70'
    settings.got10k_lmdb_path = '/home/nvidia/minh/LGTrack/data/got10k_lmdb'
    settings.got10k_path = '/home/nvidia/minh/LGTrack/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/home/nvidia/minh/LGTrack/data/itb'
    settings.lasot_extension_subset_path_path = '/home/nvidia/minh/LGTrack/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/home/nvidia/minh/LGTrack/data/lasot_lmdb'
    settings.lasot_path = '/home/nvidia/minh/LGTrack/data/lasot'
    settings.network_path = '/home/nvidia/minh/LGTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/nvidia/minh/LGTrack/data/nfs'
    settings.otb_path = '/home/nvidia/minh/LGTrack/data/otb'
    settings.prj_dir = '/home/nvidia/minh/LGTrack'
    settings.result_plot_path = '/home/nvidia/minh/LGTrack/output/test/result_plots'
    settings.results_path = '/home/nvidia/minh/LGTrack/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/nvidia/minh/LGTrack/output'
    settings.segmentation_path = '/home/nvidia/minh/LGTrack/output/test/segmentation_results'
    settings.tc128_path = '/home/nvidia/minh/LGTrack/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/home/nvidia/minh/LGTrack/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/nvidia/minh/LGTrack/data/trackingnet'
    settings.uav123_path = '/home/nvidia/datasets/UAV123'
    settings.uav_path = '/home/nvidia/datasets/UAV123'
    settings.uavdt_path = '/media/xiaoyang/new_ssd_volume/ampa_migra/F/data/uavdt'
    settings.visdrone2018_path = '/media/xiaoyang/new_ssd_volume/ampa_migra/F/data/visdrone2018'
    settings.vot18_path = '/home/nvidia/minh/LGTrack/data/vot2018'
    settings.vot22_path = '/home/nvidia/minh/LGTrack/data/vot2022'
    settings.vot_path = '/home/nvidia/minh/LGTrack/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

