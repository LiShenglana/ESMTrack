from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.art_path = 'XXX/data/arkit'
    settings.davis_dir = ''
    settings.depthtrack_path = 'XXX/data/depthtrack'
    settings.got10k_lmdb_path = 'XXX/data/got10k_lmdb'
    settings.got10k_path = 'XXX/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = 'XXX/data/itb'
    # settings.lasher_path = 'XXX/data/lasher'
    settings.lasot_extension_subset_path = 'XXX/data/lasot_extension_subset'
    settings.lasot_lmdb_path = 'XXX/data/lasot_lmdb'
    settings.lasot_path = 'XXX/data/lasot'
    settings.network_path = 'XXX/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = 'XXX/data/nfs'
    settings.otb_lang_path = 'XXX/data/otb_lang'
    settings.otb_path = 'XXX/data/otb'
    settings.prj_dir = '/home/cscv/Documents/lsl/ESMTrackRGBT'
    settings.result_plot_path = 'XXX/output/test/result_plots'
    settings.results_path = 'XXX/output/test/tracking_results'    # Where to store tracking results
    # settings.rgbt210_path = 'XXX/data/rgbt210'
    # settings.rgbt234_path = 'XXX/data/rgbt234'
    settings.gtot_path = '/home/cscv/Documents/lsl/dataset/GTOT'
    settings.rgbt210_path = '/media/cscv/d00985a0-c3e6-4ffa-9546-88c861db5ce3/02_Dataset/RGBT210'
    settings.rgbt234_path = '/home/cscv/Documents/lsl/dataset/RGB-T234'
    settings.lasher_dir = '/media/cscv/d00985a0-c3e6-4ffa-9546-88c861db5ce3/02_Dataset/LasHeR/testingset'
    settings.vtuav_dir = '/media/cscv/d00985a0-c3e6-4ffa-9546-88c861db5ce3/02_Dataset/VTUAV'
    settings.save_dir = '/home/cscv/Documents/lsl/ESMTrackRGBT'
    settings.segmentation_path = 'XXX/output/test/segmentation_results'
    settings.tc128_path = 'XXX/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = 'XXX/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = 'XXX/data/trackingnet'
    settings.uav_path = 'XXX/data/uav'
    settings.visevent_path = 'XXX/data/visevent'
    settings.vot18_path = 'XXX/data/vot2018'
    settings.vot22_path = 'XXX/data/vot2022'
    settings.vot_path = 'XXX/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

