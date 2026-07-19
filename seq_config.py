
class Config:
    phase = 'train'
    backbone = 'MotionBERT' #
    data_dir = './data/SeqHand'
    mano_path = './misc/mano'
    data_list = ['InterHand_train'] # 'DexYCB', 'ReInterHand', 'InterHand_train', 'InterHand_test','InterHand_val','UmeTrack_synthetic','UmeTrack_real'


    # Test Info
    eval_mode = 'middle'

    # Augment info
    loader_resample = True

    # seq info
    only_joint = True
    view_num = 1
    seq_len = 15 # 9 15 25 27 81 243
    min_seq_len = 30
    joint_num = 21
    data_num = None
    test_data_num = None

    # DDN model info
    global_temporal = True # 采用全局时序建模模块
    window_size = 27

    # ckp info
    add_info = '%s-Seq%d-Window%d'%(backbone, seq_len, window_size)
    checkpoint = ''
    output_root = './checkpoint/%s-%s-%s'%(data_list[0], phase, add_info)

    # training
    lr_scheduler = 'cosine'
    batch_size = 64
    lr = 1e-4
    total_epoch = 40
    input_img_shape = (256, 256)
    num_worker = 0

    # -------------
    save_epoch = 10
    eval_interval = 2
    print_iter = 1000
    draw_iter = 500
    # -------------

    vis = False
    # -------------
    experiment_name = backbone + '_{}'.format(backbone) + '_lr{}'.format(lr)+ '_Epochs{}'.format(total_epoch)

cfg = Config()
