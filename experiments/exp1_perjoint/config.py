import os
EXP_NAME = 'exp1_perjoint'
EXP_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
class Config:
    data_dir = os.path.join(EXP_ROOT, 'data/SeqHand')
    data_list = ['InterHand_train']
    test_list = ['InterHand_test']
    seq_len, min_seq_len, view_num, joint_num = 15, 30, 1, 21
    data_num = None
    test_data_num = 1000
    loader_resample = True
    dim_feat = 256
    transformer_depth = 2
    w_vel, w_accel = 0.1, 0.05
    batch_size, lr, total_epoch = 128, 1e-4, 40
    lr_scheduler = 'cosine'; num_worker = 4
    print_iter = 500; eval_interval = 2; vis_interval = 0
    exp_dir = os.path.join(os.path.dirname(__file__))
    output_root = exp_dir; add_info = EXP_NAME
