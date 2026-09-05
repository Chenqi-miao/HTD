import os
EXP_NAME = 'exp2_s30_att'
EXP_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
class Config:
    data_dir = os.path.join(EXP_ROOT, 'data/SeqHand')
    data_list = ['InterHand_train']
    test_list = ['InterHand_test']
    seq_len, min_seq_len, view_num, joint_num = 31, 31, 1, 21
    data_num = None
    test_data_num = 2000
    loader_resample = True

    dim_feat = 256
    transformer_depth = 8
    w_vel, w_accel = 0.1, 0.05

    # 从零训练 (不加载预训练)
    pretrained = ""

    # loss 权重: 关键点 > 速度 > 加速度
    w_vel, w_accel = 0.1, 0.05

    batch_size = 32
    lr = 1e-4
    total_epoch = 30
    max_batches_per_epoch = 999999    # 全量训练 (不限制)
    num_worker = 2
    print_iter = 200
    eval_interval = 5
    exp_dir = os.path.join(os.path.dirname(__file__))
    output_root = exp_dir
    add_info = EXP_NAME
