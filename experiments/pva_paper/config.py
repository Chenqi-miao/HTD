"""pva_paper — 对齐 HTD-Refine 论文架构的实验"""
import os

EXP_NAME = 'pva_paper'
EXP_ROOT = os.path.join(os.path.dirname(__file__), '..', '..')

class Config:
    # 数据
    data_dir = os.path.join(EXP_ROOT, 'data/SeqHand')
    data_list = ['InterHand_train']
    test_list = ['InterHand_test']
    seq_len = 15
    min_seq_len = 30
    view_num = 1
    joint_num = 21
    data_num = None
    test_data_num = 1000
    loader_resample = True

    # 模型（论文架构: 8层 RoPE Transformer + 3 decoder）
    dim_feat = 256
    transformer_depth = 8

    # Loss 权重
    w_vel = 0.1
    w_accel = 0.05

    # 训练
    batch_size = 128
    lr = 1e-4
    total_epoch = 40
    lr_scheduler = 'cosine'
    num_worker = 4
    print_iter = 500
    eval_interval = 2
    vis_interval = 0

    exp_dir = os.path.join(os.path.dirname(__file__))
    output_root = exp_dir
    add_info = EXP_NAME
