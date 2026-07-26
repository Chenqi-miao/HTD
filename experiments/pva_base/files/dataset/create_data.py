import os
import torch
import os.path as osp
import numpy as np
import pickle
# from utils.fix_seed import set_seed

# txt
# def generator_seq(root_path, save_path, dataset, min_seq_len=30):
#     sum_file = os.path.join(root_path, '%s.pkl' % (dataset))
#     os.makedirs(os.path.join(save_path, 'gt_joint'), exist_ok=True)
#     os.makedirs(os.path.join(save_path, 'gt_joint_valid'), exist_ok=True)
#     os.makedirs(os.path.join(save_path, 'in_joint'), exist_ok=True)
#     os.makedirs(os.path.join(save_path, 'in_joint_valid'), exist_ok=True)
#     os.makedirs(os.path.join(save_path, 'hand_type'), exist_ok=True)
#     os.makedirs(os.path.join(save_path, 'img_name'), exist_ok=True)
#
#     with open(sum_file, 'rb') as file:
#         data_dict = pickle.load(file)
#     save_index = 0
#     for capture in data_dict.keys():
#         for seq in data_dict[capture].keys():
#             img_name_list = data_dict[capture][seq]['img_name_list']
#             seq_img_num = len(img_name_list)
#             if seq_img_num < min_seq_len:
#                 continue
#
#             # 加载该序列的整体信息
#             data_path = os.path.join(root_path, dataset, capture, seq)
#             anno_file = osp.join(data_path, 'anno_info.pkl')
#             meta_file = osp.join(data_path, 'meta_info.pkl')
#             with open(meta_file, 'rb') as file:
#                 meta_info = pickle.load(file)
#             with open(anno_file, 'rb') as file:
#                 annos_info = pickle.load(file)
#
#             method_name_list = data_dict[capture][seq]['method_name_list']
#             for method_select in method_name_list:
#                 cam_name_list = data_dict[capture][seq]['cam_name_list']
#                 for cam_name_select in cam_name_list:
#                     joint_world_gt, joint_valid_gt, hand_type = load_gt(img_name_list, meta_info, annos_info)
#                     joint_world_in, joint_valid_in = load_input(data_path, method_select, cam_name_select, img_name_list, meta_info)
#                     np.savetxt(os.path.join(save_path, 'gt_joint', '%08d.txt'%save_index), joint_world_gt.reshape(-1, 63), fmt='%.3f')
#                     np.savetxt(os.path.join(save_path, 'gt_joint_valid', '%08d.txt'%save_index), joint_valid_gt.reshape(-1, 21), fmt='%d')
#                     np.savetxt(os.path.join(save_path, 'in_joint', '%08d.txt'%save_index), joint_world_in.reshape(-1, 63), fmt='%.3f')
#                     np.savetxt(os.path.join(save_path, 'in_joint_valid', '%08d.txt'%save_index), joint_valid_in.reshape(-1, 21), fmt='%d')
#                     np.savetxt(os.path.join(save_path, 'hand_type', '%08d.txt'%save_index), hand_type.reshape(-1), fmt='%d')
#                     with open(os.path.join(save_path, 'img_name', '%08d.txt'%save_index), "w") as f:
#                         for line in img_name_list:
#                             f.write(os.path.join(dataset, capture, seq, method_select,cam_name_select,line) + "\n")  # 每行末尾添加换行符
#                     save_index = save_index + 1
#             print(os.path.join(capture,seq))

# npz
def generator_seq(root_path, save_path, dataset, min_seq_len=30):
    sum_file = os.path.join(root_path, '%s.pkl' % (dataset))
    os.makedirs(os.path.join(save_path, 'gt_joint'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'gt_joint_valid'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'in_joint'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'in_joint_valid'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'hand_type'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'img_name'), exist_ok=True)

    with open(sum_file, 'rb') as file:
        data_dict = pickle.load(file)
    save_index = 0
    for capture in data_dict.keys():
        for seq in data_dict[capture].keys():
            img_name_list = data_dict[capture][seq]['img_name_list']
            seq_img_num = len(img_name_list)
            if seq_img_num < min_seq_len:
                continue

            # 加载该序列的整体信息
            data_path = os.path.join(root_path, dataset, capture, seq)
            anno_file = osp.join(data_path, 'anno_info.pkl')
            meta_file = osp.join(data_path, 'meta_info.pkl')
            with open(meta_file, 'rb') as file:
                meta_info = pickle.load(file)
            with open(anno_file, 'rb') as file:
                annos_info = pickle.load(file)

            method_name_list = data_dict[capture][seq]['method_name_list']
            for method_select in method_name_list:
                cam_name_list = data_dict[capture][seq]['cam_name_list']
                for cam_name_select in cam_name_list:
                    joint_world_gt, joint_valid_gt, hand_type = load_gt(img_name_list, meta_info, annos_info)
                    joint_world_in, joint_valid_in = load_input(data_path, method_select, cam_name_select, img_name_list, meta_info)
                    np.savez_compressed(os.path.join(save_path, 'gt_joint', '%08d.npz'%save_index), joint_world_gt.reshape(-1, 21, 3))
                    np.savez_compressed(os.path.join(save_path, 'gt_joint_valid', '%08d.npz'%save_index), joint_valid_gt.reshape(-1, 21))
                    np.savez_compressed(os.path.join(save_path, 'in_joint', '%08d.npz'%save_index), joint_world_in.reshape(-1, 21, 3))
                    np.savez_compressed(os.path.join(save_path, 'in_joint_valid', '%08d.npz'%save_index), joint_valid_in.reshape(-1, 21))
                    np.savez_compressed(os.path.join(save_path, 'hand_type', '%08d.npz'%save_index), hand_type.reshape(-1), fmt='%d')
                    with open(os.path.join(save_path, 'img_name', '%08d.txt'%save_index), "w") as f:
                        for line in img_name_list:
                            f.write(os.path.join(dataset, capture, seq, method_select,cam_name_select,line) + "\n")  # 每行末尾添加换行符
                    save_index = save_index + 1
            print(os.path.join(capture,seq))

# npz all
def generator_seq_all(root_path, save_path, dataset, min_seq_len=30):
    sum_file = os.path.join(root_path, '%s.pkl' % (dataset))
    joint_world_gt_list, joint_valid_gt_list, hand_type_list = [], [], []
    joint_world_in_list, joint_valid_in_list = [], []
    all_img_name_list = []
    key_list = []
    with open(sum_file, 'rb') as file:
        data_dict = pickle.load(file)

    for capture in data_dict.keys():
        for seq in data_dict[capture].keys():
            img_name_list = data_dict[capture][seq]['img_name_list']
            seq_img_num = len(img_name_list)
            if seq_img_num < min_seq_len:
                continue

            # 加载该序列的整体信息
            data_path = os.path.join(root_path, dataset, capture, seq)
            anno_file = osp.join(data_path, 'anno_info.pkl')
            meta_file = osp.join(data_path, 'meta_info.pkl')
            with open(meta_file, 'rb') as file:
                meta_info = pickle.load(file)
            with open(anno_file, 'rb') as file:
                annos_info = pickle.load(file)

            method_name_list = data_dict[capture][seq]['method_name_list']
            for method_select in method_name_list:
                cam_name_list = data_dict[capture][seq]['cam_name_list']
                for cam_name_select in cam_name_list:
                    joint_world_gt, joint_valid_gt, hand_type = load_gt(img_name_list, meta_info, annos_info)
                    joint_world_in, joint_valid_in = load_input(data_path, method_select, cam_name_select, img_name_list, meta_info)
                    joint_world_gt_list.append(joint_world_gt)
                    joint_valid_gt_list.append(joint_valid_gt)
                    hand_type_list.append(hand_type)
                    joint_world_in_list.append(joint_world_in)
                    joint_valid_in_list.append(joint_valid_in)
                    key_list.append(os.path.join(capture, seq, method_select, cam_name_select))
                    for line in img_name_list:
                        all_img_name_list.append(os.path.join(dataset, capture, seq, method_select, cam_name_select, line))
            print(os.path.join(capture,seq))

    joint_world_gt = dict(zip(key_list, joint_world_gt_list))
    joint_valid_gt = dict(zip(key_list, joint_valid_gt_list))
    hand_type =      dict(zip(key_list, hand_type_list))
    joint_world_in = dict(zip(key_list, joint_world_in_list))
    joint_valid_in = dict(zip(key_list, joint_valid_in_list))

    np.savez_compressed(os.path.join(save_path, 'gt_joint.npz'), **joint_world_gt)
    np.savez_compressed(os.path.join(save_path, 'gt_joint_valid.npz'), **joint_valid_gt)
    np.savez_compressed(os.path.join(save_path, 'in_joint.npz'), **joint_world_in)
    np.savez_compressed(os.path.join(save_path, 'in_joint_valid.npz'), **joint_valid_in)
    np.savez_compressed(os.path.join(save_path, 'hand_type.npz'), **hand_type)
    with open(os.path.join(save_path, 'img_name.txt'), "w") as f:
        for line in all_img_name_list:
            f.write(line + "\n")

def process_generator_seq_all_name(root_path, save_path, dataset, min_seq_len=30):
    sum_file = os.path.join(root_path, '%s.pkl' % (dataset))
    all_img_name_list = []
    key_list = []
    with open(sum_file, 'rb') as file:
        data_dict = pickle.load(file)

    for capture in data_dict.keys():
        for seq in data_dict[capture].keys():
            img_name_list = data_dict[capture][seq]['img_name_list']
            seq_img_num = len(img_name_list)
            if seq_img_num < min_seq_len:
                continue

            # 加载该序列的整体信息
            method_name_list = data_dict[capture][seq]['method_name_list']
            for method_select in method_name_list:
                cam_name_list = data_dict[capture][seq]['cam_name_list']
                for cam_name_select in cam_name_list:
                    seq_img_name_list = []
                    key_list.append(os.path.join(capture, seq, method_select, cam_name_select))
                    for line in img_name_list:
                        seq_img_name_list.append(int(line))
                    all_img_name_list.append(np.array(seq_img_name_list))
            print(os.path.join(capture,seq))

    img_name = dict(zip(key_list, all_img_name_list))
    np.savez_compressed(os.path.join(save_path, 'img_name.npz'), **img_name)


def load_gt(img_id_list, meta_info, annos_info):
    hand_types = ['right', 'left']
    # 加载全局手部坐标GT和手部类型
    joint_world_list = []
    joint_valid_list = []
    hand_type_list = []
    for img_name in img_id_list:
        if meta_info['frame_info'][img_name]['hand_type'] in ['right', 'two', 'interacting']:
            hand_type = 0  # 'right'
        else:
            hand_type = 1  # 'left'
        if annos_info[img_name][hand_types[hand_type]]['world_coord'] is not None:
            joint_world = np.array(annos_info[img_name][hand_types[hand_type]]['world_coord'], np.float64)
        else:
            joint_world = np.zeros([21, 3])
        if annos_info[img_name][hand_types[hand_type]]['joint_valid'] is not None:
            joint_valid = np.array(annos_info[img_name][hand_types[hand_type]]['joint_valid'], np.float64)
        else:
            joint_valid = np.zeros([21])
        joint_world_list.append(joint_world)
        joint_valid_list.append(joint_valid)
        hand_type_list.append(hand_type)
    joint_world_gt = np.stack(joint_world_list, axis=0)
    joint_valid_gt = np.stack(joint_valid_list, axis=0)
    joint_type = np.stack(hand_type_list, axis=0)
    return joint_world_gt, joint_valid_gt, joint_type

def load_input(data_path, method_name, cam_name, img_id_list, meta_info):
    hand_types = ['right', 'left']
    # 读取序列数据
    cam_para = meta_info['cam_params'][cam_name]
    R, T = np.array(cam_para['R']), np.array(cam_para['T'])

    joint_cam_list = []
    joint_cam_valid_list = []
    R_seq_list, T_seq_list = [], []
    for img_name in img_id_list:
        if meta_info['frame_info'][img_name]['hand_type'] in ['right', 'two', 'interacting']:
            hand_type = 0  # 'right'
        else:
            hand_type = 1  # 'left'
        data_file = os.path.join(data_path, method_name, cam_name, '%s.pkl' % (img_name))
        with open(data_file, 'rb') as file:
            data_info = pickle.load(file)

        joint_cam = np.array(data_info[hand_types[hand_type]]['cam_coord'])
        if data_info[hand_types[hand_type]]['joint_valid'] is not None:
            joint_valid = np.array(data_info[hand_types[hand_type]]['joint_valid'])
        else:
            joint_valid = np.ones([21])
        joint_cam_list.append(joint_cam)
        joint_cam_valid_list.append(joint_valid)
        R_seq_list.append(R)
        T_seq_list.append(T)

    joint_cam = np.stack(joint_cam_list, axis=0)
    joint_valid = np.stack(joint_cam_valid_list, axis=0)
    hand_joint_in = cam2world(joint_cam, R, T)

    return hand_joint_in, joint_valid

def cam2world(cam_coord, R, T):
    cam_coord = torch.from_numpy(cam_coord).float()
    batch_size = cam_coord.size(0)
    R = torch.from_numpy(R).float().repeat(batch_size, 1, 1)
    T = torch.from_numpy(T).float().repeat(batch_size, 1, 1)

    cam_coord = cam_coord - T
    world_coord = torch.matmul(torch.inverse(R), cam_coord.permute(0,2,1)).permute(0,2,1)
    return world_coord.numpy()

if __name__ == '__main__':
    # seed = 42
    # set_seed(seed)
    root_path = '/mnt/sda1/pfren/SeqHand/pkl'
    dataset = 'UmeTrack_synthetic'
    min_seq_len = 30
    save_path = '/mnt/sda1/pfren/SeqHand/npz_%d/%s'%(min_seq_len, dataset)
    os.makedirs(save_path, exist_ok=True)
    generator_seq_all(root_path, save_path, dataset, min_seq_len)
