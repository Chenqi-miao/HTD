import os
import os.path as osp
import numpy as np
import pickle
from dataset.create_data import load_gt, load_input

def generator_seq_all(root_path, save_path, dataset, min_seq_len=1):
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

if __name__ == '__main__':
    # seed = 42
    # set_seed(seed)
    root_path = '/mnt/sda1/pfren/SeqHand/pkl'
    dataset = 'DexYCB'
    save_path = '/mnt/sda1/pfren/SeqHand/npz_single/%s'%(dataset)
    os.makedirs(save_path, exist_ok=True)
    generator_seq_all(root_path, save_path, dataset)
