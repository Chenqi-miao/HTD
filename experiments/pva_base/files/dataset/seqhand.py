import time
import torch
import pickle
import cv2 as cv
import numpy as np
import os.path as osp

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import random
from utils.visualize import draw_2d_skeleton

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.video_utils import SingleJointUtils
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from utils.visualize import draw_2d_skeleton
cv.setNumThreads(0)
cv.ocl.setUseOpenCL(False)
import zipfile
from dataset.data_aug import Augmenter
import numpy.lib.format as np_format


def fix_obman_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].th_shapedirs[:, 0, :] - mano_layer['right'].th_shapedirs[:, 0, :])) < 1:
        mano_layer['left'].th_shapedirs[:, 0, :] *= -1


def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1


def random_grouping(my_list, N):
    # 复制并随机打乱列表
    shuffled_list = my_list.copy()
    random.shuffle(shuffled_list)

    # 按每组N个元素进行分组，并且抛弃不足N个元素的组
    groups = [shuffled_list[i:i + N] for i in range(0, len(shuffled_list), N) if len(shuffled_list[i:i + N]) == N]

    return groups

# Json 数据版本
class SeqHand_Json(Dataset):
    def __init__(self, data_path, dataset_list, min_seq_len=30, view_num=3, seq_len=15, aug=True, data_num=None):
        self.min_seq_len = min_seq_len
        self.seq_len = seq_len
        self.view_num = view_num
        self.augmenter = SingleJointUtils()
        self.data_path = os.path.join(data_path, 'pkl')
        self.img_path = os.path.join(data_path, 'img')

        # 'DexYCB', 'ReInterHand', 'InterHand_train', 'InterHand_test','InterHand_val','UmeTrack_synthetic','UmeTrack_real'
        self.dataset_list = ['InterHand_train']
        self.hand_types = ['right', 'left']

        self.expand = 3
        self.aug = aug
        self.data_num = data_num
        print('Loading Train data ...')
        start_time = time.time()
        self.generator_seq()
        end_time = time.time()
        total_seconds = end_time - start_time
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        print(f"Loading Time：{minutes}:{seconds}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 从 idx 开始，往后找第一个有效样本
        while idx < len(self.data_list):
            data = self._load_sample(idx)
            if data is not None:
                return data
            idx += 1
        raise RuntimeError("Dataset has no valid samples.")

    def _load_sample(self, idx):
        data_dict = self.load_seq(idx)
        # R, T = data_dict['R'], data_dict['T']  # [C T 3 3] [C T 3]
        # joint_cams_in = data_dict['joint_cams']  # [C T J 3]
        # joint_cams_valid = data_dict['joint_cams_valid'][..., np.newaxis]  # [C T J]
        # joint_valid_gt = data_dict['joint_valid_gt']  # [T J 3]
        # joint_world_gt = data_dict['joint_world_gt']  # [T J 3]
        # joint_type = data_dict['joint_type'].reshape([-1, 1, 1])  # [T 1 1]
        #
        # R, T = R[:, np.newaxis, np.newaxis], T[:, np.newaxis, np.newaxis]
        # R, T = np.tile(R, (1, self.seq_len, 21, 1, 1)), np.tile(T, (1, self.seq_len, 21, 1))
        # joint_world_in = cam2world(joint_cams_in.reshape([-1, 3]), R.reshape([-1, 3, 3]), T.reshape([-1, 3]))
        # joint_world_in = joint_world_in.reshape([self.view_num, self.seq_len, 21, 3])
        # joint_world_in = joint_world_in * joint_cams_valid
        # continuous_val = self.frame_consecutive(joint_world_gt)
        #
        # hand_joint_gt = torch.from_numpy(joint_world_gt).float()
        # hand_joint_in = torch.from_numpy(joint_world_in).float()
        # joint_type = torch.from_numpy(joint_type).float()


        R, T = torch.from_numpy(data_dict['R']).float(), torch.from_numpy(data_dict['T']).float().reshape([self.view_num, self.seq_len, 1, 3])  # [C T 3 3] [C T 3]
        joint_cam_in = torch.from_numpy(data_dict['joint_cams']).float()  # [C T J 3]
        joint_cams_valid = torch.from_numpy(data_dict['joint_cams_valid']).float()  # [C T J]
        joint_valid_gt = torch.from_numpy(data_dict['joint_valid_gt']).float()  # [T J 3]
        joint_cams_valid = self.size_valid(joint_cam_in).unsqueeze(-1) * joint_cams_valid

        joint_type = torch.from_numpy(data_dict['joint_type']).float().reshape([-1, 1, 1])  # [T 1 1]
        hand_joint_gt = torch.from_numpy(data_dict['joint_world_gt']).float()  # [T J 3]

        hand_joint_in = cam2world(joint_cam_in.reshape([-1,21,3]), R.reshape([-1, 3, 3]), T.reshape([-1,1,3]))
        hand_joint_in = hand_joint_in.reshape([self.view_num, self.seq_len, 21, 3])
        continuous_val = self.frame_consecutive(hand_joint_gt)

        center_gt = hand_joint_gt[..., 9:10, :].clone()
        center_in = hand_joint_in[..., 9:10, :].clone()

        [hand_joint_gt, hand_joint_in] = self.joint_flip([hand_joint_gt, hand_joint_in], [center_gt, center_in], [joint_type.eq(1), joint_type.eq(1)])

        split_joints = [split_tensor.squeeze(0) for split_tensor in torch.split(hand_joint_in, 1)]
        split_centers = [split_tensor.squeeze(0) for split_tensor in torch.split(center_in, 1)]
        all_joint_list = [hand_joint_gt] + split_joints
        all_center_list = [center_gt] + split_centers

        # 序列的整体增强，增加数据多样性
        all_joint_list = self.augmenter.seq_aug(all_joint_list, all_center_list)
        hand_joint_in_list = all_joint_list[1:]
        hand_joint_gt = all_joint_list[0]

        # 输入数据的增强
        hand_joint_in_list = self.augmenter.joint_aug(hand_joint_in_list, split_centers)
        hand_joint_in = torch.stack(hand_joint_in_list, dim=0)

        # 将世界坐标转换为相机坐标
        hand_joint_in_cam = world2cam(hand_joint_in.reshape([-1,21,3]), R.reshape([-1, 3, 3]), T.reshape([-1,1,3]))
        hand_joint_in_cam = hand_joint_in_cam.reshape([self.view_num, self.seq_len, 21, 3])

        hand_joint_gt_world =  hand_joint_gt.unsqueeze(0).repeat([self.view_num, 1, 1, 1])
        hand_joint_gt_cam = world2cam(hand_joint_gt_world.reshape([-1,21,3]), R.reshape([-1, 3, 3]), T.reshape([-1,1,3]))
        hand_joint_gt_cam = hand_joint_gt_cam.reshape([self.view_num, self.seq_len, 21, 3])

        # 获取center坐标
        center_gt_world = hand_joint_gt_world[..., 9:10, :].clone()
        center_gt_cam = hand_joint_gt_cam[..., 9:10, :].clone()

        if joint_valid_gt.sum() == 0 or  joint_cams_valid.sum() == 0:
            return None
        else:
            inputs = {'joint_world': np.float32(hand_joint_in), 'joint_cam': np.float32(hand_joint_in_cam)}
            targets = {'joint_world': np.float32(hand_joint_gt_world), 'joint_cam': np.float32(hand_joint_gt_cam)}
            meta_info = {"center_world": np.float32(center_gt_world), "center_cam": np.float32(center_gt_cam),
                         'continuous_val': np.float32(continuous_val), 'joint_gt_val': np.float32(joint_valid_gt),
                         'joint_in_val': np.float32(joint_cams_valid)}
            return inputs, targets, meta_info

    # 检测估计的异常帧 B T J 3
    def size_valid(self, seq):
        seq_min = torch.min(seq[..., :2], dim=2)[0]
        seq_max = torch.max(seq[..., :2], dim=2)[0]
        mask = (seq_max - seq_min).lt(50).float().sum(dim=-1).eq(2)
        return ~mask

    def frame_consecutive(self, seq):
        seq_val = torch.zeros([seq.size(0)])
        diff = (seq[1:] - seq[:-1])
        diff = torch.mean(torch.sqrt((diff * diff).sum(-1)), dim=-1)
        seq_val[:-1] += (diff > 30)
        seq_val[1:] += (diff > 30)
        return seq_val == 0

    # 按照用户和动作序列选取数据
    # 为了进行数据增强，单条数据量大于seq_len的数量
    def generator_seq(self):
        self.data_list = []
        self.info_dict = {}
        for dataset in self.dataset_list:
            self.info_dict[dataset] = {}
            sum_file = os.path.join(self.data_path, '%s.pkl' % (dataset))
            with open(sum_file, 'rb') as file:
                data_dict = pickle.load(file)
            for capture in data_dict.keys():
                self.info_dict[dataset][capture] = {}
                for seq in data_dict[capture].keys():

                    img_name_list = data_dict[capture][seq]['img_name_list']
                    seq_img_num = len(img_name_list)
                    if seq_img_num < self.min_seq_len:
                        continue
                    cam_name_list = data_dict[capture][seq]['cam_name_list']
                    cam_group_list = random_grouping(cam_name_list, self.view_num)
                    for cam_name_select in cam_group_list:
                        # 对方法进行随机采样
                        method_name_list = data_dict[capture][seq]['method_name_list']
                        method_select = random.sample(method_name_list, 1)[0]

                        data_path = os.path.join(self.data_path, dataset, capture, seq)
                        anno_file = osp.join(data_path, 'anno_info.pkl')
                        meta_file = osp.join(data_path, 'meta_info.pkl')
                        with open(meta_file, 'rb') as file:
                            meta_info = pickle.load(file)
                        with open(anno_file, 'rb') as file:
                            annos_info = pickle.load(file)
                        self.info_dict[dataset][capture][seq] = {'meta_info': meta_info, 'annos_info': annos_info}

                        first_frame = np.random.randint(0, self.seq_len)
                        seq_num = int(np.ceil((seq_img_num - first_frame) / self.seq_len))
                        seq_ids = np.arange(seq_num + 1) * self.seq_len + first_frame
                        for ii in range(seq_num):
                            img_idx = np.arange(seq_ids[ii] - self.seq_len // 2, seq_ids[ii] + self.seq_len // 2 + 1)
                            img_idx_list = np.clip(img_idx, a_min=0, a_max=seq_img_num - 1)
                            img_names = []
                            for img_idx in img_idx_list:
                                img_names.append(img_name_list[img_idx])
                            file_dict = {'dataset': dataset, 'capture': capture, 'seq': seq, 'method': method_select, 'cams': cam_name_select, 'imgs': img_names}
                            self.data_list.append(file_dict)
                            if self.data_num is not None:
                                if len(self.data_list) > self.data_num:
                                    return 0

    def load_seq(self, idx):
        data_dict = self.data_list[idx]
        dataset, capture_name, seq_name, method_name = data_dict['dataset'], data_dict['capture'], data_dict['seq'], data_dict['method']
        cam_name_list, img_id_list = data_dict['cams'], data_dict['imgs']
        cam_name_list=[cam_name_list[0]]*self.view_num
        data_path = os.path.join(self.data_path, dataset, capture_name, seq_name)

        meta_info = self.info_dict[dataset][capture_name][seq_name]['meta_info']
        annos_info = self.info_dict[dataset][capture_name][seq_name]['annos_info']

        # 加载全局手部坐标GT和手部类型
        joint_world_list = []
        joint_valid_list = []
        hand_type_list = []
        for img_name in img_id_list:
            if meta_info['frame_info'][img_name]['hand_type'] in ['right', 'two', 'interacting']:
                hand_type = 0  # 'right'
            else:
                hand_type = 1  # 'left'

            if annos_info[img_name][self.hand_types[hand_type]]['world_coord'] is not None:
                joint_world = np.array(annos_info[img_name][self.hand_types[hand_type]]['world_coord'], np.float64)
            else:
                joint_world = np.zeros([21, 3])
            if annos_info[img_name][self.hand_types[hand_type]]['joint_valid'] is not None:
                joint_valid = np.array(annos_info[img_name][self.hand_types[hand_type]]['joint_valid'], np.float64)
            else:
                joint_valid = np.ones([21])
            joint_world_list.append(joint_world)
            joint_valid_list.append(joint_valid)
            hand_type_list.append(hand_type)
        joint_world_gt = np.stack(joint_world_list, axis=0)
        joint_valid_gt = np.stack(joint_valid_list, axis=0)
        joint_type = np.stack(hand_type_list, axis=0)

        # 加载输入数据
        joint_cams_list = []
        joint_cams_valid_list = []
        R_list, T_list = [], []
        focal_list, princpt_list = [], []
        for cam_name in cam_name_list:
            cam_para = meta_info['cam_params'][cam_name]
            R, T = np.array(cam_para['R']), np.array(cam_para['T'])
            focal, princpt = np.array(cam_para['focal']), np.array(cam_para['princpt'])
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

                joint_cam = np.array(data_info[self.hand_types[hand_type]]['cam_coord'])
                if data_info[self.hand_types[hand_type]]['joint_valid'] is not None:
                    joint_valid = np.array(data_info[self.hand_types[hand_type]]['joint_valid'])
                else:
                    joint_valid = np.ones([21])
                joint_cam_list.append(joint_cam)
                joint_cam_valid_list.append(joint_valid)
                R_seq_list.append(R)
                T_seq_list.append(T)

            joint_cam = np.stack(joint_cam_list, axis=0)
            joint_cam_valid = np.stack(joint_cam_valid_list, axis=0)
            joint_cams_list.append(joint_cam)
            joint_cams_valid_list.append(joint_cam_valid)
            R_list.append(np.stack(R_seq_list, axis=0))
            T_list.append(np.stack(T_seq_list, axis=0))
            focal_list.append(focal)
            princpt_list.append(princpt)

        joint_cams = np.stack(joint_cams_list, axis=0)
        joint_cams_valid = np.stack(joint_cams_valid_list, axis=0)
        R = np.stack(R_list, axis=0)
        T = np.stack(T_list, axis=0)
        focal = np.stack(focal_list, axis=0)
        princpt = np.stack(princpt_list, axis=0)
        global_val = joint_valid_gt.sum() > len(img_id_list) * 0.6

        data_dict = {
            'R': R,
            'T': T,
            'focal': focal,
            'princpt': princpt,
            'joint_cams': joint_cams,
            'joint_cams_valid': joint_cams_valid,
            'joint_world_gt': joint_world_gt,
            'joint_valid_gt': joint_valid_gt,
            'joint_type': joint_type,
            'global_val': global_val,
        }

        return data_dict

    def evaluate(self, outs, targets, meta_info):
        cube = 1
        device = outs['pd_joint_xyz_right'].device

        joints_left_gt = targets['joint_3d_left'].to(device) * cube
        verts_left_gt = targets['mesh_3d_left'].to(device) * cube
        joints_right_gt = targets['joint_3d_right'].to(device) * cube
        verts_right_gt = targets['mesh_3d_right'].to(device) * cube

        root_left_gt = joints_left_gt[:, :, 9:10]
        root_right_gt = joints_right_gt[:, :, 9:10]
        length_left_gt = torch.linalg.norm(joints_left_gt[:, :, 9] - joints_left_gt[:, :, 0], dim=-1)
        length_right_gt = torch.linalg.norm(joints_right_gt[:, :, 9] - joints_right_gt[:, :, 0], dim=-1)
        joints_left_gt = joints_left_gt - root_left_gt
        verts_left_gt = verts_left_gt - root_left_gt
        joints_right_gt = joints_right_gt - root_right_gt
        verts_right_gt = verts_right_gt - root_right_gt

        mesh_3d_left = outs['pd_mesh_xyz_left'] * cube
        mesh_3d_right = outs['pd_mesh_xyz_right'] * cube
        joint_3d_left = outs['pd_joint_xyz_left'] * cube
        joint_3d_right = outs['pd_joint_xyz_right'] * cube

        root_left_pred = joint_3d_left[:, :, 9:10]
        root_right_pred = joint_3d_right[:, :, 9:10]
        length_left_pred = torch.linalg.norm(joint_3d_left[:, :, 9] - joint_3d_left[:, :, 0], dim=-1)
        length_right_pred = torch.linalg.norm(joint_3d_right[:, :, 9] - joint_3d_right[:, :, 0], dim=-1)
        scale_left = (length_left_gt / length_left_pred).unsqueeze(-1).unsqueeze(-1)
        scale_right = (length_right_gt / length_right_pred).unsqueeze(-1).unsqueeze(-1)

        joints_left_pred = (joint_3d_left - root_left_pred) * scale_left
        verts_left_pred = (mesh_3d_left - root_left_pred) * scale_left
        joints_right_pred = (joint_3d_right - root_right_pred) * scale_right
        verts_right_pred = (mesh_3d_right - root_right_pred) * scale_right

        joint_left_error = torch.linalg.norm((joints_left_pred - joints_left_gt), ord=2, dim=-1)
        joint_left_error = joint_left_error.detach().cpu().numpy().mean()

        joint_right_error = torch.linalg.norm((joints_right_pred - joints_right_gt), ord=2, dim=-1)
        joint_right_error = joint_right_error.detach().cpu().numpy().mean()

        vert_left_error = torch.linalg.norm((verts_left_pred - verts_left_gt), ord=2, dim=-1)
        vert_left_error = vert_left_error.detach().cpu().numpy().mean()

        vert_right_error = torch.linalg.norm((verts_right_pred - verts_right_gt), ord=2, dim=-1)
        vert_right_error = vert_right_error.detach().cpu().numpy().mean()

        return joint_left_error, joint_right_error, vert_left_error, vert_right_error

    def joint_flip(self, joint_list, center_list, flip_flag_list):
        joint_aug_list = []
        for joint, center, flag in zip(joint_list, center_list, flip_flag_list):
            joint_flip = joint.clone() - center
            joint_flip[..., 0] *= -1
            joint_flip = joint_flip + center
            joint_aug_list.append(joint_flip * flag.float() + joint * (1 - flag.float()))
        return joint_aug_list


class SeqHandTest_Json(Dataset):
    def __init__(self, data_path, dataset_list=None, min_seq_len=15, view_num=3, seq_len=15, data_num=None, vis=True):
        self.min_seq_len = min_seq_len # 最小的序列长度
        self.valid_frame = 0
        self.seq_len = seq_len
        self.view_num = view_num
        self.augmenter = SingleJointUtils()
        self.data_path = os.path.join(data_path, 'pkl')
        self.img_path = os.path.join(data_path, 'img')
        self.vis = vis

        # 'DexYCB', 'ReInterHand', 'InterHand_train', 'InterHand_test','InterHand_val','UmeTrack_synthetic','UmeTrack_real'
        self.dataset_list = ['InterHand_val']
        self.hand_types = ['right', 'left']
        self.data_num = data_num
        print('Loading Test data ...')
        start_time = time.time()
        self.generator_seq()
        end_time = time.time()
        total_seconds = end_time - start_time
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        print(f"Loading Time：{minutes}:{seconds}")


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 从 idx 开始，往后找第一个有效样本
        while idx < len(self.data_list):
            data = self._load_sample(idx)
            if data is not None:
                return data
            idx += 1
        raise RuntimeError("Dataset has no valid samples.")

    def _load_sample(self, idx):
        data_dict = self.load_seq(idx)

        R, T = torch.from_numpy(data_dict['R']).float(), torch.from_numpy(data_dict['T']).float().reshape([self.view_num, self.seq_len, 1, 3])  # [C T 3 3] [C T 3]
        focal = torch.from_numpy(data_dict['focal']).float().reshape([-1, 2]).repeat(self.seq_len, 1)
        princpt = torch.from_numpy(data_dict['princpt']).float().reshape([-1, 2]).repeat(self.seq_len, 1)
        cam_para = torch.cat((focal,princpt), dim=-1)

        joint_cam_in = torch.from_numpy(data_dict['joint_cams']).float()  # [C T J 3]
        joint_cams_valid = torch.from_numpy(data_dict['joint_cams_valid']).float()  # [C T J]
        joint_valid_gt = torch.from_numpy(data_dict['joint_valid_gt']).float()  # [T J 3]
        data_repeat = torch.from_numpy(data_dict['data_repeat']).float() # [T J]
        images = data_dict['images']  # [T J]
        joint_cams_valid = self.size_valid(joint_cam_in).unsqueeze(-1) * joint_cams_valid

        joint_type = torch.from_numpy(data_dict['joint_type']).float().reshape([-1, 1, 1])  # [T 1 1]
        hand_joint_gt = torch.from_numpy(data_dict['joint_world_gt']).float()  # [T J 3]

        hand_joint_in = cam2world(joint_cam_in.reshape([-1,21,3]), R.reshape([-1, 3, 3]), T.reshape([-1,1,3]))
        hand_joint_in = hand_joint_in.reshape([self.view_num, self.seq_len, 21, 3])
        continuous_val = self.frame_consecutive(hand_joint_gt)


        center_gt = hand_joint_gt[..., 9:10, :].clone()
        center_in = hand_joint_in[..., 9:10, :].clone()

        [hand_joint_gt, hand_joint_in] = self.joint_flip([hand_joint_gt, hand_joint_in], [center_gt, center_in], [joint_type.eq(1), joint_type.eq(1)])

        # 将世界坐标转换为相机坐标
        hand_joint_in_cam = world2cam(hand_joint_in.reshape([-1,21,3]), R.reshape([-1, 3, 3]), T.reshape([-1,1,3]))
        hand_joint_in_cam = hand_joint_in_cam.reshape([self.view_num, self.seq_len, 21, 3])

        split_joints = [split_tensor.squeeze(0) for split_tensor in torch.split(hand_joint_in_cam, 1)]
        split_centers = [split_tensor.squeeze(0) for split_tensor in torch.split(hand_joint_in_cam[..., 9:10, :].clone(), 1)]
        [hand_joint_in_cam] = self.augmenter.joint_aug(split_joints, split_centers)
        hand_joint_in_cam = hand_joint_in_cam.reshape([self.view_num, self.seq_len, 21, 3])

        hand_joint_gt_world =  hand_joint_gt.unsqueeze(0).repeat([self.view_num, 1, 1, 1])
        hand_joint_gt_cam = world2cam(hand_joint_gt_world.reshape([-1,21,3]), R.reshape([-1, 3, 3]), T.reshape([-1,1,3]))
        hand_joint_gt_cam = hand_joint_gt_cam.reshape([self.view_num, self.seq_len, 21, 3])

        # 将相机坐标转换为图像坐标
        hand_joint_gt_pixel = cam2pixel(hand_joint_gt_cam.reshape([-1,21,3]), cam_para)
        hand_joint_gt_pixel = hand_joint_gt_pixel.reshape([self.view_num, self.seq_len, 21, 3])

        hand_joint_in_pixel = cam2pixel(hand_joint_in_cam.reshape([-1,21,3]), cam_para)
        hand_joint_in_pixel = hand_joint_in_pixel.reshape([self.view_num, self.seq_len, 21, 3])

        # 获取center坐标
        center_gt_world = hand_joint_gt_world[..., 9:10, :].clone()
        center_gt_cam = hand_joint_gt_cam[..., 9:10, :].clone()

        if joint_valid_gt.sum() == 0 or joint_cams_valid.sum() == 0:
            return None
        else:
            inputs = {'joint_world': np.float32(hand_joint_in),
                      'joint_xyz': np.float32(hand_joint_in_cam),
                      'joint_pixel': np.float32(hand_joint_in_pixel),
                      'images': np.float32(images)
                      }
            targets = {'joint_world': np.float32(hand_joint_gt_world),
                       'joint_xyz': np.float32(hand_joint_gt_cam),
                       'joint_pixel': np.float32(hand_joint_gt_pixel),
                       }
            meta_info = {"center_world": np.float32(center_gt_world),
                         "center_xyz": np.float32(center_gt_cam),
                         "cam_para": np.float32(cam_para),
                         'continuous_val': np.float32(continuous_val), 'joint_gt_val': np.float32(joint_valid_gt),
                         'joint_in_val': np.float32(joint_cams_valid), 'data_repeat': np.float32(data_repeat)}
            return inputs, targets, meta_info

    # 检测异常帧
    def frame_consecutive(self, seq):
        seq_val = torch.zeros([seq.size(0)])
        diff = (seq[1:] - seq[:-1])
        diff = torch.mean(torch.sqrt((diff * diff).sum(-1)), dim=-1)
        seq_val[:-1] += (diff > 30)
        seq_val[1:] += (diff > 30)
        return seq_val == 0

    # 检测估计的异常帧 B T J 3
    def size_valid(self, seq):
        seq_min = torch.min(seq[..., :2], dim=2)[0]
        seq_max = torch.max(seq[..., :2], dim=2)[0]
        mask = (seq_max - seq_min).lt(50).float().sum(dim=-1).eq(2)
        return ~mask

    # 按照用户和动作序列选取数据
    def generator_seq(self):
        self.data_list = []
        self.info_dict = {}
        for dataset in self.dataset_list:
            self.info_dict[dataset] = {}
            sum_file = os.path.join(self.data_path, '%s.pkl' % (dataset))
            with open(sum_file, 'rb') as file:
                data_dict = pickle.load(file)
            for capture in data_dict.keys():
                self.info_dict[dataset][capture] = {}
                # for seq in data_dict[capture].keys():
                for seq in ['ROM02_Interaction_2_Hand']:
                    cam_name_list = data_dict[capture][seq]['cam_name_list']
                    img_name_list = data_dict[capture][seq]['img_name_list']
                    seq_img_num = len(img_name_list)
                    # 为保证数据的有效性，单条数据量要大于min_seq_len
                    if seq_img_num < self.min_seq_len:
                        continue
                    data_path = os.path.join(self.data_path, dataset, capture, seq)
                    anno_file = osp.join(data_path, 'anno_info.pkl')
                    meta_file = osp.join(data_path, 'meta_info.pkl')
                    with open(meta_file, 'rb') as file:
                        meta_info = pickle.load(file)
                    with open(anno_file, 'rb') as file:
                        annos_info = pickle.load(file)
                    # if meta_info['frame_info'][img_name_list[0]]['hand_type'] == 'left':
                    #     break
                    if meta_info['frame_info'][img_name_list[0]]['hand_type'] == 'left':
                        break
                    for cam_name_select in cam_name_list:
                        method_name_list = data_dict[capture][seq]['method_name_list']
                        method_select = random.sample(method_name_list, 1)[0]

                        self.info_dict[dataset][capture][seq] = {'meta_info': meta_info, 'annos_info': annos_info}
                        seq_num = int(np.ceil(seq_img_num / self.seq_len))
                        cur_id = 0
                        for ii in range(seq_num):
                            img_idx = np.arange(cur_id, cur_id + self.seq_len)
                            img_idx_list = np.clip(img_idx, a_min=0, a_max=seq_img_num - 1)
                            data_repeat = (img_idx_list - np.arange(cur_id, cur_id + self.seq_len))==0
                            img_names = []
                            for img_idx in img_idx_list:
                                img_names.append(img_name_list[img_idx])
                            file_dict = {'dataset': dataset, 'capture': capture, 'seq': seq, 'method': method_select,
                                         'cams': [cam_name_select],
                                         'imgs': img_names,
                                         'data_repeat': data_repeat # 为了保证测试的一致性，repeat数据从评估中删除
                                         }
                            self.data_list.append(file_dict)
                            cur_id = cur_id + self.seq_len
                            if self.data_num is not None:
                                if len(self.data_list) > self.data_num:
                                    return 0
    # 加载数据时需要判断输入数据和GT是否都有效

    # 加载数据
    def load_seq(self, idx):
        data_dict = self.data_list[idx]
        dataset, capture_name, seq_name, method_name = data_dict['dataset'], data_dict['capture'], data_dict['seq'], data_dict['method']
        data_repeat = data_dict['data_repeat']
        cam_name_list, img_id_list = data_dict['cams'], data_dict['imgs']
        cam_name_list=[cam_name_list[0]]*self.view_num
        data_path = os.path.join(self.data_path, dataset, capture_name, seq_name)

        meta_info = self.info_dict[dataset][capture_name][seq_name]['meta_info']
        annos_info = self.info_dict[dataset][capture_name][seq_name]['annos_info']

        # 加载全局手部坐标GT和手部类型
        joint_world_list = []
        joint_valid_list = []
        hand_type_list = []
        for img_name in img_id_list:
            if meta_info['frame_info'][img_name]['hand_type'] in ['right', 'two', 'interacting']:
                hand_type = 0  # 'right'
            else:
                hand_type = 1  # 'left'
            if annos_info[img_name][self.hand_types[hand_type]]['world_coord'] is not None:
                joint_world = np.array(annos_info[img_name][self.hand_types[hand_type]]['world_coord'], np.float64)
            else:
                joint_world = np.zeros([21, 3])
            if annos_info[img_name][self.hand_types[hand_type]]['joint_valid'] is not None:
                joint_valid = np.array(annos_info[img_name][self.hand_types[hand_type]]['joint_valid'], np.float64)
            else:
                joint_valid = np.ones([21])
            joint_world_list.append(joint_world)
            joint_valid_list.append(joint_valid)
            hand_type_list.append(hand_type)
        joint_world_gt = np.stack(joint_world_list, axis=0)
        joint_valid_gt = np.stack(joint_valid_list, axis=0)
        joint_type = np.stack(hand_type_list, axis=0)

        # 加载输入数据
        joint_cams_list = []
        joint_cams_valid_list = []
        R_list, T_list = [], []
        focal_list, princpt_list = [], []
        for cam_name in cam_name_list:
            cam_para = meta_info['cam_params'][cam_name]
            R, T = np.array(cam_para['R']), np.array(cam_para['T'])
            focal, princpt = np.array(cam_para['focal']), np.array(cam_para['princpt'])
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

                joint_cam = np.array(data_info[self.hand_types[hand_type]]['cam_coord'])
                if data_info[self.hand_types[hand_type]]['joint_valid'] is not None:
                    joint_valid = np.array(data_info[self.hand_types[hand_type]]['joint_valid'])
                else:
                    joint_valid = np.ones([21])
                joint_cam_list.append(joint_cam)
                joint_cam_valid_list.append(joint_valid)
                R_seq_list.append(R)
                T_seq_list.append(T)

            joint_cam = np.stack(joint_cam_list, axis=0)
            joint_cam_valid = np.stack(joint_cam_valid_list, axis=0)
            joint_cams_list.append(joint_cam)
            joint_cams_valid_list.append(joint_cam_valid)
            R_list.append(np.stack(R_seq_list, axis=0))
            T_list.append(np.stack(T_seq_list, axis=0))
            focal_list.append(focal)
            princpt_list.append(princpt)
        joint_cams = np.stack(joint_cams_list, axis=0)
        joint_cams_valid = np.stack(joint_cams_valid_list, axis=0)
        R = np.stack(R_list, axis=0)
        T = np.stack(T_list, axis=0)
        focal = np.stack(focal_list, axis=0)
        princpt = np.stack(princpt_list, axis=0)
        global_val = joint_valid_gt.sum() > len(img_id_list) * 0.6

        # 加载输入图像（用于可视化）
        images = np.zeros([1, 512, 334, 3])
        if self.vis:
            img_list = []
            for cam_name in cam_name_list:
                for img_name in img_id_list:
                    img_path = os.path.join(self.img_path, dataset, capture_name, seq_name, cam_name, 'image%s.jpg' % (img_name))
                    image = cv.imread(img_path)
                    if image is not None:
                        img_list.append(image)
                    else:
                        break
            if len(img_list)>1:
                images = np.stack(img_list, axis=0)
        data_dict = {
            'R': R,
            'T': T,
            'focal': focal,
            'princpt': princpt,
            'joint_cams': joint_cams,
            'joint_cams_valid': joint_cams_valid,
            'joint_world_gt': joint_world_gt,
            'joint_valid_gt': joint_valid_gt,
            'joint_type': joint_type,
            'global_val': global_val,
            'data_repeat':data_repeat,
            'images':images
        }

        return data_dict

    def evaluate(self, outs, inputs, targets, meta_info):
        batch_size, view_num, seq_len, joint_num, _ = outs['pd_joint_xyz'].size()
        device = targets['joint_xyz'].device
        con_val = meta_info['continuous_val'].view(batch_size, view_num, seq_len, 1)
        gt_val = meta_info['joint_gt_val'].view(batch_size, view_num, seq_len, 21)
        data_repeat = meta_info['data_repeat'].view(batch_size, view_num, seq_len, 1).repeat(1, 1, 1, 21)
        gt_joint_xyz = targets['joint_xyz']
        in_joint_xyz = inputs['joint_xyz']
        pd_joint_xyz = outs['pd_joint_xyz'].to(device)

        in_error = calculate_error(in_joint_xyz, gt_joint_xyz, data_repeat*gt_val*con_val)
        out_error = calculate_error(pd_joint_xyz, gt_joint_xyz, data_repeat*gt_val*con_val)
        return in_error, out_error

    def joint_flip(self, joint_list, center_list, flip_flag_list):
        joint_aug_list = []
        for joint, center, flag in zip(joint_list, center_list, flip_flag_list):
            joint_flip = joint.clone() - center
            joint_flip[..., 0] *= -1
            joint_flip = joint_flip + center
            joint_aug_list.append(joint_flip * flag.float() + joint * (1 - flag.float()))
        return joint_aug_list


def load_npz_array(zip_file, name):
    """直接读取npz中的npy数组，不经过np.load"""
    with zip_file.open(name) as f:
        return np_format.read_array(f, allow_pickle=True)

# # NPZ 整体数据版本
# class SeqHand(Dataset):
#     def __init__(self, data_path, dataset_list, min_seq_len=30, view_num=1, seq_len=15, aug=True, data_num=None):
#         self.min_seq_len = min_seq_len
#         self.seq_len = seq_len
#         self.view_num = view_num
#         self.augmenter = SingleJointUtils()
#         self.data_path = os.path.join(data_path, 'npz_single')
#         self.img_path = os.path.join(data_path, 'img')
#
#         # 'DexYCB', 'ReInterHand', 'InterHand_train', 'InterHand_test','InterHand_val','UmeTrack_synthetic','UmeTrack_real'
#         self.dataset_list = ['InterHand_train']
#         self.hand_types = ['right', 'left']
#
#         self.expand = 3
#         self.aug = aug
#         self.data_num = data_num
#         print('Loading Train data ...')
#         start_time = time.time()
#         self.generator_seq()
#         end_time = time.time()
#         total_seconds = end_time - start_time
#         minutes = int(total_seconds // 60)
#         seconds = int(total_seconds % 60)
#         print(f"Loading Time：{minutes}:{seconds}")
#
#     def __len__(self):
#         return len(self.joint_xyz_gt_list)
#
#     def __getitem__(self, idx):
#         # 从 idx 开始，往后找第一个有效样本
#         while idx < len(self.joint_xyz_gt_list):
#             data = self._load_sample(idx)
#             if data is not None:
#                 return data
#             idx += 1
#         raise RuntimeError("Dataset has no valid samples.")
#
#     def _load_sample(self, idx):
#         joint_world_in = torch.from_numpy(self.joint_xyz_in_list[idx]).float()  # [C T J 3]
#         joint_valid_in = torch.from_numpy(self.joint_valid_in_list[idx]).float()  # [C T J]
#         joint_valid_gt = torch.from_numpy(self.joint_valid_gt_list[idx]).float()  # [T J 3]
#         joint_world_valid = self.size_valid(joint_world_in).unsqueeze(-1) * joint_valid_in
#         joint_type = torch.from_numpy(self.hand_type_list[idx]).float().reshape([-1, 1, 1])  # [T 1 1]
#         hand_joint_gt = torch.from_numpy(self.joint_xyz_gt_list[idx]).float()  # [T J 3]
#         hand_joint_in = joint_world_in.reshape([self.view_num, self.seq_len, 21, 3])
#         continuous_val = self.frame_consecutive(hand_joint_gt)
#
#         center_gt = hand_joint_gt[..., 9:10, :].clone()
#         center_in = hand_joint_in[..., 9:10, :].clone()
#
#         [hand_joint_gt, hand_joint_in] = self.joint_flip([hand_joint_gt, hand_joint_in], [center_gt, center_in], [joint_type.eq(1), joint_type.eq(1)])
#
#         split_joints = [split_tensor.squeeze(0) for split_tensor in torch.split(hand_joint_in, 1)]
#         split_centers = [split_tensor.squeeze(0) for split_tensor in torch.split(center_in, 1)]
#         all_joint_list = [hand_joint_gt] + split_joints
#         all_center_list = [center_gt] + split_centers
#
#         # 序列的整体增强，增加数据多样性
#         # all_joint_list = self.augmenter.seq_aug(all_joint_list, all_center_list)
#         hand_joint_in_list = all_joint_list[1:]
#         hand_joint_gt = all_joint_list[0]
#
#         # 输入数据的增强
#         # hand_joint_in_list = self.augmenter.joint_aug(hand_joint_in_list, split_centers)
#         hand_joint_in = torch.stack(hand_joint_in_list, dim=0)
#         hand_joint_gt_world =  hand_joint_gt.unsqueeze(0).repeat([self.view_num, 1, 1, 1])
#
#         # 获取center坐标
#         center_gt_world = hand_joint_gt_world[..., 9:10, :].clone()
#
#         if joint_valid_gt.sum() == 0 or  joint_world_valid.sum() == 0:
#             return None
#         else:
#             inputs = {'joint_xyz': np.float32(hand_joint_in)}
#             targets = {'joint_xyz': np.float32(hand_joint_gt_world)}
#             meta_info = {"center_xyz": np.float32(center_gt_world),
#                          'continuous_val': np.float32(continuous_val),
#                          'joint_gt_val': np.float32(joint_valid_gt),
#                          'joint_in_val': np.float32(joint_world_valid)}
#             return inputs, targets, meta_info
#
#     # 检测估计的异常帧 B T J 3
#     def size_valid(self, seq):
#         seq_min = torch.min(seq[..., :2], dim=2)[0]
#         seq_max = torch.max(seq[..., :2], dim=2)[0]
#         mask = (seq_max - seq_min).lt(50).float().sum(dim=-1).eq(2)
#         return ~mask
#
#     def frame_consecutive(self, seq):
#         seq_val = torch.zeros([seq.size(0)])
#         diff = (seq[1:] - seq[:-1])
#         diff = torch.mean(torch.sqrt((diff * diff).sum(-1)), dim=-1)
#         seq_val[:-1] += (diff > 30)
#         seq_val[1:] += (diff > 30)
#         return seq_val == 0
#
#     # 按照用户和动作序列选取数据
#     # 为了进行数据增强，单条数据量大于seq_len的数量
#     def generator_seq(self):
#         self.joint_xyz_gt_list = []
#         self.joint_xyz_in_list = []
#         self.joint_valid_gt_list = []
#         self.joint_valid_in_list = []
#         self.hand_type_list = []
#
#         # gt_joint_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'gt_joint.npz'))
#         # in_joint_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'in_joint.npz'))
#         # hand_type_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'hand_type.npz'))
#         # in_joint_valid_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'in_joint_valid.npz'))
#         # gt_joint_valid_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'gt_joint_valid.npz'))
#         # keys = gt_joint_npz.files  # 将文件属性转换为内存中的列表
#         # data_dict = {key: (gt_joint_npz[key], in_joint_npz[key], gt_joint_valid_npz[key], in_joint_valid_npz[key], hand_type_npz[key]) for key in keys}
#
#         gt_joint_npz = os.path.join(self.data_path, self.dataset_list[0], 'gt_joint.npz')
#         in_joint_npz = os.path.join(self.data_path, self.dataset_list[0], 'in_joint.npz')
#         hand_type_npz = os.path.join(self.data_path, self.dataset_list[0], 'hand_type.npz')
#         in_joint_valid_npz = os.path.join(self.data_path, self.dataset_list[0], 'in_joint_valid.npz')
#         gt_joint_valid_npz = os.path.join(self.data_path, self.dataset_list[0], 'gt_joint_valid.npz')
#
#         with (zipfile.ZipFile(gt_joint_npz, 'r') as gt_zip, zipfile.ZipFile(in_joint_npz, 'r') as in_zip,
#               zipfile.ZipFile(gt_joint_valid_npz, 'r') as gt_v_zip, zipfile.ZipFile(in_joint_valid_npz, 'r') as in_v_zip,
#               zipfile.ZipFile(hand_type_npz, 'r') as type_zip):
#             keys = gt_zip.namelist()  # 假设每个npz包含同名键
#             data_dict = {key.strip('.npy'): (load_npz_array(gt_zip, key), load_npz_array(in_zip, key), load_npz_array(gt_v_zip, key), load_npz_array(in_v_zip, key), load_npz_array(type_zip, key),) for key in keys}
#
#         for key in data_dict.keys():
#             joint_xyz_gt, joint_xyz_in, joint_valid_gt, joint_valid_in, hand_type = data_dict[key]
#             seq_img_num = joint_xyz_gt.shape[0]
#             first_frame = np.random.randint(0, self.seq_len)
#             seq_num = int(np.ceil((seq_img_num - first_frame) / self.seq_len))
#             seq_ids = np.arange(seq_num + 1) * self.seq_len + first_frame
#             for ii in range(seq_num):
#                 img_idx = np.arange(seq_ids[ii] - self.seq_len // 2, seq_ids[ii] + self.seq_len // 2 + 1)
#                 img_idx_clip = np.clip(img_idx, a_min=0, a_max=seq_img_num - 1)
#                 self.joint_xyz_gt_list.append(joint_xyz_gt[img_idx_clip])
#                 self.joint_xyz_in_list.append(joint_xyz_in[img_idx_clip])
#                 self.joint_valid_gt_list.append(joint_valid_gt[img_idx_clip])
#                 self.joint_valid_in_list.append(joint_valid_in[img_idx_clip])
#                 self.hand_type_list.append(hand_type[img_idx_clip])
#
#     def evaluate(self, outs, targets, meta_info):
#         cube = 1
#         device = outs['pd_joint_xyz_right'].device
#
#         joints_left_gt = targets['joint_3d_left'].to(device) * cube
#         verts_left_gt = targets['mesh_3d_left'].to(device) * cube
#         joints_right_gt = targets['joint_3d_right'].to(device) * cube
#         verts_right_gt = targets['mesh_3d_right'].to(device) * cube
#
#         root_left_gt = joints_left_gt[:, :, 9:10]
#         root_right_gt = joints_right_gt[:, :, 9:10]
#         length_left_gt = torch.linalg.norm(joints_left_gt[:, :, 9] - joints_left_gt[:, :, 0], dim=-1)
#         length_right_gt = torch.linalg.norm(joints_right_gt[:, :, 9] - joints_right_gt[:, :, 0], dim=-1)
#         joints_left_gt = joints_left_gt - root_left_gt
#         verts_left_gt = verts_left_gt - root_left_gt
#         joints_right_gt = joints_right_gt - root_right_gt
#         verts_right_gt = verts_right_gt - root_right_gt
#
#         mesh_3d_left = outs['pd_mesh_xyz_left'] * cube
#         mesh_3d_right = outs['pd_mesh_xyz_right'] * cube
#         joint_3d_left = outs['pd_joint_xyz_left'] * cube
#         joint_3d_right = outs['pd_joint_xyz_right'] * cube
#
#         root_left_pred = joint_3d_left[:, :, 9:10]
#         root_right_pred = joint_3d_right[:, :, 9:10]
#         length_left_pred = torch.linalg.norm(joint_3d_left[:, :, 9] - joint_3d_left[:, :, 0], dim=-1)
#         length_right_pred = torch.linalg.norm(joint_3d_right[:, :, 9] - joint_3d_right[:, :, 0], dim=-1)
#         scale_left = (length_left_gt / length_left_pred).unsqueeze(-1).unsqueeze(-1)
#         scale_right = (length_right_gt / length_right_pred).unsqueeze(-1).unsqueeze(-1)
#
#         joints_left_pred = (joint_3d_left - root_left_pred) * scale_left
#         verts_left_pred = (mesh_3d_left - root_left_pred) * scale_left
#         joints_right_pred = (joint_3d_right - root_right_pred) * scale_right
#         verts_right_pred = (mesh_3d_right - root_right_pred) * scale_right
#
#         joint_left_error = torch.linalg.norm((joints_left_pred - joints_left_gt), ord=2, dim=-1)
#         joint_left_error = joint_left_error.detach().cpu().numpy().mean()
#
#         joint_right_error = torch.linalg.norm((joints_right_pred - joints_right_gt), ord=2, dim=-1)
#         joint_right_error = joint_right_error.detach().cpu().numpy().mean()
#
#         vert_left_error = torch.linalg.norm((verts_left_pred - verts_left_gt), ord=2, dim=-1)
#         vert_left_error = vert_left_error.detach().cpu().numpy().mean()
#
#         vert_right_error = torch.linalg.norm((verts_right_pred - verts_right_gt), ord=2, dim=-1)
#         vert_right_error = vert_right_error.detach().cpu().numpy().mean()
#
#         return joint_left_error, joint_right_error, vert_left_error, vert_right_error
#
#     def joint_flip(self, joint_list, center_list, flip_flag_list):
#         joint_aug_list = []
#         for joint, center, flag in zip(joint_list, center_list, flip_flag_list):
#             joint_flip = joint.clone() - center
#             joint_flip[..., 0] *= -1
#             joint_flip = joint_flip + center
#             joint_aug_list.append(joint_flip * flag.float() + joint * (1 - flag.float()))
#         return joint_aug_list
#
# class SeqHandTest(Dataset):
#     def __init__(self, data_path, dataset_list, min_seq_len=15, view_num=3, seq_len=15, data_num=None):
#         self.min_seq_len = min_seq_len # 最小的序列长度
#         self.valid_frame = 0
#         self.seq_len = seq_len
#         self.view_num = view_num
#         self.augmenter = SingleJointUtils()
#         self.data_path = os.path.join(data_path, 'npz_single')
#         self.img_path = os.path.join(data_path, 'img')
#
#         # 'DexYCB', 'ReInterHand', 'InterHand_train', 'InterHand_test','InterHand_val','UmeTrack_synthetic','UmeTrack_real'
#         self.dataset_list = ['InterHand_test']
#         self.hand_types = ['right', 'left']
#         self.data_num = data_num
#         print('Loading Test data ...')
#         start_time = time.time()
#         self.generator_seq()
#         end_time = time.time()
#         total_seconds = end_time - start_time
#         minutes = int(total_seconds // 60)
#         seconds = int(total_seconds % 60)
#         print(f"Loading Time：{minutes}:{seconds}")
#
#
#     def __len__(self):
#         return len(self.joint_xyz_gt_list)
#
#     def __getitem__(self, idx):
#         # 从 idx 开始，往后找第一个有效样本
#         while idx < len(self.joint_xyz_gt_list):
#             data = self._load_sample(idx)
#             if data is not None:
#                 return data
#             idx += 1
#         raise RuntimeError("Dataset has no valid samples.")
#
#     def _load_sample(self, idx):
#         joint_world_in = torch.from_numpy(self.joint_xyz_in_list[idx]).float()  # [C T J 3]
#         joint_valid_in = torch.from_numpy(self.joint_valid_in_list[idx]).float()  # [C T J]
#         joint_world_valid = self.size_valid(joint_world_in).unsqueeze(-1) * joint_valid_in
#         hand_joint_gt = torch.from_numpy(self.joint_xyz_gt_list[idx]).float()  # [T J 3]
#         joint_valid_gt = torch.from_numpy(self.joint_valid_gt_list[idx]).float()  # [T J 3]
#         data_repeat = torch.from_numpy(self.data_repeat_list[idx]).float()  # [T J 1]
#         joint_type = torch.from_numpy(self.hand_type_list[idx]).float().reshape([-1, 1, 1])  # [T 1 1]
#         hand_joint_in = joint_world_in.reshape([self.view_num, self.seq_len, 21, 3])
#         continuous_val = self.frame_consecutive(hand_joint_gt)
#
#         center_gt = hand_joint_gt[..., 9:10, :].clone()
#         center_in = hand_joint_in[..., 9:10, :].clone()
#
#         [hand_joint_gt, hand_joint_in] = self.joint_flip([hand_joint_gt, hand_joint_in], [center_gt, center_in], [joint_type.eq(1), joint_type.eq(1)])
#
#         # 将世界坐标转换为相机坐标
#         hand_joint_gt_world =  hand_joint_gt.unsqueeze(0).repeat([self.view_num, 1, 1, 1])
#         center_gt_world = hand_joint_gt_world[..., 9:10, :].clone()
#
#         if joint_valid_gt.sum() == 0 or joint_world_valid.sum() == 0:
#             return None
#         else:
#             inputs = {'joint_xyz': np.float32(hand_joint_in)}
#             targets = {'joint_xyz': np.float32(hand_joint_gt_world)}
#             meta_info = {"center_xyz": np.float32(center_gt_world),
#                          'continuous_val': np.float32(continuous_val),
#                          'joint_gt_val': np.float32(joint_valid_gt),
#                          'joint_in_val': np.float32(joint_world_valid),
#                          'data_repeat': np.float32(data_repeat)}
#             return inputs, targets, meta_info
#
#     # 检测异常帧
#     def frame_consecutive(self, seq):
#         seq_val = torch.zeros([seq.size(0)])
#         diff = (seq[1:] - seq[:-1])
#         diff = torch.mean(torch.sqrt((diff * diff).sum(-1)), dim=-1)
#         seq_val[:-1] += (diff > 30)
#         seq_val[1:] += (diff > 30)
#         return seq_val == 0
#
#     # 检测估计的异常帧 B T J 3
#     def size_valid(self, seq):
#         seq_min = torch.min(seq[..., :2], dim=2)[0]
#         seq_max = torch.max(seq[..., :2], dim=2)[0]
#         mask = (seq_max - seq_min).lt(50).float().sum(dim=-1).eq(2)
#         return ~mask
#
#     # 按照用户和动作序列选取数据
#     def generator_seq(self):
#         self.joint_xyz_gt_list = []
#         self.joint_xyz_in_list = []
#         self.joint_valid_gt_list = []
#         self.joint_valid_in_list = []
#         self.hand_type_list = []
#         self.data_repeat_list  = []
#
#         gt_joint_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'gt_joint.npz'))
#         in_joint_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'in_joint.npz'))
#         hand_type_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'hand_type.npz'))
#         in_joint_valid_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'in_joint_valid.npz'))
#         gt_joint_valid_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'gt_joint_valid.npz'))
#
#         for key in gt_joint_npz.files:
#             joint_xyz_gt, joint_xyz_in = gt_joint_npz[key], in_joint_npz[key]
#             joint_valid_gt, joint_valid_in = gt_joint_valid_npz[key], in_joint_valid_npz[key]
#             hand_type = hand_type_npz[key]
#             seq_img_num = joint_xyz_gt.shape[0]
#             seq_num = int(np.ceil(seq_img_num / self.seq_len))
#             cur_id = 0
#             for ii in range(seq_num):
#                 img_idx = np.arange(cur_id, cur_id + self.seq_len)
#                 img_idx_clip = np.clip(img_idx, a_min=0, a_max=seq_img_num - 1)
#                 data_repeat = (img_idx_clip - np.arange(cur_id, cur_id + self.seq_len)) == 0
#                 self.joint_xyz_gt_list.append(joint_xyz_gt[img_idx_clip])
#                 self.joint_xyz_in_list.append(joint_xyz_in[img_idx_clip])
#                 self.joint_valid_gt_list.append(joint_valid_gt[img_idx_clip])
#                 self.joint_valid_in_list.append(joint_valid_in[img_idx_clip])
#                 self.hand_type_list.append(hand_type[img_idx_clip])
#                 self.data_repeat_list.append(data_repeat)
#                 cur_id = cur_id + self.seq_len
#
#     def evaluate(self, outs, inputs, targets, meta_info):
#         batch_size, view_num, seq_len, joint_num, _ = outs['pd_joint_xyz'].size()
#         device = targets['joint_xyz'].device
#         con_val = meta_info['continuous_val'].view(batch_size, view_num, seq_len, 1)
#         gt_val = meta_info['joint_gt_val'].view(batch_size, view_num, seq_len, 21)
#         data_repeat = meta_info['data_repeat'].view(batch_size, view_num, seq_len, 1).repeat(1, 1, 1, 21)
#         gt_joint_xyz = targets['joint_xyz']
#         in_joint_xyz = inputs['joint_xyz']
#         pd_joint_xyz = outs['pd_joint_xyz'].to(device)
#
#         in_error = calculate_error(in_joint_xyz, gt_joint_xyz, data_repeat*gt_val*con_val)
#         out_error = calculate_error(pd_joint_xyz, gt_joint_xyz, data_repeat*gt_val*con_val)
#         return in_error, out_error
#
#     def joint_flip(self, joint_list, center_list, flip_flag_list):
#         joint_aug_list = []
#         for joint, center, flag in zip(joint_list, center_list, flip_flag_list):
#             joint_flip = joint.clone() - center
#             joint_flip[..., 0] *= -1
#             joint_flip = joint_flip + center
#             joint_aug_list.append(joint_flip * flag.float() + joint * (1 - flag.float()))
#         return joint_aug_list


# NPZ 高效数据增强版本
class SeqHand(Dataset):
    def __init__(self, data_path, dataset_list, joint_num=21, min_seq_len=30, view_num=1, seq_len=15, aug=True, data_num=None):
        self.min_seq_len = min_seq_len
        self.seq_len = seq_len
        self.view_num = view_num
        self.joint_num = joint_num
        self.augmenter = Augmenter()
        self.data_path = os.path.join(data_path, 'npz_30')
        self.img_path = os.path.join(data_path, 'img')

        # 'DexYCB', 'ReInterHand', 'InterHand_train', 'InterHand_test','InterHand_val','UmeTrack_synthetic','UmeTrack_real'
        self.dataset_list = dataset_list
        self.hand_types = ['right', 'left']

        self.expand = 3
        self.aug = aug
        self.data_num = data_num
        print('Loading Train data ...')
        start_time = time.time()
        self.generator_seq()
        end_time = time.time()
        total_seconds = end_time - start_time
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        print(f"Loading Time：{minutes}:{seconds}")

    def __len__(self):
        return len(self.joint_xyz_gt_list)

    def __getitem__(self, idx):
        # 从 idx 开始，往后找第一个有效样本
        while idx < len(self.joint_xyz_gt_list):
            data = self._load_sample(idx)
            if data is not None:
                return data
            idx += 1
        raise RuntimeError("Dataset has no valid samples.")

    def _load_sample(self, idx):
        joint_in = self.joint_xyz_in_list[idx].reshape([self.view_num, self.seq_len, self.joint_num, 3])  # [V T J 3]
        joint_gt = self.joint_xyz_gt_list[idx].reshape([self.view_num, self.seq_len, self.joint_num, 3])   # [V T J 3]

        joint_valid_in = self.joint_valid_in_list[idx].reshape([self.view_num, self.seq_len, self.joint_num])   # [C T J]
        joint_valid_gt = self.joint_valid_gt_list[idx].reshape([self.view_num, self.seq_len, self.joint_num])   # [T J 3]
        joint_world_valid = self.size_valid(joint_in) * joint_valid_in
        joint_type = self.hand_type_list[idx].reshape([self.seq_len, 1, 1])  # [T 1 1]
        continuous_val = self.frame_consecutive(joint_gt[0])

        center_gt = joint_gt[..., 9:10, :].copy()
        center_in = joint_in[..., 9:10, :].copy()
        [joint_gt, joint_in] = self.joint_flip([joint_gt, joint_in], [center_gt, center_in], [joint_type==1, joint_type==1])

        all_joint = np.concatenate([joint_gt, joint_in], axis=0)
        all_center = np.concatenate([center_gt, center_in], axis=0)

        # 序列的整体增强，增加数据多样性
        all_joint_aug = self.augmenter.seq_aug(all_joint, all_center)
        joint_in_aug = all_joint_aug[1:]

        # 构建多视角的gt
        joint_gt_aug = all_joint_aug[0:1]
        joint_gt_aug = np.repeat(joint_gt_aug, self.view_num, axis=0)
        center_gt = joint_gt_aug[..., 9:10, :].copy()

        # 输入数据的增强
        joint_in_aug = self.augmenter.frame_aug(joint_in_aug, center_in)
        if joint_valid_gt.sum() == 0 or  joint_world_valid.sum() == 0:
            return None
        else:
            inputs = {'joint_xyz': np.float32(joint_in_aug)}
            targets = {'joint_xyz': np.float32(joint_gt_aug)}
            meta_info = {"center_xyz": np.float32(center_gt),
                         'continuous_val': np.float32(continuous_val),
                         'joint_gt_val': np.float32(joint_valid_gt),
                         'joint_in_val': np.float32(joint_world_valid)}
            return inputs, targets, meta_info


    # 按照用户和动作序列选取数据
    # 为了进行数据增强，单条数据量大于seq_len的数量
    def generator_seq(self):
        self.joint_xyz_gt_list = []
        self.joint_xyz_in_list = []
        self.joint_valid_gt_list = []
        self.joint_valid_in_list = []
        self.hand_type_list = []

        # gt_joint_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'gt_joint.npz'))
        # in_joint_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'in_joint.npz'))
        # hand_type_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'hand_type.npz'))
        # in_joint_valid_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'in_joint_valid.npz'))
        # gt_joint_valid_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'gt_joint_valid.npz'))
        # keys = gt_joint_npz.files  # 将文件属性转换为内存中的列表
        # data_dict = {key: (gt_joint_npz[key], in_joint_npz[key], gt_joint_valid_npz[key], in_joint_valid_npz[key], hand_type_npz[key]) for key in keys}

        data_dict = {}
        for dataset_info in self.dataset_list:
            gt_joint_npz = os.path.join(self.data_path, dataset_info, 'gt_joint.npz')
            in_joint_npz = os.path.join(self.data_path, dataset_info, 'in_joint.npz')
            hand_type_npz = os.path.join(self.data_path, dataset_info, 'hand_type.npz')
            in_joint_valid_npz = os.path.join(self.data_path, dataset_info, 'in_joint_valid.npz')
            gt_joint_valid_npz = os.path.join(self.data_path, dataset_info, 'gt_joint_valid.npz')

            with (zipfile.ZipFile(gt_joint_npz, 'r') as gt_zip, zipfile.ZipFile(in_joint_npz, 'r') as in_zip,
                  zipfile.ZipFile(gt_joint_valid_npz, 'r') as gt_v_zip, zipfile.ZipFile(in_joint_valid_npz, 'r') as in_v_zip,
                  zipfile.ZipFile(hand_type_npz, 'r') as type_zip):
                keys = gt_zip.namelist()  # 假设每个npz包含同名键
                data_dict_item = {key.strip('.npy'): (load_npz_array(gt_zip, key), load_npz_array(in_zip, key), load_npz_array(gt_v_zip, key), load_npz_array(in_v_zip, key), load_npz_array(type_zip, key),) for key in keys}
                data_dict.update(data_dict_item)

        for key in data_dict.keys():
            joint_xyz_gt, joint_xyz_in, joint_valid_gt, joint_valid_in, hand_type = data_dict[key]
            seq_img_num = joint_xyz_gt.shape[0]
            first_frame = np.random.randint(0, self.seq_len)
            seq_num = int(np.ceil((seq_img_num - first_frame) / self.seq_len))
            seq_ids = np.arange(seq_num + 1) * self.seq_len + first_frame
            for ii in range(seq_num):
                img_idx = np.arange(seq_ids[ii] - self.seq_len // 2, seq_ids[ii] + self.seq_len // 2 + 1)
                img_idx_clip = np.clip(img_idx, a_min=0, a_max=seq_img_num - 1)
                self.joint_xyz_gt_list.append(joint_xyz_gt[img_idx_clip])
                self.joint_xyz_in_list.append(joint_xyz_in[img_idx_clip])
                self.joint_valid_gt_list.append(joint_valid_gt[img_idx_clip])
                self.joint_valid_in_list.append(joint_valid_in[img_idx_clip])
                self.hand_type_list.append(hand_type[img_idx_clip])

    def joint_flip(self, joint_list, center_list, flip_flag_list):
        joint_aug_list = []
        for joint, center, flag in zip(joint_list, center_list, flip_flag_list):
            joint_flip = joint - center
            joint_flip[..., 0] *= -1
            joint_flip = joint_flip + center
            joint_aug_list.append(joint_flip * flag + joint * (1 - flag))
        return joint_aug_list

    # 检测估计的异常帧 V T J 3
    def size_valid(self, seq):
        seq_min = np.min(seq[..., :2], axis=2)
        seq_max = np.max(seq[..., :2], axis=2)
        mask = (seq_max - seq_min)<50
        mask = mask.sum(axis=-1)<2
        return mask.reshape([self.view_num, self.seq_len, 1])

    def frame_consecutive(self, seq):
        seq_val = np.zeros([seq.shape[0]])
        diff = (seq[1:] - seq[:-1])
        diff = np.mean(np.sqrt((diff * diff).sum(-1)), axis=-1)
        seq_val[:-1] += (diff > 30)
        seq_val[1:] += (diff > 30)
        return seq_val == 0

class SeqHandTest(Dataset):
    def __init__(self, data_path, dataset_list, joint_num=21, min_seq_len=15, view_num=3, seq_len=15, data_num=None):
        self.min_seq_len = min_seq_len # 最小的序列长度
        self.valid_frame = 0
        self.seq_len = seq_len
        self.view_num = view_num
        self.joint_num = joint_num
        self.augmenter = SingleJointUtils()
        self.data_path = os.path.join(data_path, 'npz_30')
        self.img_path = os.path.join(data_path, 'img')

        # 'DexYCB', 'ReInterHand', 'InterHand_train', 'InterHand_test','InterHand_val','UmeTrack_synthetic','UmeTrack_real'
        self.dataset_list = ['InterHand_test']
        self.hand_types = ['right', 'left']
        self.data_num = data_num
        print('Loading Test data ...')
        start_time = time.time()
        self.generator_seq()
        end_time = time.time()
        total_seconds = end_time - start_time
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        print(f"Loading Time：{minutes}:{seconds}")


    def __len__(self):
        return len(self.joint_xyz_gt_list)

    def __getitem__(self, idx):
        # 从 idx 开始，往后找第一个有效样本
        while idx < len(self.joint_xyz_gt_list):
            data = self._load_sample(idx)
            if data is not None:
                return data
            idx += 1
        raise RuntimeError("Dataset has no valid samples.")

    def _load_sample(self, idx):
        joint_in = self.joint_xyz_in_list[idx].reshape([self.view_num, self.seq_len, self.joint_num, 3])  # [V T J 3]
        joint_gt = self.joint_xyz_gt_list[idx].reshape([self.view_num, self.seq_len, self.joint_num, 3])   # [V T J 3]

        joint_valid_in = self.joint_valid_in_list[idx].reshape([self.view_num, self.seq_len, self.joint_num])   # [C T J]
        joint_valid_gt = self.joint_valid_gt_list[idx].reshape([self.view_num, self.seq_len, self.joint_num])   # [T J 3]
        joint_world_valid = self.size_valid(joint_in) * joint_valid_in
        joint_type = self.hand_type_list[idx].reshape([self.seq_len, 1, 1])  # [T 1 1]
        continuous_val = self.frame_consecutive(joint_gt[0])
        data_repeat = self.data_repeat_list[idx].reshape([self.view_num, self.seq_len, 1])

        center_gt = joint_gt[..., 9:10, :].copy()
        center_in = joint_in[..., 9:10, :].copy()
        [joint_gt, joint_in] = self.joint_flip([joint_gt, joint_in], [center_gt, center_in], [joint_type==1, joint_type==1])

        joint_gt = np.repeat(joint_gt, self.view_num, axis=0)
        center_gt = joint_gt[..., 9:10, :].copy()

        if joint_valid_gt.sum() == 0 or joint_world_valid.sum() == 0:
            return None
        else:
            inputs = {'joint_xyz': np.float32(joint_in)}
            targets = {'joint_xyz': np.float32(joint_gt)}
            meta_info = {"center_xyz": np.float32(center_gt),
                         'continuous_val': np.float32(continuous_val),
                         'joint_gt_val': np.float32(joint_valid_gt),
                         'joint_in_val': np.float32(joint_world_valid),
                         'data_repeat': np.float32(data_repeat)}
            return inputs, targets, meta_info

    # 按照用户和动作序列选取数据
    def generator_seq(self):
        self.joint_xyz_gt_list = []
        self.joint_xyz_in_list = []
        self.joint_valid_gt_list = []
        self.joint_valid_in_list = []
        self.hand_type_list = []
        self.data_repeat_list  = []

        gt_joint_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'gt_joint.npz'))
        in_joint_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'in_joint.npz'))
        hand_type_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'hand_type.npz'))
        in_joint_valid_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'in_joint_valid.npz'))
        gt_joint_valid_npz = np.load(os.path.join(self.data_path, self.dataset_list[0], 'gt_joint_valid.npz'))

        for key in gt_joint_npz.files:
            joint_xyz_gt, joint_xyz_in = gt_joint_npz[key], in_joint_npz[key]
            joint_valid_gt, joint_valid_in = gt_joint_valid_npz[key], in_joint_valid_npz[key]
            hand_type = hand_type_npz[key]
            seq_img_num = joint_xyz_gt.shape[0]
            seq_num = int(np.ceil(seq_img_num / self.seq_len))
            cur_id = 0
            for ii in range(seq_num):
                img_idx = np.arange(cur_id, cur_id + self.seq_len)
                img_idx_clip = np.clip(img_idx, a_min=0, a_max=seq_img_num - 1)
                data_repeat = (img_idx_clip - np.arange(cur_id, cur_id + self.seq_len)) == 0
                self.joint_xyz_gt_list.append(joint_xyz_gt[img_idx_clip])
                self.joint_xyz_in_list.append(joint_xyz_in[img_idx_clip])
                self.joint_valid_gt_list.append(joint_valid_gt[img_idx_clip])
                self.joint_valid_in_list.append(joint_valid_in[img_idx_clip])
                self.hand_type_list.append(hand_type[img_idx_clip])
                self.data_repeat_list.append(data_repeat)
                cur_id = cur_id + self.seq_len

    def evaluate(self, outs, inputs, targets, meta_info):
        batch_size, view_num, seq_len, joint_num, _ = outs['pd_joint_xyz'].size()
        device = targets['joint_xyz'].device
        con_val = meta_info['continuous_val'].view(batch_size, view_num, seq_len, 1)
        gt_val = meta_info['joint_gt_val'].view(batch_size, view_num, seq_len, 21)
        data_repeat = meta_info['data_repeat'].view(batch_size, view_num, seq_len, 1).repeat(1, 1, 1, 21)
        gt_joint_xyz = targets['joint_xyz']
        in_joint_xyz = inputs['joint_xyz']
        pd_joint_xyz = outs['pd_joint_xyz'].to(device)

        in_error = calculate_error(in_joint_xyz, gt_joint_xyz, data_repeat*gt_val*con_val)
        out_error = calculate_error(pd_joint_xyz, gt_joint_xyz, data_repeat*gt_val*con_val)
        return in_error, out_error

    def joint_flip(self, joint_list, center_list, flip_flag_list):
        joint_aug_list = []
        for joint, center, flag in zip(joint_list, center_list, flip_flag_list):
            joint_flip = joint - center
            joint_flip[..., 0] *= -1
            joint_flip = joint_flip + center
            joint_aug_list.append(joint_flip * flag + joint * (1 - flag))
        return joint_aug_list

    # 检测估计的异常帧 V T J 3
    def size_valid(self, seq):
        seq_min = np.min(seq[..., :2], axis=2)
        seq_max = np.max(seq[..., :2], axis=2)
        mask = (seq_max - seq_min) < 50
        mask = mask.sum(axis=-1) < 2
        return mask.reshape([self.view_num, self.seq_len, 1])

    def frame_consecutive(self, seq):
        seq_val = np.zeros([seq.shape[0]])
        diff = (seq[1:] - seq[:-1])
        diff = np.mean(np.sqrt((diff * diff).sum(-1)), axis=-1)
        seq_val[:-1] += (diff > 30)
        seq_val[1:] += (diff > 30)
        return seq_val == 0


def cam2pixel(joint_xyz, paras):
    joint_uvd = torch.zeros_like(joint_xyz).to(joint_xyz.device)
    joint_uvd[:, :, 0] = (joint_xyz[:, :, 0] * paras[..., 0:1] / (joint_xyz[:, :, 2] + 1e-8) + paras[..., 2:3])
    joint_uvd[:, :, 1] = (joint_xyz[:, :, 1] * paras[..., 1:2] / (joint_xyz[:, :, 2]) + paras[..., 3:4])
    joint_uvd[:, :, 2] = joint_xyz[:, :, 2]
    return joint_uvd

"""
world_coord N J 3
R N 3 3
T N 1 3
"""
def world2cam(world_coord, R, t):
    cam_coord = torch.matmul(R, world_coord.permute(0, 2, 1)).permute(0, 2, 1)
    cam_coord = cam_coord + t
    return cam_coord

"""
world_coord N J 3
R N 3 3
T N 1 3
"""
def cam2world(cam_coord, R, t):
    cam_coord = cam_coord - t
    world_coord = torch.matmul(torch.inverse(R), cam_coord.permute(0,2,1)).permute(0,2,1)
    return world_coord


def calculate_error(joint, gt, mask=None):
    diff = (joint - gt)
    error = torch.sqrt(torch.sum(diff * diff, dim=-1))
    if mask is not None:
        return (error * mask).sum() / (mask.sum() + 1e-8)
    else:
        return error

def calculate_error_numpy(joint, gt, mask=None):
    diff = (joint - gt)
    error = np.sqrt(np.sum(diff * diff, axis=-1))
    if mask is not None:
        return (error * mask).sum() / (mask.sum() + 1e-8)
    else:
        return error

def draw_error_fig(frames_list, errors_list, name_list, fig_name):
    plt.figure(dpi=600)
    # color_list = ['lightcoral', 'lightskyblue', 'lightgreen']
    # pastel_colors = ['LightSkyBlue', 'PaleGreen', 'LightCoral', 'Plum', 'Khaki']
    pastel_colors = ['darkred', 'DeepSkyBlue']
    for idx in range(len(frames_list)):
        plt.plot(frames_list[idx], errors_list[idx], linestyle='-', color=pastel_colors[idx], label=name_list[idx])
    plt.xlabel('Frame')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(False)
    plt.savefig('%s.png'%(fig_name))
    plt.close()

def draw_pose_fig(frames, joints, name_list, fig_name):
    plt.figure(figsize=(10, 6))
    pastel_colors = ['LightSkyBlue', 'PaleGreen', 'LightCoral', 'Plum', 'Khaki', 'Sienna']
    # pastel_colors = ['LightCoral', 'Tomato', 'DarkOrange', 'Gold', 'Chocolate', 'Sienna']
    select_joint_list = [4, 8, 12, 16, 20]
    for idx in range(len(name_list)):
        joint_id = select_joint_list[idx]
        joint_offset = joints[:, joint_id, :] - joints[0:1, joint_id, :]
        joint_x, joint_y, joint_z = joint_offset[:, 0],joint_offset[:, 1],joint_offset[:, 2]
        plt.plot(frames, joint_x, linestyle='-', color=pastel_colors[idx], label=name_list[idx])

    joint_offset = joints.mean(1) - joints.mean(1) [0:1, :]
    joint_x, joint_y, joint_z = joint_offset[:, 0], joint_offset[:, 1], joint_offset[:, 2]
    plt.plot(frames, joint_x, linestyle='-', color=pastel_colors[-1], label='Global')

    plt.xlabel('Frame')
    plt.ylabel('Offset Relative to Initial Position')
    plt.legend()
    plt.grid(False)
    plt.savefig('%s.png'%(fig_name))
    plt.close()

def draw_joint_pose_fig(frames, joints_list, name_list, fig_name, joint_id):
    plt.figure(figsize=(10, 6))
    pastel_colors = ['LightSkyBlue', 'PaleGreen', 'LightCoral', 'Plum', 'Khaki', 'Sienna']
    # pastel_colors = ['LightCoral', 'Tomato', 'DarkOrange', 'Gold', 'Chocolate', 'Sienna']
    for idx in range(len(name_list)):
        joint_offset = joints_list[idx][:, joint_id, :] - joints_list[idx][0:1, joint_id, :]
        joint_x, joint_y, joint_z = joint_offset[:, 0],joint_offset[:, 1],joint_offset[:, 2]
        plt.plot(frames, joint_x, linestyle='-', color=pastel_colors[idx], label=name_list[idx])

    plt.xlabel('Frame')
    plt.ylabel('Offset Relative to Initial Position')
    plt.legend()
    plt.grid(False)
    plt.savefig('%s.png'%(fig_name))
    plt.close()



@torch.no_grad()
def draw_joint_data():
    batch_size = 1
    num_workers = 0
    # seq_len = 81
    seq_len = 15
    anno_dir = '../data/SeqHand/'

    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 为GPU设置种子
    torch.cuda.manual_seed_all(seed)  # 当有多张GPU时，为所有GPU设置种子
    np.random.seed(seed)  # 为Numpy设置随机种子
    random.seed(seed)  #

    dataset = SeqHandTest(anno_dir,'', seq_len=seq_len, view_num=1, data_num=5000)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True, drop_last=True)
    idx = 0
    for inputs, targets, meta_info in tqdm(dataloader):
        idx += 1
        center = meta_info['center_xyz'].view(batch_size, 1, seq_len, 1, 3)
        con_val = meta_info['continuous_val'].view(batch_size, 1, seq_len, 1)
        gt_val = meta_info['joint_gt_val'].view(batch_size, 1, seq_len, 21)
        gt_joint_xyz = targets['joint_xyz'] - center
        in_joint_xyz = inputs['joint_xyz'] - center
        error = calculate_error(in_joint_xyz, gt_joint_xyz, gt_val*con_val)
        print(error)
        gt_joint_pixel = targets['joint_pixel'].numpy()
        in_joint_pixel = inputs['joint_pixel'].numpy()
        img = inputs['images'].numpy()
        for ii in range(batch_size):
            for jj in range(seq_len):
                img_gt = draw_2d_skeleton(img[ii, jj], gt_joint_pixel[ii, 0, jj])
                img_pd = draw_2d_skeleton(img[ii, jj], in_joint_pixel[ii, 0, jj])
                cv.imwrite('./mean/joint_gt_%d_%d.png' % (ii, jj), img_gt)
                cv.imwrite('./mean/joint_pd_%d_%d.png' % (ii, jj), img_pd)
        break
    #     # 高斯去噪
    #     in_joint_xyz_np = in_joint_xyz.numpy()
    #     gs_refine_list = []
    #     for b_index in range(in_joint_xyz.size(0)):
    #         refine_joints = gaussian_filter1d(in_joint_xyz_np[b_index, 0], sigma=1.0, axis=0, radius=7)
    #         gs_refine_list.append(refine_joints)
    #     gs_refine_joint_xyz_np = np.stack(gs_refine_list, axis=0)
    #     gs_refine_joint_xyz = torch.from_numpy(gs_refine_joint_xyz_np).unsqueeze(1)
    #     gs_refine_error = calculate_error(gs_refine_joint_xyz, gt_joint_xyz, data_repeat*gt_val*con_val)
    #     gs_refine_error_list.append(gs_refine_error)
    #
    #     # savgol 去噪
    #     savgol_refine_list = []
    #     for b_index in range(in_joint_xyz.size(0)):
    #         refine_joints = savgol_filter(in_joint_xyz[b_index, 0], 7, 2, axis=0, mode='nearest')
    #         savgol_refine_list.append(refine_joints)
    #     savgol_refine_joint_xyz_np = np.stack(savgol_refine_list, axis=0)
    #     savgol_refine_joint_xyz = torch.from_numpy(savgol_refine_joint_xyz_np).unsqueeze(1)
    #     savgol_refine_error = calculate_error(savgol_refine_joint_xyz, gt_joint_xyz, data_repeat*gt_val*con_val)
    #     savgol_refine_error_list.append(savgol_refine_error)
    #
    #     # 为了绘制每帧的误差
    #     init_error = calculate_error(in_joint_xyz, gt_joint_xyz).mean(-1)
    #     gs_refine_error = calculate_error(gs_refine_joint_xyz, gt_joint_xyz).mean(-1)
    #     savgol_refine_error = calculate_error(savgol_refine_joint_xyz, gt_joint_xyz).mean(-1)
    #
    #     frames = np.arange(1, seq_len+1)
    #     draw_error_fig([frames, frames, frames], [init_error[0,0], gs_refine_error[0,0],savgol_refine_error[0,0]], ['Init', 'GS', 'savgol'], 'Error'+str(idx))
    #     # draw_pose_fig([frames, frames, frames, frames], [gt_joint_xyz[0,0,:,0,0], in_joint_xyz[0,0,:,0,0], savgol_refine_joint_xyz[0,0,:,0,0], gs_refine_joint_xyz[0,0,:,0,0]], ['GT','Init', 'savgol', 'GS'], 'pose'+str(idx))
    #     draw_pose_fig(frames, gt_joint_xyz[0,0,...], ['Thumb','Index','Middle','Ring','Little'], 'pose'+str(idx))
    #
    #
    # print(np.stack(error_list, axis=0).mean())
    # print(np.stack(gs_refine_error_list, axis=0).mean())
    # print(np.stack(savgol_refine_error_list, axis=0).mean())
        # for view_id in range(1):
        #     gt_joint_xyz = ((gt_joint_xyz[:, 0, ...] / 150 + 1) * 128).numpy()
        #     in_joint_xyz = ((in_joint_xyz[:, view_id, ...] / 150 + 1) * 128).numpy()
        #     img = np.ones([256, 256, 3]) * 255
        #     for ii in range(batch_size):
        #         for jj in range(seq_len):
        #             img_gt = draw_2d_skeleton(img, gt_joint_xyz[ii, jj])
        #             img_pd = draw_2d_skeleton(img, in_joint_xyz[ii, jj])
        #             cv.imwrite('./mean/joint_gt_%d_%d.png' % (ii, jj), img_gt)
        #             cv.imwrite('./mean/joint_pd_%d_%d.png' % (ii, jj), img_pd)
        #     print(error)
        # break


def vis_error():
    data_dir = '/home/rpf/pycharm/SeqHand/'
    init_pose_all = np.loadtxt(data_dir+'init.txt').reshape([-1,21,3])
    gt_pose_all = np.loadtxt(data_dir+'gt.txt').reshape([-1,21,3])
    MotionBERT_pose_all = np.loadtxt(data_dir+'MotionBERT.txt').reshape([-1,21,3])
    MotionAGFormer_pose_all = np.loadtxt(data_dir+'MotionAGFormer.txt').reshape([-1,21,3])
    img = np.ones([256, 256, 3]) * 255
    seq_len = 500
    iter_num = init_pose_all.shape[0] // seq_len
    for iter in range(iter_num):
        init_pose = init_pose_all[iter*seq_len:(iter+1)*seq_len]
        gt_pose = gt_pose_all[iter*seq_len:(iter+1)*seq_len]
        MotionBERT_refine_pose = MotionBERT_pose_all[iter*seq_len:(iter+1)*seq_len]
        MotionAGFormer_refine_pose = MotionAGFormer_pose_all[iter*seq_len:(iter+1)*seq_len]
        gaussian_refine_pose = gaussian_filter1d(init_pose, sigma=1.0, axis=0, radius=7)
        savgol_refine_pose = savgol_filter(init_pose, 7, 2, axis=0, mode='nearest')

        init_error = calculate_error_numpy(init_pose, gt_pose).mean(-1)
        MotionBERT_refine_error = calculate_error_numpy(MotionBERT_refine_pose, gt_pose).mean(-1)
        MotionAGFormer_refine_error = calculate_error_numpy(MotionAGFormer_refine_pose, gt_pose).mean(-1)
        # gaussian_refine_error = calculate_error_numpy(gaussian_refine_pose, gt_pose).mean(-1)
        # savgol_refine_error = calculate_error_numpy(savgol_refine_pose, gt_pose).mean(-1)

        # frames = np.arange(1, seq_len+1)
        # draw_error_fig([frames, frames, frames, frames, frames],
        #               [init_error,gaussian_refine_error,savgol_refine_error, MotionAGFormer_refine_error, MotionBERT_refine_error],
        #                ['Init', 'Gaussian1d', 'Savitzky-Golay', 'MotionBERT', 'Ours'],
        #                './debug/Error'+str(iter))
        frames = np.arange(1, seq_len+1)
        draw_error_fig([frames, frames],
                      [init_error, MotionBERT_refine_error],
                       ['Init Pose', 'Refined Pose'],
                       './debug/Error'+str(iter))

        gt_pose = gt_pose - np.mean(gt_pose, axis=1, keepdims=True)
        init_pose = init_pose- np.mean(init_pose, axis=1, keepdims=True)
        MotionBERT_refine_pose = MotionBERT_refine_pose - np.mean(MotionBERT_refine_pose, axis=1, keepdims=True)
        # if iter == 26:
        #     for jj in range(seq_len):
        #         pose = ((gt_pose[jj] / 150 + 1) * 128)
        #         img_gt = draw_2d_skeleton(img, pose)
        #         pose = ((init_pose[jj]  / 150 + 1) * 128)
        #         img_init = draw_2d_skeleton(img, pose)
        #         pose = ((MotionBERT_refine_pose[jj] / 150 + 1) * 128)
        #         img_refine = draw_2d_skeleton(img, pose)
        #         cv.imwrite('./debug/joint_gt_%d.png' % (jj), img_gt)
        #         cv.imwrite('./debug/joint_init_%d.png' % (jj), img_init)
        #         cv.imwrite('./debug/joint_pd_%d.png' % (jj), img_refine)

def vis_position():
    data_dir = '/home/rpf/pycharm/SeqHand/'
    init_pose_all = np.loadtxt(data_dir+'init.txt').reshape([-1,21,3])
    gt_pose_all = np.loadtxt(data_dir+'gt.txt').reshape([-1,21,3])
    MotionBERT_pose_all = np.loadtxt(data_dir+'MotionBERT.txt').reshape([-1,21,3])
    MotionAGFormer_pose_all = np.loadtxt(data_dir+'MotionAGFormer.txt').reshape([-1,21,3])
    img = np.ones([256, 256, 3]) * 255
    seq_len = 500
    iter_num = init_pose_all.shape[0] // seq_len
    for iter in range(iter_num):
        init_pose = init_pose_all[iter*seq_len:(iter+1)*seq_len]
        gt_pose = gt_pose_all[iter*seq_len:(iter+1)*seq_len]
        MotionBERT_refine_pose = MotionBERT_pose_all[iter*seq_len:(iter+1)*seq_len]
        MotionAGFormer_refine_pose = MotionAGFormer_pose_all[iter*seq_len:(iter+1)*seq_len]
        gaussian_refine_pose = gaussian_filter1d(init_pose, sigma=1.0, axis=0, radius=7)
        savgol_refine_pose = savgol_filter(init_pose, 7, 2, axis=0, mode='nearest')

        frames = np.arange(1, seq_len+1)
        draw_joint_pose_fig(frames,
                      [init_pose, gt_pose, gaussian_refine_pose, savgol_refine_pose,MotionAGFormer_refine_pose, MotionBERT_refine_pose],
                       ['Init Pose', 'GT', 'Gaussian1d', 'Savitzky-Golay',        'MotionBERT', 'Ours'],
                       './debug/Error'+str(iter), joint_id=4)


if __name__ == '__main__':
    # generate_single_pred_data()
    draw_joint_data()
    # vis_error()
    # vis_position()