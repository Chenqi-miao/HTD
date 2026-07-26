import json
import torch
import pickle
import cv2 as cv
import numpy as np
import os.path as osp
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.manolayer import ManoLayer
from utils.visualize import draw_2d_skeleton
from utils.video_utils import get_mano_path,get_mano_dir, JointUtils,SingleJointUtils
from manopth.manolayer import ManoLayer as ObmanManoLayer

cv.setNumThreads(0)
cv.ocl.setUseOpenCL(False)


def fix_obman_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].th_shapedirs[:, 0, :] - mano_layer['right'].th_shapedirs[:, 0, :])) < 1:
        mano_layer['left'].th_shapedirs[:, 0, :] *= -1


def fix_shape(mano_layer):
    if torch.sum(torch.abs(mano_layer['left'].shapedirs[:, 0, :] - mano_layer['right'].shapedirs[:, 0, :])) < 1:
        print('Fix shapedirs bug of MANO')
        mano_layer['left'].shapedirs[:, 0, :] *= -1


class VideoJointInterHand(Dataset):
    def __init__(self, anno_path, split, seq_len=15, data_len=None, aug=True):
        assert split in ['train', 'test', 'val']
        self.split = split
        self.seq_len = seq_len
        self.augmenter = JointUtils()
        mano_path = get_mano_path()
        mano_root_path = get_mano_dir()
        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        fix_shape(self.mano_layer)
        self.obman_layer = {'left': ObmanManoLayer(root_rot_mode='rotmat', mano_root=mano_root_path, side='left', use_pca=True,ncomps=45, center_idx=9, flat_hand_mean=False),
                           'right': ObmanManoLayer(root_rot_mode='rotmat', mano_root=mano_root_path, side='right', use_pca=True,ncomps=45, center_idx=9, flat_hand_mean=False)}
        fix_obman_shape(self.obman_layer)
        self.expand = 3
        self.anno_path = anno_path
        self.aug = aug
        self.single_hand_anno_file = osp.join(anno_path, split, 'single_data_info.pkl')
        self.two_hand_anno_file = osp.join(anno_path, split, 'data_info.pkl')
        self.generator_expand_seq()
        self.size = len(self.seq_info)
        if data_len is not None:
            self.size = data_len

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        annos = self.load_seq(idx)

        gt_R_left = torch.from_numpy(np.concatenate(annos['R_left'], axis=0)).float()
        gt_R_left = gt_R_left.reshape([-1, 9])
        gt_pose_left = torch.from_numpy(np.concatenate(annos['pose_left'], axis=0)).float()
        gt_shape_left = torch.from_numpy(np.concatenate(annos['shape_left'], axis=0)).float()
        gt_trans_left = torch.from_numpy(np.concatenate(annos['trans_left'], axis=0)).float()

        gt_R_right = torch.from_numpy(np.concatenate(annos['R_right'], axis=0)).float()
        gt_R_right = gt_R_right.reshape([-1, 9])
        gt_pose_right = torch.from_numpy(np.concatenate(annos['pose_right'], axis=0)).float()
        gt_shape_right = torch.from_numpy(np.concatenate(annos['shape_right'], axis=0)).float()
        gt_trans_right = torch.from_numpy(np.concatenate(annos['trans_right'], axis=0)).float()

        joint_val_left = torch.from_numpy(annos['joint_val_left']).view(-1, 1).float()
        joint_val_right = torch.from_numpy(annos['joint_val_right']).view(-1, 1).float()

        # 数据帧率增强
        [gt_pose_left, gt_shape_left, gt_trans_left,
         gt_pose_right, gt_shape_right, gt_trans_right], \
            [gt_R_left, gt_R_right], \
            [joint_val_left, joint_val_right] = \
            self.augmenter.pose_rate_aug([gt_pose_left, gt_shape_left, gt_trans_left,
                                          gt_pose_right, gt_shape_right, gt_trans_right],
                                         [gt_R_left, gt_R_right], [joint_val_left, joint_val_right],
                                         self.seq_len, expand=self.expand, inter_num=2)

        R = torch.from_numpy(np.stack(annos['R'], axis=0)).float()[0:1]
        T = torch.from_numpy(np.stack(annos['T'], axis=0)).float()[0:1]
        cam = torch.from_numpy(np.stack(annos['cam'], axis=0)).float()[0:1].repeat(self.seq_len,1,1)

        handV, handJ = self.mano_layer['left'](gt_R_left.reshape([-1, 3, 3]), gt_pose_left, gt_shape_left, trans=gt_trans_left)
        handV_left = torch.matmul(handV, R.permute(0, 2, 1)) + T.unsqueeze(1)
        handJ_left = torch.matmul(handJ, R.permute(0, 2, 1)) + T.unsqueeze(1)

        handV, handJ = self.mano_layer['right'](gt_R_right.reshape([-1, 3, 3]), gt_pose_right, gt_shape_right, trans=gt_trans_right)
        handV_right = torch.matmul(handV, R.permute(0, 2, 1)) + T.unsqueeze(1)
        handJ_right = torch.matmul(handJ, R.permute(0, 2, 1)) + T.unsqueeze(1)

        continuous_val = self.frame_consecutive(handJ_right) & self.frame_consecutive(handJ_left)
        center_left = handJ_left[:, 9:10, :].clone()
        center_right = handJ_right[:, 9:10, :].clone()

        if self.split == 'train' and self.aug:
            aug_center = (center_left * joint_val_left.unsqueeze(1) + center_right * joint_val_right.unsqueeze(1)) / 2
            [handJ_left, handV_left], [handJ_right, handV_right], flip = \
                self.augmenter.seq_aug([handJ_left, handV_left],
                                       [handJ_right, handV_right],
                                       [aug_center] * 3)
            center_left = handJ_left[:, 9:10, :].clone()
            center_right = handJ_right[:, 9:10, :].clone()
            if flip:
                joint_val_right, joint_val_left = joint_val_left, joint_val_right
            [in_joint_left], [in_joint_right] = self.augmenter.joint_aug([handJ_left], [handJ_right], [aug_center])

        inputs = {'in_joint_left': np.float32(in_joint_left),
                  'in_joint_right': np.float32(in_joint_right),}
        targets = {
            'joint_3d_left': np.float32(handJ_left),
            'mesh_3d_left': np.float32(handV_left),
            'joint_3d_right': np.float32(handJ_right),
            'mesh_3d_right': np.float32(handV_right),
        }
        meta_info = {"center_left": np.float32(center_left),
                     "center_right": np.float32(center_right),
                     'camera': np.float32(cam),
                     'continuous_val': np.float32(continuous_val),
                     'img_val': np.float32(torch.ones_like(joint_val_left.view(-1))),
                     'joint_val_left': np.float32(joint_val_left.view(-1)),
                     'joint_val_right': np.float32(joint_val_right.view(-1)),
                     }
        return inputs, targets, meta_info

    def frame_consecutive(self, seq):
        seq_val = torch.zeros([seq.size(0)])
        diff = (seq[1:] - seq[:-1]) * 1000
        diff = torch.mean(torch.sqrt((diff * diff).sum(-1)), dim=-1)
        seq_val[:-1] += (diff > 30).float()
        seq_val[1:] += (diff > 30).float()
        return seq_val.eq(0)

    def generator_stride_seq(self):
        f = open(self.single_hand_anno_file, 'rb')
        single_hand_seq_list = pickle.load(f)
        f.close()
        f = open(self.two_hand_anno_file, 'rb')
        two_hand_seq_list = pickle.load(f)
        f.close()
        seq_list = single_hand_seq_list + two_hand_seq_list
        self.seq_info = []
        for seq in seq_list:
            seq_name = seq['seq_name']
            for cam in seq['camera_list']:
                seq_img_num = seq['image_num']
                first_frame = np.random.randint(0, self.seq_len)
                seq_num = int(np.ceil((seq_img_num - first_frame) / self.seq_len))
                seq_bound_start = np.arange(seq_num + 1) * self.seq_len + first_frame
                if self.split == 'train':
                    seq_bound_stride = [1] * seq_num
                else:
                    seq_bound_stride = [1] * seq_num
                img_id_list = []
                for ii in range(seq_num):
                    stride = seq_bound_stride[ii]
                    img_idx = np.clip(0, seq_img_num - 1,np.arange(seq_bound_start[ii], seq_bound_start[ii] + self.seq_len * stride, stride))
                    img_names = []
                    for img_idx in img_idx:
                        img_names.append(seq['image_list'][img_idx])
                    img_id_list.append(img_names)
                self.seq_info += zip([seq_name] * seq_num, [cam] * seq_num, img_id_list)
        f.close()

    def generator_expand_seq(self):
        seq_list = []
        f = open(self.single_hand_anno_file, 'rb')
        single_hand_seq_list = pickle.load(f)
        f.close()
        seq_list += single_hand_seq_list
        # f = open(self.two_hand_anno_file, 'rb')
        # two_hand_seq_list = pickle.load(f)
        # f.close()
        # seq_list = two_hand_seq_list
        self.seq_info = []
        expand_len = int((self.seq_len-1) / 2 * self.expand)
        for seq in seq_list:
            seq_name = seq['seq_name']
            for cam in seq['camera_list']:
                seq_img_num = seq['image_num']
                img_id_list = []
                if self.split == 'train':
                    first_frame = np.random.randint(0, self.seq_len)
                    seq_num = int(np.ceil((seq_img_num - first_frame) / self.seq_len))
                    seq_ids = np.arange(seq_num + 1) * self.seq_len + first_frame
                    for ii in range(seq_num):
                        img_idx = np.arange(seq_ids[ii]-expand_len, seq_ids[ii]+expand_len)
                        img_idx = np.clip(img_idx, a_min=0, a_max=seq_img_num - 1)
                        img_names = []
                        for img_idx in img_idx:
                            img_names.append(seq['image_list'][img_idx])
                        img_id_list.append(img_names)
                else:
                    seq_num = int(np.ceil(seq_img_num / self.seq_len))
                    cur_id = 0
                    for ii in range(seq_num):
                        img_idx = np.arange(cur_id, cur_id + self.seq_len)
                        img_idx = np.clip(img_idx, a_min=0, a_max=seq_img_num - 1)
                        img_names = []
                        for img_idx in img_idx:
                            img_names.append(seq['image_list'][img_idx])
                        img_id_list.append(img_names)
                        cur_id += self.seq_len

                self.seq_info += zip([seq_name] * seq_num, [cam] * seq_num, img_id_list)
        f.close()

    def load_seq(self, idx):
        seq_name, cam_id, img_id_list = self.seq_info[idx]

        # load anno info
        R_list, T_list, cam_list = [], [], []
        R_left_list, pose_left_list, shape_left_list, trans_left_list = [], [], [], []
        R_right_list, pose_right_list, shape_right_list, trans_right_list = [], [], [], []
        val_left_list, val_right_list = [], []
        for img_id in img_id_list:
            with open(os.path.join(self.anno_path, self.split, seq_name, cam_id, 'anno', '{}.pkl'.format(img_id)), 'rb') as file:
                data = pickle.load(file)
            R_list.append(data['camera']['R'])
            T_list.append(data['camera']['t'])
            cam_list.append(data['camera']['camera'])
            params_left = data['mano_params']['left']

            if params_left is not None:
                R_left_list.append(params_left['R'])
                pose_left_list.append(params_left['pose'])
                shape_left_list.append(params_left['shape'])
                trans_left_list.append(params_left['trans'])
                val_left_list.append(1)
            else:
                R_left_list.append(np.eye(3)[np.newaxis, :, :])
                pose_left_list.append(np.zeros([1, 45]))
                shape_left_list.append(np.zeros([1, 10]))
                trans_left_list.append(np.zeros([1, 3]))
                val_left_list.append(0)

            params_right = data['mano_params']['right']
            if params_right is not None:
                R_right_list.append(params_right['R'])
                pose_right_list.append(params_right['pose'])
                shape_right_list.append(params_right['shape'])
                trans_right_list.append(params_right['trans'])
                val_right_list.append(1)
            else:
                R_right_list.append(np.eye(3)[np.newaxis, :, :])
                pose_right_list.append(np.zeros([1, 45]))
                shape_right_list.append(np.zeros([1, 10]))
                trans_right_list.append(np.zeros([1, 3]))
                val_right_list.append(0)
        anno_dict = {
            'R': R_list,
            'T': T_list,
            'cam': cam_list,
            'R_left': R_left_list,
            'pose_left': pose_left_list,
            'shape_left': shape_left_list,
            'trans_left': trans_left_list,
            'joint_val_left': np.array(val_left_list),
            'R_right': R_right_list,
            'pose_right': pose_right_list,
            'shape_right': shape_right_list,
            'trans_right': trans_right_list,
            'joint_val_right': np.array(val_right_list),
            'img_val': np.ones_like(np.array(val_right_list)),
        }
        return anno_dict

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
        joint_left_error = joint_left_error.detach().cpu().numpy().mean() * 1000  # m -> mm

        joint_right_error = torch.linalg.norm((joints_right_pred - joints_right_gt), ord=2, dim=-1)
        joint_right_error = joint_right_error.detach().cpu().numpy().mean() * 1000

        vert_left_error = torch.linalg.norm((verts_left_pred - verts_left_gt), ord=2, dim=-1)
        vert_left_error = vert_left_error.detach().cpu().numpy().mean() * 1000

        vert_right_error = torch.linalg.norm((verts_right_pred - verts_right_gt), ord=2, dim=-1)
        vert_right_error = vert_right_error.detach().cpu().numpy().mean() * 1000

        return joint_left_error, joint_right_error, vert_left_error, vert_right_error


class SingleVideoJointInterHand(Dataset):
    def __init__(self, anno_path, split, seq_len=15, data_len=None, aug=True):
        assert split in ['train', 'test', 'val']
        self.split = split
        self.seq_len = seq_len
        self.augmenter = SingleJointUtils()
        mano_path = get_mano_path()
        mano_root_path = get_mano_dir()
        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        fix_shape(self.mano_layer)
        self.obman_layer = {'left': ObmanManoLayer(root_rot_mode='rotmat', mano_root=mano_root_path, side='left', use_pca=True,ncomps=45, center_idx=9, flat_hand_mean=False),
                           'right': ObmanManoLayer(root_rot_mode='rotmat', mano_root=mano_root_path, side='right', use_pca=True,ncomps=45, center_idx=9, flat_hand_mean=False)}
        fix_obman_shape(self.obman_layer)
        self.expand = 3
        self.anno_path = anno_path
        self.aug = aug
        self.single_hand_anno_file = osp.join(anno_path, split, 'single_data_info.pkl')
        self.two_hand_anno_file = osp.join(anno_path, split, 'two_data_info.pkl')
        self.generator_expand_seq()
        self.size = len(self.seq_info)
        # if data_len is not None:
        #     self.size = data_len
        # else:
        #     self.size = data_len
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        hand_type, annos = self.load_seq(idx)

        gt_R = torch.from_numpy(np.concatenate(annos['R'], axis=0)).float()
        gt_R = gt_R.reshape([-1, 9])
        gt_pose = torch.from_numpy(np.concatenate(annos['pose'], axis=0)).float()
        gt_shape = torch.from_numpy(np.concatenate(annos['shape'], axis=0)).float()
        gt_trans = torch.from_numpy(np.concatenate(annos['trans'], axis=0)).float()
        joint_val = torch.from_numpy(annos['joint_val']).view(-1, 1).float()

        # 数据帧率增强
        if self.split == 'train':
            [gt_pose, gt_shape, gt_trans], [gt_R], [joint_val] = \
            self.augmenter.pose_rate_aug([gt_pose, gt_shape, gt_trans], [gt_R], [joint_val],
                                         self.seq_len, expand=self.expand, inter_num=2)
        R = torch.from_numpy(np.stack(annos['cam_R'], axis=0)).float()[0:1]
        T = torch.from_numpy(np.stack(annos['cam_T'], axis=0)).float()[0:1]
        cam = torch.from_numpy(np.stack(annos['cam'], axis=0)).float()[0:1].repeat(self.seq_len,1,1)

        # 获取手部节点和mesh
        handV, handJ = self.mano_layer[hand_type](gt_R.reshape([-1, 3, 3]), gt_pose, gt_shape, trans=gt_trans)
        handV = torch.matmul(handV, R.permute(0, 2, 1)) + T.unsqueeze(1)
        handJ = torch.matmul(handJ, R.permute(0, 2, 1)) + T.unsqueeze(1)

        continuous_val = self.frame_consecutive(handJ)
        center = handJ[:, 9:10, :].clone()

        aug_center = center * joint_val.unsqueeze(1)
        [handJ, handV] = self.augmenter.seq_aug([handJ, handV], [aug_center,aug_center])
        center = handJ[:, 9:10, :].clone()

        if hand_type =='right':
            [handJ, handV] = self.augmenter.joint_flip([handJ, handV], [aug_center,aug_center])
        [in_joint] = self.augmenter.joint_aug([handJ], [aug_center])
        inputs = {'joint': np.float32(in_joint)}
        targets = {'joint_3d': np.float32(handJ),'mesh_3d': np.float32(handV)}
        meta_info = {"center": np.float32(center),
                     'camera': np.float32(cam),
                     'continuous_val': np.float32(continuous_val),
                     'joint_val': np.float32(joint_val.view(-1))
                     }
        return inputs, targets, meta_info

    def frame_consecutive(self, seq):
        seq_val = torch.zeros([seq.size(0)])
        diff = (seq[1:] - seq[:-1]) * 1000
        diff = torch.mean(torch.sqrt((diff * diff).sum(-1)), dim=-1)
        seq_val[:-1] += (diff > 30).float()
        seq_val[1:] += (diff > 30).float()
        return seq_val.eq(0)

    def generator_stride_seq(self):
        f = open(self.single_hand_anno_file, 'rb')
        single_hand_seq_list = pickle.load(f)
        f.close()
        f = open(self.two_hand_anno_file, 'rb')
        two_hand_seq_list = pickle.load(f)
        f.close()
        seq_list = single_hand_seq_list + two_hand_seq_list
        self.seq_info = []
        for seq in seq_list:
            seq_name = seq['seq_name']
            for cam in seq['camera_list']:
                seq_img_num = seq['image_num']
                first_frame = np.random.randint(0, self.seq_len)
                seq_num = int(np.ceil((seq_img_num - first_frame) / self.seq_len))
                seq_bound_start = np.arange(seq_num + 1) * self.seq_len + first_frame
                if self.split == 'train':
                    seq_bound_stride = [1] * seq_num
                else:
                    seq_bound_stride = [1] * seq_num
                img_id_list = []
                for ii in range(seq_num):
                    stride = seq_bound_stride[ii]
                    img_idx = np.clip(0, seq_img_num - 1,np.arange(seq_bound_start[ii], seq_bound_start[ii] + self.seq_len * stride, stride))
                    img_names = []
                    for img_idx in img_idx:
                        img_names.append(seq['image_list'][img_idx])
                    img_id_list.append(img_names)
                self.seq_info += zip([seq_name] * seq_num, [cam] * seq_num, img_id_list)
        f.close()

    def generator_expand_seq(self):
        seq_list = []
        f = open(self.single_hand_anno_file, 'rb')
        single_hand_seq_list = pickle.load(f)
        f.close()
        seq_list += single_hand_seq_list
        f = open(self.two_hand_anno_file, 'rb')
        two_hand_seq_list = pickle.load(f)
        f.close()
        seq_list += two_hand_seq_list
        self.seq_info = []
        expand_len = int((self.seq_len-1) / 2 * self.expand)
        for seq in seq_list:
            seq_name = seq['seq_name']
            for cam in seq['camera_list']:
                seq_img_num = seq['image_num']
                img_id_list = []
                if self.split == 'train':
                    first_frame = np.random.randint(0, self.seq_len)
                    seq_num = int(np.ceil((seq_img_num - first_frame) / self.seq_len))
                    seq_ids = np.arange(seq_num + 1) * self.seq_len + first_frame
                    for ii in range(seq_num):
                        img_idx = np.arange(seq_ids[ii]-expand_len, seq_ids[ii]+expand_len)
                        img_idx = np.clip(img_idx, a_min=0, a_max=seq_img_num - 1)
                        img_names = []
                        for img_idx in img_idx:
                            img_names.append(seq['image_list'][img_idx])
                        img_id_list.append(img_names)
                else:
                    seq_num = int(np.ceil(seq_img_num / self.seq_len))
                    cur_id = 0
                    for ii in range(seq_num):
                        img_idx = np.arange(cur_id, cur_id + self.seq_len)
                        img_idx = np.clip(img_idx, a_min=0, a_max=seq_img_num - 1)
                        img_names = []
                        for img_idx in img_idx:
                            img_names.append(seq['image_list'][img_idx])
                        img_id_list.append(img_names)
                        cur_id += self.seq_len

                self.seq_info += zip([seq_name] * seq_num, [cam] * seq_num, img_id_list)

    def load_seq(self, idx):
        seq_name, cam_id, img_id_list = self.seq_info[idx]

        # load anno info
        R_list, T_list, cam_list = [], [], []
        R_left_list, pose_left_list, shape_left_list, trans_left_list = [], [], [], []
        R_right_list, pose_right_list, shape_right_list, trans_right_list = [], [], [], []
        val_left_list, val_right_list = [], []
        for img_id in img_id_list:
            with open(os.path.join(self.anno_path, self.split, seq_name, cam_id, 'anno', '{}.pkl'.format(img_id)), 'rb') as file:
                data = pickle.load(file)
            R_list.append(data['camera']['R'])
            T_list.append(data['camera']['t'])
            cam_list.append(data['camera']['camera'])
            params_left = data['mano_params']['left']

            if params_left is not None:
                R_left_list.append(params_left['R'])
                pose_left_list.append(params_left['pose'])
                shape_left_list.append(params_left['shape'])
                trans_left_list.append(params_left['trans'])
                val_left_list.append(1)
            else:
                R_left_list.append(np.eye(3)[np.newaxis, :, :])
                pose_left_list.append(np.zeros([1, 45]))
                shape_left_list.append(np.zeros([1, 10]))
                trans_left_list.append(np.zeros([1, 3]))
                val_left_list.append(0)

            params_right = data['mano_params']['right']
            if params_right is not None:
                R_right_list.append(params_right['R'])
                pose_right_list.append(params_right['pose'])
                shape_right_list.append(params_right['shape'])
                trans_right_list.append(params_right['trans'])
                val_right_list.append(1)
            else:
                R_right_list.append(np.eye(3)[np.newaxis, :, :])
                pose_right_list.append(np.zeros([1, 45]))
                shape_right_list.append(np.zeros([1, 10]))
                trans_right_list.append(np.zeros([1, 3]))
                val_right_list.append(0)

        left_val = np.array(val_left_list).sum() > len(img_id_list) * 0.6
        right_val = np.array(val_right_list).sum() > len(img_id_list) * 0.6
        anno_dict_left = {
            'cam_R': R_list,
            'cam_T': T_list,
            'cam': cam_list,
            'R': R_left_list,
            'pose': pose_left_list,
            'shape': shape_left_list,
            'trans': trans_left_list,
            'joint_val': np.array(val_left_list),
            'img_val': np.ones_like(np.array(val_left_list)),
        }

        anno_dict_right = {
            'cam_R': R_list,
            'cam_T': T_list,
            'cam': cam_list,
            'R': R_right_list,
            'pose': pose_right_list,
            'shape': shape_right_list,
            'trans': trans_right_list,
            'joint_val': np.array(val_right_list),
            'img_val': np.ones_like(np.array(val_left_list))
        }
        if left_val and right_val:
            if np.random.rand() > 0.5:
                anno_dict = anno_dict_left
                hand_type = 'left'
            else:
                anno_dict = anno_dict_right
                hand_type = 'right'
        elif left_val:
            anno_dict = anno_dict_left
            hand_type = 'left'
        else:
            anno_dict = anno_dict_right
            hand_type = 'right'

        return hand_type, anno_dict

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
        joint_left_error = joint_left_error.detach().cpu().numpy().mean() * 1000  # m -> mm

        joint_right_error = torch.linalg.norm((joints_right_pred - joints_right_gt), ord=2, dim=-1)
        joint_right_error = joint_right_error.detach().cpu().numpy().mean() * 1000

        vert_left_error = torch.linalg.norm((verts_left_pred - verts_left_gt), ord=2, dim=-1)
        vert_left_error = vert_left_error.detach().cpu().numpy().mean() * 1000

        vert_right_error = torch.linalg.norm((verts_right_pred - verts_right_gt), ord=2, dim=-1)
        vert_right_error = vert_right_error.detach().cpu().numpy().mean() * 1000

        return joint_left_error, joint_right_error, vert_left_error, vert_right_error


# 单手时序加载
class SeqHand(Dataset):
    def __init__(self, data_path, split, view_len=3, seq_len=15, data_len=None, aug=True):
        assert split in ['train', 'test', 'val']
        self.split = split
        self.seq_len = seq_len
        self.view_len = view_len
        self.augmenter = SingleJointUtils()
        self.data_path = os.path.join(data_path, 'data')
        self.img_path = os.path.join(data_path, 'img')
        self.dataset_list = ['HanCo']

        self.expand = 3
        self.aug = aug
        self.generator_expand_seq()
        self.size = len(self.data_list)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        data_dict = self.load_seq(idx)

        R, T = data_dict['R'],data_dict['T'] # [C T 3 3] [C T 3]
        joint_cams_in = data_dict['joint_cams'] # [C T J 3]
        joint_cams_valid = data_dict['joint_cams_valid'][..., np.newaxis] # [C T J]
        joint_type =  data_dict['joint_type'].reshape([1, 1, -1, 1]) # [T 1]

        joint_valid_gt = data_dict['joint_valid_gt']  # [C T J 3]
        joint_world_gt = data_dict['joint_world_gt'] # [T J 3]

        R, T = R[:, np.newaxis] , T[:, np.newaxis]
        R, T = np.tile(R, (1, self.seq_len, 1, 1)), T.tile(T, (1, self.seq_len, 1))
        joint_world_in = cam2world(joint_cams_in.reshape([-1, 3]), R.reshape([-1, 3, 3]), T.reshape([-1,3]))
        joint_world_in = joint_world_in*joint_cams_valid
        continuous_val = self.frame_consecutive(joint_world_gt)


        hand_joint_gt = torch.from_numpy(joint_world_gt)
        hand_joint_in = torch.from_numpy(joint_world_in)
        center_gt = hand_joint_gt[:, 9:10, :].clone()
        center_in = hand_joint_in[:, 9:10, :].clone()

        #
        self.joint_flip([hand_joint_gt, hand_joint_in], [center_gt, center_in], [joint_type==0, joint_type==0])

        # 序列的整体增强，增加数据多样性
        [hand_joint_gt, hand_joint_in] = self.augmenter.seq_aug([hand_joint_gt, hand_joint_in], [center_gt, center_in])
        # 输入数据的增强
        [hand_joint_in] = self.augmenter.joint_aug([hand_joint_in], [center_in])

        inputs = {'joint': np.float32(hand_joint_gt)}
        targets = {'joint_3d': np.float32(hand_joint_in)}
        meta_info = {"center": np.float32(center_gt),
                     'continuous_val': np.float32(continuous_val),
                     'joint_val': np.float32(joint_valid_gt)}
        return inputs, targets, meta_info

    def frame_consecutive(self, seq):
        seq_val = np.zeros([seq.shape[0]])
        diff = (seq[1:] - seq[:-1]) * 1000
        diff = np.mean(np.sqrt((diff * diff).sum(-1)), axis=-1)
        seq_val[:-1] += (diff > 30)
        seq_val[1:] += (diff > 30)
        return seq_val==0

    # 从数据按照用户和动作序列选取数据
    # 为了进行数据增强，单条数据量大于seq_len的数量
    def generator_expand_seq(self):

        expand_len = int((self.seq_len-1) / 2 * self.expand)
        for dataset in self.dataset_list:
            f = open(os.path.join(self.data_path, '%s.json'%(dataset)), 'rb')
            data_dict = pickle.load(f)
            f.close()
            self.data_list = []
            for capture in data_dict.keys():
                for seq in data_dict[capture].keys():
                    # 对方法和相机进行随机采样
                    method_name_list = data_dict[capture][seq]['method_name_list']
                    cam_name_list = data_dict[capture][seq]['cam_name_list']
                    method_id = np.random.choice(len(method_name_list), 1)
                    cam_id = np.random.choice(len(cam_name_list), 1)
                    method_select = method_name_list[method_id]
                    cam_name_select = cam_name_list[cam_id]

                    img_name_list = data_dict[capture][seq]['img_name_list']
                    seq_img_num = len(img_name_list)
                    if seq_img_num < self.seq_len:
                        continue
                    img_id_list = []

                    if self.split == 'train':
                        first_frame = np.random.randint(0, self.seq_len)
                        seq_num = int(np.ceil((seq_img_num - first_frame) / self.seq_len))
                        seq_ids = np.arange(seq_num + 1) * self.seq_len + first_frame
                        for ii in range(seq_num):
                            img_idx = np.arange(seq_ids[ii] - expand_len, seq_ids[ii] + expand_len)
                            img_idx = np.clip(img_idx, a_min=0, a_max=seq_img_num - 1)
                            img_names = []
                            for img_idx in img_idx:
                                img_names.append(img_name_list[img_idx])
                            img_id_list.append(img_names)
                            file_dict = {'dataset': dataset,
                                         'capture': capture,
                                         'seq': seq,
                                         'method': method_select,
                                         'cams': cam_name_select,
                                         'imgs': img_id_list}
                            self.data_list.append(file_dict)
                    else:
                        seq_num = int(np.ceil(seq_img_num / self.seq_len))
                        cur_id = 0
                        for ii in range(seq_num):
                            img_idx = np.arange(cur_id, cur_id + self.seq_len)
                            img_idx = np.clip(img_idx, a_min=0, a_max=seq_img_num - 1)
                            img_names = []
                            for img_idx in img_idx:
                                img_names.append(img_name_list[img_idx])
                            img_id_list.append(img_names)
                            cur_id += self.seq_len


    # 仅加载单手数据
    def load_seq(self, idx):
        data_dict = self.data_list[idx]
        dataset, capture_name, seq_name, method_name = data_dict['dataset'],data_dict['capture'], data_dict['seq'],data_dict['method']
        cam_name_list, img_id_list = data_dict['cams'],data_dict['imgs']
        data_path = os.path.join(self.data_path, dataset, capture_name, seq_name)
        hand_types = ['right', 'left']

        # load anno info
        anno_file = osp.join(data_path, 'anno_info.json')
        meta_file = osp.join(data_path, 'meta_info.json')
        with open(meta_file, 'r') as file:
            meta_info = json.load(file)
        with open(anno_file, 'r') as file:
            annos_info = json.load(file)

        # 加载全局坐标和手部类型
        joint_world_list = []
        joint_valid_list = []
        hand_type_list = []
        for img_name in img_id_list:
            if meta_info[img_name]['hand_type'] in ['right', 'two', 'interacting']:
                hand_type = 0 # 'right'
            else:
                hand_type = 1 # 'left'
            if annos_info[img_name][hand_types[hand_type]]['world_coord'] is not None:
                joint_world = np.array(annos_info[img_name][hand_types[hand_type]]['world_coord'], np.float64)
                joint_valid = np.array(annos_info[img_name][hand_types[hand_type]]['joint_valid'], np.float64)
            else:
                joint_world = np.zeros([21, 3])
                joint_valid = np.zeros([21])

            joint_world_list.append(joint_world)
            joint_valid_list.append(joint_valid)
            hand_type_list.append(hand_type)
        joint_world_gt = np.stack(joint_world_list, axis=0)
        joint_valid_gt = np.stack(joint_valid_list, axis=0)
        joint_type = np.stack(hand_type_list, axis=0)

        # load input info
        joint_cams_list = []
        joint_cams_valid_list = []
        R_list, T_list = [], []
        focal_list, princpt_list = [], []
        for cam_name in cam_name_list:
            cam_para = meta_info['cam_params'][cam_name]
            R, T = np.array(cam_para['R']), np.array(cam_para['T'])
            focal, princpt = np.array(cam_para['focal']), np.array(cam_para['princpt'])
            data_file = os.path.join(data_path, method_name, '%s.json'%(cam_name))
            with open(data_file, 'r') as file:
                data_info = json.load(file)
            joint_cam_list = []
            joint_cam_valid_list = []
            for img_name in img_id_list:
                if meta_info[img_name]['hand_type'] in ['right', 'two', 'interacting']:
                    hand_type = 0  # 'right'
                else:
                    hand_type = 1  # 'left'
                joint_cam = np.array(data_info[img_name][hand_types[hand_type]]['cam_coord'])
                joint_valid = np.array(data_info[img_name][hand_types[hand_type]]['joint_valid'])
                joint_cam_list.append(joint_cam)
                joint_cam_valid_list.append(joint_valid)
            joint_cam = np.stack(joint_cam_list, axis=0)
            joint_cam_valid = np.stack(joint_cam_valid_list, axis=0)
            joint_cams_list.append(joint_cam)
            joint_cams_valid_list.append(joint_cam_valid)
            R_list.append(R)
            T_list.append(T)
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
            'joint_cams_valid':joint_cams_valid,
            'joint_world_gt':joint_world_gt,
            'joint_valid_gt':joint_valid_gt,
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
        joint_left_error = joint_left_error.detach().cpu().numpy().mean() * 1000  # m -> mm

        joint_right_error = torch.linalg.norm((joints_right_pred - joints_right_gt), ord=2, dim=-1)
        joint_right_error = joint_right_error.detach().cpu().numpy().mean() * 1000

        vert_left_error = torch.linalg.norm((verts_left_pred - verts_left_gt), ord=2, dim=-1)
        vert_left_error = vert_left_error.detach().cpu().numpy().mean() * 1000

        vert_right_error = torch.linalg.norm((verts_right_pred - verts_right_gt), ord=2, dim=-1)
        vert_right_error = vert_right_error.detach().cpu().numpy().mean() * 1000

        return joint_left_error, joint_right_error, vert_left_error, vert_right_error

    def joint_flip(self, joint_list, center_list, flip_flag_list):
        joint_aug_list = []
        for joint, center, flag in zip(joint_list, center_list, flip_flag_list):
            joint_flip = joint.clone()-center
            joint_flip[..., 0] *= -1
            joint_flip = joint_flip+center
            joint_aug_list.append(joint_flip * flag + joint * (1-flag))
        return joint_aug_list

def world2cam(world_coord, R, t):
    cam_coord = np.matmul(R, world_coord.transpose(2, 1))
    cam_coord = cam_coord.transpose(2, 1) + t.reshape(-1, 3)
    return cam_coord

def cam2world(cam_coord, R, t):
    cam_coord = cam_coord - t.reshape(-1, 3)
    world_coord = np.matmul(np.linalg.inv(R), cam_coord.transpose(2, 1)).transpose(2, 1)
    return world_coord

def calculate_error(joint, gt):
    diff = (joint - gt) * 1000
    error = torch.sqrt(torch.sum(diff * diff, dim=-1))
    return error.mean()


@torch.no_grad()
def draw_joint_data():
    batch_size = 8
    num_workers = 4
    seq_len = 27
    anno_dir = '/data/dataset/interhand2.6m_30fps'

    dataset = SingleVideoJointInterHand(anno_dir, 'test', seq_len=seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False, pin_memory=True)
    idx = 0
    for inputs, targets, meta_info in tqdm(dataloader):
        idx+=1
        center_left = meta_info['center']
        val_left = meta_info['joint_val'].view(-1, seq_len, 1, 1)
        gt_joint_xyz_left = targets['joint_3d'] - center_left
        in_joint_xyz_left = inputs['joint'] - center_left

        error_left = calculate_error(in_joint_xyz_left * val_left, gt_joint_xyz_left * val_left)
        gt_joint_xyz_left = ((gt_joint_xyz_left / 0.15 + 1) * 128).numpy()
        in_joint_xyz_left = ((in_joint_xyz_left / 0.15 + 1) * 128).numpy()

        img = np.ones([256, 256, 3]) * 255
        for ii in range(batch_size):
            for jj in range(seq_len):
                img_gt = draw_2d_skeleton(img, gt_joint_xyz_left[ii, jj])
                img_pd = draw_2d_skeleton(img, in_joint_xyz_left[ii, jj])
                cv.imwrite('./mean/joint_left_gt_%d_%d.png' % (ii, jj), img_gt)
                cv.imwrite('./mean/joint_left_pd_%d_%d.png' % (ii, jj), img_pd)
        print(error_left)
        break

@torch.no_grad()
def draw_mano_data():
    batch_size = 8
    num_workers = 0
    seq_len = 27
    data_dir = 'C:\\Users\\Admin\\Desktop\\dataset\\interHand\\InterHand2.6M_30fps_batch1'
    anno_dir = 'C:\\Users\\Admin\\Desktop\\dataset\\interHand\\interhand2.6m_30fps'

    dataset = VideoJointInterHand_dataset(data_dir, anno_dir, 'test', seq_len=seq_len, aug=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False, pin_memory=True)

    for inputs, targets, meta_info in tqdm(dataloader):
        center_left = meta_info['center_left']
        center_right = meta_info['center_right']
        val_left = meta_info['joint_val_left'].view(-1, seq_len, 1, 1)
        val_right = meta_info['joint_val_right'].view(-1, seq_len, 1, 1)

        gt_joint_xyz_left = targets['joint_3d_left'] - center_left
        gt_joint_xyz_right = targets['joint_3d_right'] - center_right

        # left_mano_para = inputs['in_mano_left'].view(-1, 68)
        # in_mesh_xyz_left, in_joint_xyz_left = mano['left'](left_mano_para[:, :9+45], left_mano_para[:, 9+45:9+45+10])
        # in_joint_xyz_left = in_joint_xyz_left.view(-1, seq_len, 21, 3)
        # right_mano_para = inputs['in_mano_right'].view(-1, 68)
        # in_mesh_xyz_right, in_joint_xyz_right = mano['right'](right_mano_para[:, :9 + 45], right_mano_para[:, 9 + 45:9 + 45 + 10])
        # in_joint_xyz_right = in_joint_xyz_right.view(-1, seq_len, 21, 3)

        in_joint_xyz_left = inputs['in_joint_left'] - center_left
        in_joint_xyz_right = inputs['in_joint_right'] - center_right

        error_left = calculate_error(in_joint_xyz_left * val_left, gt_joint_xyz_left * val_left)
        error_right = calculate_error(in_joint_xyz_right * val_right, gt_joint_xyz_right * val_right)

        gt_joint_xyz_left = ((gt_joint_xyz_left / 0.15 + 1) * 128).numpy()
        gt_joint_xyz_right = ((gt_joint_xyz_right / 0.15 + 1) * 128).numpy()

        in_joint_xyz_left = ((in_joint_xyz_left / 0.15 + 1) * 128).numpy()
        in_joint_xyz_right = ((in_joint_xyz_right / 0.15 + 1) * 128).numpy()

        img = np.ones([256, 256, 3]) * 255
        for ii in range(batch_size):
            for jj in range(seq_len):
                img_gt = draw_2d_skeleton(img, gt_joint_xyz_left[ii, jj])
                img_pd = draw_2d_skeleton(img, in_joint_xyz_left[ii, jj])
                cv.imwrite('./mean/joint_left_gt_%d_%d.png' % (ii, jj), img_gt)
                cv.imwrite('./mean/joint_left_pd_%d_%d.png' % (ii, jj), img_pd)

                img_gt = draw_2d_skeleton(img, gt_joint_xyz_right[ii, jj])
                img_pd = draw_2d_skeleton(img, in_joint_xyz_right[ii, jj])
                cv.imwrite('./mean/joint_right_pd_%d_%d.png' % (ii, jj), img_pd)
                cv.imwrite('./mean/joint_right_gt_%d_%d.png' % (ii, jj), img_gt)

        print(error_left)
        print(error_right)
        break

if __name__ == '__main__':
    # generate_single_pred_data()
    draw_joint_data()
    # draw_mano_data()
