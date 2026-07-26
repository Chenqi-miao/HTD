"""
seqhand_motion.py — 运动编码器专用的数据加载器
==============================================
复用 SeqHand（NPZ 版）的数据加载逻辑，额外产出速度/加速度伪 GT。

与 SeqHand 的接口兼容：
  - 同样的 NPZ 文件读取
  - 同样的序列切分逻辑
  - 额外的 `pseudo_gt` 和 `mask` 返回字段
"""

import os
import time
import numpy as np
import torch
import zipfile
from torch.utils.data import Dataset
import numpy.lib.format as np_format

from dataset.seqhand import load_npz_array


class SeqHandMotion(Dataset):
    """
    运动编码器数据集

    以和 SeqHand 相同的方式加载 NPZ 数据，额外产出:
      pseudo_gt:
        joint_vel:   3D 速度 [V, F-1, J, 3]  从 GT 坐标差分
        joint_accel: 3D 加速度 [V, F-2, J, 3]  从 GT 速度差分
        joint_2d:    伪 2D 关键点 [V, F, J, 2] 从 GT 坐标取 (x, y)
        joint_xyz:   同 SeqHand 的 GT [V, F, J, 3]
        joint_in:    同 SeqHand 的输入 [V, F, J, 3]
      mask:
        vel:   速度有效 [V, F-1, J, 1]  (连续帧且 joint 有效)
        accel: 加速度有效 [V, F-2, J, 1] (连续帧且 joint 有效)
        kp:    2D 关键点有效 [V, F, J, 1] (同 joint_gt_val)

    Args:
        data_path: 数据路径 (./data/SeqHand)
        dataset_list: 数据集名称列表，如 ['InterHand_train']
        min_seq_len: 序列最小长度 (默认 30)
        seq_len: 输出序列长度 (默认 15)
        view_num: 视角数 (默认 1)
        data_num: 限制样本数 (默认 None)
    """
    def __init__(self, data_path, dataset_list, joint_num=21,
                 min_seq_len=30, view_num=1, seq_len=15, data_num=None):
        self.min_seq_len = min_seq_len
        self.seq_len = seq_len
        self.view_num = view_num
        self.joint_num = joint_num
        self.data_path = os.path.join(data_path, 'npz_30')
        self.dataset_list = dataset_list if isinstance(dataset_list, list) else [dataset_list]
        self.data_num = data_num

        print('Loading Motion dataset ...')
        t0 = time.time()
        self.generator_seq()
        print(f'  ✓ {len(self.joint_xyz_gt_list)} samples  ({time.time()-t0:.0f}s)')

    def __len__(self):
        return len(self.joint_xyz_gt_list)

    def __getitem__(self, idx):
        while idx < len(self.joint_xyz_gt_list):
            data = self._load_sample(idx)
            if data is not None:
                return data
            idx += 1
        raise RuntimeError("No valid samples in dataset.")

    def _load_sample(self, idx):
        joint_in = self.joint_xyz_in_list[idx].reshape(
            [self.view_num, self.seq_len, self.joint_num, 3])
        joint_gt = self.joint_xyz_gt_list[idx].reshape(
            [self.view_num, self.seq_len, self.joint_num, 3])

        joint_valid_in = self.joint_valid_in_list[idx].reshape(
            [self.view_num, self.seq_len, self.joint_num])
        joint_valid_gt = self.joint_valid_gt_list[idx].reshape(
            [self.view_num, self.seq_len, self.joint_num])
        joint_world_valid = self.size_valid(joint_in) * joint_valid_in
        joint_type = self.hand_type_list[idx].reshape([self.seq_len, 1, 1])
        continuous_val = self.frame_consecutive(joint_gt[0])

        # 手型翻转（和 SeqHand 一致）
        center_gt = joint_gt[..., 9:10, :].copy()
        center_in = joint_in[..., 9:10, :].copy()
        joint_gt, joint_in = self.joint_flip(
            [joint_gt, joint_in], [center_gt, center_in],
            [joint_type == 1, joint_type == 1])

        joint_gt = np.repeat(joint_gt, self.view_num, axis=0)
        center_gt = joint_gt[..., 9:10, :].copy()

        if joint_valid_gt.sum() == 0 or joint_world_valid.sum() == 0:
            return None

        # ─── 构造 motion encoder 的伪 GT ─────────────────

        # joint_2d: 取 3D GT 的 (x, y) 作为伪 2D（无相机参数时的替代）
        pseudo_joint_2d = joint_gt[..., :2]  # [V, F, J, 2]

        # joint_vel: GT 坐标差分
        pseudo_joint_vel = joint_gt[:, 1:, :, :] - joint_gt[:, :-1, :, :]  # [V, F-1, J, 3]

        # joint_accel: GT 速度差分
        pseudo_joint_accel = (pseudo_joint_vel[:, 1:, :, :] -
                              pseudo_joint_vel[:, :-1, :, :])  # [V, F-2, J, 3]

        # ─── Mask ───────────────────────────────────────

        # continuous_val 标记连续帧: [F]
        cv = continuous_val  # [F] bool

        # 2D mask: 同 joint_gt_val
        kp_mask = (joint_valid_gt > 0).astype(np.float32)  # [V, F, J]

        # vel mask: 需要前后帧都连续且有效
        vel_mask_f = (cv[:-1] & cv[1:]).astype(np.float32)  # [F-1]
        vel_joint_mask = (joint_valid_gt[:, :-1, :] > 0) & (joint_valid_gt[:, 1:, :] > 0)
        vel_mask = (vel_mask_f[None, :, None] * vel_joint_mask).astype(np.float32)  # [V, F-1, J]

        # accel mask: 需要前后三帧都连续且有效
        accel_mask_f = (cv[:-2] & cv[1:-1] & cv[2:]).astype(np.float32)  # [F-2]
        accel_joint_mask = (joint_valid_gt[:, :-2, :] > 0) & \
                           (joint_valid_gt[:, 1:-1, :] > 0) & \
                           (joint_valid_gt[:, 2:, :] > 0)
        accel_mask = (accel_mask_f[None, :, None] * accel_joint_mask).astype(np.float32)

        # ─── 返回 ──────────────────────────────────────

        # 和 SeqHand 一致的字段
        inputs = {'joint_xyz': np.float32(joint_in)}
        targets = {'joint_xyz': np.float32(joint_gt)}
        meta_info = {
            "center_xyz": np.float32(center_gt),
            'continuous_val': np.float32(continuous_val),
            'joint_gt_val': np.float32(joint_valid_gt),
            'joint_in_val': np.float32(joint_world_valid),
        }

        # motion encoder 专用的额外 GT
        pseudo_gt = {
            'joint_xyz':  np.float32(joint_gt),              # [V, F, J, 3]
            'joint_in':   np.float32(joint_in),              # [V, F, J, 3]
            'joint_2d':   np.float32(pseudo_joint_2d),       # [V, F, J, 2]
            'joint_vel':  np.float32(pseudo_joint_vel),      # [V, F-1, J, 3]
            'joint_accel': np.float32(pseudo_joint_accel),   # [V, F-2, J, 3]
        }

        masks = {
            'kp':    np.float32(kp_mask[..., None]),          # [V, F, J, 1]
            'vel':   np.float32(vel_mask[..., None]),         # [V, F-1, J, 1]
            'accel': np.float32(accel_mask[..., None]),       # [V, F-2, J, 1]
        }

        return inputs, targets, meta_info, pseudo_gt, masks

    # ─── 序列生成 ───────────────────────────────────

    def generator_seq(self):
        self.joint_xyz_gt_list = []
        self.joint_xyz_in_list = []
        self.joint_valid_gt_list = []
        self.joint_valid_in_list = []
        self.hand_type_list = []

        data_dict = {}
        for dataset_name in self.dataset_list:
            gt_path = os.path.join(self.data_path, dataset_name, 'gt_joint.npz')
            in_path = os.path.join(self.data_path, dataset_name, 'in_joint.npz')
            gt_v_path = os.path.join(self.data_path, dataset_name, 'gt_joint_valid.npz')
            in_v_path = os.path.join(self.data_path, dataset_name, 'in_joint_valid.npz')
            type_path = os.path.join(self.data_path, dataset_name, 'hand_type.npz')

            with (zipfile.ZipFile(gt_path, 'r') as gt_zip,
                  zipfile.ZipFile(in_path, 'r') as in_zip,
                  zipfile.ZipFile(gt_v_path, 'r') as gt_v_zip,
                  zipfile.ZipFile(in_v_path, 'r') as in_v_zip,
                  zipfile.ZipFile(type_path, 'r') as type_zip):
                keys = gt_zip.namelist()
                for key in keys:
                    k = key.strip('.npy')
                    data_dict[k] = (
                        load_npz_array(gt_zip, key),
                        load_npz_array(in_zip, key),
                        load_npz_array(gt_v_zip, key),
                        load_npz_array(in_v_zip, key),
                        load_npz_array(type_zip, key),
                    )

        for key in data_dict.keys():
            joint_xyz_gt, joint_xyz_in, joint_valid_gt, joint_valid_in, hand_type = data_dict[key]
            seq_img_num = joint_xyz_gt.shape[0]
            first_frame = np.random.randint(0, self.seq_len)
            seq_num = int(np.ceil((seq_img_num - first_frame) / self.seq_len))
            seq_ids = np.arange(seq_num + 1) * self.seq_len + first_frame
            for ii in range(seq_num):
                img_idx = np.arange(seq_ids[ii] - self.seq_len // 2,
                                   seq_ids[ii] + self.seq_len // 2 + 1)
                img_idx_clip = np.clip(img_idx, a_min=0, a_max=seq_img_num - 1)
                self.joint_xyz_gt_list.append(joint_xyz_gt[img_idx_clip])
                self.joint_xyz_in_list.append(joint_xyz_in[img_idx_clip])
                self.joint_valid_gt_list.append(joint_valid_gt[img_idx_clip])
                self.joint_valid_in_list.append(joint_valid_in[img_idx_clip])
                self.hand_type_list.append(hand_type[img_idx_clip])

                if self.data_num is not None and len(self.joint_xyz_gt_list) > self.data_num:
                    return

    def frame_consecutive(self, seq):
        """检测连续帧（与 SeqHand 一致）"""
        seq_val = np.zeros([seq.shape[0]])
        diff = (seq[1:] - seq[:-1])
        diff = np.mean(np.sqrt((diff * diff).sum(-1)), axis=-1)
        seq_val[:-1] += (diff > 30)
        seq_val[1:] += (diff > 30)
        return seq_val == 0

    def size_valid(self, seq):
        """检测异常帧（与 SeqHand 一致）"""
        seq_min = np.min(seq[..., :2], axis=2)
        seq_max = np.max(seq[..., :2], axis=2)
        mask = (seq_max - seq_min) < 50
        mask = mask.sum(axis=-1) < 2
        return mask.reshape([self.view_num, self.seq_len, 1])

    def joint_flip(self, joint_list, center_list, flip_flag_list):
        """手型翻转（与 SeqHand 一致）"""
        joint_aug_list = []
        for joint, center, flag in zip(joint_list, center_list, flip_flag_list):
            joint_flip = joint - center
            joint_flip[..., 0] *= -1
            joint_flip = joint_flip + center
            joint_aug_list.append(joint_flip * flag + joint * (1 - flag))
        return joint_aug_list
