import numpy as np
import random
import math
import cv2 as cv
import pickle
import torch

import os
import sys
import imgaug.augmenters as iaa

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.config import get_cfg_defaults
import roma
import torch.nn.functional as F


def projection(scale, trans2d, label3d, img_size=256):
    scale = scale * img_size
    trans2d = trans2d * img_size / 2 + img_size / 2
    trans2d = trans2d

    label2d = scale * label3d[:, :2] + trans2d
    return label2d

# -1, 1 ->
def inv_projection_batch_uv(scale, trans2d, label2d):
    """orthodox projection
    Input:
        scale: (B)
        trans2d: (B, 2)
        label2d: (B x N x 3)
    Returns:
        (B, N, 3)
    """
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1).unsqueeze(-1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(-1)
    trans2d = trans2d.unsqueeze(1)

    label3d = label2d * scale + trans2d
    return label3d


def projection_batch_xy(scale, trans2d, label3d):
    """orthodox projection
    Input:
        scale: (B)
        trans2d: (B, 2)
        label3d: (B x N x 3)
    Returns:
        (B, N, 2)
    """
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1).unsqueeze(-1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(-1)
    trans2d = trans2d.unsqueeze(1)

    label2d = scale * label3d[..., :2] + trans2d
    return label2d

# -1, 1 ->
def inv_projection_batch(scale, trans2d, label2d, img_size=256):
    """orthodox projection
    Input:
        scale: (B)
        trans2d: (B, 2)
        label2d: (B x N x 3)
    Returns:
        (B, N, 3)
    """
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1).unsqueeze(-1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(-1)
    trans2d = trans2d.unsqueeze(1)

    label3d = label2d[:, :, :2] * scale + trans2d
    label3d = torch.cat((label3d, label2d[..., 2:]), dim=-1)
    return label3d


def projection_batch(scale, trans2d, label3d, img_size=256):
    """orthodox projection
    Input:
        scale: (B)
        trans2d: (B, 2)
        label3d: (B x N x 3)
    Returns:
        (B, N, 2)
    """
    scale = scale * img_size  # bs
    if scale.dim() == 1:
        scale = scale.unsqueeze(-1).unsqueeze(-1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(-1)
    trans2d = trans2d * img_size / 2 + img_size / 2  # bs x 2
    trans2d = trans2d.unsqueeze(1)

    label2d = scale * label3d[..., :2] + trans2d
    return label2d


def projection_batch_np(scale, trans2d, label3d, img_size=256):
    """orthodox projection
    Input:
        scale: (B)
        trans2d: (B, 2)
        label3d: (B x N x 3)
    Returns:
        (B, N, 2)
    """
    scale = scale * img_size  # bs
    if scale.dim() == 1:
        scale = scale[..., np.newaxis, np.newaxis]
    if scale.dim() == 2:
        scale = scale[..., np.newaxis]
    trans2d = trans2d * img_size / 2 + img_size / 2  # bs x 2
    trans2d = trans2d[:, np.newaxis, :]

    label2d = scale * label3d[..., :2] + trans2d
    return label2d


def get_mano_path():
    cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.path.join(abspath, cfg.MISC.MANO_PATH)
    mano_path = {'left': os.path.join(path, 'MANO_LEFT.pkl'),
                 'right': os.path.join(path, 'MANO_RIGHT.pkl')}
    return mano_path

def get_mano_dir():
    cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    path = os.path.join(abspath, cfg.MISC.MANO_PATH)
    return path

def get_graph_dict_path():
    cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    graph_path = {'left': os.path.join(abspath, cfg.MISC.GRAPH_LEFT_DICT_PATH),
                  'right': os.path.join(abspath, cfg.MISC.GRAPH_RIGHT_DICT_PATH)}
    return graph_path


def get_dense_color_path():
    cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dense_path = os.path.join(abspath, cfg.MISC.DENSE_COLOR)
    return dense_path


def get_mano_seg_path():
    cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    seg_path = os.path.join(abspath, cfg.MISC.MANO_SEG_PATH)
    return seg_path


def get_upsample_path():
    cfg = get_cfg_defaults()
    abspath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    upsample_path = os.path.join(abspath, cfg.MISC.UPSAMPLE_PATH)
    return upsample_path



def uvd2xyz(coord, camera):
    fx, fy, fu, fv = camera[:, 0:1, 0:1], camera[:, 1:2, 1:2], camera[:, 0:1, 2:3], camera[:, 1:2, 2:3]
    x = (coord[:, :, 0:1] - fu) * coord[:, :, 2:3] / fx
    y = (coord[:, :, 1:2] - fv) * coord[:, :, 2:3] / fy
    coord_xyz = np.concatenate((x, y, coord[:, :, 2:3]), axis=-1)
    return coord_xyz


class imgUtils():
    def __init__(self, coco_path=''):
        super(imgUtils, self).__init__()
        self.seq = iaa.Sequential([
            iaa.Sometimes(0.3, iaa.MotionBlur(k=(3, 15), order=0))
        ])

    @ staticmethod
    def get_scale_mat(center, scale=1.0):
        scaleMat = np.zeros((3, 3), dtype='float32')
        scaleMat[0, 0] = scale
        scaleMat[1, 1] = scale
        scaleMat[2, 2] = 1.0
        t = np.matmul((np.identity(3, dtype='float32') - scaleMat), center)
        scaleMat[0, 2] = t[0]
        scaleMat[1, 2] = t[1]
        return scaleMat

    @ staticmethod
    def get_rotation_mat(center, theta=0):
        # t = theta * (3.14159 / 180)
        t = np.deg2rad(theta)
        rotationMat = np.zeros((3, 3), dtype='float32')
        rotationMat[0, 0] = math.cos(t)
        rotationMat[0, 1] = -math.sin(t)
        rotationMat[1, 0] = math.sin(t)
        rotationMat[1, 1] = math.cos(t)
        rotationMat[2, 2] = 1.0
        t = np.matmul((np.identity(3, dtype='float32') - rotationMat), center)
        rotationMat[0, 2] = t[0]
        rotationMat[1, 2] = t[1]
        return rotationMat

    @ staticmethod
    def get_rotation_mat3d(theta=0):
        # t = theta * (3.14159 / 180)
        t = np.deg2rad(theta)
        rotationMat = np.zeros((3, 3), dtype='float32')
        rotationMat[0, 0] = math.cos(t)
        rotationMat[0, 1] = -math.sin(t)
        rotationMat[1, 0] = math.sin(t)
        rotationMat[1, 1] = math.cos(t)
        rotationMat[2, 2] = 1.0
        return rotationMat

    @ staticmethod
    def get_affine_mat(theta=0, scale=1.0,
                       u=0, v=0,
                       height=480, width=640):
        center = np.array([width / 2, height / 2, 1], dtype='float32')
        rotationMat = imgUtils.get_rotation_mat(center, theta)
        scaleMat = imgUtils.get_scale_mat(center, scale)
        trans = np.identity(3, dtype='float32')
        trans[0, 2] = u
        trans[1, 2] = v
        affineMat = np.matmul(scaleMat, rotationMat)
        affineMat = np.matmul(trans, affineMat)
        return affineMat

    @staticmethod
    def img_trans(theta, scale, u, v, img):
        size = img.shape[0]
        u = int(u * size / 2)
        v = int(v * size / 2)
        affineMat = imgUtils.get_affine_mat(theta=theta, scale=scale,
                                            u=u, v=v,
                                            height=256, width=256)
        return cv.warpAffine(src=img,
                             M=affineMat[0:2, :],
                             dsize=(256, 256),
                             dst=img,
                             flags=cv.INTER_LINEAR,
                             borderMode=cv.BORDER_REPLICATE,
                             borderValue=(0, 0, 0)
                             )

    # 考虑3D空间中的平移
    @staticmethod
    def data_augmentation_3D(theta, scale, u, v, cam,
                          img_list=None, label2d_list=None, label3d_list=None,
                          R=None,
                          img_size=(256,256)):
        height, width = img_size
        affineMat = imgUtils.get_affine_mat(theta=theta, scale=scale,
                                            u=u, v=v,
                                            height=height, width=width)
        if img_list is not None:
            img_list_out = []
            for img in img_list:
                img_list_out.append(cv.warpAffine(src=img,
                                                  M=affineMat[0:2, :],
                                                  dsize=(height, width)))
        else:
            img_list_out = None
        seq_len = len(img_list)
        affineMat = np.stack([affineMat]*seq_len, axis=0)
        if label2d_list is not None:
            label2d_list_out = []
            label3d_list_out = []
            for i in range(len(label2d_list)):
                label2d = label2d_list[i]
                label3d = label3d_list[i]
                label2d_aug = np.matmul(label2d[:, :, 0:2], affineMat[:, 0:2, 0:2].transpose(0, 2, 1)) + affineMat[:, 0:2, 2:3].transpose(0, 2, 1)
                label2d_list_out.append(label2d_aug)
                labeluvd_aug = np.concatenate((label2d_aug, label3d[:, :, 2:3]), axis=-1)
                label3d_list_out.append(uvd2xyz(labeluvd_aug, cam))
        else:
            label2d_list_out = None
            label3d_list_out = None

        if R is not None:
            R_delta = imgUtils.get_rotation_mat3d(theta)
            R_delta = np.stack([R_delta] * seq_len, axis=0)
            R = np.matmul(R_delta, R)
        else:
            R = imgUtils.get_rotation_mat3d(theta)

        return img_list_out, label2d_list_out, label3d_list_out, R

    # 考虑3D空间中的平移
    @staticmethod
    def trans_augmentation_3D(u, v, cam,
                          img_list=None, label2d_list=None, label3d_list=None,
                          img_size=(256, 256)):
        height, width = img_size
        M = np.identity(3, dtype='float32')
        M = np.expand_dims(M, 0).repeat(len(img_list), axis=0)
        M[:, 0, 2] = u
        M[:, 1, 2] = v
        if img_list is not None:
            img_list_out = []
            for index, img in enumerate(img_list):
                img_list_out.append(cv.warpAffine(src=img, M=M[index, 0:2, :], dsize=(height, width)))
        else:
            img_list_out = None
        if label2d_list is not None:
            label2d_list_out = []
            label3d_list_out = []
            for i in range(len(label2d_list)):
                label2d = label2d_list[i]
                label3d = label3d_list[i]
                label2d_aug = label2d[:, :, 0:2] + M[:, 0:2, 2:3].transpose(0, 2, 1)
                label2d_list_out.append(label2d_aug)
                labeluvd_aug = np.concatenate((label2d_aug, label3d[:, :, 2:3]), axis=-1)
                label3d_list_out.append(uvd2xyz(labeluvd_aug, cam))
        else:
            label2d_list_out = None
            label3d_list_out = None

        return img_list_out, label2d_list_out, label3d_list_out

    @ staticmethod
    def add_noise(img, noise=0.00, scale=255.0, alpha=0.3, beta=0.05):
        # add brightness noise & add random gaussian noise
        a = np.random.uniform(1 - alpha, 1 + alpha, 3)
        b = scale * beta * (2 * random.random() - 1)
        img = a * img + b + scale * np.random.normal(loc=0.0, scale=noise, size=img.shape)
        img = np.clip(img, 0, scale).astype(np.uint8)
        return img

    @ staticmethod
    def aug_color(img, color_factor=0.2):
        c_up = 1.0 + color_factor
        c_low = 1.0 - color_factor
        color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])
        img = np.clip(img * color_scale[None, None, :], 0, 255).astype(np.uint8)
        return img

    @staticmethod
    def get_aug_config(scale_factor=0.1, rot_factor=180, transl_factor=10, flip=True):
        scale = 1 + (np.random.rand() * 2 - 1) * scale_factor
        rot = (np.random.rand() * 2 - 1) * rot_factor
        transl_x = (np.random.rand() * 2 - 1) * transl_factor
        transl_y = (np.random.rand() * 2 - 1) * transl_factor
        if flip:
            do_flip = random.random() <= 0.5
        else:
            do_flip = False

        return scale, rot, transl_x, transl_y, do_flip

    @staticmethod
    def flip(img_list=None, label2d_list=None, img_size=256):
        if img_list is not None:
            img_list_out = []
            for img in img_list:
                img_list_out.append(img[:, ::-1, :])
        else:
            img_list_out = None

        if label2d_list is not None:
            label2d_list_out = []
            for label2d in label2d_list:
                label2d_out = label2d.copy()
                label2d_out[:, :, 0:1] = img_size - label2d_out[:, :, 0:1] - 1
                label2d_list_out.append(label2d_out)
        else:
            label2d_list_out = None

        return img_list_out, label2d_list_out

    @staticmethod
    def bi_flip(img_list=None, label2d_list=None, img_size=256):
        flip_direction = 0
        if random.random() <= 0.5:
            flip_direction = 1

        if img_list is not None:
            img_list_out = []
            for img in img_list:
                if flip_direction == 0:
                    img_list_out.append(img[:, ::-1, :])
                else:
                    img_list_out.append(img[::-1, :, :])
        else:
            img_list_out = None

        if label2d_list is not None:
            label2d_list_out = []
            for label2d in label2d_list:
                label2d_out = label2d.copy()
                if flip_direction == 0:
                    label2d_out[:, 0:1] = img_size - label2d_out[:, 0:1] - 1
                else:
                    label2d_out[:, 1:2] = img_size - label2d_out[:, 1:2] - 1
                label2d_list_out.append(label2d_out)
        else:
            label2d_list_out = None

        return img_list_out, label2d_list_out

    def blur(self, img):
        return self.seq(image=img)


def rodrigues_batch(axis):
    # axis : bs * 3
    # return: bs * 3 * 3
    bs = axis.shape[0]
    Imat = torch.eye(3, dtype=axis.dtype, device=axis.device).repeat(bs, 1, 1)  # bs * 3 * 3
    angle = torch.norm(axis, p=2, dim=1, keepdim=True) + 1e-8  # bs * 1
    axes = axis / angle  # bs * 3
    sin = torch.sin(angle).unsqueeze(2)  # bs * 1 * 1
    cos = torch.cos(angle).unsqueeze(2)  # bs * 1 * 1
    L = torch.zeros((bs, 3, 3), dtype=axis.dtype, device=axis.device)
    L[:, 2, 1] = axes[:, 0]
    L[:, 1, 2] = -axes[:, 0]
    L[:, 0, 2] = axes[:, 1]
    L[:, 2, 0] = -axes[:, 1]
    L[:, 1, 0] = axes[:, 2]
    L[:, 0, 1] = -axes[:, 2]
    return Imat + sin * L + (1 - cos) * L.bmm(L)


def axis2Rmat(axis):
    # axis: bs x 3
    rotation_mat = rodrigues_batch(axis.view(-1, 3))
    rotation_mat = rotation_mat.view(-1, 3, 3)
    return rotation_mat


class JointUtils():
    def __init__(self):
        super(JointUtils, self).__init__()

    def seq_rotation(self, joint_left_list, joint_right_list, center_list, sigma=1.0):
        t, _, _ = joint_left_list[0].size()
        theta = torch.rand([1, 3]) * np.pi * sigma
        R = axis2Rmat(theta)
        rot_joint_left_list = []
        rot_joint_right_list = []
        for joint_left, joint_right, center in zip(joint_left_list, joint_right_list, center_list):
            joint_left_norm, joint_right_norm = joint_left - center, joint_right - center
            joint_rot_left = torch.matmul(joint_left_norm, R) + center
            joint_rot_right = torch.matmul(joint_right_norm, R) + center
            rot_joint_left_list.append(joint_rot_left)
            rot_joint_right_list.append(joint_rot_right)
        return rot_joint_left_list, rot_joint_right_list

    def part_rotation(self, joint_left_list, joint_right_list, center_list):
        T, J, _ = joint_left_list[0].size()
        part_num = np.random.randint(1, 4)
        steps = np.random.choice(np.arange(int(T//6), int(T//3)), [part_num-1], replace=False)
        steps = np.array(list(steps) + [T-np.sum(steps)-1])

        rand_deg = (torch.rand([part_num, 3])-0.5)*90/180*np.pi
        rand_deg = torch.cat((rand_deg, torch.zeros([1, 3])), dim=0)
        deg_select_id = np.random.choice(part_num + 1, [part_num], replace=True)
        select_rot_deg = rand_deg[deg_select_id]
        init_rot = (torch.rand([3])-0.5)*np.pi*2
        rand_quats = [init_rot]
        for deg in select_rot_deg:
            cur_rot = rand_quats[-1] + deg
            rand_quats.append(cur_rot)
        rand_quats = torch.stack(rand_quats, dim=0)
        rand_quats = roma.rotvec_to_unitquat(rand_quats)
        # rand_quats = roma.random_unitquat(part_num+3)
        # quats_select_id = np.random.choice(part_num+3, [part_num + 1], replace=True)
        # rand_quats = rand_quats[quats_select_id]

        quat_list = []
        for index in range(part_num):
            step = torch.linspace(0, 1.0, steps[index])
            quat_list.append(roma.unitquat_slerp(rand_quats[index:index+1], rand_quats[index+1:index+2], step).squeeze(1))
        quat_list.append(rand_quats[-1:])
        quats = torch.cat(quat_list, dim=0)
        R = roma.unitquat_to_rotmat(quats)
        rot_joint_left_list = []
        rot_joint_right_list = []
        for joint_left, joint_right, center in zip(joint_left_list, joint_right_list, center_list):
            joint_left_norm, joint_right_norm = joint_left - center, joint_right - center
            joint_rot_left = torch.matmul(joint_left_norm, R) + center
            joint_rot_right = torch.matmul(joint_right_norm, R) + center
            rot_joint_left_list.append(joint_rot_left)
            rot_joint_right_list.append(joint_rot_right)
        return rot_joint_left_list, rot_joint_right_list

    def seq_scale(self, joint_left_list, joint_right_list, center_list, sigma=0.4):
        scale = 1 + (torch.rand([1])*sigma - sigma/2)
        scale_joint_left_list = []
        scale_joint_right_list = []
        for joint_left, joint_right, center in zip(joint_left_list, joint_right_list, center_list):
            joint_left_norm, joint_right_norm = joint_left - center, joint_right - center
            joint_left_scale = joint_left_norm * scale + center
            joint_right_scale = joint_right_norm * scale + center
            scale_joint_left_list.append(joint_left_scale)
            scale_joint_right_list.append(joint_right_scale)
        return scale_joint_left_list, scale_joint_right_list

    def flip(self, joint_left_list, joint_right_list, center_list):
        joint_aug_left_list = []
        joint_aug_right_list = []
        for joint_left, joint_right, center in zip(joint_left_list, joint_right_list, center_list):
            joint_left_flip, joint_right_flip = joint_left.clone()-center, joint_right.clone()-center
            joint_left_flip[..., 0] *= -1
            joint_right_flip[..., 0] *= -1
            joint_left_flip, joint_right_flip = joint_right_flip+center, joint_left_flip+center
            joint_aug_left_list.append(joint_left_flip)
            joint_aug_right_list.append(joint_right_flip)
        return joint_aug_left_list, joint_aug_right_list

    def seq_flip(self, joint_left_list, joint_right_list):
        joint_aug_left_list = []
        joint_aug_right_list = []
        for joint_left, joint_right in zip(joint_left_list, joint_right_list):
            joint_aug_left_list.append(torch.flip(joint_left, dims=[0]))
            joint_aug_right_list.append(torch.flip(joint_right, dims=[0]))
        return joint_aug_left_list, joint_aug_right_list

    def seq_aug(self, joint_left_list, joint_right_list, center_list):
        if np.random.rand() > 0.5:
            joint_left_list, joint_right_list = self.seq_flip(joint_left_list, joint_right_list)
        if np.random.rand() > 0.5:
            joint_left_list, joint_right_list = self.part_rotation(joint_left_list, joint_right_list, center_list)
        joint_left_list, joint_right_list = self.seq_scale(joint_left_list, joint_right_list, center_list)
        return joint_left_list, joint_right_list

    def jitter(self, joint_left_list, joint_right_list, sigma=0.01):
        t, j, c = joint_left_list[0].size()
        noise_left = (torch.randn([t, j, c])-0.5)*sigma
        noise_right = (torch.randn([t, j, c])-0.5)*sigma
        jitter_joint_left_list = []
        jitter_joint_right_list = []
        for joint_left, joint_right in zip(joint_left_list, joint_right_list):
            joint_left_jitter = joint_left + noise_left
            joint_right_jitter = joint_right + noise_right
            jitter_joint_left_list.append(joint_left_jitter)
            jitter_joint_right_list.append(joint_right_jitter)
        return jitter_joint_left_list, jitter_joint_right_list

    def joint_jitter(self, joint_left_list, joint_right_list, sigma=0.01):
        jitter_sigma = [
            0.5,
            0.5, 1, 1.5, 2,
            0.5, 1, 1.5, 2,
            0.5, 1, 1.5, 2,
            0.5, 1, 1.5, 2,
            0.5, 1, 1.5, 2,
        ]
        device = joint_left_list[0].device
        jitter_sigma = torch.Tensor(jitter_sigma).to(device).view(1, 21, 1)
        t, j, c = joint_left_list[0].size()
        noise_left = (torch.randn([t, j, c])-0.5)*sigma*jitter_sigma
        noise_right = (torch.randn([t, j, c])-0.5)*sigma*jitter_sigma
        jitter_joint_left_list = []
        jitter_joint_right_list = []
        for joint_left, joint_right in zip(joint_left_list, joint_right_list):
            joint_left_jitter = joint_left + noise_left
            joint_right_jitter = joint_right + noise_right
            jitter_joint_left_list.append(joint_left_jitter)
            jitter_joint_right_list.append(joint_right_jitter)
        return jitter_joint_left_list, jitter_joint_right_list

    def joint_center_jitter(self, joint_left_list, joint_right_list, sigma=0.01):
        t, j, c = joint_left_list[0].size()
        noise_left = (torch.randn([t, 1, c])-0.5)*sigma
        noise_right = (torch.randn([t, 1, c])-0.5)*sigma
        jitter_joint_left_list = []
        jitter_joint_right_list = []
        for joint_left, joint_right in zip(joint_left_list, joint_right_list):
            joint_left_jitter = joint_left + noise_left
            joint_right_jitter = joint_right + noise_right
            jitter_joint_left_list.append(joint_left_jitter)
            jitter_joint_right_list.append(joint_right_jitter)
        return jitter_joint_left_list, jitter_joint_right_list

    def rotation(self, joint_left_list, joint_right_list, center_list, sigma=1.0):
        t, j, _ = joint_left_list[0].size()
        theta = torch.rand([t, 3]) * np.pi * sigma
        R = axis2Rmat(theta)
        rot_joint_left_list = []
        rot_joint_right_list = []
        for joint_left, joint_right, center in zip(joint_left_list, joint_right_list, center_list):
            joint_left_norm, joint_right_norm = joint_left - center, joint_right - center
            joint_rot_left = torch.matmul(joint_left_norm, R) + center
            joint_rot_right = torch.matmul(joint_right_norm, R) + center
            rot_joint_left_list.append(joint_rot_left)
            rot_joint_right_list.append(joint_rot_right)
        return rot_joint_left_list, rot_joint_right_list

    def scale(self, joint_left_list, joint_right_list, center_list, sigma=0.4):
        t, j, _ = joint_left_list[0].size()
        scale = 1 + (torch.randn([t, 1, 1])*sigma - sigma/2)
        scale_joint_left_list = []
        scale_joint_right_list = []
        for joint_left, joint_right, center in zip(joint_left_list, joint_right_list, center_list):
            joint_left_norm, joint_right_norm = joint_left - center, joint_right - center
            joint_left_scale = joint_left_norm * scale + center
            joint_right_scale = joint_right_norm * scale + center
            scale_joint_left_list.append(joint_left_scale)
            scale_joint_right_list.append(joint_right_scale)
        return scale_joint_left_list, scale_joint_right_list

    def finger_ambiguity(self, joints):
        T, J, _ = joints.size()
        finger_id = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]])
        select_finger_id = np.random.randint(0, 5)
        frame_aug_joint = joints.clone().reshape(-1, 3)

        aug_frame_len = np.random.randint(2, int(T*0.4))
        start_id = np.random.randint(2, T-aug_frame_len)
        end_id = min(start_id + aug_frame_len + 1, T)
        percent = np.random.uniform(0.7, 1)
        select_frame_id = np.random.choice(np.arange(start_id, end_id), int(np.round((end_id-start_id)*percent)), replace=False)
        finger_a = select_finger_id
        finger_b = (select_finger_id + np.random.choice([-1, 1, -2, 2])) % 5
        finger_noise = torch.rand([select_frame_id.shape[0], 4]) * torch.Tensor([0.1, 0.2, 0.6, 0.8])
        select_id_a = (select_frame_id.reshape(-1, 1)*J + finger_id[finger_a].reshape(1, -1)).reshape(-1)
        select_id_b = (select_frame_id.reshape(-1, 1)*J + finger_id[finger_b].reshape(1, -1)).reshape(-1)
        finger_noise = finger_noise.view(-1, 1)

        joint_a = frame_aug_joint[select_id_a, :].clone()
        joint_b = frame_aug_joint[select_id_b, :].clone()
        frame_aug_joint[select_id_a, :] = joint_a + (joint_b - joint_a) * finger_noise
        return frame_aug_joint.reshape(T, J, 3)

    def finger_aug_prob(self, joints_list):
        aug_joints_list = []
        for joint in joints_list:
            if np.random.rand() > 0.5:
                aug_joints_list.append(self.finger_ambiguity(joint))
            else:
                aug_joints_list.append(joint)
        return aug_joints_list

    def joint_aug(self, joint_left_list, joint_right_list, center_list, prob=0.8):
        if np.random.rand() > 0.5:
            joint_left_list, joint_right_list = self.rotation(joint_left_list, joint_right_list, center_list, 0.01)
        if np.random.rand() > 0.5:
            joint_left_list, joint_right_list = self.scale(joint_left_list, joint_right_list, center_list, 0.01)
        if np.random.rand() > 0.5:
            joint_left_list, joint_right_list = self.joint_jitter(joint_left_list, joint_right_list, 0.001)
        joint_left_list = self.finger_aug_prob(joint_left_list)
        joint_right_list = self.finger_aug_prob(joint_right_list)
        joint_left_list, joint_right_list = self.joint_center_jitter(joint_left_list, joint_right_list, 0.0005)
        return joint_left_list, joint_right_list

    def pose_rate_aug(self, mano_list, R_list, val_list, seq_len, expand=4, inter_num=4):
        T, _ = mano_list[0].size()
        high_rate_seq_len = (T-1)*(inter_num+1) + 1

        mano_split_len = []
        for list_iter in mano_list:
            mano_split_len.append(list_iter.size(-1))
        mano = torch.cat(mano_list, dim=-1)
        mano_inter = F.interpolate(mano.unsqueeze(0).permute(0, 2, 1), size=high_rate_seq_len, mode='linear', align_corners=True)

        R_split_len = [9]*len(R_list)
        R = torch.cat(R_list, dim=-1)
        R_inter = F.interpolate(R.unsqueeze(0).permute(0, 2, 1), size=high_rate_seq_len, mode='nearest').squeeze(0).permute(1, 0)

        val_split_len = [1]*len(val_list)
        val = torch.cat(val_list, dim=-1)
        val_inter = F.interpolate(val.unsqueeze(0).permute(0, 2, 1), size=high_rate_seq_len, mode='nearest').squeeze(0).permute(1, 0)

        # 生成时序分段，模拟一个时序中的动作速率可能发生变化，最低慢放插值的倍率, 最高快放扩展的倍率
        part_num = np.random.randint(1, 4)
        part_frame_num = [int(seq_len/part_num)]*(part_num-1)
        part_frame_num = np.array(part_frame_num + [seq_len - sum(part_frame_num)])
        stride = np.random.choice(np.arange(1, int((inter_num+1)*(expand-1))), [part_num], replace=False)
        start_id = 0
        id_list = []
        for i in range(part_num):
            part_ids = np.arange(start=start_id, step=stride[i], stop=start_id+stride[i]*part_frame_num[i])
            id_list.append(part_ids)
            start_id = part_ids[-1] + stride[i]
        ids = np.concatenate(id_list, axis=0)
        start_id = np.random.randint(0, high_rate_seq_len - ids[-1] -1)
        ids = ids + start_id

        mano_inter_list = torch.split(mano_inter.squeeze(0).permute(1, 0)[ids], tuple(mano_split_len), dim=-1)
        quat_inter_list = torch.split(R_inter[ids], tuple(R_split_len), dim=-1)
        val_inter_list = torch.split(val_inter[ids], tuple(val_split_len), dim=-1)
        return mano_inter_list, quat_inter_list, val_inter_list


class SingleJointUtils():
    def __init__(self):
        super(SingleJointUtils, self).__init__()

    def seq_rotation(self, joint_list, center_list, sigma=1.0):
        t, _, _ = joint_list[0].size()
        theta = torch.rand([1, 3]) * np.pi * sigma
        R = axis2Rmat(theta)
        rot_joint_list = []
        rot_joint_right_list = []
        for joint, center in zip(joint_list, center_list):
            joint_norm = joint - center
            joint_rot = torch.matmul(joint_norm, R) + center
            rot_joint_list.append(joint_rot)
        return rot_joint_list

    def part_rotation(self, joint_list, center_list):
        T, J, _ = joint_list[0].size()
        part_num = np.random.randint(1, 4)
        steps = np.random.choice(np.arange(int(T//6), int(T//3)), [part_num-1], replace=False)
        steps = np.array(list(steps) + [T-np.sum(steps)-1])

        rand_deg = (torch.rand([part_num, 3])-0.5)*90/180*np.pi
        rand_deg = torch.cat((rand_deg, torch.zeros([1, 3])), dim=0)
        deg_select_id = np.random.choice(part_num + 1, [part_num], replace=True)
        select_rot_deg = rand_deg[deg_select_id]
        init_rot = (torch.rand([3])-0.5)*np.pi*2
        rand_quats = [init_rot]
        for deg in select_rot_deg:
            cur_rot = rand_quats[-1] + deg
            rand_quats.append(cur_rot)
        rand_quats = torch.stack(rand_quats, dim=0)
        rand_quats = roma.rotvec_to_unitquat(rand_quats)
        # rand_quats = roma.random_unitquat(part_num+3)
        # quats_select_id = np.random.choice(part_num+3, [part_num + 1], replace=True)
        # rand_quats = rand_quats[quats_select_id]

        quat_list = []
        for index in range(part_num):
            step = torch.linspace(0, 1.0, steps[index])
            quat_list.append(roma.unitquat_slerp(rand_quats[index:index+1], rand_quats[index+1:index+2], step).squeeze(1))
        quat_list.append(rand_quats[-1:])
        quats = torch.cat(quat_list, dim=0)
        R = roma.unitquat_to_rotmat(quats)
        rot_joint_list = []
        rot_joint_right_list = []
        for joint, center in zip(joint_list, center_list):
            joint_norm = joint - center
            joint_rot = torch.matmul(joint_norm, R) + center
            rot_joint_list.append(joint_rot)
        return rot_joint_list

    def seq_scale(self, joint_list, center_list, sigma=0.4):
        scale = 1 + (torch.rand([1])*sigma - sigma/2)
        scale_joint_list = []
        for joint, center in zip(joint_list, center_list):
            joint_norm = joint - center
            joint_scale = joint_norm * scale + center
            scale_joint_list.append(joint_scale)
        return scale_joint_list

    def joint_flip(self, joint_list, center_list):
        joint_aug_list = []
        for joint, center in zip(joint_list, center_list):
            joint_flip = joint.clone()-center
            joint_flip[..., 0] *= -1
            joint_flip = joint_flip+center
            joint_aug_list.append(joint_flip)
        return joint_aug_list

    def seq_flip(self, joint_list):
        joint_aug_list = []
        for joint in joint_list:
            joint_aug_list.append(torch.flip(joint, dims=[0]))
        return joint_aug_list

    def seq_aug(self, joint_list, center_list):
        if np.random.rand() > 0.5:
            joint_list = self.seq_flip(joint_list)
        if np.random.rand() > 0.5:
            joint_list = self.part_rotation(joint_list, center_list)
        # joint_list = self.seq_scale(joint_list, center_list)
        return joint_list


    def frame_mask(self, joint_list, center_list, ratio=0.1):
        t, _, _ = joint_list[0].size()
        mask_id = np.random.rand(t) > ratio
        mask_id = mask_id.reshape([t, 1, 1])
        mask_joint_list = []
        for joints, centers in zip(joint_list, center_list):
            joints_norm = joints - centers
            joint_mask = joints_norm*mask_id
            joint_mask = joint_mask + centers
            mask_joint_list.append(joint_mask)
        return mask_joint_list

    def joint_jitter(self, joint_list, sigma=0.01):
        jitter_sigma = [
            0.5,
            0.5, 1, 1.5, 2,
            0.5, 1, 1.5, 2,
            0.5, 1, 1.5, 2,
            0.5, 1, 1.5, 2,
            0.5, 1, 1.5, 2,
        ]
        device = joint_list[0].device
        jitter_sigma = torch.Tensor(jitter_sigma).to(device).view(1, 21, 1)
        t, j, c = joint_list[0].size()
        noise = (torch.randn([t, j, c])-0.5)*sigma*jitter_sigma
        jitter_joint_list = []
        for joint in joint_list:
            joint_jitter = joint + noise
            jitter_joint_list.append(joint_jitter)
        return jitter_joint_list

    def joint_center_jitter(self, joint_list, sigma=0.01):
        t, j, c = joint_list[0].size()
        noise = (torch.randn([t, 1, c])-0.5)*sigma
        jitter_joint_list = []
        for joint in joint_list:
            joint_jitter = joint + noise
            jitter_joint_list.append(joint_jitter)
        return jitter_joint_list

    def rotation(self, joint_list, center_list, sigma=1.0):
        t, j, _ = joint_list[0].size()
        theta = torch.rand([t, 3]).float() * np.pi * sigma
        R = axis2Rmat(theta)
        rot_joint_list = []
        for joint, center in zip(joint_list, center_list):
            joint_norm = joint - center
            joint_rot = torch.matmul(joint_norm, R) + center
            rot_joint_list.append(joint_rot)
        return rot_joint_list

    def scale(self, joint_list, center_list, sigma=0.4):
        t, j, _ = joint_list[0].size()
        scale = 1 + (torch.randn([t, 1, 1])*sigma - sigma/2)
        scale_joint_list = []
        scale_joint_right_list = []
        for joint, center in zip(joint_list, center_list):
            joint_norm = joint - center
            joint_scale = joint_norm * scale + center
            scale_joint_list.append(joint_scale)
        return scale_joint_list

    def finger_ambiguity(self, joints):
        T, J, _ = joints.size()
        finger_id = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]])
        select_finger_id = np.random.randint(0, 5)
        frame_aug_joint = joints.clone().reshape(-1, 3)

        aug_frame_len = np.random.randint(2, int(T*0.4))
        start_id = np.random.randint(2, T-aug_frame_len)
        end_id = min(start_id + aug_frame_len + 1, T)
        percent = np.random.uniform(0.7, 1)
        select_frame_id = np.random.choice(np.arange(start_id, end_id), int(np.round((end_id-start_id)*percent)), replace=False)
        finger_a = select_finger_id
        finger_b = (select_finger_id + np.random.choice([-1, 1, -2, 2])) % 5
        finger_noise = torch.rand([select_frame_id.shape[0], 4]) * torch.Tensor([0.1, 0.2, 0.6, 0.8])
        select_id_a = (select_frame_id.reshape(-1, 1)*J + finger_id[finger_a].reshape(1, -1)).reshape(-1)
        select_id_b = (select_frame_id.reshape(-1, 1)*J + finger_id[finger_b].reshape(1, -1)).reshape(-1)
        finger_noise = finger_noise.view(-1, 1)

        joint_a = frame_aug_joint[select_id_a, :].clone()
        joint_b = frame_aug_joint[select_id_b, :].clone()
        frame_aug_joint[select_id_a, :] = joint_a + (joint_b - joint_a) * finger_noise
        return frame_aug_joint.reshape(T, J, 3)

    def finger_aug_prob(self, joints_list):
        aug_joints_list = []
        for joint in joints_list:
            if np.random.rand() > 0.5:
                aug_joints_list.append(self.finger_ambiguity(joint))
            else:
                aug_joints_list.append(joint)
        return aug_joints_list

    def joint_aug(self, joint_list, center_list):
        if np.random.rand() > 0.5:
            joint_list = self.rotation(joint_list, center_list, 0.01)
        # if np.random.rand() > 0.5:
        #     joint_list = self.scale(joint_list, center_list, 0.01)
        if np.random.rand() > 0.5:
            joint_list = self.joint_jitter(joint_list, 0.004)
        joint_list = self.finger_aug_prob(joint_list)
        joint_list = self.joint_center_jitter(joint_list, 0.0005)
        # joint_list = self.frame_mask(joint_list, center_list, 0.2)
        return joint_list

    def pose_rate_aug(self, mano_list, R_list, val_list, seq_len, expand=4, inter_num=4):
        T, _ = mano_list[0].size()
        high_rate_seq_len = (T-1)*(inter_num+1) + 1

        mano_split_len = []
        for list_iter in mano_list:
            mano_split_len.append(list_iter.size(-1))
        mano = torch.cat(mano_list, dim=-1)
        mano_inter = F.interpolate(mano.unsqueeze(0).permute(0, 2, 1), size=high_rate_seq_len, mode='linear', align_corners=True)

        R_split_len = [9]*len(R_list)
        R = torch.cat(R_list, dim=-1)
        R_inter = F.interpolate(R.unsqueeze(0).permute(0, 2, 1), size=high_rate_seq_len, mode='nearest').squeeze(0).permute(1, 0)

        val_split_len = [1]*len(val_list)
        val = torch.cat(val_list, dim=-1)
        val_inter = F.interpolate(val.unsqueeze(0).permute(0, 2, 1), size=high_rate_seq_len, mode='nearest').squeeze(0).permute(1, 0)

        # 生成时序分段，模拟一个时序中的动作速率可能发生变化，最低慢放插值的倍率, 最高快放扩展的倍率
        part_num = np.random.randint(1, 4)
        part_frame_num = [int(seq_len/part_num)]*(part_num-1)
        part_frame_num = np.array(part_frame_num + [seq_len - sum(part_frame_num)])
        stride = np.random.choice(np.arange(1, int((inter_num+1)*(expand-1))), [part_num], replace=False)
        start_id = 0
        id_list = []
        for i in range(part_num):
            part_ids = np.arange(start=start_id, step=stride[i], stop=start_id+stride[i]*part_frame_num[i])
            id_list.append(part_ids)
            start_id = part_ids[-1] + stride[i]
        ids = np.concatenate(id_list, axis=0)
        start_id = np.random.randint(0, high_rate_seq_len - ids[-1] -1)
        ids = ids + start_id

        mano_inter_list = torch.split(mano_inter.squeeze(0).permute(1, 0)[ids], tuple(mano_split_len), dim=-1)
        quat_inter_list = torch.split(R_inter[ids], tuple(R_split_len), dim=-1)
        val_inter_list = torch.split(val_inter[ids], tuple(val_split_len), dim=-1)
        return mano_inter_list, quat_inter_list, val_inter_list

if __name__ == '__main__':
    # joint = torch.rand([27, 21, 3])
    # center = torch.rand([27, 1, 3])
    a = JointUtils()
    # a.part_rotation([joint], [joint], [center])
    joint = torch.rand([26*4+1, 21, 3])
    center = torch.rand([26*4+1, 45])
    # a1, a2 = a.pose_rate_aug([joint], [joint], 27, inter_num=4)
    # print('finish')
    a.finger_ambiguity(joint)
