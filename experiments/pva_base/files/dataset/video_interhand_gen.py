import json
import torch
import pickle
import cv2 as cv
import numpy as np
import os.path as osp
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader, Dataset

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.manolayer import ManoLayer, rodrigues_batch
from dataset.dataset_utils import IMG_SIZE, HAND_BBOX_RATIO, HEATMAP_SIGMA, HEATMAP_SIZE, cut_img, video_cut_img
from utils.visualize import draw_2d_skeleton
from utils.video_utils import get_mano_path, imgUtils, JointUtils
import torchvision.transforms as transforms
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


class InterHandLoader():
    def __init__(self, data_path, split='train', mano_path=None):
        assert split in ['train', 'test', 'val']

        self.root_path = data_path
        self.img_root_path = os.path.join(self.root_path, 'images')
        self.annot_root_path = os.path.join(self.root_path, 'annotations')

        self.mano_layer = {'right': ManoLayer(mano_path['right'], center_idx=None),
                           'left': ManoLayer(mano_path['left'], center_idx=None)}
        fix_shape(self.mano_layer)

        self.split = split

        with open(osp.join(self.annot_root_path, self.split,
                           'InterHand2.6M_' + self.split + '_data.json')) as f:
            self.data_info = json.load(f)
        with open(osp.join(self.annot_root_path, self.split,
                           'InterHand2.6M_' + self.split + '_camera.json')) as f:
            self.cam_params = json.load(f)
        with open(osp.join(self.annot_root_path, self.split,
                           'InterHand2.6M_' + self.split + '_joint_3d.json')) as f:
            self.joints = json.load(f)
        with open(osp.join(self.annot_root_path, self.split,
                           'InterHand2.6M_' + self.split + '_MANO_NeuralAnnot.json')) as f:
            self.mano_params = json.load(f)

        self.data_size = len(self.data_info['images'])

    def __len__(self):
        return self.data_size

    def show_data(self, idx):
        for k in self.data_info['images'][idx].keys():
            print(k, self.data_info['images'][idx][k])
        for k in self.data_info['annotations'][idx].keys():
            print(k, self.data_info['annotations'][idx][k])

    def load_camera(self, idx):
        img_info = self.data_info['images'][idx]
        capture_idx = img_info['capture']
        cam_idx = img_info['camera']

        capture_idx = str(capture_idx)
        cam_idx = str(cam_idx)
        cam_param = self.cam_params[str(capture_idx)]
        cam_t = np.array(cam_param['campos'][cam_idx], dtype=np.float32).reshape(3)
        cam_R = np.array(cam_param['camrot'][cam_idx], dtype=np.float32).reshape(3, 3)
        cam_t = -np.dot(cam_R, cam_t.reshape(3, 1)).reshape(3) / 1000  # -Rt -> t

        # add camera intrinsics
        focal = np.array(cam_param['focal'][cam_idx], dtype=np.float32).reshape(2)
        princpt = np.array(cam_param['princpt'][cam_idx], dtype=np.float32).reshape(2)
        cameraIn = np.array([[focal[0], 0, princpt[0]],
                             [0, focal[1], princpt[1]],
                             [0, 0, 1]])
        return cam_R, cam_t, cameraIn

    def load_mano(self, idx):
        img_info = self.data_info['images'][idx]
        capture_idx = img_info['capture']
        frame_idx = img_info['frame_idx']

        capture_idx = str(capture_idx)
        frame_idx = str(frame_idx)
        mano_dict = {}
        coord_dict = {}
        for hand_type in ['left', 'right']:
            try:
                mano_param = self.mano_params[capture_idx][frame_idx][hand_type]
                mano_pose = torch.FloatTensor(mano_param['pose']).view(-1, 3)
                root_pose = mano_pose[0].view(1, 3)
                hand_pose = mano_pose[1:, :].view(1, -1)
                # hand_pose = hand_pose.view(1, -1, 3)
                mano = self.mano_layer[hand_type]
                mean_pose = mano.hands_mean
                hand_pose = mano.axis2pca(hand_pose + mean_pose)
                shape = torch.FloatTensor(mano_param['shape']).view(1, -1)
                trans = torch.FloatTensor(mano_param['trans']).view(1, 3)
                root_pose = rodrigues_batch(root_pose)

                handV, handJ = self.mano_layer[hand_type](root_pose, hand_pose, shape, trans=trans)
                mano_dict[hand_type] = {'R': root_pose.numpy(), 'pose': hand_pose.numpy(), 'shape': shape.numpy(),
                                        'trans': trans.numpy()}
                coord_dict[hand_type] = {'verts': handV, 'joints': handJ}
            except:
                mano_dict[hand_type] = None
                coord_dict[hand_type] = None

        return mano_dict, coord_dict

    def load_mano_data(self, idx):
        img_info = self.data_info['images'][idx]
        capture_idx = img_info['capture']
        frame_idx = img_info['frame_idx']

        capture_idx = str(capture_idx)
        frame_idx = str(frame_idx)
        mano_dict = {}
        for hand_type in ['left', 'right']:
            try:
                mano_param = self.mano_params[capture_idx][frame_idx][hand_type]
                mano_pose = torch.FloatTensor(mano_param['pose']).view(-1, 3)
                root_pose = mano_pose[0].view(1, 3)
                hand_pose = mano_pose[1:, :].view(1, -1)
                mano = self.mano_layer[hand_type]
                mean_pose = mano.hands_mean
                hand_pose = mano.axis2pca(hand_pose + mean_pose)
                shape = torch.FloatTensor(mano_param['shape']).view(1, -1)
                trans = torch.FloatTensor(mano_param['trans']).view(1, 3)
                root_pose = rodrigues_batch(root_pose)
                mano_dict[hand_type] = {'R': root_pose.numpy(), 'pose': hand_pose.numpy(), 'shape': shape.numpy(),
                                        'trans': trans.numpy()}
            except:
                mano_dict[hand_type] = None

        return mano_dict

    def load_img(self, idx):
        img_info = self.data_info['images'][idx]
        img = cv.imread(osp.join(self.img_root_path, self.split, img_info['file_name']))
        return img


InteractionHandSeqList = ['rightclaspleft', 'leftclaspright', 'fingergun', 'rightfistcoverleft', 'leftfistcoverright',
                          'interlockedfingers', 'pray', 'rightfistoverleft', 'leftfistoverright', 'rightbabybird',
                          'leftbabybird',
                          'interlockedfingerspread', 'fingersqueeze',
                          'palmerrub', 'knucklecrack', 'fingernoodle', 'itsybitsyspider', 'nontouchROM', 'touchROM',
                          'rightfingercountindexpoint', 'leftfingercountindexpoint',
                          'rightreceivethewafer', 'leftreceivethewafer', 'pointingtowardsfeatures',
                          'handscratch', 'rockpaperscissors', 'golfclaprol', 'golfclaplor', 'sarcasticclap',
                          'Interaction', '2_Hand']


def if_seq_continuous(seq_path):
    img_name_list = os.listdir(seq_path)
    img_num_list = []
    for img_name in img_name_list:
        img_num_list.append(int(img_name.split('.')[0][5:]))
    if len(img_num_list) <= 1:
        return False
    else:
        img_num = np.sort(np.array(img_num_list))
        img_diff = img_num[1:] - img_num[:-1]
        return np.sum(img_diff <= 3) == len(img_num[1:])


def if_inter_seq(seq_name):
    for name in InteractionHandSeqList:
        if name in seq_name:
            return True
    return False


def select_data(data_path, save_path, split, record_type):
    loader = InterHandLoader(data_path, split=split, mano_path=get_mano_path())
    record_data(loader, data_path, save_path, split, record_type)


def record_data(loader, data_path, save_path, split, record_type):
    seq_dict = {}  # 序列长度，相机数量
    idx = 0
    for i in tqdm(range(len(loader))):
        annotation = loader.data_info['annotations'][i]
        images_info = loader.data_info['images'][i]
        hand_type = annotation['hand_type']
        seq_name = images_info['seq_name']
        capture = "Capture" + str(images_info['capture'])
        camera = "cam" + images_info['camera']
        frame_idx = images_info['frame_idx']
        seq_path = osp.join(data_path, 'images', split, capture, seq_name, camera)

        if record_type == 'two':
            if hand_type == 'interacting' and if_inter_seq(seq_name) and if_seq_continuous(seq_path):
                os.makedirs(osp.join(save_path, split, capture, seq_name, camera, 'anno'), exist_ok=True)
                seq_full = '{}/{}/'.format(capture, seq_name)

                if not seq_full in seq_dict:
                    seq_dict.update({seq_full: {}})
                if not camera in seq_dict[seq_full]:
                    seq_dict[seq_full].update({camera: []})
                seq_dict[seq_full][camera].append(str(frame_idx))

                mano_dict = loader.load_mano_data(i)
                cam_R, cam_t, cameraIn = loader.load_camera(i)

                data_info = {}
                data_info['inter_idx'] = frame_idx
                data_info['image'] = images_info
                data_info['annotation'] = annotation
                data_info['mano_params'] = mano_dict
                data_info['camera'] = {'R': cam_R, 't': cam_t, 'camera': cameraIn}
                data_info['hand_type'] = hand_type
                with open(osp.join(save_path, split, capture, seq_name, camera, 'anno', '{}.pkl'.format(frame_idx)), 'wb') as file:
                    pickle.dump(data_info, file)
                idx += 1
        else:
            if if_seq_continuous(seq_path) and not if_inter_seq(seq_name) and not hand_type == 'interacting':
                os.makedirs(osp.join(save_path, split, capture, seq_name, camera, 'anno'), exist_ok=True)
                seq_full = '{}/{}/'.format(capture, seq_name)

                if not seq_full in seq_dict:
                    seq_dict.update({seq_full: {}})
                if not camera in seq_dict[seq_full]:
                    seq_dict[seq_full].update({camera: []})
                seq_dict[seq_full][camera].append(str(frame_idx))

                mano_dict = loader.load_mano_data(i)
                cam_R, cam_t, cameraIn = loader.load_camera(i)

                data_info = {}
                data_info['inter_idx'] = frame_idx
                data_info['image'] = images_info
                data_info['annotation'] = annotation
                data_info['mano_params'] = mano_dict
                data_info['camera'] = {'R': cam_R, 't': cam_t, 'camera': cameraIn}
                data_info['hand_type'] = hand_type
                with open(osp.join(save_path, split, capture, seq_name, camera, 'anno', '{}.pkl'.format(frame_idx)),'wb') as file:
                    pickle.dump(data_info, file)
                idx += 1
    data_info = []
    str_pad = '-'
    for key in seq_dict.keys():
        diff_str_list = []
        diff_str_num = []
        for camera in seq_dict[key].keys():
            img_srt = str_pad.join(seq_dict[key][camera])
            if img_srt in diff_str_list:
                diff_str_num[diff_str_list.index(img_srt)] += 1
            else:
                diff_str_list.append(img_srt)
                diff_str_num.append(1)
        max_seq_id = np.argmax(np.array(diff_str_num))
        main_seq = diff_str_list[max_seq_id]
        cam_name_list = []
        for camera in seq_dict[key].keys():
            img_srt = str_pad.join(seq_dict[key][camera])
            if img_srt == main_seq:
                cam_name_list.append(camera)
            else:
                print('Remove: ' + key + camera)
        img_name_list = main_seq.split(str_pad)
        image_num = len(img_name_list)
        cam_num = len(cam_name_list)
        seq_info = {}
        seq_info['seq_name'] = key
        seq_info['image_list'] = img_name_list
        seq_info['image_num'] = image_num
        seq_info['camera_num'] = cam_num
        seq_info['camera_list'] = cam_name_list
        data_info.append(seq_info)

    with open(osp.join(save_path, split, '%s_data_info.pkl' % (record_type)), 'wb') as file:
        pickle.dump(data_info, file)


def generate_video_data():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='/data/dataset/InterHand2.6M_30fps_batch1')
    parser.add_argument("--save_path", type=str, default='/data/dataset/interhand2.6m_30fps')
    opt = parser.parse_args()

    for record_type in ['single', 'two']:
        for split in ['train', 'test', 'val']:
            select_data(opt.data_path, opt.save_path, split=split, record_type=record_type)


def calculate_error(joint, gt):
    diff = (joint - gt) * 1000
    error = torch.sqrt(torch.sum(diff * diff, dim=-1))
    return error.mean()


if __name__ == '__main__':
    generate_video_data()