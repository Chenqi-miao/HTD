import os
import cv2
import torch
import shutil
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from seq_config import cfg
from utils.logger import setup_logger
from utils.visualize import draw_2d_skeleton
from dataset.seqhand import SeqHand, SeqHandTest
from model.FusionFormer import FusionModel
# from model.FusionFormer import DecoupleFusionModel as FusionModel
import json
from torch.utils.tensorboard import SummaryWriter
from utils.fix_seed import set_seed, seed_worker

def vis(inputs, targets, meta_infos, outs, error, id, seq_len):
    img = np.zeros([cfg.batch_size, seq_len, 256, 256, 3])
    sample_list = [0]
    center = meta_infos['center_xyz'].cuda()
    for index in sample_list:
        init_joints_pd = (inputs['joint_xyz'].cuda() - center)[...,:2].detach().cpu().numpy() / 150 * 128 + 128
        joints_pd = (outs['pd_joint_xyz'] - center)[..., :2].detach().cpu().numpy() / 150 * 128 + 128
        joints_gt = (targets['joint_xyz'].cuda() - center)[...,:2].detach().cpu().numpy() / 150 * 128 + 128
        seq_path = cfg.output_root + '/vis/%d/' % (id * cfg.batch_size + index)
        if not os.path.exists(seq_path):
            os.makedirs(seq_path)
        for seq_id in range(cfg.seq_len):
            img_draw = draw_2d_skeleton(img[index, seq_id], joints_pd[index, 0, seq_id])
            cv2.imwrite(seq_path + 'joint_pd_%d.png' % (seq_id), img_draw)
            img_draw = draw_2d_skeleton(img[index, seq_id], joints_gt[index, 0, seq_id])
            cv2.imwrite(seq_path + 'joint_gt_%d.png' % (seq_id), img_draw)
            img_draw = draw_2d_skeleton(img[index, seq_id], init_joints_pd[index, 0, seq_id])
            cv2.imwrite(seq_path + 'joint_init_%d.png' % (seq_id), img_draw)

def train():
    trainer = Trainer()
    trainer._make_model()
    trainer._make_batch_loader()
    min_error = 100
    global_step_num = 0
    for epoch in range(trainer.start_epoch, cfg.total_epoch):
        for iteration, (inputs, targets, meta_infos) in tqdm(enumerate(trainer.trian_loader)):
            trainer.optimizer.zero_grad()
            outs, loss, error = trainer.model(inputs, targets, meta_infos)
            sum(loss[k] for k in loss).backward()
            trainer.optimizer.step()
            for k, v in loss.items():
                trainer.writer.add_scalar('train_loss_'+k, v.detach(), global_step=global_step_num)
            for k, v in error.items():
                trainer.writer.add_scalar('train_error_'+k, v.detach(), global_step=global_step_num)
            global_step_num = global_step_num + 1
            if iteration % cfg.print_iter == 0:
                screen = ['[Epoch %d/%d]' % (epoch, cfg.total_epoch),
                          '[Batch %d/%d]' % (iteration, len(trainer.trian_loader)),
                          '[lr %f]' % (trainer.get_lr())]
                screen += ['[%s: %.4f]' % ('loss_' + k, v.detach()) for k, v in loss.items()]
                screen += ['[%s: %.4f]' % ('error_' + k, v.detach()) for k, v in error.items()]
                trainer.logger.info(''.join(screen))
            if iteration % cfg.draw_iter == 0:
                vis(inputs, targets, meta_infos, outs, error, iteration, cfg.seq_len)

        trainer.schedule.step()
        trainer.save_model(trainer.model, trainer.optimizer, trainer.schedule, epoch, 'latest')

        if cfg.loader_resample:
            trainer.train_dataset.generator_seq() # 重新采样一批序列

        if epoch % cfg.eval_interval == 0:
            error = trainer.test_model(epoch)
            if error < min_error:
                trainer.save_model(trainer.model, trainer.optimizer, trainer.schedule, epoch, 'best')
                min_error = error

def test():
    tester = Tester()
    tester._make_model()
    tester._make_batch_loader()
    tester.test_model()

def test_vis():
    tester = Visualer()
    tester._make_model()
    tester._make_batch_loader()
    tester.save_data()

def test_wild():
    tester = WildTester()
    tester._make_model()
    tester.test_model()


class Trainer:
    def __init__(self):
        log_folder = os.path.join(cfg.output_root, 'log')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        logfile = os.path.join(log_folder, 'train_' + cfg.experiment_name + '.log')
        self.writer = SummaryWriter('runs/%s'%(cfg.add_info))

        vis_folder = os.path.join(cfg.output_root, 'vis')
        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder)

        file_folder = os.path.join(cfg.output_root, 'files')
        if not os.path.exists(file_folder):
            os.makedirs(file_folder)
        shutil.copytree('./model', file_folder + '/model/', dirs_exist_ok=True)
        shutil.copytree('./dataset', file_folder + '/dataset/', dirs_exist_ok=True)
        shutil.copytree('./utils', file_folder + '/utils/', dirs_exist_ok=True)
        shutil.copy('seq_train.py', file_folder + '/seq_train.py')
        shutil.copy('seq_config.py', file_folder + '/seq_config.py')
        self.logger = setup_logger(output=logfile, name="Training")
        self.logger.info('Start training: %s' % ('train_' + cfg.experiment_name))

    def load_model(self, checkpoint, model):
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['net'])
        return model

    def save_model(self, model, optimizer, schedule, epoch, name):
        save = {
            'net': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'schedule': schedule.state_dict(),
            'last_epoch': epoch
        }
        path_checkpoint = os.path.join(cfg.output_root, 'checkpoint')
        if not os.path.exists(path_checkpoint):
            os.makedirs(path_checkpoint)
        save_path = os.path.join(path_checkpoint, "%s.pth" % (name))
        torch.save(save, save_path)

    def get_lr(self):
        for g in self.optimizer.param_groups:
            cur_lr = g['lr']
        return cur_lr

    def _make_batch_loader(self):
        seed_value = 42
        g = torch.Generator()
        g.manual_seed(seed_value)  # 创建一个生成器并设置种子，推荐方式
        self.train_dataset = SeqHand(cfg.data_dir, cfg.data_list , min_seq_len=cfg.min_seq_len, seq_len=cfg.seq_len, view_num=cfg.view_num, data_num=cfg.data_num)
        self.trian_loader = DataLoader(self.train_dataset,
                                       batch_size=cfg.batch_size,
                                       num_workers=cfg.num_worker,
                                       worker_init_fn=seed_worker,
                                       generator=g,
                                       shuffle=True,
                                       drop_last=True)
        self.test_dataset = SeqHandTest(cfg.data_dir, cfg.data_list, min_seq_len=cfg.min_seq_len, seq_len=cfg.seq_len, view_num=cfg.view_num, data_num=cfg.test_data_num)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=cfg.batch_size,
                                      shuffle=False,
                                      worker_init_fn=seed_worker,
                                      generator=g,
                                      num_workers=cfg.num_worker)

    def _make_model(self):
        static_model = None
        # 时序模型
        model = FusionModel(model_name=cfg.backbone, num_frame=cfg.seq_len, num_joints=cfg.joint_num,
                            num_view=cfg.view_num, global_temporal=cfg.global_temporal, window_size=cfg.window_size).cuda()
        model.train()
        params_to_optimize = [{"params": model.parameters(), 'initial_lr': cfg.lr}]
        optimizer = optim.AdamW(params_to_optimize, cfg.lr)

        if cfg.lr_scheduler == 'cosine':
            schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.total_epoch, eta_min=0)
        elif cfg.lr_scheduler == 'step':
            schedule = optim.lr_scheduler.MultiStepLR(optimizer, [30, 40], gamma=0.1, last_epoch=-1)
        start_epoch = 0

        self.start_epoch = start_epoch
        self.model = model
        self.static_model = static_model
        self.optimizer = optimizer
        self.schedule = schedule

    @torch.no_grad()
    def test_model(self, epoch):
        self.model.eval()
        init_errors_list = []
        refine_errors_list = []
        for iteration, (inputs, targets, meta_infos) in tqdm(enumerate(self.test_loader)):
            outs = self.model(inputs, targets, meta_infos)
            init_errors, refine_errors = self.test_dataset.evaluate(outs, inputs, targets, meta_infos)
            init_errors_list.append(init_errors)
            refine_errors_list.append(refine_errors)

        refine_joints_error = np.stack(refine_errors_list, axis=0).mean()
        init_joints_error = np.stack(init_errors_list, axis=0).mean()

        self.writer.add_scalar('test_Init_MPJPE', init_joints_error, global_step=epoch)
        self.writer.add_scalar('test_Refine_MPVPE', refine_joints_error, global_step=epoch)

        print('Init MPJPE: {:.4} mm'.format(init_joints_error))
        print('Refine MPJPE:{:.4} mm'.format(refine_joints_error))

        self.logger.info('Init MPJPE: left {:.4} mm'.format(init_joints_error))
        self.logger.info('Refine MPJPE: left {:.4} mm'.format(refine_joints_error))
        self.model.train()
        return refine_joints_error


class Tester:
    def __init__(self):
        log_folder = os.path.join(cfg.output_root, 'log')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        logfile = os.path.join(log_folder, 'test_' + cfg.experiment_name + '.log')
        self.logger = setup_logger(output=logfile, name="Testing")
        self.logger.info('Start testing: %s' % ('test_' + cfg.experiment_name))

    def load_model(self, checkpoint, model):
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['net'])
        return model

    def _make_batch_loader(self):
        self.test_dataset = SeqHandTest(cfg.data_dir, cfg.data_list, min_seq_len=cfg.min_seq_len, seq_len=cfg.seq_len, view_num=cfg.view_num, data_num=cfg.test_data_num)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=cfg.batch_size,
                                      shuffle=False,
                                      num_workers=cfg.num_worker)

    def _make_model(self):
        # 时序模型
        model = FusionModel(model_name=cfg.backbone, num_frame=cfg.seq_len, num_joints=cfg.joint_num, num_view=cfg.view_num, global_temporal=cfg.global_temporal, window_size=cfg.window_size).cuda()
        for para in model.parameters():
            para.requires_grad = False
        model = self.load_model(cfg.checkpoint, model)
        model.eval()
        self.model = model

    @torch.no_grad()
    def test_model(self):
        self.model.eval()
        init_errors_list = []
        refine_errors_list = []
        for iteration, (inputs, targets, meta_infos) in tqdm(enumerate(self.test_loader)):
            outs = self.model(inputs, targets, meta_infos)
            init_errors, refine_errors = self.test_dataset.evaluate(outs, inputs, targets, meta_infos)
            init_errors_list.append(init_errors)
            refine_errors_list.append(refine_errors)

        refine_joints_error = np.stack(refine_errors_list, axis=0).mean()
        init_joints_error = np.stack(init_errors_list, axis=0).mean()

        print('Init MPJPE: {:.4} mm'.format(init_joints_error))
        print('Refine MPJPE:{:.4} mm'.format(refine_joints_error))

        self.logger.info('Init MPJPE: left {:.4} mm'.format(init_joints_error))
        self.logger.info('Refine MPJPE: left {:.4} mm'.format(refine_joints_error))
        self.model.train()
        return refine_joints_error


class WildTester:
    def __init__(self):
        log_folder = os.path.join(cfg.output_root, 'log')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        logfile = os.path.join(log_folder, 'test_' + cfg.experiment_name + '.log')
        self.logger = setup_logger(output=logfile, name="Testing")
        self.logger.info('Start Wild testing: %s' % ('test_' + cfg.experiment_name))

    def load_model(self, checkpoint, model):
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['net'])
        return model

    def _make_model(self):
        model = FusionModel(model_name=cfg.backbone, num_frame=cfg.seq_len, num_joints=cfg.joint_num, num_view=cfg.view_num, global_temporal=cfg.global_temporal, window_size=cfg.window_size).cuda()
        for para in model.parameters():
            para.requires_grad = False
        model = self.load_model(cfg.checkpoint, model)
        model.eval()
        self.model = model

    @torch.no_grad()
    def test_model(self):
        anno_dir = 'seq.json'
        with open(anno_dir, 'r') as file:
            annos = json.load(file)
        B, V, F, J = 1, 3, 15, 21
        in_joint=torch.from_numpy(np.array(annos[0])).float().cuda()
        in_joint = in_joint.reshape([1,1,15,21,3]).repeat([B,V,1,1,1])
        center = in_joint[..., 9:10, :].clone()
        gt_joint=in_joint.clone()
        in_joint_val = torch.ones([B, V, F, 21]).float().cuda()
        gt_joint_val = torch.ones([B, 1, F, 21]).float().cuda()
        continuous_val = torch.ones([B, F, 1]).float().cuda()
        inputs = {'joint_xyz': in_joint}
        targets = {'joint_xyz': gt_joint}
        meta_info = {"center_xyz": center,
                     'continuous_val': continuous_val,
                     'joint_gt_val': gt_joint_val,
                     'joint_in_val': in_joint_val
                     }
        outs, errors = self.model(inputs, targets, meta_info)
        vis(inputs, targets, meta_info, outs, 0, 0, cfg.seq_len)

        return 0

class Visualer:
    def __init__(self):
        log_folder = os.path.join(cfg.output_root, 'log')
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        logfile = os.path.join(log_folder, 'test_' + cfg.experiment_name + '.log')
        self.logger = setup_logger(output=logfile, name="Testing")
        self.logger.info('Start testing: %s' % ('test_' + cfg.experiment_name))

        self.init_pose_list = []
        self.refine_pose_list = []
        self.gt_pose_list = []

    def load_model(self, checkpoint, model):
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['net'])
        return model

    def _make_batch_loader(self):
        self.test_dataset = SeqHandTest(cfg.data_dir, cfg.data_list, min_seq_len=cfg.min_seq_len, seq_len=cfg.seq_len, view_num=cfg.view_num)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=cfg.batch_size,
                                      shuffle=False,
                                      num_workers=cfg.num_worker)

    def _make_model(self):
        # 时序模型
        model = FusionModel(model_name=cfg.backbone, num_frame=cfg.seq_len, num_joints=cfg.joint_num, num_view=cfg.view_num, global_temporal=cfg.global_temporal, window_size=cfg.window_size).cuda()
        for para in model.parameters():
            para.requires_grad = False
        model = self.load_model(cfg.checkpoint, model)
        model.eval()
        self.model = model

    @torch.no_grad()
    def save_data(self):
        self.model.eval()
        init_errors_list = []
        refine_errors_list = []
        for iteration, (inputs, targets, meta_infos) in tqdm(enumerate(self.test_loader)):
            outs = self.model(inputs, targets, meta_infos)
            init_errors, refine_errors = self.test_dataset.evaluate(outs, inputs, targets, meta_infos)
            init_errors_list.append(init_errors)
            refine_errors_list.append(refine_errors)
            self.record_data(outs, inputs, targets)

        self.save_list()
        refine_joints_error = np.stack(refine_errors_list, axis=0).mean()
        init_joints_error = np.stack(init_errors_list, axis=0).mean()

        print('Init MPJPE: {:.4} mm'.format(init_joints_error))
        print('Refine MPJPE:{:.4} mm'.format(refine_joints_error))

        self.logger.info('Init MPJPE: left {:.4} mm'.format(init_joints_error))
        self.logger.info('Refine MPJPE: left {:.4} mm'.format(refine_joints_error))
        self.model.train()
        return refine_joints_error

    def record_data(self, outs, inputs, targets):
        gt_joint_xyz = targets['joint_xyz'].detach().cpu().numpy()
        in_joint_xyz = inputs['joint_xyz'].detach().cpu().numpy()
        pd_joint_xyz = outs['pd_joint_xyz'].detach().cpu().numpy()
        self.init_pose_list.append(in_joint_xyz)
        self.gt_pose_list.append(gt_joint_xyz)
        self.refine_pose_list.append(pd_joint_xyz)

    def save_list(self):
        in_joint_xyz = np.concatenate(self.init_pose_list, axis=0).reshape([-1, 21*3])
        gt_joint_xyz = np.concatenate(self.gt_pose_list, axis=0).reshape([-1, 21*3])
        pd_joint_xyz = np.concatenate(self.refine_pose_list, axis=0).reshape([-1, 21*3])
        np.savetxt('init.txt', in_joint_xyz, fmt='%.4f')
        np.savetxt('gt.txt', gt_joint_xyz, fmt='%.4f')
        np.savetxt('refine.txt', pd_joint_xyz, fmt='%.4f')

if __name__ == '__main__':
    seed = 42
    set_seed(seed)
    if cfg.phase == 'train':
        train()
    elif cfg.phase == 'test':
        test()
    elif cfg.phase == 'vis':
        test_vis()
    else:
        test_wild()