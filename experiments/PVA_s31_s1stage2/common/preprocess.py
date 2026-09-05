"""多帧统一预处理: 中心归一化 + 帧间速度/加速度 -> 9维模型输入.

供多个实验的模型调用 (exp2 基线 / 分组时空网络), 避免各自重复实现"切帧/多帧处理"。
输入均为相机空间 mm 坐标, 形状 [B,V,F,J,3] (B批/V视角/F帧/J关节)。
"""
import torch


def center_normalize(ji, center, joint_valid, norm=300):
    """以 center 的 joint_valid 加权均值为中心, 归一化到 /norm.

    作用: 去掉全局平移 (DC), 统一尺度 (手约 300mm -> O(1))。
    ji:[B,V,F,J,3], center:[B,V,F,1,3], joint_valid:[B,V,F,J,1] -> [B,V,F,J,3]。
    """
    sc = (center * joint_valid).sum(dim=(2, 3)) / (joint_valid.sum(dim=(2, 3)) + 1e-8)
    sc = sc.reshape(*ji.shape[:2], 1, 1, 3)
    return (ji - sc) / norm, sc    # 返回 sc 供模型反归一化用


def frame_dynamics(jn):
    """多帧处理: 从归一化坐标算帧间速度/加速度, 边界补全, 拼接成 9 维.

    速度 = 相邻帧差分; 加速度 = 速度差分; 最后一帧用最后一个差分补齐。
    jn:[B,V,F,J,3] -> [B,V,F,J,9] = (pos, vel, acc)。
    """
    in_v = torch.cat([jn[:, :, 1:] - jn[:, :, :-1],
                      jn[:, :, -1:] - jn[:, :, -2:-1]], dim=2)
    in_a = torch.cat([in_v[:, :, 1:] - in_v[:, :, :-1],
                      in_v[:, :, -1:] - in_v[:, :, -2:-1]], dim=2)
    return torch.cat([jn, in_v, in_a], dim=-1)


def build_model_input(ji, center, joint_valid, norm=300):
    """统一入口: 中心归一化 + 帧间差分 -> [B,V,F,J,9] 模型输入.

    模型 forward 只调这一个函数即可拿到多帧 9 维输入。
    """
    jn = center_normalize(ji, center, joint_valid, norm)
    return frame_dynamics(jn)
