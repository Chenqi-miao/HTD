import numpy as np
from scipy.spatial.transform import Rotation as R  # 用于替代roma库的四元数操作


import numpy as np
from scipy.spatial.transform import Rotation, Slerp

def axis2Rmat(theta):
    """theta: (...,3) array of rotation angles; returns (...,3,3) rotation matrices"""
    rot = R.from_euler('xyz', theta.reshape(-1,3))
    mats = rot.as_matrix()
    return mats.reshape(theta.shape[:-1] + (3,3))

class Augmenter():
    def __init__(self):
        pass

    def seq_rotation(self, joints, centers, sigma=1.0):
        """
        Apply a single random rotation to a batch of sequences in parallel.
        joints: numpy array of shape (B, T, J, 3)
        centers: numpy array of shape (B, T, J, 3)
        sigma: rotation scale
        Returns: numpy array of shape (B, T, J, 3)
        """
        # Generate one random rotation
        theta = np.random.rand(1,3) * np.pi * sigma            # (1,3)
        Rmat = axis2Rmat(theta)[0]                              # (3,3)
        norm = joints - centers                                 # (B,T,J,3)
        B, T, J, C = norm.shape
        norm_flat = norm.reshape(-1, 3)                         # (B*T*J,3)
        rot_flat = norm_flat @ Rmat.T                           # (B*T*J,3)
        rot = rot_flat.reshape(B, T, J, 3) + centers
        return rot

    def seq_part_rotation(self, joints: np.ndarray, centers: np.ndarray, use_global_center: bool = True) -> np.ndarray:
        """
        Apply segmented smooth random rotations to a batch of joint sequences using NumPy and SciPy.

        Args:
            joints: np.ndarray of shape (B, T, J, 3)
            centers: np.ndarray of shape (B, T, 1, 3)
        Returns:
            rotated: np.ndarray of shape (B, T, J, 3)
        """
        B, T, J, _ = joints.shape

        # 1. Determine random segmentation parameters (same for all batch samples)
        part_num = np.random.randint(1, 4)
        if part_num > 1:
            splits = np.random.choice(np.arange(T // 6, T // 3, dtype=int), size=part_num - 1, replace=False)
            lengths = np.concatenate([splits, [T - splits.sum()]])
        else:
            lengths = np.array([T])

        # 2. Random axis-angle increments per segment and cumulative rotvecs
        increments = (np.random.rand(part_num, 3) - 0.5) * (np.pi / 2)
        increments = np.vstack([increments, np.zeros(3)])  # end with zero increment
        init_vec = (np.random.rand(3) - 0.5) * 2 * np.pi
        rotvecs = [init_vec]
        for inc in increments[:-1]:
            rotvecs.append(rotvecs[-1] + inc)
        rotvecs = np.stack(rotvecs)  # shape (part_num+1, 3)

        # 3. Create Rotation objects and interpolate (SLERP)
        all_rots = []
        for i, seg_len in enumerate(lengths):
            if seg_len <= 0:
                continue
            key_rots = Rotation.from_rotvec([rotvecs[i], rotvecs[i + 1]])
            slerp = Slerp([0, 1], key_rots)
            times = np.linspace(0, 1, seg_len, endpoint=False)
            all_rots.append(slerp(times))
        # pad if needed to reach T frames
        total = sum(r.as_quat().shape[0] for r in all_rots)
        if total < T:
            extra_rotvecs = np.tile(rotvecs[-1], (T - total, 1))
            all_rots.append(Rotation.from_rotvec(extra_rotvecs))

        # concatenate and convert to rotation matrices
        concat_rots = Rotation.concatenate(all_rots)
        R_mats = concat_rots.as_matrix()  # shape (T, 3, 3)

        # 4. Apply to batch
        # Choose center: global or individual
        if use_global_center:
            # average center over batch to keep consistency
            cn = centers.mean(axis=0, keepdims=True)  # (1, T, 1, 3)
        else:
            cn = centers  # per-sample center

        rel = joints - cn  # (B, T, J, 3)
        rotated_rel = np.einsum('btji,tik->btjk', rel, R_mats)
        rotated = rotated_rel + cn
        return rotated

    def seq_scale(self, joints, centers, sigma=0.4):
        scale = 1 + (np.random.rand() * sigma - sigma/2)
        out = (joints - centers) * scale + centers
        return out

    def joint_flip(self, joint_list, center_list):
        joints = np.stack(joint_list, axis=0)
        centers = np.stack(center_list, axis=0)
        flipped = joints - centers
        flipped[...,0] *= -1
        out = flipped + centers
        return list(out)

    def seq_flip(self, joints):
        return np.flip(joints, axis=1)

    def seq_aug(self, joints, centers):
        if np.random.rand() > 0.5:
            joints = self.seq_flip(joints)
        if np.random.rand() > 0.5:
            joints = self.seq_part_rotation(joints, centers)
        # joints = self.seq_scale(joints, centers)
        return joints

    def frame_mask(self, joints, centers, ratio=0.1):
        N, T, J, C = joints.shape
        mask = (np.random.rand(N, T) > ratio).astype(float)[:, :, None, None]
        out = (joints - centers) * mask + centers
        return out

    def finger_ambiguity(self, joints):
        T, J, _ = joints.shape
        finger_id = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]])
        select_finger_id = np.random.randint(0, 5)
        frame_aug_joint = joints.copy().reshape([-1, 3])

        aug_frame_len = np.random.randint(2, int(T*0.4))
        start_id = np.random.randint(2, T-aug_frame_len)
        end_id = min(start_id + aug_frame_len + 1, T)
        percent = np.random.uniform(0.7, 1)
        select_frame_id = np.random.choice(np.arange(start_id, end_id), int(np.round((end_id-start_id)*percent)), replace=False)
        finger_a = select_finger_id
        finger_b = (select_finger_id + np.random.choice([-1, 1, -2, 2])) % 5
        finger_noise = np.random.rand(select_frame_id.shape[0], 4) * np.array([[0.2, 0.4, 0.6, 0.8]])
        select_id_a = (select_frame_id.reshape(-1, 1)*J + finger_id[finger_a].reshape(1, -1)).reshape(-1)
        select_id_b = (select_frame_id.reshape(-1, 1)*J + finger_id[finger_b].reshape(1, -1)).reshape(-1)
        finger_noise = finger_noise.reshape([-1, 1])

        joint_a = frame_aug_joint[select_id_a, :].copy()
        joint_b = frame_aug_joint[select_id_b, :].copy()
        frame_aug_joint[select_id_a, :] = joint_a + (joint_b - joint_a) * finger_noise
        return frame_aug_joint.reshape(T, J, 3)

    def frame_joint_jitter(self, joints, sigma=0.01):
        """
        Add jitter noise to each sequence in a batched numpy array.
        joints: numpy array of shape (N, T, J, C)
        sigma: noise scale
        Returns: numpy array of same shape
        """
        # Batch dimensions
        N, T, J, C = joints.shape
        # Original per-joint sigma factors
        jitter_sigma = np.array([
            0.5,
            0.5, 1, 1.5, 2,
            0.5, 1, 1.5, 2,
            0.5, 1, 1.5, 2,
            0.5, 1, 1.5, 2,
            0.5, 1, 1.5, 2,
        ]).reshape(1, 1, J, 1)
        # Generate batch noise and apply per-joint sigma
        # noise shape: (N, T, J, C)
        noise = (np.random.randn(N, T, J, C) - 0.5) * 2 * sigma * jitter_sigma
        return joints + noise


    def frame_center_jitter(self, joints, sigma=0.01):
        N, T, J, C = joints.shape
        noise = (np.random.randn(N,T,1,C) - 0.5) * sigma
        return joints + noise

    def frame_rotation(self, joints, centers, sigma=1.0):
        N, T, J, C = joints.shape
        theta = np.random.rand(T,3) * np.pi * sigma
        Rmats = axis2Rmat(theta)               # (T,3,3)
        norm = joints - centers
        out = np.einsum('tij,ntpj->ntpi', Rmats, norm) + centers
        return out

    def frame_scale(self, joints, centers, sigma=0.4):
        N,T,J,C = joints.shape
        scale = 1 + (np.random.randn(T,1,1) * sigma - sigma/2)
        out = (joints - centers) * scale + centers
        return out

    def frame_finger_aug(self, joints):
        aug_joints_list = []
        for joint in joints:
            if np.random.rand() > 0.3:
                aug_joints_list.append(self.finger_ambiguity(joint))
            else:
                aug_joints_list.append(joint)
        return np.stack(aug_joints_list, axis=0)

    def frame_aug(self, joints, centers):
        if np.random.rand()>0.5:
            joints = self.frame_rotation(joints, centers,0.01)
        joints = self.frame_finger_aug(joints)
        if np.random.rand()>0.5:
            joints = self.frame_joint_jitter(joints,0.004)
        joints = self.frame_center_jitter(joints, 0.0005)
        joints = self.frame_mask(joints, centers,0.2)
        return joints

