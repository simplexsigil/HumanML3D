import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

def qrot_np(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    q: (..., 4), quaternion in wxyz format
    v: (..., 3), vector
    Returns rotated vector(s): same shape as v
    """
    qvec = q[..., 1:]  # xyz
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2 * (q[..., :1] * uv + uuv)


def qinv_np(q):
    """
    Return the inverse of a quaternion
    q: (..., 4)
    """
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def qmul_np(q, r):
    """
    Quaternion multiplication
    q, r: (..., 4)
    Returns: (..., 4)
    """
    w1, x1, y1, z1 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w2, x2, y2, z2 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ], axis=-1)


def qbetween_np(v1, v2):
    """
    Compute quaternion that rotates v1 to v2
    v1, v2: (..., 3)
    Returns: (..., 4) quaternion
    """
    v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)
    c = np.cross(v1, v2)
    d = (v1 * v2).sum(axis=-1, keepdims=True)
    s = np.sqrt((1 + d) * 2)
    q = np.concatenate([s / 2, c / s], axis=-1)
    return q


def quaternion_to_cont6d_np(q):
    """
    Convert quaternions (wxyz) to 6D continuous representation
    q: (N, 4)
    Returns: (N, 6)
    """
    r = R.from_quat(q[..., [1, 2, 3, 0]])  # Convert wxyz to xyzw
    rotmat = r.as_matrix()  # (N, 3, 3)
    return rotmat[..., :, :2].reshape(q.shape[0], -1)


def cont6d_to_rotation(cont6d):
    """
    Convert 6D continuous representation to rotation matrices
    cont6d: (N, 6)
    Returns: (N, 3, 3)
    """
    a1 = cont6d[..., :3]
    a2 = cont6d[..., 3:6]

    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)

    return np.stack([b1, b2, b3], axis=-1)  # (N, 3, 3)


class Skeleton:
    def __init__(self, offsets, kinematic_chain, device="cpu"):
        self.offsets = offsets.numpy() if torch.is_tensor(offsets) else offsets
        self.kinematic_chain = kinematic_chain
        self.device = device

    def forward_kinematics_cont6d(self, cont6d, root_positions):
        T, J, _ = cont6d.shape
        rot_mats = cont6d_to_rotation(cont6d.reshape(-1, 6)).reshape(T, J, 3, 3)

        positions = np.zeros((T, J, 3))
        positions[:, 0, :] = root_positions

        for t in range(T):
            for parent, children in enumerate(self.kinematic_chain):
                for child in children:
                    offset = self.offsets[child]
                    parent_pos = positions[t, parent]
                    parent_rot = rot_mats[t, parent]
                    positions[t, child] = parent_pos + parent_rot @ offset

        return positions
