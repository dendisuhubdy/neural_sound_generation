"""

Copyrigt Dendi Suhubdy, 2018
All rights reserved

"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from torch.autograd import Function


import numpy as np


def load_motion_leap():

    return hand_motion_data

def pca_on_motion_leap(hand_motion_data):
    thumb_finger_direction, index_finger_direction, ring_finger_direction\
            ring_finger_direction, pinky_finger_direction = hand_motion_data

    joint_angle_thumb_index = np.dot(thumb_finger_direction, index_finger_direction.T)
    joint_angle_index_ring = np.dot(index_finger_direction, ring_finger_direction.T)
    joint_angle_ring_pinky = np.dot(ring_finger_direction, pinky_finger_direction.T)
    return (joint_angle_thumb_index, joint_angle_index_ring, joint_angle_ring_pinky)


def load_model():
    pass


def inference():
    pass


if __name__ == "__main__":
    real_time()
