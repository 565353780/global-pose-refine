#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import cv2
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from global_pose_refine.Method.device import toCuda
from global_pose_refine.Model.gcnn.gcnn import GCNN


class Detector(object):

    def __init__(self, model_file_path=None):
        self.model = GCNN(True).cuda()

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path):
        assert os.path.exists(model_file_path)

        print("[INFO][Detector::loadModel]")
        print("\t start loading model from :")
        print("\t", model_file_path)
        model_dict = torch.load(model_file_path)
        self.model.load_state_dict(model_dict['model'])
        return True

    def detectSceneObjects(self, data):
        self.model.eval()

        toCuda(data)

        wall_position = data['inputs']['wall_position']
        trans_object_obb = data['inputs']['trans_object_obb']

        wall_num = wall_position.shape[0]
        floor_num = 1
        object_num = trans_object_obb.shape[0]

        data['inputs']['floor_position'] = data['inputs'][
            'floor_position'].reshape(1, floor_num, -1)
        data['inputs']['floor_normal'] = data['inputs'][
            'floor_normal'].reshape(1, floor_num, -1)
        data['inputs']['floor_z_value'] = data['inputs'][
            'floor_z_value'].reshape(1, floor_num, -1)

        data['inputs']['wall_position'] = data['inputs'][
            'wall_position'].reshape(1, wall_num, -1)
        data['inputs']['wall_normal'] = data['inputs']['wall_normal'].reshape(
            1, wall_num, -1)

        data['inputs']['trans_object_obb'] = data['inputs'][
            'trans_object_obb'].reshape(1, object_num, -1)
        data['inputs']['trans_object_abb'] = data['inputs'][
            'trans_object_abb'].reshape(1, object_num, -1)
        data['inputs']['trans_object_obb_center'] = data['inputs'][
            'trans_object_obb_center'].reshape(1, object_num, -1)

        data = self.model(data)
        return data
