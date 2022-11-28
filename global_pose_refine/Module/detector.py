#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import cv2
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from global_pose_refine.Model.gcnn.gcnn import GCNN


class Detector(object):

    def __init__(self, model_file_path=None):
        self.model = GCNN().cuda()

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

    def detectSceneObjects(self, scene_object):
        self.model.eval()

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        data['predictions']['layout_position'] = torch.randn(1, 3, 3).cuda()
        data['predictions']['object_position'] = torch.randn(1, 8, 3).cuda()
        data['predictions']['object_bbox'] = torch.randn(1, 8, 3).cuda()
        data['predictions']['position_dist'] = torch.randn(1, 8 * 8, 1).cuda()
        data['predictions']['bbox_eiou'] = torch.randn(1, 8 * 8, 1).cuda()
        data['predictions']['bbox_diff'] = torch.randn(1, 8 * 8, 6).cuda()

        data = self.model(data)
        return data
