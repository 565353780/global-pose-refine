#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from auto_cad_recon.Module.dataset_manager import DatasetManager


class ObjectPositionDataset(Dataset):

    def __init__(self, training=True, training_percent=0.8):
        self.training = training
        self.training_percent = training_percent

        self.object_position_set_list = []
        self.train_idx_list = []
        self.eval_idx_list = []

        self.loadScan2CAD()
        self.updateIdx()
        return

    def reset(self):
        self.cad_model_file_path_list = []
        return True

    def updateIdx(self, random=False):
        model_num = len(self.cad_model_file_path_list)
        if model_num == 1:
            self.train_idx_list = [0]
            self.eval_idx_list = [0]
            return True

        assert model_num > 0

        train_model_num = int(model_num * self.training_percent)
        if train_model_num == 0:
            train_model_num += 1
        elif train_model_num == model_num:
            train_model_num -= 1

        if random:
            random_idx_list = np.random.choice(np.arange(model_num),
                                               size=model_num,
                                               replace=False)
        else:
            random_idx_list = np.arange(model_num)

        self.train_idx_list = random_idx_list[:train_model_num]
        self.eval_idx_list = random_idx_list[train_model_num:]
        return True

    def loadScan2CAD(self):
        scannet_dataset_folder_path = "/home/chli/chLi/ScanNet/scans/"
        scannet_object_dataset_folder_path = "/home/chli/chLi/ScanNet/objects/"
        scannet_bbox_dataset_folder_path = "/home/chli/chLi/ScanNet/bboxes/"
        scan2cad_dataset_folder_path = "/home/chli/chLi/Scan2CAD/scan2cad_dataset/"
        scan2cad_object_model_map_dataset_folder_path = "/home/chli/chLi/Scan2CAD/object_model_maps/"
        shapenet_dataset_folder_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/"
        shapenet_udf_dataset_folder_path = "/home/chli/chLi/ShapeNet/udfs/"
        print_progress = True

        dataset_manager = DatasetManager(
            scannet_dataset_folder_path, scannet_object_dataset_folder_path,
            scannet_bbox_dataset_folder_path, scan2cad_dataset_folder_path,
            scan2cad_object_model_map_dataset_folder_path,
            shapenet_dataset_folder_path, shapenet_udf_dataset_folder_path)

        scene_name_list = dataset_manager.getScanNetSceneNameList()

        print("[INFO][ObjectPositionDataset::loadScan2CAD]")
        print("\t start load scan2cad dataset...")
        for scene_name in tqdm(scene_name_list):
            object_file_name_list = dataset_manager.getScanNetObjectFileNameList(
                scene_name)
            object_position_set = []
            for object_file_name in object_file_name_list:
                shapenet_model_dict = dataset_manager.getShapeNetModelDict(
                    scene_name, object_file_name)
                print(shapenet_model_dict.keys())
                exit()
        return True

    def __getitem__(self, idx, training=True):
        if self.training:
            idx = self.train_idx_list[idx]
        else:
            idx = self.eval_idx_list[idx]

        object_position_set = self.object_position_set_list[idx]

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        translate = (np.random.rand(3) - 0.5) * 0.1
        euler_angle = np.random.rand(3) * 36.0
        scale = 1.0 + (np.random.rand(3) - 0.5) * 0.1

        trans_point_array = transPointArray(origin_point_array, translate,
                                            euler_angle, scale)
        data['inputs']['trans_point_array'] = torch.from_numpy(
            trans_point_array).float()

        if training:
            rotate_matrix = getRotateMatrix(euler_angle)
            data['inputs']['rotate_matrix'] = torch.from_numpy(
                rotate_matrix).to(torch.float32)
        return data

    def __len__(self):
        if self.training:
            return len(self.train_idx_list)
        else:
            return len(self.eval_idx_list)
