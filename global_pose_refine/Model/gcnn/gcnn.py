#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from points_shape_detect.Loss.ious import IoULoss
from points_shape_detect.Method.trans import transPointArray

from global_pose_refine.Method.weight import setWeight
from global_pose_refine.Model.gcnn.gclayer_collect import \
    GraphConvolutionLayerCollect
from global_pose_refine.Model.gcnn.gclayer_update import \
    GraphConvolutionLayerUpdate


class GCNN(nn.Module):

    def __init__(self, infer=False):
        super().__init__()

        self.floor_features = {
            'floor_position': 4 * 3,
            'floor_normal': 3,
            'floor_z_value': 1,
        }
        self.wall_features = {
            'wall_position': 4 * 3,
            'wall_normal': 3,
        }
        self.object_features = {
            'trans_object_obb': 8 * 3,
            'trans_object_abb': 2 * 3,
            'trans_object_obb_center': 3,
        }
        self.relation_features = {
            'trans_object_obb_center_dist': 1,
            'trans_object_abb_eiou': 1,
        }

        self.feature_dim = 512
        self.feat_update_step = 4
        self.feat_update_group = 1

        object_features_len = sum(self.object_features.values())
        relation_features_len = sum(self.relation_features.values())
        floor_features_len = sum(self.floor_features.values())
        wall_features_len = sum(self.wall_features.values())

        self.object_embedding = nn.Sequential(
            nn.Linear(object_features_len, self.feature_dim),
            nn.ReLU(True),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.relation_embedding = nn.Sequential(
            nn.Linear(relation_features_len, self.feature_dim),
            nn.ReLU(True),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.floor_embedding = nn.Sequential(
            nn.Linear(floor_features_len, self.feature_dim),
            nn.ReLU(True),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.wall_embedding = nn.Sequential(
            nn.Linear(wall_features_len, self.feature_dim),
            nn.ReLU(True),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        self.gcn_collect_feat = nn.ModuleList([
            GraphConvolutionLayerCollect(self.feature_dim, self.feature_dim)
            for i in range(self.feat_update_group)
        ])
        self.gcn_update_feat = nn.ModuleList([
            GraphConvolutionLayerUpdate(self.feature_dim, self.feature_dim)
            for i in range(self.feat_update_group)
        ])

        self.translate_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(self.feature_dim // 2, 3),
        )

        self.euler_angle_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(self.feature_dim // 2, 3),
        )

        self.scale_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.5),
            nn.Linear(self.feature_dim // 2, 3),
        )

        # initiate weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

        self.l1_loss = nn.SmoothL1Loss()

        self.infer = infer
        return

    def buildMap(self, data):
        device = data['inputs'][next(iter(self.object_features.keys()))].device

        object_num = data['inputs'][next(iter(
            self.object_features.keys()))].shape[1]
        floor_num = data['inputs'][next(iter(
            self.floor_features.keys()))].shape[1]
        wall_num = data['inputs'][next(iter(
            self.wall_features.keys()))].shape[1]
        total_num = object_num + wall_num + floor_num

        total_map = torch.ones([total_num, total_num]).to(device)

        object_mask = torch.zeros(total_num, dtype=torch.bool).to(device)
        wall_mask = torch.zeros(total_num, dtype=torch.bool).to(device)
        floor_mask = torch.zeros(total_num, dtype=torch.bool).to(device)
        object_mask[:object_num] = True
        wall_mask[object_num:object_num + wall_num] = True
        floor_mask[object_num + wall_num:] = True

        object_index = torch.arange(0, total_num, dtype=torch.long).to(device)
        subject_index_grid, object_index_grid = torch.meshgrid(object_index,
                                                               object_index,
                                                               indexing='ij')
        edges = torch.stack(
            [subject_index_grid.reshape(-1),
             object_index_grid.reshape(-1)], -1)

        relation_mask = edges[:, 0] != edges[:, 1]
        relation_index = edges[relation_mask]

        # [sum(Ni + 1), sum((Ni + 1) ** 2)]
        subj_pred_map = torch.zeros(total_num,
                                    relation_index.shape[0]).to(device)
        obj_pred_map = torch.zeros(total_num,
                                   relation_index.shape[0]).to(device)

        # map from subject (an object or layout vertex) to predicate (a relation vertex)
        subj_pred_map.scatter_(0, (relation_index[:, 0].view(1, -1)), 1)
        # map from object (an object or layout vertex) to predicate (a relation vertex)
        obj_pred_map.scatter_(0, (relation_index[:, 1].view(1, -1)), 1)

        data['predictions']['relation_mask'] = relation_mask
        data['predictions']['object_mask'] = object_mask
        data['predictions']['wall_mask'] = wall_mask
        data['predictions']['floor_mask'] = floor_mask
        data['predictions']['total_map'] = total_map
        data['predictions']['subj_pred_map'] = subj_pred_map
        data['predictions']['obj_pred_map'] = obj_pred_map
        return data

    def embedObjectFeature(self, data):
        object_feature_list = []
        for key in self.object_features.keys():
            object_feature_list.append(data['inputs'][key])

        cat_object_feature = torch.cat(object_feature_list, -1)

        embed_object_feature = self.object_embedding(cat_object_feature)

        data['predictions']['cat_object_feature'] = cat_object_feature
        data['predictions']['embed_object_feature'] = embed_object_feature
        return data

    def embedWallFeature(self, data):
        wall_feature_list = []
        for key in self.wall_features.keys():
            wall_feature_list.append(data['inputs'][key])

        cat_wall_feature = torch.cat(wall_feature_list, -1)

        embed_wall_feature = self.wall_embedding(cat_wall_feature)

        data['predictions']['cat_wall_feature'] = cat_wall_feature
        data['predictions']['embed_wall_feature'] = embed_wall_feature
        return data

    def embedFloorFeature(self, data):
        floor_feature_list = []
        for key in self.floor_features.keys():
            floor_feature_list.append(data['inputs'][key])

        cat_floor_feature = torch.cat(floor_feature_list, -1)

        embed_floor_feature = self.floor_embedding(cat_floor_feature)

        data['predictions']['cat_floor_feature'] = cat_floor_feature
        data['predictions']['embed_floor_feature'] = embed_floor_feature
        return data

    def embedRelationFeature(self, data):
        relation_feature_list = []
        for key in self.relation_features.keys():
            relation_feature_list.append(data['inputs'][key])

        cat_relation_feature = torch.cat(relation_feature_list, -1)

        embed_relation_feature = self.relation_embedding(cat_relation_feature)

        data['predictions']['cat_relation_feature'] = cat_relation_feature
        data['predictions']['embed_relation_feature'] = embed_relation_feature
        return data

    def embedFeatures(self, data):
        data = self.embedObjectFeature(data)
        data = self.embedWallFeature(data)
        data = self.embedFloorFeature(data)
        data = self.embedRelationFeature(data)

        relation_mask = data['predictions']['relation_mask']
        embed_object_feature = data['predictions']['embed_object_feature']
        embed_wall_feature = data['predictions']['embed_wall_feature']
        embed_floor_feature = data['predictions']['embed_floor_feature']
        embed_relation_feature = data['predictions']['embed_relation_feature']

        object_num = embed_object_feature.shape[1]
        wall_num = embed_wall_feature.shape[1]
        floor_num = embed_floor_feature.shape[1]
        total_num = object_num + wall_num + floor_num

        # representation of object and layout vertices
        cat_total_feature_list = [
            embed_object_feature, embed_wall_feature, embed_floor_feature
        ]
        embed_total_feature = torch.cat(cat_total_feature_list, 1)

        # representation of relation vertices connecting obj/lo vertices
        if total_num > object_num:
            embed_relation_feature_matrix = embed_relation_feature.reshape(
                object_num, object_num, -1)
            total_relation_feature_matrix = F.pad(
                embed_relation_feature_matrix.permute(2, 0, 1),
                [0, total_num - object_num, 0, total_num - object_num],
                "constant", 0.001).permute(1, 2, 0)
            total_relation_feature = total_relation_feature_matrix.reshape(
                total_num**2, -1)
        else:
            total_relation_feature = embed_relation_feature.reshape(
                total_num**2, -1)

        # from here, for compatibility with graph-rcnn, x_obj corresponds to obj/lo vertices
        mask_total_relation_feature = total_relation_feature[relation_mask]

        data['predictions']['embed_total_feature'] = embed_total_feature
        data['predictions'][
            'mask_total_relation_feature'] = mask_total_relation_feature
        return data

    def updateFeature(self, data):
        object_mask = data['predictions']['object_mask']
        wall_mask = data['predictions']['wall_mask']
        floor_mask = data['predictions']['floor_mask']
        total_map = data['predictions']['total_map']
        subj_pred_map = data['predictions']['subj_pred_map']
        obj_pred_map = data['predictions']['obj_pred_map']
        embed_total_feature = data['predictions']['embed_total_feature']
        mask_total_relation_feature = data['predictions'][
            'mask_total_relation_feature']

        # FIXME: now can only training on batch_size = 0
        total_feature_list = [embed_total_feature[0]]
        relation_feature_list = [mask_total_relation_feature]

        latest_feature_idx = 0
        for group, (gcn_collect_feat, gcn_update_feat) in enumerate(
                zip(self.gcn_collect_feat, self.gcn_update_feat)):
            for t in range(latest_feature_idx,
                           latest_feature_idx + self.feat_update_step):

                # update object features
                # message from other objects
                source_obj = gcn_collect_feat(total_feature_list[t],
                                              total_feature_list[t], total_map,
                                              4)

                # message from predicate
                source_rel_sub = gcn_collect_feat(total_feature_list[t],
                                                  relation_feature_list[t],
                                                  subj_pred_map, 0)
                source_rel_obj = gcn_collect_feat(total_feature_list[t],
                                                  relation_feature_list[t],
                                                  obj_pred_map, 1)
                source2obj_all = (source_obj + source_rel_sub +
                                  source_rel_obj) / 3
                total_feature_list.append(
                    gcn_update_feat(total_feature_list[t], source2obj_all, 0))

                # update relation features
                source_obj_sub = gcn_collect_feat(relation_feature_list[t],
                                                  total_feature_list[t],
                                                  subj_pred_map.t(), 2)
                source_obj_obj = gcn_collect_feat(relation_feature_list[t],
                                                  total_feature_list[t],
                                                  obj_pred_map.t(), 3)
                source2rel_all = (source_obj_sub + source_obj_obj) / 2
                relation_feature_list.append(
                    gcn_update_feat(relation_feature_list[t], source2rel_all,
                                    1))
            latest_feature_idx += self.feat_update_step

        obj_feats_object = total_feature_list[-1][object_mask[0]]
        obj_feats_wall = total_feature_list[-1][wall_mask[0]]
        obj_feats_floor = total_feature_list[-1][floor_mask[0]]

        data['predictions']['update_total_feature'] = total_feature_list[-1]
        data['predictions']['obj_feats_object'] = obj_feats_object
        data['predictions']['obj_feats_wall'] = obj_feats_wall
        data['predictions']['obj_feats_floor'] = obj_feats_floor
        return data

    def decodeFeature(self, data):
        obj_feats_object = data['predictions']['obj_feats_object']
        #  obj_feats_wall = data['predictions']['obj_feats_wall']
        #  obj_feats_floor = data['predictions']['obj_feats_floor']

        translate_inv = self.translate_encoder(obj_feats_object)
        euler_angle_inv = self.euler_angle_encoder(obj_feats_object)
        scale_inv = self.scale_encoder(obj_feats_object)

        data['predictions']['refine_translate_inv'] = translate_inv
        data['predictions']['refine_euler_angle_inv'] = euler_angle_inv
        data['predictions']['refine_scale_inv'] = scale_inv
        return data

    def updatePose(self, data):
        trans_object_obb = data['inputs']['trans_object_obb']
        trans_object_obb_center = data['inputs']['trans_object_obb_center']

        refine_translate_inv = data['predictions']['refine_translate_inv']
        refine_euler_angle_inv = data['predictions']['refine_euler_angle_inv']
        refine_scale_inv = data['predictions']['refine_scale_inv']

        object_num = trans_object_obb.shape[1]

        object_translate_inv = refine_translate_inv[:, :object_num]
        object_euler_angle_inv = refine_euler_angle_inv[:, :object_num]
        object_scale_inv = refine_scale_inv[:, :object_num]

        trans_back_object_obb_list = []
        trans_back_object_abb_list = []
        trans_back_object_obb_center_list = []
        for batch_idx in range(trans_object_obb.shape[0]):
            current_obb_list = []
            current_abb_list = []
            current_obb_center_list = []
            for i in range(trans_object_obb.shape[1]):
                trans_obb = trans_object_obb[batch_idx][i].reshape(-1, 3)
                trans_obb_center = trans_object_obb_center[batch_idx][i]
                translate_inv = object_translate_inv[batch_idx][i]
                euler_angle_inv = object_euler_angle_inv[batch_idx][i]
                scale_inv = object_scale_inv[batch_idx][i]

                trans_back_object_obb = transPointArray(trans_obb,
                                                        translate_inv,
                                                        euler_angle_inv,
                                                        scale_inv,
                                                        is_inverse=True)
                trans_back_object_abb = torch.hstack(
                    (torch.min(trans_back_object_obb,
                               0)[0], torch.max(trans_back_object_obb, 0)[0]))
                trans_back_object_obb_center = trans_obb_center + translate_inv

                trans_back_object_obb = trans_back_object_obb.reshape(1, -1)
                trans_back_object_abb = trans_back_object_abb.unsqueeze(0)
                trans_back_object_obb_center = trans_back_object_obb_center.unsqueeze(
                    0)

                current_obb_list.append(trans_back_object_obb)
                current_abb_list.append(trans_back_object_abb)
                current_obb_center_list.append(trans_back_object_obb_center)

            current_obb = torch.cat(current_obb_list, 0).unsqueeze(0)
            current_abb = torch.cat(current_abb_list, 0).unsqueeze(0)
            current_obb_center = torch.cat(current_obb_center_list,
                                           0).unsqueeze(0)

            trans_back_object_obb_list.append(current_obb)
            trans_back_object_abb_list.append(current_abb)
            trans_back_object_obb_center_list.append(current_obb_center)

        trans_back_object_obb = torch.cat(trans_back_object_obb_list, 0)
        trans_back_object_abb = torch.cat(trans_back_object_abb_list, 0)
        trans_back_object_obb_center = torch.cat(
            trans_back_object_obb_center_list, 0)

        data['predictions'][
            'refine_object_translate_inv'] = object_translate_inv
        data['predictions'][
            'refine_object_euler_angle_inv'] = object_euler_angle_inv
        data['predictions']['refine_object_scale_inv'] = object_scale_inv
        data['predictions']['refine_object_obb'] = trans_back_object_obb
        data['predictions']['refine_object_abb'] = trans_back_object_abb
        data['predictions'][
            'refine_object_obb_center'] = trans_back_object_obb_center
        if not self.infer:
            data = self.loss(data)
        return data

    def loss(self, data):
        refine_object_translate_inv = data['predictions'][
            'refine_object_translate_inv']
        refine_object_euler_angle_inv = data['predictions'][
            'refine_object_euler_angle_inv']
        refine_object_scale_inv = data['predictions'][
            'refine_object_scale_inv']
        refine_object_obb = data['predictions']['refine_object_obb']
        refine_object_abb = data['predictions']['refine_object_abb']
        refine_object_obb_center = data['predictions'][
            'refine_object_obb_center']
        gt_object_translate_inv = data['inputs']['translate_inv']
        gt_object_euler_angle_inv = data['inputs']['euler_angle_inv']
        gt_object_scale_inv = data['inputs']['scale_inv']
        gt_object_obb = data['inputs']['object_obb']
        gt_object_abb = data['inputs']['object_abb']
        gt_object_obb_center = data['inputs']['object_obb_center']

        loss_refine_translate_inv_l1 = self.l1_loss(
            refine_object_translate_inv, gt_object_translate_inv)
        loss_refine_euler_angle_inv_l1 = self.l1_loss(
            refine_object_euler_angle_inv, gt_object_euler_angle_inv)
        loss_refine_scale_inv_l1 = self.l1_loss(refine_object_scale_inv,
                                                gt_object_scale_inv)
        loss_refine_object_obb_l1 = self.l1_loss(refine_object_obb,
                                                 gt_object_obb)
        loss_refine_object_abb_l1 = self.l1_loss(refine_object_abb,
                                                 gt_object_abb)
        loss_refine_object_obb_center_l1 = self.l1_loss(
            refine_object_obb_center, gt_object_obb_center)
        loss_refine_object_abb_eiou = torch.mean(
            IoULoss.EIoU(refine_object_abb.reshape(-1, 6),
                         gt_object_abb.reshape(-1, 6)))

        data['losses'][
            'loss_refine_translate_inv_l1'] = loss_refine_translate_inv_l1
        data['losses'][
            'loss_refine_euler_angle_inv_l1'] = loss_refine_euler_angle_inv_l1
        data['losses']['loss_refine_scale_inv_l1'] = loss_refine_scale_inv_l1
        data['losses']['loss_refine_object_obb_l1'] = loss_refine_object_obb_l1
        data['losses']['loss_refine_object_abb_l1'] = loss_refine_object_abb_l1
        data['losses'][
            'loss_refine_object_obb_center_l1'] = loss_refine_object_obb_center_l1
        data['losses'][
            'loss_refine_object_abb_eiou'] = loss_refine_object_abb_eiou
        return data

    def setWeight(self, data):
        if self.infer:
            return data

        data = setWeight(data, 'loss_refine_translate_inv_l1', 1000)
        data = setWeight(data, 'loss_refine_euler_angle_inv_l1', 1000)
        data = setWeight(data, 'loss_refine_scale_inv_l1', 1000)
        data = setWeight(data, 'loss_refine_object_obb_l1', 1000)
        data = setWeight(data, 'loss_refine_object_abb_l1', 1000)
        data = setWeight(data, 'loss_refine_object_obb_center_l1', 1000)
        data = setWeight(data, 'loss_refine_object_abb_eiou', 1, max_value=1)
        return data

    def forward(self, data):
        data = self.buildMap(data)

        data = self.embedFeatures(data)

        data = self.updateFeature(data)

        data = self.decodeFeature(data)

        data = self.updatePose(data)

        data = self.setWeight(data)
        return data
