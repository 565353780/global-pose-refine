#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from global_pose_refine.Model.gcnn.gclayer_collect import GraphConvolutionLayerCollect
from global_pose_refine.Model.gcnn.gclayer_update import GraphConvolutionLayerUpdate


class GCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.lo_features = ['layout_center', 'layout_bbox']
        self.obj_features = ['object_center', 'object_bbox']
        self.rel_features = ['center_dist', 'bbox_eiou']

        self.feature_dim = 512
        self.feat_update_step = 4
        self.feat_update_group = 1

        self.feature_length = {
            'layout_center': 3,
            'layout_bbox': 6,
            'object_center': 3,
            'object_bbox': 6,
            'center_dist': 1,
            'bbox_eiou': 1,
        }

        obj_features_len = sum(
            [self.feature_length[k] for k in self.obj_features])
        rel_features_len = sum(
            [self.feature_length[k] for k in self.rel_features])
        lo_features_len = sum(
            [self.feature_length[k] for k in self.lo_features])

        self.obj_embedding = nn.Sequential(
            nn.Linear(obj_features_len, self.feature_dim),
            nn.ReLU(True),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.rel_embedding = nn.Sequential(
            nn.Linear(rel_features_len, self.feature_dim),
            nn.ReLU(True),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.lo_embedding = nn.Sequential(
            nn.Linear(lo_features_len, self.feature_dim),
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

        # branch to predict the bbox_scale
        self.fc1 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc2 = nn.Linear(self.feature_dim // 2, 3)

        # branch to predict the orientation
        self.fc3 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc4 = nn.Linear(self.feature_dim // 2, 6)

        # branch to predict the centroid
        self.fc5 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc_centroid = nn.Linear(self.feature_dim // 2, 3)

        self.relu_1 = nn.LeakyReLU(0.2)
        self.dropout_1 = nn.Dropout(p=0.5)

        # initiate weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()
        return

    def getMap(self, data):
        device = data['inputs'][self.obj_features[0]].device

        object_num = data['inputs'][self.obj_features[0]].shape[1]
        layout_num = data['inputs'][self.lo_features[0]].shape[1]
        total_num = layout_num + object_num

        total_map = torch.ones([total_num, total_num]).to(device)

        object_mask = torch.zeros(total_num, dtype=torch.bool).to(device)
        layout_mask = torch.zeros(total_num, dtype=torch.bool).to(device)
        object_mask[:object_num] = True
        layout_mask[object_num:] = True

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
        data['predictions']['layout_mask'] = layout_mask
        data['predictions']['total_map'] = total_map
        data['predictions']['subj_pred_map'] = subj_pred_map
        data['predictions']['obj_pred_map'] = obj_pred_map
        return data

    def embedObjectFeature(self, data):
        object_feature_list = []
        for key in self.obj_features:
            object_feature_list.append(data['inputs'][key])

        cat_object_feature = torch.cat(object_feature_list, -1)

        embed_object_feature = self.obj_embedding(cat_object_feature)

        data['predictions']['cat_object_feature'] = cat_object_feature
        data['predictions']['embed_object_feature'] = embed_object_feature
        return data

    def embedLayoutFeature(self, data):
        layout_feature_list = []
        for key in self.lo_features:
            layout_feature_list.append(data['inputs'][key])

        cat_layout_feature = torch.cat(layout_feature_list, -1)

        embed_layout_feature = self.lo_embedding(cat_layout_feature)

        data['predictions']['cat_layout_feature'] = cat_layout_feature
        data['predictions']['embed_layout_feature'] = embed_layout_feature
        return data

    def embedRelationFeature(self, data):
        relation_feature_list = []
        for key in self.rel_features:
            relation_feature_list.append(data['inputs'][key])

        cat_relation_feature = torch.cat(relation_feature_list, -1)

        embed_relation_feature = self.rel_embedding(cat_relation_feature)

        data['predictions']['cat_relation_feature'] = cat_relation_feature
        data['predictions']['embed_relation_feature'] = embed_relation_feature
        return data

    def embedFeatures(self, data):
        data = self.embedObjectFeature(data)
        data = self.embedLayoutFeature(data)
        data = self.embedRelationFeature(data)

        relation_mask = data['predictions']['relation_mask']
        embed_object_feature = data['predictions']['embed_object_feature']
        embed_layout_feature = data['predictions']['embed_layout_feature']
        embed_relation_feature = data['predictions']['embed_relation_feature']

        object_num = embed_object_feature.shape[1]
        layout_num = embed_layout_feature.shape[1]
        total_num = object_num + layout_num

        # representation of object and layout vertices
        cat_total_feature_list = [embed_object_feature, embed_layout_feature]
        embed_total_feature = torch.cat(cat_total_feature_list, 1)

        # representation of relation vertices connecting obj/lo vertices
        if layout_num > 0:
            embed_relation_feature_matrix = embed_relation_feature.reshape(
                object_num, object_num, -1)
            total_relation_feature_matrix = F.pad(
                embed_relation_feature_matrix.permute(2, 0, 1),
                [0, layout_num, 0, layout_num], "constant",
                0.001).permute(1, 2, 0)
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
        total_map = data['predictions']['total_map']
        subj_pred_map = data['predictions']['subj_pred_map']
        obj_pred_map = data['predictions']['obj_pred_map']
        embed_total_feature = data['predictions']['embed_total_feature']
        mask_total_relation_feature = data['predictions'][
            'mask_total_relation_feature']

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

        obj_feats_wolo = total_feature_list[-1][object_mask[0]]

        data['predictions']['update_total_feature'] = total_feature_list[-1]
        data['predictions']['obj_feats_wolo'] = obj_feats_wolo
        return data

    def decodeFeature(self, data):
        layout_mask = data['predictions']['layout_mask']
        update_total_feature = data['predictions']['update_total_feature']
        obj_feats_wolo = data['predictions']['obj_feats_wolo']

        # branch to predict the bbox_scale
        size = self.fc1(obj_feats_wolo)
        size = self.relu_1(size)
        size = self.dropout_1(size)
        size = self.fc2(size)

        # branch to predict the orientation
        ori = self.fc3(obj_feats_wolo)
        ori = self.relu_1(ori)
        ori = self.dropout_1(ori)
        ori = self.fc4(ori)

        # branch to predict the centroid
        centroid = self.fc5(obj_feats_wolo)
        centroid = self.relu_1(centroid)
        centroid = self.dropout_1(centroid)
        centroid = self.fc_centroid(centroid)

        obj_feats_lo = update_total_feature[layout_mask[0]]

        data['predictions']['refine_bbox_scale_diff'] = size
        data['predictions']['refine_rotation_diff'] = ori
        data['predictions']['refine_center_diff'] = centroid

        #  if self.training:
        data = self.loss(data)
        return data

    def loss(self, data):
        return data

    def setWeight(self, data):
        #  if self.training:
        #  return

        return data

    def forward(self, data):
        data = self.getMap(data)

        data = self.embedFeatures(data)

        data = self.updateFeature(data)

        data = self.decodeFeature(data)

        data = self.setWeight(data)
        return data
