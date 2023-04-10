#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from global_pose_refine.Data.obb import OBB
from global_pose_refine.Method.dist import getOBBPoseDist, getOBBSupportDist


class RelationCalculator(object):

    def __init__(self):
        return

    def calculateRelations(self, obb_list):
        obb_num = len(obb_list)

        if obb_num == 0:
            return None

        relation_matrix = np.zeros([obb_num, obb_num], dtype=float)

        for i in range(obb_num - 1):
            for j in range(i + 1, obb_num):
                support_dist = getOBBSupportDist(obb_list[i], obb_list[j])
                pose_dist = getOBBPoseDist(obb_list[i], obb_list[j])
                relation_value = support_dist + pose_dist

                relation_matrix[i, j] = relation_value
                relation_matrix[j, i] = relation_value

        return relation_matrix

    def calculateRelationsByABBValueList(self, abb_value_list):
        obb_list = [OBB.fromABBList(abb_value) for abb_value in abb_value_list]
        relation_matrix = self.calculateRelations(obb_list)
        return relation_matrix

    def calculateRelationsByOBBValueList(self, obb_value_list):
        obb_list = [OBB(obb_value) for obb_value in obb_value_list]
        return self.calculateRelations(obb_list)
