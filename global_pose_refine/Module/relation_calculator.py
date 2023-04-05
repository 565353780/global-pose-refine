#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from global_pose_refine.Method.dist import getOBBSupportDist


class RelationCalculator(object):

    def __init__(self):
        return

    def calculateRelations(self, obb_list):
        obb_num = len(obb_list)

        if obb_num < 2:
            return None

        relation_matrix = np.zeros([obb_num, obb_num], dtype=float)

        for i in range(obb_num - 1):
            for j in range(i + 1, obb_num):
                relation_value = 0

                support_dist = getOBBSupportDist(obb_list[i], obb_list[j])
                relation_value += support_dist

                relation_matrix[i, j] = relation_value
                relation_matrix[j, i] = relation_value

        return relation_matrix
