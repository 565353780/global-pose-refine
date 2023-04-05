#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class RelationCalculator(object):

    def __init__(self):
        return

    def calculateRelations(self, obb_list):
        obb_num = len(obb_list)

        if obb_num < 2:
            return None

        relation_matrix = np.zeros([obb_num, obb_num], dtype=float)
        return relation_matrix
