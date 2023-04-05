#!/usr/bin/env python
# -*- coding: utf-8 -*-

from global_pose_refine.Data.obb import OBB
from global_pose_refine.Module.relation_calculator import RelationCalculator


def demo():
    obb_list = [
        OBB.fromABBList([0, 0, 0, 1, 1, 1]),
        OBB.fromABBList([0.5, 0.5, 2, 1.5, 1.5, 3]),
        OBB.fromABBList([0.25, 0.75, 4, 0.75, 1.25, 5]),
        OBB.fromABBList([2, 2, 4, 3, 3, 5]),
    ]

    relation_calculator = RelationCalculator()
    relation_matrix = relation_calculator.calculateRelations(obb_list)
    print('relation_matrix is')
    print(relation_matrix)
    if relation_matrix is not None:
        print(relation_matrix.shape)
    return True