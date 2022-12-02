#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d

from points_shape_detect.Data.bbox import BBox
from points_shape_detect.Method.bbox import (getOpen3DBBox,
                                             getOpen3DBBoxFromBBox)


def getPCDFromPointArray(point_array, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_array)

    if color is not None:
        colors = np.array([color for _ in range(point_array.shape[0])],
                          dtype=float) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def renderPointArray(point_array):
    if isinstance(point_array, np.ndarray):
        pcd = getPCDFromPointArray(point_array)
    else:
        pcd = getPCDFromPointArray(point_array.detach().cpu().numpy())

    o3d.visualization.draw_geometries([pcd])
    return True


def renderPointArrayList(point_array_list):
    if isinstance(point_array_list[0], np.ndarray):
        points = np.vstack(point_array_list)
        return renderPointArray(points)

    points = torch.vstack(point_array_list)
    return renderPointArray(points)


def renderRefineBBox(data):
    assert 'object_bbox' in data['inputs'].keys()
    assert 'object_center' in data['inputs'].keys()
    assert 'refine_object_bbox' in data['predictions'].keys()
    assert 'refine_object_center' in data['predictions'].keys()

    pcd_list = []

    gt_bbox_list_list = data['inputs']['gt_object_bbox'][0].cpu().numpy(
    ).reshape(-1, 2, 3)
    for gt_bbox_list in gt_bbox_list_list:
        gt_bbox = BBox.fromList(gt_bbox_list)
        open3d_gt_bbox = getOpen3DBBoxFromBBox(gt_bbox, [0, 255, 0])
        pcd_list.append(open3d_gt_bbox)

    gt_center_list = data['inputs']['gt_object_center'][0].cpu().numpy(
    ).reshape(-1, 1, 3)
    for gt_center in gt_center_list:
        gt_center_pcd = getPCDFromPointArray(gt_center, [0, 255, 0])
        pcd_list.append(gt_center_pcd)

    bbox_list_list = data['inputs']['object_bbox'][0].cpu().numpy().reshape(
        -1, 2, 3)
    for bbox_list in bbox_list_list:
        bbox = BBox.fromList(bbox_list)
        open3d_bbox = getOpen3DBBoxFromBBox(bbox, [255, 0, 0])
        pcd_list.append(open3d_bbox)

    center_list = data['inputs']['object_center'][0].cpu().numpy().reshape(
        -1, 1, 3)
    for center in center_list:
        center_pcd = getPCDFromPointArray(center, [255, 0, 0])
        pcd_list.append(center_pcd)

    #  bbox_list = data['predictions']['refine_object_bbox'][0].detach().cpu(
    #  ).numpy().reshape(2, 3)
    #  bbox = BBox.fromList(bbox_list)
    #  open3d_bbox = getOpen3DBBoxFromBBox(bbox, [255, 0, 0])
    #  pcd_list.append(open3d_bbox)

    #  center = data['predictions']['refine_object_center'][0].detach().cpu(
    #  ).numpy().reshape(1, 3)
    #  center_pcd = getPCDFromPointArray(center, [255, 0, 0])
    #  pcd_list.append(center_pcd)

    o3d.visualization.draw_geometries(pcd_list)
    return True
