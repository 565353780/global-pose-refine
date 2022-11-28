#!/usr/bin/env python
# -*- coding: utf-8 -*-

from global_pose_refine.Module.relation_detector import RelationDetector


def demo():
    model_file_path = None

    relation_detector = RelationDetector(model_file_path)

    relation_detector.detectSceneObjects(None)
    return True
