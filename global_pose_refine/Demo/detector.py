#!/usr/bin/env python
# -*- coding: utf-8 -*-

from global_pose_refine.Module.detector import Detector


def demo():
    model_file_path = None

    detector = Detector(model_file_path)

    detector.detectSceneObjects(None)
    return True
