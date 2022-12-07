#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../udf-generate")
sys.path.append("../auto-cad-recon")
sys.path.append("../mesh-manage/")
sys.path.append("../scannet-dataset-manage")
sys.path.append("../scan2cad-dataset-manage")
sys.path.append("../shapenet-dataset-manage")
sys.path.append("../points-shape-detect")
sys.path.append("../scene-layout-detect")

from global_pose_refine.Module.trainer import Trainer


def demo():
    model_file_path = "./output/pretrained_gcnn/model_best.pth"
    model_file_path = ""
    resume_model_only = True
    print_progress = True

    trainer = Trainer()
    trainer.loadModel(model_file_path, resume_model_only)
    trainer.testTrain()
    #  trainer.train(print_progress)
    return True
