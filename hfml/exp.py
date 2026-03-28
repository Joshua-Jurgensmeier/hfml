#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 11
        self.data_num_workers = 4
        self.input_size = (1024, 1024)  # (height, width)
        self.multiscale_range = 0
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = "hfml"

        self.data_dir = None
        self.train_ann = "result.json"
        self.val_ann = "result.json"
        self.test_ann = "result.json"

        # Disable transforms that don't make sense
        # prob of applying hsv aug
        self.hsv_prob = 0
        # prob of applying flip aug
        self.flip_prob = 0
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.shear = 0