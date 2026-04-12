#!/usr/bin/env python3
# -*- coding:utf-8 -*-

# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 python3 train.py -b 288 -f exp.py

import os

from yolox.exp import Exp as MyExp
from yolox.core import Trainer as YoloxTrainer
from yolox.utils import all_reduce_norm

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 8
        self.data_num_workers = 8 # Max with 6 GPUs and 64 cores would be ~10
        self.input_size = (1024, 1024)  # (height, width)
        self.multiscale_range = 0
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = "hfml"

        self.data_dir = "/workspaces/hfml/dataset"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        # self.test_ann = "result.json"

        # Disable transforms that don't make sense
        # self.mosaic_prob = 0.50
        # self.mosaic_scale = (0.5, 1.5)
        self.hsv_prob = 0.
        self.flip_prob = 0.
        self.degrees = 0.
        self.translate = 0.05
        self.shear = 0.
        # self.enable_mixup = True
        # self.mixup_prob = 0.50
        # self.mixup_scale = (0.5, 1.5)
        self.no_aug_epochs = 50

        # TODO: Add time/frequency masks to emulate fades. Noise injection, circular shift

        self.test_size = (1024, 1024)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.65

        self.print_interval = 1 # Have to set this lower because batch size is high.
        self.eval_interval = 15
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = False
    
    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        """
        Get dataset according to cache and cache_type parameters.
        Args:
            cache (bool): Whether to cache imgs to ram or disk.
            cache_type (str, optional): Defaults to "ram".
                "ram" : Caching imgs to ram for fast training.
                "disk": Caching imgs to disk for fast training.
        """
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name="train",
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
        )
    
    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform
        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name="val" if not testdev else "test",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

# Move ckpt inside eval interval
class Trainer(YoloxTrainer):
    def after_epoch(self):
        if (self.epoch + 1) % self.exp.eval_interval == 0:
            self.save_ckpt(ckpt_name="latest")
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()
    
    def train_in_iter(self):
        print(f"{self.max_iter}")
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()