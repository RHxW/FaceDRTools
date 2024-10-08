import os
import torch

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
class CONFIG():
    def __init__(self):

        #############################################
        #                                           #
        #              detection API                #
        #                                           #
        #############################################
        self.det_backbone = "resnet"  # ["resnet", "mobile0.25"]
        self.det_device = DEVICE
        self.det_checkpoint_path = ""

        self.det_infer_size = (320, 320)  # -1 表示原图， 不做缩放
        self.rgb_mean = (104, 117, 123)  # 实际图像是BGR, 这里顺序也是BGR
        self.confidence_threshold = 0.3
        self.top_k = 500
        self.nms_threshold = 0.4
        self.keep_top_k = 10
        self.vis_thres = 0.3
        self.use_box_scale = True
        self.box_scale = 1.5

        self.aligned_size = (96, 112)  # 对齐图像目标 宽， 高
        self.aligned_scale = 1.0


        #############################################
        #                                           #
        #               quality API                 #
        #                                           #
        #############################################
        self.q_device = DEVICE
        self.q_checkpoint_path = ""

        self.q_size = (112, 96)  # H, W

        #############################################
        #                                           #
        #             recognition API               #
        #                                           #
        #############################################
        self.rec_device = DEVICE
        self.rec_checkpoint_path = ""

        self.rec_size = (112, 96)  # H, W