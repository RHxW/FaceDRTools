import os

class CONFIG():
    def __init__(self):

        #############################################
        #                                           #
        #              detection API                #
        #                                           #
        #############################################
        self.det_backbone = "resnet"  # ["resnet", "mobile0.25"]
        self.det_device = "cuda:0"
        self.det_checkpoint_path = ""
        if not os.path.exists(self.det_checkpoint_path):
            raise RuntimeError("Detection checkpoint does not exist!!!")
        self.det_size = (-1, -1)  # -1 表示原图， 不做缩放
        self.rgb_mean = (104, 117, 123)  # 实际图像是BGR, 这里顺序也是BGR
        self.confidence_threshold = 0.3
        self.top_k = 500
        self.nms_threshold = 0.4
        self.keep_top_k = 10
        self.vis_thres = 0.3
        self.use_box_scale = True
        self.box_scale = 1.5

        self.save_det = False
        self.det_save_root = ""
        if self.save_det:
            if not os.path.exists(self.det_save_root):
                os.mkdir(self.det_save_root)
            if self.det_save_root[-1] != "/":
                self.det_save_root += "/"

        self.save_align = True
        self.align_save_root = ""
        if self.save_det:
            if not os.path.exists(self.align_save_root):
                os.mkdir(self.align_save_root)
            if self.align_save_root[-1] != "/":
                self.align_save_root += "/"

        self.aligned_size = (96, 112)  # 对齐图像目标 宽， 高
        self.aligned_scale = 1.0


        #############################################
        #                                           #
        #               quality API                 #
        #                                           #
        #############################################
        self.q_device = "cuda:0"
        self.q_checkpoint_path = ""
        if not os.path.exists(self.q_checkpoint_path):
            raise RuntimeError("Quality checkpoint does not exist!!!")
        self.q_size = (112, 96)  # H, W

        #############################################
        #                                           #
        #             recognition API               #
        #                                           #
        #############################################
        self.rec_device = "cuda:0"
        self.rec_checkpoint_path = ""
        if not os.path.exists(self.rec_checkpoint_path):
            raise RuntimeError("Recognition checkpoint does not exist!!!")
        self.rec_size = (112, 96)  # H, W