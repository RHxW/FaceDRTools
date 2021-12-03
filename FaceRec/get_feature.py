import os
import torch
from FaceRec.FRAPI import FRAPI
from Config.config import CONFIG as cfg

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output

if __name__ == "__main__":
    cfg = cfg()
    cfg.rec_checkpoint_path = 'F:/partial_fc/result/rar_real_29w/backbone/Backbone_IR_SE_50_Epoch_30_Time_2021-10-07-18-47.pth'
    FRAPI = FRAPI(cfg)
    img_path = 'F:/face_rec_testdata/lfw/data/1_true/2.png'
    feature = FRAPI.get_feature(img_path)
    feature = l2_norm(feature)
    print(feature)