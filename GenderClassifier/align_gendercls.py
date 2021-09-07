import os
import shutil
import torch
import numpy as np
import cv2
import tqdm

from FaceDet.FDAPI import FDAPI
from Config.config import CONFIG

from GenderClassifier.config import GC_CONFIG as GC_cfg
from GenderClassifier.model import GenderClsNetwork


def align_and_gendercls(API_cfg, GC_cfg, data_root, res_root):
    """
    保存两个txt文件(male_list, female_list)
    :param API_cfg:
    :param GC_cfg:
    :param data_root:
    :param res_root:
    :return:
    """
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    RGB_MEAN = np.array([0.5507809, 0.4314252, 0.37640554], dtype=np.float32)
    RGB_STD = np.array([0.29048514, 0.25262007, 0.24208826], dtype=np.float32)

    FD = FDAPI(API_cfg)

    device = GC_cfg["device"]

    network = GenderClsNetwork().to(device)
    network.eval()
    if os.path.exists(GC_cfg["checkpoint_file"]):
        checkpoint = torch.load(GC_cfg["checkpoint_file"], map_location=device)
        network.load_state_dict(checkpoint)
    else:
        raise RuntimeError('Gender cls ckpt does not exist!!!')

    if not os.path.exists(data_root):
        return
    if data_root[-1] != "/":
        data_root += "/"

    male_imgs = []
    female_imgs = []

    for name in tqdm.tqdm(os.listdir(data_root)):
        if name.split(".")[-1] not in ['jpg', 'png']:
            continue
        img_path = data_root + name
        img_aligned = FD.get_align(img_path)[0]
        img = cv2.cvtColor(img_aligned, cv2.COLOR_BGR2RGB)
        img = (np.array(img, dtype=np.float32) / 255 - RGB_MEAN) / RGB_STD
        img = img.transpose((2, 0, 1))
        img = torch.tensor(img).unsqueeze(0).to(device)
        pred = network(img).argmax().item()
        if pred == 1:
            male_imgs.append(name + '\n')
        else:
            female_imgs.append(name + '\n')

    if not os.path.exists(res_root):
        os.mkdir(res_root)
    if res_root[-1] != '/':
        res_root += '/'
    male_list_file = res_root + 'male_list.txt'
    female_list_file = res_root + 'female_list.txt'

    with open(male_list_file,'w') as f:
        f.writelines(male_imgs)
    with open(female_list_file,'w') as f:
        f.writelines(female_imgs)

if __name__ == "__main__":
    data_root = ''
    res_root = ''
    API_cfg = CONFIG()
    align_and_gendercls(API_cfg, GC_cfg, data_root, res_root)