import os
import shutil

from FaceDet.FDAPI import FDAPI
from FaceQ.FQAPI import FQAPI
from Config.config import CONFIG

def test(cfg):

    FD = FDAPI(cfg)

    test_data_dir = "F:/FaceDRTools/test/test_data/"

    test_res_dir = "F:/FaceDRTools/test/test_res/"
    if not os.path.exists(test_res_dir):
        os.mkdir(test_res_dir)

    det_res_dir = "F:/FaceDRTools/test/test_det_res/"
    if not os.path.exists(det_res_dir):
        os.mkdir(det_res_dir)

    imgs = os.listdir(test_data_dir)
    for img in imgs:
        img_path = test_data_dir + img
        FD.save_det(img_path, det_res_dir, True)
        FD.save_align(img_path, test_res_dir)

    FQ = FQAPI(cfg)
    imgs = os.listdir(test_res_dir)
    q_res_dir = "F:/FaceDRTools/test/test_q/"
    if not os.path.exists(q_res_dir):
        os.mkdir(q_res_dir)
    for img in imgs:
        img_path = test_res_dir + img
        qs = FQ.get_q_score(img_path)
        shutil.copy(img_path, q_res_dir + "%.6f_%s" % (qs, img))

def test_save_anno_batch(cfg):
    FD = FDAPI(cfg)
    img_root = "E:/StyleGAN/generators-with-stylegan2-master/results/"
    files = os.listdir(img_root)
    img_paths = []
    for f in files:
        if f.split(".")[-1] in ["jpg", "png"]:
            img_paths.append(img_root + f)

    save_file = img_root + "annos.txt"
    FD.save_annotations_batch(img_paths, save_file)

def test_save_det(cfg):
    FD = FDAPI(cfg)
    img_root = "E:/StyleGAN/generators-with-stylegan2-master/results/"
    files = os.listdir(img_root)
    img_paths = []
    for f in files:
        if f.split(".")[-1] in ["jpg", "png"]:
            img_paths.append(img_root + f)

    save_dir = img_root + "save_det/"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    for img_path in img_paths:
        FD.save_det(img_path, save_dir, show_kpts=True)

if __name__ == "__main__":
    cfg = CONFIG()
    # test_save_anno_batch(cfg)
    test(cfg)