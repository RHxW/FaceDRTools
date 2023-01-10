import os
import cv2
import numpy as np

from FaceDet.FDAPI import FDAPI
from VideoTools.audio_tools import *


def video_face_det(cfg, video_path, save_res=False, save_path=""):
    if not os.path.exists(video_path):
        return

    fdapi = FDAPI(cfg)

    video_capture = cv2.VideoCapture(video_path)
    fps = int(video_capture.get(5))  # 获取帧率
    H = int(video_capture.get(4))
    W = int(video_capture.get(3))
    videowriter = None
    if save_res:
        videowriter = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (W, H))
    while True:
        # Capture frame-by-frame
        success, frame_image = video_capture.read()
        if not success:
            break
        oriImg = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)

        img_det, kpts, fbox, text, kpts_src = fdapi.detect(oriImg)
        new_kpts = []
        for kpt in kpts:
            e1x, e1y = kpt[0]
            e2x, e2y = kpt[1]
            emx = (e1x + e2x) / 2
            emy = (e1y + e2y) / 2
            new_p = np.array([[emx, emy]])
            new_kpt = np.concatenate([kpt, new_p])
            new_kpts.append(new_p)

        out = fdapi.draw_annos_all(frame_image, new_kpts, fbox, text, show_kpts=False)

        # Display the resulting frame
        # cv2.imshow('Video', out)
        if save_res:
            videowriter.write(out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    videowriter.release()
    cv2.destroyAllWindows()


def video_face_det_with_audio(cfg, video_path, output_dir):
    if not os.path.exists(video_path):
        return

    fdapi = FDAPI(cfg)

    _audio = extract(video_path, output_dir, 'wav')

    video_capture = cv2.VideoCapture(video_path)
    fps = int(video_capture.get(5))  # 获取帧率
    H = int(video_capture.get(4))
    W = int(video_capture.get(3))

    _ext = os.path.basename(video_path).strip().split('.')[-1]
    result = os.path.join(output_dir, '{}.{}'.format(uuid.uuid1().hex, _ext))

    videowriter = cv2.VideoWriter(result, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (W, H))
    while True:
        # Capture frame-by-frame
        success, frame_image = video_capture.read()
        if not success:
            break
        oriImg = cv2.cvtColor(frame_image, cv2.COLOR_BGR2RGB)

        img_det, kpts, fbox, text, kpts_src = fdapi.detect(oriImg)
        new_kpts = []
        for kpt in kpts:
            e1x, e1y = kpt[0]
            e2x, e2y = kpt[1]
            emx = (e1x + e2x) / 2
            emy = (e1y + e2y) / 2
            new_p = np.array([[emx, emy]])
            new_kpt = np.concatenate([kpt, new_p])
            new_kpts.append(new_p)

        out = fdapi.draw_annos_all(frame_image, new_kpts, fbox, text, show_kpts=False)

        # Display the resulting frame
        cv2.imshow('Video', out)

        videowriter.write(out)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    videowriter.release()
    _final_video = video_add_audio(result, _audio, output_dir)
    return _final_video

if __name__ == "__main__":
    from Config.config import CONFIG

    cfg = CONFIG()
    cfg.det_checkpoint_path = "E:/FaceDRTools/FaceDet/checkpoints/Resnet50_Final.pth"
    video_path = "E:/FaceDRTools/77.mp4"
    save_path = "E:/FaceDRTools/77_det.mp4"
    output_dir = "E:/FaceDRTools/"
    video_face_det(cfg, video_path, save_res=True, save_path=save_path)
    # video_face_det_with_audio(cfg, video_path, output_dir)
