import uuid
import os
from ffmpy3 import FFmpeg


def _run_ffmpeg(video_path: str, audio_path: str, format: str):
    """
    返回的是保存的音频路径
    """
    ff = FFmpeg(inputs={video_path: None},
                outputs={audio_path: '-f {} -vn'.format(format)})
    print(ff.cmd)
    ff.run()
    return audio_path


def extract(video_path: str, tmp_dir: str, ext: str):
    """
    返回的是临时保存的音频路径
    """
    file_name = '.'.join(os.path.basename(video_path).split('.')[0:-1])
    print('文件名:{}，提取音频'.format(file_name))
    if ext == 'mp3':
        return _run_ffmpeg(video_path, os.path.join(tmp_dir, '{}.{}'.format(uuid.uuid1().hex, ext)), 'mp3')
    if ext == 'wav':
        return _run_ffmpeg(video_path, os.path.join(tmp_dir, '{}.{}'.format(uuid.uuid1().hex, ext)), 'wav')


# 视频添加音频
def video_add_audio(video_path: str, audio_path: str, output_dir: str):
    _ext_video = os.path.basename(video_path).strip().split('.')[-1]
    _ext_audio = os.path.basename(audio_path).strip().split('.')[-1]
    if _ext_audio not in ['mp3', 'wav']:
        raise Exception('audio format not support')
    _codec = 'copy'
    if _ext_audio == 'wav':
        _codec = 'aac'
    result = os.path.join(
        output_dir, '{}.{}'.format(
            uuid.uuid4(), _ext_video))
    ff = FFmpeg(
        inputs={video_path: None, audio_path: None},
        outputs={result: '-map 0:v -map 1:a -c:v copy -c:a {} -shortest'.format(_codec)})
    print(ff.cmd)
    ff.run()
    return result
