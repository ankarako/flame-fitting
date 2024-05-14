import argparse
import os

import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import cv2

import submodules
import util
import log

from tqdm import tqdm


# inference device
k_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def extract_frames(filepath: str, output: str):
    """
    @brief Extract video frames from the specified filepath

    :param filepath The video filepath to load
    :param output the output root directory
    """
    log.INFO(f"Extracting frames from: {filepath}")
    video_state = util.video.read_video(filepath)
    frame_output_dir = os.path.join(output, 'image')

    if not os.path.exists(frame_output_dir):
        os.mkdir(frame_output_dir)

    for frame_id in tqdm(range(len(video_state.frames)), desc="extracting frames", total=len(video_state.frames)):
        frame = video_state.frames[frame_id]
        filepath = os.path.join(frame_output_dir, f"frame-{frame_id}.png")
        cv2.imwrite(filepath, frame[..., [2, 1, 0]])
    return video_state


def matting(video_state: util.video.VideoState, output: str, chkp_path: str):
    """
    @brief Perform robust matting for the specified video.

    :param video_state A valid ``VideoState`` object holding frame videos.
    :param output The root output directory
    """
    log.INFO(f"Performing video matting.")
    if not os.path.exists(chkp_path):
        log.ERROR(f"The specified rvm checkpoint path is invalid: {chkp_path}")
        return None
    
    mask_output_dir = os.path.join(output, 'mask')
    if not os.path.exists(mask_output_dir):
        os.mkdir(mask_output_dir)

    # load rvm model
    model = submodules.RobustVideoMatting.model.MattingNetwork('mobilenetv3').eval().to(k_device)
    model.load_state_dict(torch.load(chkp_path))

    rec = [None] * 4
    bgr = torch.zeros([3]).view(3, 1, 1).to(k_device)
    with torch.no_grad():
        for frame_id in tqdm(range(len(video_state.frames)), desc='infering rvm', total=len(video_state.frames)):
            frame = video_state.frames[frame_id].astype(np.float32) / 255.0
            frame = torch.from_numpy(frame).permute(2, 0, 1).to(k_device)
            fgr, pha, *rec = model(frame.unsqueeze(0), *rec, 1.0)
            comp = fgr * pha + bgr * (1 - pha)
            comp = torch.where(comp != 0, 1, 0)
            comp = (comp.squeeze().permute(1, 2, 0)  * 255.0).cpu().numpy().astype(np.uint8)
            output_path = os.path.join(mask_output_dir, f"{frame_id}.png")
            cv2.imwrite(output_path, comp)
            


def infer_deca():
    """
    """

def detect_lmks():
    """
    """

def detect_iris():
    """
    """

def optimmize_frame():
    """
    """

def prepare_output_dir(input_fpath: str, output: str) -> None:
    """
    Create a video specific folder in the output directory.

    :param input_fpath The input video filepath.
    :param output The output directory
    """
    filename = os.path.basename(input_fpath)
    output_video_dir = os.path.join(output, filename).split('.')[0]
    if not os.path.exists(output_video_dir):
        log.INFO(f"Creating output directory {output_video_dir}")
        os.mkdir(output_video_dir)
    return output_video_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Flame-Video-Fitting")
    parser.add_argument("--conf", type=str, help="Path to the configuration file to load.")
    args = parser.parse_args()

    conf = util.conf.read_conf(args.conf)

    # prepare output
    output_dir = prepare_output_dir(conf.input_video, conf.output_dir)

    # extract frames
    video_state = extract_frames(conf.input_video, output_dir)

    # run video matting
    dunno = matting(video_state, output_dir, conf.rvm_chkp)