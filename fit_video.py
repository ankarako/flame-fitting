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
import json

import face_alignment
from skimage import io
from tqdm import tqdm

from fdlite import (
    FaceDetection,
    FaceLandmark,
    face_detection_to_roi,
    IrisLandmark,
    iris_roi_from_face_landmarks
)

from PIL import Image
import re

import optimization as optim


# inference device
k_device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
k_device = torch.device(k_device_name)


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
        if os.path.exists(filepath):
            continue

        cv2.imwrite(filepath, frame[..., [2, 1, 0]])
        video_state.extracted_filepaths += [filepath]
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
            output_path = os.path.join(mask_output_dir, f"{frame_id}.png")
            if os.path.exists(output_path):
                continue
            frame = video_state.frames[frame_id].astype(np.float32) / 255.0
            frame = torch.from_numpy(frame).permute(2, 0, 1).to(k_device)
            fgr, pha, *rec = model(frame.unsqueeze(0), *rec, 1.0)
            comp = fgr * pha + bgr * (1 - pha)
            comp = torch.where(comp != 0, 1, 0)
            comp = (comp.squeeze().permute(1, 2, 0)  * 255.0).cpu().numpy().astype(np.uint8)
            cv2.imwrite(output_path, comp)
            


def infer_deca(input_video_path: str, output_dir: str):
    """
    Infer the specified video_state with DECA and 
    export the results
    """
    log.INFO(f"Running DECA on: {input_video_path}")

    # construct DECA arguments
    k_default_DECA_args = {
        'inputpath': input_video_path,
        'savefolder': os.path.join(output_dir, 'DECA'), # help='path to the output 
        'device': k_device_name,
        'iscrop': True, # help='whether to crop input 
        'sample_step': 1, # help='sample images from video 
        'detector': 'fan', # help='detector for cropping face,
        'rasterizer_type': 'pytorch3d', # help='rasterizer type: pytorch3d 
        'render_orig': False, # help='whether to render results 
        'useTex': False, # help='whether to use FLAME set it to True only if set it to True only if you downloaded texture model
        'extractTex': True, # help='whether to extract texture from input image as the uv texture map, set false if you want albeo map from FLAME mode
        'saveVis': False, # help='whether to save visualization of output
        'saveKpt': False, # help='whether to save 2D and 3D keypoints
        'saveDepth': False, # help='whether to save depth 
        'saveObj': False, # help='whether to save outputs as .obj, detail mesh will end with _detail.obj
        'saveMat': False, # help='whether to save outputs as .mat
        'saveImages': False, # help='whether to save visualization output as seperate images
        'saveCode': True,
    }
    if os.path.exists(os.path.join(output_dir, 'DECA', 'flame_space_deca.json')):
        return

    k_default_DECA_args = util.conf.make_easy(k_default_DECA_args)
    submodules.DECA.demo_reconstruct(k_default_DECA_args)



def detect_lmks(output_dir: str):
    """
    Detect facial landmarks
    """
    keypoints_path = os.path.join(output_dir, 'keypoints.json')
    if os.path.exists(keypoints_path):
        return
    
    input_path = os.path.join(output_dir, 'image')
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    preds = fa.get_landmarks_from_directory(input_path)

    save = {}
    for k, v in preds.items():
        k = k.split('/')[-1]
        save[k] = v[0].tolist()

    with open(keypoints_path, 'w') as outfd:
        json.dump(save, outfd)



def detect_iris(output_dir: str):
    """
    Detect landmarks
    """
    output_path = os.path.join(output_dir, 'iris.json')
    if os.path.exists(output_path):
        return
    
    detect_faces = FaceDetection()
    detect_face_landmarks = FaceLandmark()
    detect_iris_landmarks = IrisLandmark()

    images_path = os.path.join(output_dir, 'image')
    image_filenames = os.listdir(images_path)
    image_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

    landmarks = { }
    for image_filename in tqdm(image_filenames, desc='detecting iris lmkks', total=len(image_filenames)):
        img = Image.open(os.path.join(images_path, image_filename))
        width, height = img.size
        img_size = (width, height)

        face_detections = detect_faces(img)
        landmarks[image_filename] = None
        if len(face_detections) != 1:
            log.WARN(f'Failed to detect faces: {image_filename}')
        else:
            for face_detection in face_detections:
                try:
                    face_roi = face_detection_to_roi(face_detection, img_size)
                except ValueError:
                    log.WARN(f"Failed to detect face landmarks: {image_filename}")
                    break

                face_landmarks = detect_face_landmarks(img, face_roi)
                if len(face_landmarks) == 0:
                    log.WARN(f"Failed to detect face landmarks: {image_filename}")
                    break

                iris_rois = iris_roi_from_face_landmarks(face_landmarks, img_size)
                if len(iris_rois) != 2:
                    log.WARN(f"Failed to detect face landmarks: {image_filename}")
                    break

                lmks = []
                for iris_roi in iris_rois[::-1]:
                    try:
                        iris_landmarks = detect_iris_landmarks(img, iris_roi).iris[0:1]
                    except np.linalg.LinAlgError:
                        log.WARN(f"Failed to detect face landmarks: {image_filename}")
                        break

                    for landmark in iris_landmarks:
                        lmks.append(landmark.x * width)
                        lmks.append(landmark.y * height)
        landmarks[image_filename] = lmks
    with open(output_path, 'w') as outfd:
        json.dump(landmarks, outfd)



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

def fit_video(input_video: str, output_dir: str, rvm_chkp: str):
    """
    Fit everything for a single video
    """
    # prepare output
    output_dir = prepare_output_dir(conf.input_video, conf.output_dir)

    # extract frames
    video_state = extract_frames(conf.input_video, output_dir)

    # run video matting
    matting(video_state, output_dir, conf.rvm_chkp)

    # run DECA
    infer_deca(conf.input_video, output_dir)

    # detect facial lmks for each frame
    detect_lmks(output_dir)

    # detect iris lmks
    detect_iris(output_dir)

    # optimize for every frame
    optim.optimize_directory(output_dir, video_state.width, video_state.height, k_device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Flame-Video-Fitting")
    parser.add_argument("--conf", type=str, help="Path to the configuration file to load.")
    args = parser.parse_args()

    conf = util.conf.read_conf(args.conf)
    fit_video(conf.input_video, conf.output_dir, conf.rvm_chkp)
    log.INFO(f"Flame-Video-Fitting terminated.")
    log.INFO(f"Output directory: {conf.output_dir}")