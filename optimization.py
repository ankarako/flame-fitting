from dataclasses import dataclass
import os
import json

import torch


k_deca_dirname = 'DECA'
k_image_dirname = 'image'
k_iris_filename = 'iris.json'
k_lmk_filename = 'keypoints.json'

@dataclass
class OptimizationState:
    device: str = 'cuda'
    shape: torch.Tensor = None


def optimization_loop(
    shape: torch.Tensor,
    expressions: torch.Tensor,
    lmks: torch.Tensor,
    poses: torch.Tensor,
    filename: str,
    output_dir: str
) -> None:
    """
    """




def optimize_directory(output_dir: str, size) -> None:
    """
    """
    output_filepath = os.path.join(output_dir, 'flame_data_dict.json')
    deca_code_filepath = os.path.join(output_dir, k_deca_dirname, 'flame_space_deca.json')
    lmk_filepath = os.path.join(output_dir, k_lmk_filename)
    iris_file = os.path.join(output_dir, k_iris_filename)

    # read input files
    with open(deca_code_filepath, 'r') as infd:
        deca_data = json.load(infd)
        
    with open(lmk_filepath, 'r') as infd:
        lmk_data = json.load(infd)

    with open(iris_file, 'r') as infd:
        iris_data = json.load(infd)

    # parse input data in a single batch
    shapes = []
    expressions = []
    lmks = []
    poses = []
    cams = []
    filenames = []

    for filename in lmk_data:
        frame_filepath = os.path.join(output_dir, k_image_dirname, filename)
        video_name = os.path.basename(output_dir)
        frame_idx = int(filename.split('-')[-1].replace('.png', ''))
        deca_data_key = video_name + '_frame{:04d}'.format(frame_idx)

        shape = deca_data[deca_data_key]['shape']
        expression = deca_data[deca_data_key]['exp']
        pose = deca_data[deca_data_key]['pose']
        cam = deca_data[deca_data_key]['cam']
        lmk = lmk_data[filename]
        iris = iris_data[filename]

        shapes += [torch.tensor(shape)]
        expressions += [torch.tensor(expression)]
        poses += [torch.tensor(pose)]
        cams += [torch.tensor(cam)]
        
        lmk = torch.tensor(lmk)
        iris = torch.tensor(iris)
        lmk = torch.cat([lmk, iris[[1, 0], :]], dim=0)

        lmk = lmk / size * 2 - 1
        lmks += [lmk.float()]
        filenames += [filename]
    
    shapes = torch.cat(shapes, dim=0)
    shape = torch.mean(shapes, dim=0).unsqueeze(0)
    expressions = torch.cat(expressions, dim=0)
    lmks = torch.cat(lmks, dim=0)
    poses = torch.cat(poses, dim=0)

    # run optimization loop for all data
    optimization_loop(shape, expressions, lmks, poses, filenames, output_dir)