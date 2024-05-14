import os
import json

import torch
import numpy as np
import log

import submodules
from submodules.DECA import cfg as deca_cfg


k_deca_dirname = 'DECA'
k_image_dirname = 'image'
k_iris_filename = 'iris.json'
k_lmk_filename = 'keypoints.json'

focal_l = 32.0e-3 # 32mm
sensor_width = 23.5e-3  # 23.5mm
sensor_height = 15.6e-3 # 15.6mm

def projection(points, K, w2c, no_intrinsics=False):
    rot = w2c[:, np.newaxis, :3, :3]
    points_cam = torch.sum(points[..., np.newaxis, :] * rot, -1) + w2c[:, np.newaxis, :3, 3]
    if no_intrinsics:
        return points_cam

    points_cam_projected = points_cam
    points_cam_projected[..., :2] /= points_cam[..., [2]]
    points_cam[..., [2]] *= -1

    i = points_cam_projected[..., 0] * K[0] + K[2]
    j = points_cam_projected[..., 1] * K[1] + K[3]
    points2d = torch.stack([i, j, points_cam_projected[..., -1]], dim=-1)
    return points2d


def inverse_projection(points2d, K, c2w):
    i = points2d[:, :, 0]
    j = points2d[:, :, 1]
    dirs = torch.stack([(i - K[2]) / K[0], (j - K[3]) / K[1], torch.ones_like(i) * -1], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:, np.newaxis, :3, :3], -1)
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
    rays_o = c2w[:, np.newaxis, :3, -1].expand(rays_d.shape)

    return rays_o, rays_d

def optimization_loop(
    shape: torch.Tensor,
    expressions: torch.Tensor,
    lmks: torch.Tensor,
    poses: torch.Tensor,
    filename: str,
    width, height,
    output_dir: str,
    lmk_emb_weyes,
    device=torch.device('cuda')
) -> None:
    """
    main optimization loop
    """
    # create a camera
    # we assume (like in IMavatar) that the camera
    # is a little bit on the back of the face
    # and it doesn't move (so we optimize only
    # camera translation)
    # The camera intrinsics are kept constant
    # per video
    
    pixel_size_x = sensor_width / width
    pixel_size_y = sensor_height / height

    cx = (width - 1) / 2
    cy = (height - 1) / 2

    intrinsics = np.array([focal_l / pixel_size_x, focal_l / pixel_size_y, cx, cy])
    intrinsics = torch.from_numpy(intrinsics).to(device)
    log.INFO(f"Using intrinsics matrix: {intrinsics}")

    lmks = lmks.to(device)
    opt_cam_trans = torch.tensor([0.0, 0.0, -1.], dtype=torch.float32, device=device)
    opt_flame_pose = torch.cat([torch.zeros_like(poses[:, :3]), poses], dim=1)

    use_iris = False
    if lmks.shape[1] == 70:
        use_iris = True
    
    if use_iris:
        opt_flame_pose = torch.cat([opt_flame_pose, torch.zeros_like(opt_flame_pose[:, :6])], dim=1)
    
    opt_cam_trans = torch.nn.Parameter(opt_cam_trans.to(device))
    opt_flame_pose = torch.nn.Parameter(opt_flame_pose.to(device))
    opt_flame_exp = torch.nn.Parameter(expressions.to(device))
    opt_flame_shape = torch.nn.Parameter(shape.to(device))

    # create our optimizer
    optim = torch.optim.Adam(
        [opt_cam_trans, opt_flame_pose, opt_flame_exp, opt_flame_shape],
        lr=1.0e-2
    )

    # load DECA
    
    deca_cfg.model.use_tex = False
    deca_cfg.model.flame_lmk_embedding_path = lmk_emb_weyes
    deca_cfg.rasterizer_type = 'pytorch3d'
    # lol
    deca = submodules.DECA.deca.DECA(config=deca_cfg, device=device)

    nframes = opt_flame_exp.shape[0]
    for step in range(10001):
        verts_p, lmks2d_p, lmks3d_p = deca.flame(
            shape_params=opt_flame_shape.expand(nframes, -1),
            expression_params=opt_flame_exp,
            full_pose=opt_flame_pose
        )

        # perspective projection
        # global rotation is handled by FLAME.
        # set camera rotation to identity
        eye = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).expand(nframes, -1, -1)
        cam_trans = opt_cam_trans.unsqueeze(0).expand(nframes, -1).unsqueeze(2)
        w2c_p = torch.cat([eye, cam_trans], dim=2)

        lmks2d_trans = projection(lmks2d_p, intrinsics, w2c_p)[..., :2]

        # calc loss
        lmk_loss = torch.mean(torch.square(lmks2d_trans - lmks))
        total_loss = lmk_loss + torch.mean(torch.square(opt_flame_shape)) * 1e-2 + torch.mean(torch.square(opt_flame_exp)) * 1e-2
        total_loss += torch.mean(torch.square(opt_flame_exp[1:] - opt_flame_exp[:-1])) * 1e-1
        total_loss += torch.mean(torch.square(opt_flame_pose[1:] - opt_flame_pose[:-1])) * 10

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        # visualize
        if step % 100 == 0:
            with torch.no_grad():
                log.INFO(f"step: {step} | loss: {total_loss.detach().cpu().item():.3f} | lmk-loss: {lmk_loss.detach().cpu().item():.3f}")
                trans_verts = projection(verts_p[::50], intrinsics, w2c_p[::50])
                shape_images = deca.render.render_shape(verts_p[::50], trans_verts)







def optimize_directory(
    output_dir: str, 
    width: int, 
    height: int, 
    lmk_emb_weyes: str,
    device=torch.device('cuda')
) -> None:
    log.INFO(f"Running video per frame optimization: {output_dir}")
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
        
        lmk = np.array(lmk)
        iris = np.array(iris).reshape(2, 2)
        lmk = np.concatenate([lmk, iris[[1, 0], :]], 0)

        lmk = lmk / [width, height] * 2 - 1
        lmks += [torch.from_numpy(lmk).float().unsqueeze(0)]
        filenames += [filename]
    
    shapes = torch.cat(shapes, dim=0)
    shape = torch.mean(shapes, dim=0).unsqueeze(0)
    expressions = torch.cat(expressions, dim=0)
    lmks = torch.cat(lmks, dim=0)
    poses = torch.cat(poses, dim=0)

    # run optimization loop for all data
    optimization_loop(
        shape, expressions, lmks, poses, filenames, width, height, output_dir, lmk_emb_weyes, device
    )