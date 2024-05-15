import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import json

import log

# GLOBAL_POSE: if true, optimize global rotation, otherwise, only optimize head rotation (shoulder stays un-rotated)
# if GLOBAL_POSE is set to false, global translation is used.
GLOBAL_POSE = True
from submodules.DECA.decalib.deca import DECA
from submodules.DECA.decalib.utils import util
# from decalib.utils import util
from submodules.DECA.decalib.utils.config import cfg as deca_cfg
from submodules.DECA.decalib.utils import lossfunc
# from decalib.utils import lossfunc

import cv2
import argparse

np.random.seed(0)


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
    rays_d = F.normalize(rays_d, dim=-1)
    rays_o = c2w[:, np.newaxis, :3, -1].expand(rays_d.shape)

    return rays_o, rays_d


class Optimizer(object):
    def __init__(self, device='cuda:0'):
        deca_cfg.model.use_tex = False
        # TODO: landmark_embedding.npy with eyes to optimize iris parameters
        deca_cfg.model.flame_lmk_embedding_path = os.path.join(deca_cfg.deca_dir, 'data',
                                                               'landmark_embedding.npy')
        deca_cfg.rasterizer_type = 'pytorch3d' # or 'standard'
        self.deca = DECA(config=deca_cfg, device=device)

    def optimize(self, shape, exp, landmark, pose, name, visualize_images, savefolder, intrinsics, json_path, size,
                 save_name):
        num_img = pose.shape[0]
        # we need to project to [-1, 1] instead of [0, size], hence modifying the cam_intrinsics as below
        cam_intrinsics = torch.tensor(
            [-1 * intrinsics[0] / size * 2, intrinsics[1] / size * 2, intrinsics[2] / size * 2 - 1,
             intrinsics[3] / size * 2 - 1]).float().cuda()

        if GLOBAL_POSE:
            translation_p = torch.tensor([0, 0, -2]).float().cuda()
        else:
            translation_p = torch.tensor([0, 0, -2]).unsqueeze(0).expand(num_img, -1).float().cuda()

        if GLOBAL_POSE:
            pose = torch.cat([torch.zeros_like(pose[:, :3]), pose], dim=1)
        if landmark.shape[1] == 70:
            # use iris landmarks, optimize gaze direction
            use_iris = True
        if use_iris:
            pose = torch.cat([pose, torch.zeros_like(pose[:, :6])], dim=1)

        pose = torch.zeros_like(pose)
        # actually better to start with zero pose, converges faster
        translation_p = nn.Parameter(translation_p)
        pose = nn.Parameter(pose)
        exp = nn.Parameter(exp)
        shape = nn.Parameter(shape)

        # set optimizer
        if json_path is None:
            opt_p = torch.optim.Adam(
                [translation_p, pose, exp, shape],
                lr=1e-2)
        else:
            opt_p = torch.optim.Adam(
                [translation_p, pose, exp],
                lr=1e-2)

        # optimization steps
        len_landmark = landmark.shape[1]
        for k in range(1001):
            full_pose = pose
            if not use_iris:
                full_pose = torch.cat([full_pose, torch.zeros_like(full_pose[..., :6])], dim=1)
            if not GLOBAL_POSE:
                full_pose = torch.cat([torch.zeros_like(full_pose[:, :3]), full_pose], dim=1)
            verts_p, landmarks2d_p, landmarks3d_p = self.deca.flame(shape_params=shape.expand(num_img, -1),
                                                                    expression_params=exp,
                                                                    full_pose=full_pose)
            # CAREFUL: FLAME head is scaled by 4 to fit unit sphere tightly
            # verts_p *= 4
            # landmarks3d_p *= 4
            # landmarks2d_p *= 4

            # perspective projection
            # Global rotation is handled in FLAME, set camera rotation matrix to identity
            ident = torch.eye(3).float().cuda().unsqueeze(0).expand(num_img, -1, -1)
            if GLOBAL_POSE:
                w2c_p = torch.cat([ident, translation_p.unsqueeze(0).expand(num_img, -1).unsqueeze(2)], dim=2)
            else:
                w2c_p = torch.cat([ident, translation_p.unsqueeze(2)], dim=2)

            trans_landmarks2d = projection(landmarks2d_p, cam_intrinsics, w2c_p)
            ## landmark loss
            landmark_loss2 = lossfunc.l2_distance(trans_landmarks2d[:, :len_landmark, :2], landmark[:, :len_landmark])
            total_loss = landmark_loss2 + torch.mean(torch.square(shape)) * 1e-2 + torch.mean(torch.square(exp)) * 1e-2
            total_loss += torch.mean(torch.square(exp[1:] - exp[:-1])) * 1e-1
            total_loss += torch.mean(torch.square(pose[1:] - pose[:-1])) * 10
            if not GLOBAL_POSE:
                total_loss += torch.mean(torch.square(translation_p[1:] - translation_p[:-1])) * 10

            opt_p.zero_grad()
            total_loss.backward()
            opt_p.step()

            # visualize
            if k % 100 == 0:
                with torch.no_grad():
                    loss_info = '----iter: {}, time: {}\n'.format(k,
                                                                  datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
                    loss_info = loss_info + f'landmark_loss: {landmark_loss2}'
                    print(loss_info)
                    trans_verts = projection(verts_p[::50], cam_intrinsics, w2c_p[::50])
                    # trans_landmarks2d_for_visual = projection(landmarks2d_p, cam_intrinsics, w2c_p)
                    shape_images = self.deca.render.render_shape(verts_p[::50], trans_verts)
                    visdict = {
                        'inputs': visualize_images,
                        'gt_landmarks2d': util.tensor_vis_landmarks(visualize_images, landmark[::50]),
                        'landmarks2d': util.tensor_vis_landmarks(visualize_images, trans_landmarks2d.detach()[::50]),
                        'shape_images': shape_images
                    }
                    cv2.imwrite(os.path.join(savefolder, 'optimize_vis.jpg'), self.deca.visualize(visdict))

                    # shape_images = self.deca.render.render_shape(verts_p, trans_verts)
                    # print(shape_images.shape)

                    save = True
                    if save:
                        save_intrinsics = [-1 * intrinsics[0] / size, intrinsics[1] / size, intrinsics[2] / size,
                                           intrinsics[3] / size]
                        dict = {}
                        frames = []
                        for i in range(num_img):
                            frames.append({'file_path': './image/' + name[i],
                                           'world_mat': w2c_p[i].detach().cpu().numpy().tolist(),
                                           'expression': exp[i].detach().cpu().numpy().tolist(),
                                           'pose': full_pose[i].detach().cpu().numpy().tolist(),
                                           'bbox': torch.stack(
                                               [torch.min(landmark[i, :, 0]), torch.min(landmark[i, :, 1]),
                                                torch.max(landmark[i, :, 0]), torch.max(landmark[i, :, 1])],
                                               dim=0).detach().cpu().numpy().tolist(),
                                           'flame_keypoints': trans_landmarks2d[i, :,
                                                              :2].detach().cpu().numpy().tolist()
                                           })

                        dict['frames'] = frames
                        dict['intrinsics'] = save_intrinsics
                        dict['shape_params'] = shape[0].cpu().numpy().tolist()
                        with open(os.path.join(savefolder, save_name + '.json'), 'w') as fp:
                            json.dump(dict, fp)


    def run(self, deca_code_file, face_kpts_file, iris_file, savefolder, image_path, json_path, intrinsics, size,
            save_name):
        deca_code = json.load(open(deca_code_file, 'r'))
        face_kpts = json.load(open(face_kpts_file, 'r'))
        try:
            iris_kpts = json.load(open(iris_file, 'r'))
        except:
            iris_kpts = None
            print("Not using Iris keypoint")
        visualize_images = []
        shape = []
        exps = []
        landmarks = []
        poses = []
        name = []
        num_img = len(deca_code)
        # ffmpeg extracted frames, index starts with 1
        video_name = os.path.basename(os.path.dirname(image_path))
        for k in range(1, num_img + 1):
            deca_key = video_name + '_frame' + '{:04d}'.format(k - 1)
            shape.append(torch.tensor(deca_code[deca_key]['shape']).float().cuda())
            exps.append(torch.tensor(deca_code[deca_key]['exp']).float().cuda())
            poses.append(torch.tensor(deca_code[deca_key]['pose']).float().cuda())
            name.append(f"frame-{k-1}")
            landmark = np.array(face_kpts['frame-{}.png'.format(str(k-1))]).astype(np.float32)
            if iris_kpts is not None:
                iris = np.array(iris_kpts['frame-{}.png'.format(str(k-1))]).astype(np.float32).reshape(2, 2)
                landmark = np.concatenate([landmark, iris[[1,0], :]], 0)
            landmark = landmark / size * 2 - 1
            landmarks.append(torch.tensor(landmark).float().cuda())
            if k % 50 == 1:
                image = cv2.imread(image_path + '/frame-{}.png'.format(str(k-1))).astype(np.float32) / 255.
                image = image[:, :, [2, 1, 0]].transpose(2, 0, 1)
                visualize_images.append(torch.from_numpy(image[None, :, :, :]).cuda())

        shape = torch.cat(shape, dim=0)
        if json_path is None:
            shape = torch.mean(shape, dim=0).unsqueeze(0)
        else:
            shape = torch.tensor(json.load(open(json_path, 'r'))['shape_params']).float().cuda().unsqueeze(0)
        exps = torch.cat(exps, dim=0)
        landmarks = torch.stack(landmarks, dim=0)
        poses = torch.cat(poses, dim=0)
        visualize_images = torch.cat(visualize_images, dim=0)
        # optimize
        self.optimize(shape, exps, landmarks, poses, name, visualize_images, savefolder, intrinsics, json_path, size,
                      save_name)


def optimize_directory(
    output_dir: str, 
    width: int, 
    height: int, 
    device=torch.device('cuda')
) -> None:
    log.INFO(f"Running video per frame optimization: {output_dir}")
    model = Optimizer()
    json_path = os.path.join(output_dir, 'DECA', "flame_deca_space.json")
    intrinsics = [1500, 1500, 256, 256]
    size = width
    model.run(
        deca_code_file=os.path.join(output_dir, 'DECA', 'flame_space_deca.json'),
        face_kpts_file=os.path.join(output_dir, 'keypoints.json'),
        iris_file=os.path.join(output_dir, 'iris.json'),
        json_path=None, 
        intrinsics=intrinsics,
        savefolder=output_dir,
        image_path=os.path.join(output_dir, 'image'),
        size=size,
        save_name='flame_params'
    )
