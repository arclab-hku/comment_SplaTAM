import argparse
import os
import shutil
import sys
import time
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

print("System Paths:")
for p in sys.path:
    print(p)

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import wandb

from datasets.gradslam_datasets import (
    load_dataset_config,
    ICLDataset,
    ReplicaDataset,
    ReplicaV2Dataset,
    AzureKinectDataset,
    ScannetDataset,
    Ai2thorDataset,
    Record3DDataset,
    RealsenseDataset,
    TUMDataset,
    ScannetPPDataset,
    NeRFCaptureDataset
)
from utils.common_utils import seed_everything, save_params_ckpt, save_params
from utils.eval_helpers import report_loss, report_progress, eval
from utils.keyframe_selection import keyframe_selection_overlap
from utils.recon_helpers import setup_camera
from utils.slam_helpers import (
    transformed_params2rendervar, transformed_params2depthplussilhouette,
    transform_to_frame, l1_loss_v1, matrix_to_quaternion
)
from utils.slam_external import calc_ssim, build_rotation, prune_gaussians, densify

from diff_gaussian_rasterization import GaussianRasterizer as Renderer


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicav2"]:
        return ReplicaV2Dataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


# 从给定的颜色图像、深度图像、相机内参和相机到世界坐标系的变换矩阵中获取点云。
# 输入参数：
# color：RGB颜色图像，形状为 (C, H, W)，表示通道数、高度和宽度。
# depth：深度图像，形状为 (1, H, W)，只使用深度信息的第一个通道。
# intrinsics：相机内参矩阵，形状为 (3, 3)。
# w2c：相机到世界坐标系的变换矩阵，形状为 (4, 4)。
# transform_pts：一个布尔值，指示是否对点进行坐标变换，默认为 True。
# mask：可选的掩码，形状为 (H * W,)，用于选择特定的点云点。
# compute_mean_sq_dist：一个布尔值，指示是否计算均方距离，默认为 False。
# mean_sq_dist_method：均方距离计算方法，目前仅支持 "projective"。
def get_pointcloud(color, depth, intrinsics, w2c, transform_pts=True, 
                   mask=None, compute_mean_sq_dist=False, mean_sq_dist_method="projective"):
    
    # 从颜色图像中提取宽度和高度，并计算相机内参的各个分量。
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of pixels
    # 计算像素坐标和深度信息：

    # 利用网格生成像素坐标 xx 和 yy。
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    # 计算深度信息 depth_z。
    depth_z = depth[0].reshape(-1)

    # Initialize point cloud
    # 初始化相机坐标系下的点云
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1) #利用像素坐标和深度信息初始化相机坐标系下的点云

    # 如果 transform_pts 为 True（默认为true且没有传入参数），则进行坐标变换，将点云从相机坐标系变换到世界坐标系。
    if transform_pts:
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        c2w = torch.inverse(w2c)
        pts = (c2w @ pts4.T).T[:, :3]
    else:
        pts = pts_cam

    # Compute mean squared distance for initializing the scale of the Gaussians
    # 根据指定的方法计算均方距离。
    if compute_mean_sq_dist:
        if mean_sq_dist_method == "projective":
            # Projective Geometry (this is fast, farther -> larger radius)
            scale_gaussian = depth_z / ((FX + FY)/2)
            mean3_sq_dist = scale_gaussian**2
        else:
            raise ValueError(f"Unknown mean_sq_dist_method {mean_sq_dist_method}")
    
    # Colorize point cloud
    # 将点云与颜色信息结合，形成彩色的点云。
    cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    point_cld = torch.cat((pts, cols), -1) #是一个张量（tensor），包含了彩色的点云数据。它的格式是一个二维张量，形状为 (N, 6)，其中 N 是点云中点的数量。每一行代表一个点，包含了点的三维坐标（x、y、z）以及颜色信息（R、G、B）

    # Select points based on mask
    # 如果提供了掩码 mask，则基于掩码选择特定的点
    if mask is not None:
        point_cld = point_cld[mask]
        if compute_mean_sq_dist:
            mean3_sq_dist = mean3_sq_dist[mask]

    if compute_mean_sq_dist:
        return point_cld, mean3_sq_dist
    else:
        return point_cld


def initialize_params(init_pt_cld, num_frames, mean3_sq_dist):
    num_pts = init_pt_cld.shape[0]
    means3D = init_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    params = {
        'means3D': means3D,
        'rgb_colors': init_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }

    # Initialize a single gaussian trajectory to model the camera poses relative to the first frame
    cam_rots = np.tile([1, 0, 0, 0], (1, 1))
    cam_rots = np.tile(cam_rots[:, :, None], (1, 1, num_frames))
    params['cam_unnorm_rots'] = cam_rots
    params['cam_trans'] = np.zeros((1, 3, num_frames))

    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    variables = {'max_2D_radius': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'means2D_gradient_accum': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'denom': torch.zeros(params['means3D'].shape[0]).cuda().float(),
                 'timestep': torch.zeros(params['means3D'].shape[0]).cuda().float()}

    return params, variables


def initialize_optimizer(params, lrs_dict, tracking):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    if tracking:
        return torch.optim.Adam(param_groups)
    else:
        return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)

# 这段代码的目的是在初始化时间步骤时设置相机和场景参数，并获取初始点云。
    # 输入参数：
    # dataset：包含RGB-D数据和相机参数的数据集。
    # num_frames：时间步骤数。
    # scene_radius_depth_ratio：用于初始化场景半径的深度比率。
    # mean_sq_dist_method：均方距离计算方法。
    # densify_dataset：可选的用于密集化的数据集。
def initialize_first_timestep(dataset, num_frames, scene_radius_depth_ratio, mean_sq_dist_method, densify_dataset=None):

    # Get RGB-D Data & Camera Parameters
    # 从数据集中获取第一帧RGB-D数据（颜色、深度）、相机内参和相机姿态。
    color, depth, intrinsics, pose = dataset[0] 

    # Process RGB-D Data
    # 将颜色数据调整为PyTorch的形状和范围。
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    # 调整深度数据的形状。
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    # 提取相机内参并计算相机到世界坐标系的逆矩阵。
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    # 使用提取的相机参数设置相机。
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())

    if densify_dataset is not None: #如果提供了密集化数据集，获取第一帧RGB-D数据和相机内参，并进行相应的处理。
        # Get Densification RGB-D Data & Camera Parameters
        color, depth, densify_intrinsics, _ = densify_dataset[0]
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        densify_intrinsics = densify_intrinsics[:3, :3]
        densify_cam = setup_camera(color.shape[2], color.shape[1], densify_intrinsics.cpu().numpy(), w2c.detach().cpu().numpy())
    else:
        densify_intrinsics = intrinsics

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)
    # 根据颜色、深度、相机内参、相机到世界坐标系的逆矩阵等信息，使用 get_pointcloud 函数获取初始点云。
    # 通过 mask 过滤掉无效深度值。
    init_pt_cld, mean3_sq_dist = get_pointcloud(color, depth, densify_intrinsics, w2c, 
                                                mask=mask, compute_mean_sq_dist=True, 
                                                mean_sq_dist_method=mean_sq_dist_method)

    # Initialize Parameters
    # 利用初始点云和其他信息，使用 initialize_params 函数初始化模型参数和变量。
    params, variables = initialize_params(init_pt_cld, num_frames, mean3_sq_dist)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    # 估计场景半径，用于后续的高斯光斑密集化。
    variables['scene_radius'] = torch.max(depth)/scene_radius_depth_ratio

    if densify_dataset is not None:
        return params, variables, intrinsics, w2c, cam, densify_intrinsics, densify_cam
    else:
        return params, variables, intrinsics, w2c, cam

# 主要用于在跟踪（tracking）或映射（mapping）过程中评估当前帧的损失。
# 函数接受一系列输入参数，包括相机参数 params、当前数据 curr_data、一些中间变量 variables、迭代的时间步 iter_time_idx、损失权重 loss_weights、是否使用深度图用于损失计算 use_sil_for_loss、阈值 sil_thres 等等。
def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1,ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir=None, visualize_tracking_loss=False, tracking_iteration=None):
    # Initialize Loss Dictionary
    losses = {}

    # 根据输入的参数和当前迭代的时间步，调用 transform_to_frame 函数将世界坐标系中的点转换为相机坐标系中的高斯分布中心点，并考虑是否需要计算梯度。不同的模式（tracking、mapping）会影响对哪些参数计算梯度。
    # transform_to_frame执行了从世界坐标系到相机坐标系的高斯分布中心点的转换操作，同时考虑了是否需要计算梯度。
    # tracking的时候camera pose需要计算梯度,mapping的时候BA优化,则高斯和pose的梯度都要优化,而单纯的mapping则只需要优化高斯的梯度
    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_pts = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_pts = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_pts = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)
        

    # Initialize Render Variables (初始化一些渲染的变量)
    #将输入的参数 params 转换成一个包含渲染相关变量的字典 rendervar与depth_sil_rendervar
    rendervar = transformed_params2rendervar(params, transformed_pts)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts)

    # RGB Rendering
    rendervar['means2D'].retain_grad() #在进行RGB渲染时，保留其梯度信息(means2D)。
    # 使用渲染器 Renderer 对当前帧进行RGB渲染，得到RGB图像 im、半径信息 radius。
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar) #这里的Renderer是import from diff_gaussian_rasterization,也就是高斯光栅化的渲染
    # 将 means2D 的梯度累积到 variables 中，这是为了在颜色渲染过程中进行密集化（densification）。
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    # 使用渲染器 Renderer 对当前帧进行深度和轮廓渲染，得到深度轮廓图 depth_sil。
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    # 从深度轮廓图中提取深度信息 depth，轮廓信息 silhouette，以及深度的平方 depth_sq。
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    # 计算深度的不确定性，即深度平方的差值，然后将其分离出来并进行 detach 操作(不计算梯度)。
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    # 建一个 nan_mask，用于标记深度和不确定性的有效值，避免处理异常值。
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss: #如果开启了 ignore_outlier_depth_loss，则基于深度误差生成一个新的掩码 mask，并且该掩码会剔除深度值异常的区域。
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else: #如果没有开启 ignore_outlier_depth_loss，则直接使用深度大于零的区域作为 mask。
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    # 如果在跟踪模式下且开启了使用轮廓图进行损失计算 (use_sil_for_loss)，则将 mask 与轮廓图的存在性掩码 presence_sil_mask 相与。
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # 至此,生成RGB图像、深度图、并根据需要进行掩码处理，以便后续在计算损失时使用。

    # Depth loss(计算深度的loss)
    if use_l1: #如果使用L1损失 (use_l1)，则将 mask 进行 detach 操作，即不计算其梯度。
        mask = mask.detach()
        if tracking: #如果在跟踪模式下 (tracking)，计算深度损失 (losses['depth']) 为当前深度图与渲染深度图之间差值的绝对值之和（只考虑掩码内的区域）。
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else: #如果不在跟踪模式下，计算深度损失为当前深度图与渲染深度图之间差值的绝对值的平均值（只考虑掩码内的区域）。上下一模一样
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()
    
    # RGB Loss(计算RGB的loss)
    # 如果在跟踪模式下 (tracking) 并且使用轮廓图进行损失计算 (use_sil_for_loss) 或者忽略异常深度值 (ignore_outlier_depth_loss)，计算RGB损失 (losses['im']) 为当前图像与渲染图像之间差值的绝对值之和（只考虑掩码内的区域）。
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking: #如果在跟踪模式下，但没有使用轮廓图进行损失计算，计算RGB损失为当前图像与渲染图像之间差值的绝对值之和。
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else: #如果不在跟踪模式下，计算RGB损失为L1损失和结构相似性损失的加权和，其中 l1_loss_v1 是L1损失的计算函数，calc_ssim 是结构相似性损失的计算函数。
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))

    # Visualize the Diff Images
    if tracking and visualize_tracking_loss:
        fig, ax = plt.subplots(2, 4, figsize=(12, 6))
        weighted_render_im = im * color_mask
        weighted_im = curr_data['im'] * color_mask
        weighted_render_depth = depth * mask
        weighted_depth = curr_data['depth'] * mask
        diff_rgb = torch.abs(weighted_render_im - weighted_im).mean(dim=0).detach().cpu()
        diff_depth = torch.abs(weighted_render_depth - weighted_depth).mean(dim=0).detach().cpu()
        viz_img = torch.clip(weighted_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[0, 0].imshow(viz_img)
        ax[0, 0].set_title("Weighted GT RGB")
        viz_render_img = torch.clip(weighted_render_im.permute(1, 2, 0).detach().cpu(), 0, 1)
        ax[1, 0].imshow(viz_render_img)
        ax[1, 0].set_title("Weighted Rendered RGB")
        ax[0, 1].imshow(weighted_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[0, 1].set_title("Weighted GT Depth")
        ax[1, 1].imshow(weighted_render_depth[0].detach().cpu(), cmap="jet", vmin=0, vmax=6)
        ax[1, 1].set_title("Weighted Rendered Depth")
        ax[0, 2].imshow(diff_rgb, cmap="jet", vmin=0, vmax=0.8)
        ax[0, 2].set_title(f"Diff RGB, Loss: {torch.round(losses['im'])}")
        ax[1, 2].imshow(diff_depth, cmap="jet", vmin=0, vmax=0.8)
        ax[1, 2].set_title(f"Diff Depth, Loss: {torch.round(losses['depth'])}")
        ax[0, 3].imshow(presence_sil_mask.detach().cpu(), cmap="gray")
        ax[0, 3].set_title("Silhouette Mask")
        ax[1, 3].imshow(mask[0].detach().cpu(), cmap="gray")
        ax[1, 3].set_title("Loss Mask")
        # Turn off axis
        for i in range(2):
            for j in range(4):
                ax[i, j].axis('off')
        # Set Title
        fig.suptitle(f"Tracking Iteration: {tracking_iteration}", fontsize=16)
        # Figure Tight Layout
        fig.tight_layout()
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"tmp.png"), bbox_inches='tight')
        plt.close()
        plot_img = cv2.imread(os.path.join(plot_dir, f"tmp.png"))
        cv2.imshow('Diff Images', plot_img)
        cv2.waitKey(1)
        ## Save Tracking Loss Viz
        # save_plot_dir = os.path.join(plot_dir, f"tracking_%04d" % iter_time_idx)
        # os.makedirs(save_plot_dir, exist_ok=True)
        # plt.savefig(os.path.join(save_plot_dir, f"%04d.png" % tracking_iteration), bbox_inches='tight')
        # plt.close()

    # 下面代码进行了损失的加权和最终的损失值计算
    # 对每个损失项按照其权重进行加权，得到 weighted_losses 字典，其中 k 是损失项的名称，v 是对应的损失值，loss_weights 是各个损失项的权重。
    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()} 
    # 最终损失值 loss 是加权损失项的和。
    loss = sum(weighted_losses.values())

    seen = radius > 0 #创建一个布尔掩码 seen，其中对应的位置为 True 表示在当前迭代中观察到了某个点。
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen]) #更新 variables['max_2D_radius'] 中已观察到的点的最大半径。
    variables['seen'] = seen #将 seen 存储在 variables 字典中。
    weighted_losses['loss'] = loss #最终，将总损失值存储在 weighted_losses 字典中的 'loss' 键下。

    return loss, variables, weighted_losses

# 初始化新的高斯分布参数
# mean3_sq_dist：新点云的均方距离，用于初始化高斯分布的尺度参数。
def initialize_new_params(new_pt_cld, mean3_sq_dist):
    num_pts = new_pt_cld.shape[0] #点云
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3] #点云对应的位置信息xyz
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]  高斯球的旋转，四元数的未归一化旋转表示，暗示高斯分布没有旋转。
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda") #透明度，初始化为0
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }
    # 构建参数字典 params：params 包含了高斯分布的均值 means3D、颜色 rgb_colors、未归一化旋转 unnorm_rotations、不透明度的对数 logit_opacities 以及尺度的对数 log_scales。
    for k, v in params.items(): #遍历 params 字典，将其值转换为 torch.Tensor 或 torch.nn.Parameter 类型。
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))

    return params #返回初始化后的高斯分布参数字典。


# 现了在建图过程中根据当前帧的数据进行高斯分布的密集化，
def add_new_gaussians(params, variables, curr_data, sil_thres, time_idx, mean_sq_dist_method):
    # Silhouette Rendering
    transformed_pts = transform_to_frame(params, time_idx, gaussians_grad=False, camera_grad=False)#将高斯模型转换到frame坐标系下
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_pts) #获取深度的渲染变量
    # 通过渲染器 Renderer 得到深度图和轮廓图，其中 depth_sil 包含了深度信息和轮廓信息。
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    silhouette = depth_sil[1, :, :]
    # non_presence_sil_mask代表当前帧中未出现的区域？
    non_presence_sil_mask = (silhouette < sil_thres) #通过设置阈值 sil_thres（输入参数为0.5），创建一个轮廓图的非存在掩码 

    # Check for new foreground objects by using GT depth
    # 利用当前深度图和渲染后的深度图，通过 depth_error 计算深度误差，并生成深度非存在掩码 non_presence_depth_mask。
    gt_depth = curr_data['depth'][0, :, :]
    render_depth = depth_sil[0, :, :]
    depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
    non_presence_depth_mask = (render_depth > gt_depth) * (depth_error > 50*depth_error.median())

    # Determine non-presence mask
    # 将轮廓图非存在掩码和深度非存在掩码合并生成整体的非存在掩码 non_presence_mask。
    non_presence_mask = non_presence_sil_mask | non_presence_depth_mask
    # Flatten mask
    non_presence_mask = non_presence_mask.reshape(-1)

    # Get the new frame Gaussians based on the Silhouette
    # 检测到非存在掩码中有未出现的点时，根据当前帧的数据生成新的高斯分布参数，并将这些参数添加到原有的高斯分布参数中
    if torch.sum(non_presence_mask) > 0:
        # Get the new pointcloud in the world frame
        # 获取当前相机的旋转和平移信息:
        curr_cam_rot = torch.nn.functional.normalize(params['cam_unnorm_rots'][..., time_idx].detach()) #获取当前帧的相机未归一化旋转信息。
        curr_cam_tran = params['cam_trans'][..., time_idx].detach() #对旋转信息进行归一化。
        # 构建当前帧相机到世界坐标系的变换矩阵:
        curr_w2c = torch.eye(4).cuda().float() #创建一个单位矩阵
        # 利用归一化后的旋转信息和当前帧的相机平移信息，更新变换矩阵的旋转和平移部分。
        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
        curr_w2c[:3, 3] = curr_cam_tran
        # 生成有效深度掩码:
        valid_depth_mask = (curr_data['depth'][0, :, :] > 0) #生成当前帧的有效深度掩码 valid_depth_mask。
        # 更新非存在掩码:
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1) #将 non_presence_mask 和 valid_depth_mask 进行逐元素与操作，得到更新后的非存在掩码。
        # 获取新的点云和平均平方距离:
        #利用 get_pointcloud 函数，传入当前帧的图像、深度图、内参、变换矩阵和非存在掩码，生成新的点云 new_pt_cld。同时计算这些新点云到已存在高斯分布的平均平方距离 mean3_sq_dist。
        new_pt_cld, mean3_sq_dist = get_pointcloud(curr_data['im'], curr_data['depth'], curr_data['intrinsics'], 
                                    curr_w2c, mask=non_presence_mask, compute_mean_sq_dist=True,
                                    mean_sq_dist_method=mean_sq_dist_method) #参数文件中定义mean_sq_dist_method为projective
        # 初始化新的高斯分布参数:
        # 利用新的点云和平均平方距离，调用 initialize_new_params 函数生成新的高斯分布参数 new_params。
        new_params = initialize_new_params(new_pt_cld, mean3_sq_dist)
        # 将新的高斯分布参数添加到原有参数中:
        for k, v in new_params.items(): #对于每个键值对 (k, v)，其中 k 是高斯分布参数的键，v 是对应的值，在 params 中将其与新参数 v 拼接，并转换为可梯度的 torch.nn.Parameter 对象。
            params[k] = torch.nn.Parameter(torch.cat((params[k], v), dim=0).requires_grad_(True))
        # (更新相关的统计信息)初始化一些统计信息，如梯度累积、分母、最大2D半径等。
        num_pts = params['means3D'].shape[0]
        variables['means2D_gradient_accum'] = torch.zeros(num_pts, device="cuda").float()
        variables['denom'] = torch.zeros(num_pts, device="cuda").float()
        variables['max_2D_radius'] = torch.zeros(num_pts, device="cuda").float()
        # (更新时间步信息)将新的点云对应的时间步信息 new_timestep（都是当前帧的时间步）拼接到原有的时间步信息中。
        new_timestep = time_idx*torch.ones(new_pt_cld.shape[0],device="cuda").float()
        variables['timestep'] = torch.cat((variables['timestep'],new_timestep),dim=0)

    # 将更新后的模型参数 params 和相关的统计信息 variables 返回。
    return params, variables

# 用于初始化相机姿态的函数 
# 根据当前时间初始化相机的旋转和平移参数。（根据前两帧对当前帧的初始pose进行预测）
def initialize_camera_pose(params, curr_time_idx, forward_prop): #参数文件中，forward_prop是true
    with torch.no_grad(): #此用来确保在这个上下文中没有梯度计算。
        if curr_time_idx > 1 and forward_prop: #检查当前时间步 curr_time_idx 是否大于 1，以及是否使用了向前传播
            # Initialize the camera pose for the current frame based on a constant velocity model
            # 使用常速度模型初始化相机姿态。
            # Rotation（通过前两帧的旋转计算出当前帧的新旋转。）
            prev_rot1 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-1].detach())
            prev_rot2 = F.normalize(params['cam_unnorm_rots'][..., curr_time_idx-2].detach())
            new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2))
            params['cam_unnorm_rots'][..., curr_time_idx] = new_rot.detach()
           
            # Translation（通过前两帧的平移计算出当前帧的新平移。）
            prev_tran1 = params['cam_trans'][..., curr_time_idx-1].detach()
            prev_tran2 = params['cam_trans'][..., curr_time_idx-2].detach()
            new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
            params['cam_trans'][..., curr_time_idx] = new_tran.detach()
        else: #如果条件不满足，则直接复制前一帧的相机姿态到当前帧。这是为了处理初始化的特殊情况，确保在开始时有初始姿态。
            # Initialize the camera pose for the current frame
            params['cam_unnorm_rots'][..., curr_time_idx] = params['cam_unnorm_rots'][..., curr_time_idx-1].detach()
            params['cam_trans'][..., curr_time_idx] = params['cam_trans'][..., curr_time_idx-1].detach()
    
    return params


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store


def rgbd_slam(config: dict):
    # Print Config
    print("Loaded Config:")
    if "use_depth_loss_thres" not in config['tracking']:
        config['tracking']['use_depth_loss_thres'] = False
        config['tracking']['depth_loss_thres'] = 100000
    if "visualize_tracking_loss" not in config['tracking']:
        config['tracking']['visualize_tracking_loss'] = False
    print(f"{config}")

    # Create Output Directories
    output_dir = os.path.join(config["workdir"], config["run_name"])
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Init WandB
    if config['use_wandb']:
        wandb_time_step = 0
        wandb_tracking_step = 0
        wandb_mapping_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)

    # Get Device
    device = torch.device(config["primary_device"])

    # Load Dataset
    print("Loading Dataset ...") #输入数据
    dataset_config = config["data"] #读入config中的数据路径
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"]) #内参
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "densification_image_height" not in dataset_config:
        dataset_config["densification_image_height"] = dataset_config["desired_image_height"]
        dataset_config["densification_image_width"] = dataset_config["desired_image_width"]
        seperate_densification_res = False
    else:
        if dataset_config["densification_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["densification_image_width"] != dataset_config["desired_image_width"]:
            seperate_densification_res = True
        else:
            seperate_densification_res = False
    if "tracking_image_height" not in dataset_config:
        dataset_config["tracking_image_height"] = dataset_config["desired_image_height"]
        dataset_config["tracking_image_width"] = dataset_config["desired_image_width"]
        seperate_tracking_res = False
    else:
        if dataset_config["tracking_image_height"] != dataset_config["desired_image_height"] or \
            dataset_config["tracking_image_width"] != dataset_config["desired_image_width"]:
            seperate_tracking_res = True
        else:
            seperate_tracking_res = False
    # Poses are relative to the first frame
    dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
    ) #定义一个类，这个类为数据集的内容？（是一个数据集对象）
    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(dataset)

    # Init seperate dataloader for densification if required
    if seperate_densification_res:
        densify_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["densification_image_height"],
            desired_width=dataset_config["densification_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        # Initialize Parameters, Canonical & Densification Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam, \
            densify_intrinsics, densify_cam = initialize_first_timestep(dataset, num_frames,
                                                                        config['scene_radius_depth_ratio'],
                                                                        config['mean_sq_dist_method'],
                                                                        densify_dataset=densify_dataset)                                                                                                                  
    else:
        # Initialize Parameters & Canoncial Camera parameters
        params, variables, intrinsics, first_frame_w2c, cam = initialize_first_timestep(dataset, num_frames, 
                                                                                        config['scene_radius_depth_ratio'],
                                                                                        config['mean_sq_dist_method'])
    
    # Init seperate dataloader for tracking if required
    if seperate_tracking_res:
        tracking_dataset = get_dataset(
            config_dict=gradslam_data_cfg,
            basedir=dataset_config["basedir"],
            sequence=os.path.basename(dataset_config["sequence"]),
            start=dataset_config["start"],
            end=dataset_config["end"],
            stride=dataset_config["stride"],
            desired_height=dataset_config["tracking_image_height"],
            desired_width=dataset_config["tracking_image_width"],
            device=device,
            relative_pose=True,
            ignore_bad=dataset_config["ignore_bad"],
            use_train_split=dataset_config["use_train_split"],
        )
        tracking_color, _, tracking_intrinsics, _ = tracking_dataset[0]
        tracking_color = tracking_color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        tracking_intrinsics = tracking_intrinsics[:3, :3]
        tracking_cam = setup_camera(tracking_color.shape[2], tracking_color.shape[1], 
                                    tracking_intrinsics.cpu().numpy(), first_frame_w2c.detach().cpu().numpy())
    
    # Initialize list to keep track of Keyframes
    keyframe_list = []
    keyframe_time_indices = []
    
    # Init Variables to keep track of ground truth poses and runtimes
    gt_w2c_all_frames = []
    tracking_iter_time_sum = 0
    tracking_iter_time_count = 0
    mapping_iter_time_sum = 0
    mapping_iter_time_count = 0
    tracking_frame_time_sum = 0
    tracking_frame_time_count = 0
    mapping_frame_time_sum = 0
    mapping_frame_time_count = 0

    # Load Checkpoint
    if config['load_checkpoint']:
        checkpoint_time_idx = config['checkpoint_time_idx']
        print(f"Loading Checkpoint for Frame {checkpoint_time_idx}")
        ckpt_path = os.path.join(config['workdir'], config['run_name'], f"params{checkpoint_time_idx}.npz")
        params = dict(np.load(ckpt_path, allow_pickle=True))
        params = {k: torch.tensor(params[k]).cuda().float().requires_grad_(True) for k in params.keys()}
        variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
        # Load the keyframe time idx list
        keyframe_time_indices = np.load(os.path.join(config['workdir'], config['run_name'], f"keyframe_time_indices{checkpoint_time_idx}.npy"))
        keyframe_time_indices = keyframe_time_indices.tolist()
        # Update the ground truth poses list
        for time_idx in range(checkpoint_time_idx):
            # Load RGBD frames incrementally instead of all frames
            color, depth, _, gt_pose = dataset[time_idx]
            # Process poses
            gt_w2c = torch.linalg.inv(gt_pose)
            gt_w2c_all_frames.append(gt_w2c)
            # Initialize Keyframe List
            if time_idx in keyframe_time_indices:
                # Get the estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                color = color.permute(2, 0, 1) / 255
                depth = depth.permute(2, 0, 1)
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
    else:
        checkpoint_time_idx = 0
    
    # Iterate over Scan （迭代扫描，迭代处理RGB-D帧，进行跟踪（Tracking）和建图（Mapping））
    for time_idx in tqdm(range(checkpoint_time_idx, num_frames)): #通过循环迭代处理 RGB-D 帧，循环的起始索引是 checkpoint_time_idx（也就是是否从某帧开始，一般都是0开始），终止索引是 num_frames。
        # Load RGBD frames incrementally instead of all frames
        color, depth, _, gt_pose = dataset[time_idx] #从数据集 dataset 中加载 RGB-D 帧的颜色、深度、姿态等信息。
        # Process poses
        gt_w2c = torch.linalg.inv(gt_pose)#对姿态信息进行处理，计算pose的逆，也就是世界到相机的变换矩阵 gt_w2c。
        
        # Process RGB-D Data
        # 使用了PyTorch中的permute函数，将颜色数据的维度进行重新排列。
        # 在这里，color是一个张量（tensor），通过permute(2, 0, 1)操作，将原始颜色数据的维度顺序从 (height, width, channels) 调整为 (channels, height, width)。
        color = color.permute(2, 0, 1) / 255 #将颜色归一化，归一化到0~1范围
        depth = depth.permute(2, 0, 1)

        # 将当前帧的pose gt_w2c 添加到列表 gt_w2c_all_frames 中。
        gt_w2c_all_frames.append(gt_w2c)
        curr_gt_w2c = gt_w2c_all_frames
        # Optimize only current time step for tracking
        iter_time_idx = time_idx

        # Initialize Mapping Data for selected frame
        # 初始化当前帧的数据 curr_data 包括相机参数、颜色数据、深度数据等。
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': iter_time_idx, 'intrinsics': intrinsics, 
                     'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        
        # Initialize Data for Tracking（根据配置，初始化跟踪数据 tracking_curr_data。）
        if seperate_tracking_res:
            tracking_color, tracking_depth, _, _ = tracking_dataset[time_idx]
            tracking_color = tracking_color.permute(2, 0, 1) / 255
            tracking_depth = tracking_depth.permute(2, 0, 1)
            tracking_curr_data = {'cam': tracking_cam, 'im': tracking_color, 'depth': tracking_depth, 'id': iter_time_idx,
                                  'intrinsics': tracking_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
        else:
            tracking_curr_data = curr_data #初始化跟踪数据

        # Optimization Iterations（设置建图迭代次数）
        num_iters_mapping = config['mapping']['num_iters']
        
        # Initialize the camera pose for the current frame
        if time_idx > 0: #如果当前帧索引大于 0，则初始化相机姿态参数。
            params = initialize_camera_pose(params, time_idx, forward_prop=config['tracking']['forward_prop']) #参数文件中，forward_prop是true

 #################### Tracking （进入跟踪阶段，根据当前帧进行优化迭代，包括重置优化器、学习率、迭代过程中的损失计算和优化器更新等。）
        tracking_start_time = time.time() #记录跟踪阶段的开始时间，用于计时
        # 判断是否采用真值的pose
        if time_idx > 0 and not config['tracking']['use_gt_poses']: #如果当前时间步 time_idx 大于 0 且不使用真实姿态
            # Reset Optimizer & Learning Rates for tracking(重置优化器和学习率，这通常是为了跟踪阶段使用不同的优化设置。)
            optimizer = initialize_optimizer(params, config['tracking']['lrs'], tracking=True)

            # Keep Track of Best Candidate Rotation & Translation(初始化变量 candidate_cam_unnorm_rot 和 candidate_cam_tran 以跟踪最佳的相机旋转和平移。)
            candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
            candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()

            # 初始化变量 current_min_loss 用于跟踪当前迭代中的最小损失。
            current_min_loss = float(1e20)

            # Tracking Optimization(开始进行tracking的优化)
            iter = 0 #设置迭代次数初始值为 0。
            do_continue_slam = False #是否进行运行,用于判断是否满足终止的条件
            num_iters_tracking = config['tracking']['num_iters'] #定义的跟踪迭代次数,参数文件中为200
            progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}") #使用 tqdm 创建一个进度条，显示当前跟踪迭代的进度
            while True:
                iter_start_time = time.time() #记录迭代开始的时间，用于计算迭代的运行时间。

                # Loss for current frame
                # 计算当前帧的损失（loss）
                loss, variables, losses = get_loss(params, tracking_curr_data, variables, iter_time_idx, config['tracking']['loss_weights'],
                                                   config['tracking']['use_sil_for_loss'], config['tracking']['sil_thres'],
                                                   config['tracking']['use_l1'], config['tracking']['ignore_outlier_depth_loss'], tracking=True, 
                                                   plot_dir=eval_dir, visualize_tracking_loss=config['tracking']['visualize_tracking_loss'],
                                                   tracking_iteration=iter)
                
                # 检查是否使用 Weights and Biases（W&B）进行记录和可视化。
                if config['use_wandb']:
                    # Report Loss
                    wandb_tracking_step = report_loss(losses, wandb_run, wandb_tracking_step, tracking=True)
                
                # Backprop(将loss进行反向传播。计算梯度)
                loss.backward()

                # Optimizer Update(更新优化器。根据计算的梯度更新模型参数。)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True) #清零梯度，以便下一次迭代重新计算梯度。

                with torch.no_grad(): #进入没有梯度的上下文，下面的操作不会影响梯度计算。
                    # Save the best candidate rotation & translation(记录最小损失对应的相机旋转和平移。)
                    if loss < current_min_loss: #如果当前损失小于 current_min_loss，更新最小损失对应的相机旋转和平移。
                        current_min_loss = loss
                        candidate_cam_unnorm_rot = params['cam_unnorm_rots'][..., time_idx].detach().clone()
                        candidate_cam_tran = params['cam_trans'][..., time_idx].detach().clone()
                    # Report Progress
                    if config['report_iter_progress']: #如果配置中启用了报告迭代进度 (config['report_iter_progress'])，执行报告进度的操作。
                        if config['use_wandb']:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                            wandb_run=wandb_run, wandb_step=wandb_tracking_step, wandb_save_qual=config['wandb']['save_qual'])
                        else:
                            report_progress(params, tracking_curr_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                    else:
                        progress_bar.update(1)
                
                # Update the runtime numbers （更新迭代次数和计算迭代的运行时间。）
                iter_end_time = time.time()
                tracking_iter_time_sum += iter_end_time - iter_start_time
                tracking_iter_time_count += 1
                
                # Check if we should stop tracking（检查是否最大迭代次数，满足终止计算）
                iter += 1
                if iter == num_iters_tracking: #(如果配置中定义的条件满足，则终止跟踪迭代。)
                    if losses['depth'] < config['tracking']['depth_loss_thres'] and config['tracking']['use_depth_loss_thres']:#如果启用了深度损失门限 (config['tracking']['use_depth_loss_thres']) 且深度损失小于门限，则终止迭代。
                        break
                    elif config['tracking']['use_depth_loss_thres'] and not do_continue_slam: #如果启用了深度损失门限且 do_continue_slam为false，则将 do_continue_slam 设置为 True，并增加迭代次数。
                        do_continue_slam = True
                        progress_bar = tqdm(range(num_iters_tracking), desc=f"Tracking Time Step: {time_idx}")
                        num_iters_tracking = 2*num_iters_tracking
                        if config['use_wandb']:
                            wandb_run.log({"Tracking/Extra Tracking Iters Frames": time_idx,
                                        "Tracking/step": wandb_time_step})
                    else:
                        break

            progress_bar.close()
            # Copy over the best candidate rotation & translation
            with torch.no_grad():
                params['cam_unnorm_rots'][..., time_idx] = candidate_cam_unnorm_rot
                params['cam_trans'][..., time_idx] = candidate_cam_tran
        elif time_idx > 0 and config['tracking']['use_gt_poses']: #采用真值的pose来做tracking
            with torch.no_grad(): #进入没有梯度的上下文，下面的操作不会影响梯度计算。（用真值的pose也确实不应该进行梯度的计算）
                # Get the ground truth pose relative to frame 0
                rel_w2c = curr_gt_w2c[-1] #获取当前时间帧的真值姿态相对于第 0 帧的相机到世界坐标系的变换矩阵。
                rel_w2c_rot = rel_w2c[:3, :3].unsqueeze(0).detach() #提取相机旋转矩阵，并在第 0 维度上增加一个维度，转换成形状为 (1, 3, 3) 的张量。
                rel_w2c_rot_quat = matrix_to_quaternion(rel_w2c_rot) #将相机旋转矩阵转换为四元数。
                rel_w2c_tran = rel_w2c[:3, 3].detach() #提取相机平移矩阵。
                # Update the camera parameters
                params['cam_unnorm_rots'][..., time_idx] = rel_w2c_rot_quat #将真值姿态的四元数赋值给相机旋转参数。
                params['cam_trans'][..., time_idx] = rel_w2c_tran #将真值姿态的平移矩阵赋值给相机平移参数（参数用作全局变量了）。
        # Update the runtime numbers
        tracking_end_time = time.time()
        tracking_frame_time_sum += tracking_end_time - tracking_start_time
        tracking_frame_time_count += 1
 #################### ####################

        # 如果当前帧索引是第一帧或者符合全局报告进度的条件，则报告跟踪进度。
        if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0: #确定是否需要报告全局进度。条件满足的情况包括当前帧索引是第一帧，或者当前帧索引符合全局报告进度的条件
            try:
                # Report Final Tracking Progress
                progress_bar = tqdm(range(1), desc=f"Tracking Result Time Step: {time_idx}") #创建一个进度条 progress_bar，用于显示当前进度。
                with torch.no_grad(): #调用 report_progress 函数报告跟踪进度。这里的 with torch.no_grad() 确保在这个过程中不会记录梯度信息，因为报告进度通常不需要进行梯度计算。
                    if config['use_wandb']: #如果使用了 WandB（Weights & Biases）工具，会将相关信息记录到 WandB 的运行中，以便进行可视化和追踪。
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True,
                                        wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'], global_logging=True)
                    else:
                        report_progress(params, tracking_curr_data, 1, progress_bar, iter_time_idx, sil_thres=config['tracking']['sil_thres'], tracking=True)
                progress_bar.close()
            except: #如果报告进度出现异常（except 块），会进行异常处理：
                ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                save_params_ckpt(params, ckpt_output_dir, time_idx) #将当前模型参数保存到检查点文件，以便后续恢复。
                print('Failed to evaluate trajectory.') #输出一条提示信息，指示评估轨迹失败。

 #################### ####################
        # Densification （致密化） & KeyFrame-based Mapping
        # 进入建图阶段，包括密集化和基于关键帧的建图。
        if time_idx == 0 or (time_idx+1) % config['map_every'] == 0: #如果当前帧索引是第一帧或者满足 config['map_every'] （每多少帧进行mapping一次）条件时
            
            # Densification（首先进行密集化）
            if config['mapping']['add_new_gaussians'] and time_idx > 0: #如果开启了 config['mapping']['add_new_gaussians']，并且当前帧索引大于0，则根据当前帧的数据密集化场景中的新高斯分布。
                # Setup Data for Densification
                if seperate_densification_res:
                    # Load RGBD frames incrementally instead of all frames
                    densify_color, densify_depth, _, _ = densify_dataset[time_idx] #从 densify_dataset 中加载RGBD帧的数据
                    # 对RGB和深度数据进行处理，将RGB数据的维度调整为(3, H, W)，将RGB数据的范围缩放到[0, 1]。
                    densify_color = densify_color.permute(2, 0, 1) / 255
                    densify_depth = densify_depth.permute(2, 0, 1)
                    # 构建 densify_curr_data 字典，包含了用于密集化的数据，如相机矩阵、RGB图像、深度图、帧索引等信息。
                    densify_curr_data = {'cam': densify_cam, 'im': densify_color, 'depth': densify_depth, 'id': time_idx, 
                                 'intrinsics': densify_intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c}
                else:
                    densify_curr_data = curr_data #使用当前帧的数据。

                # Add new Gaussians to the scene based on the Silhouette（轮廓，剪影）
                # 高斯分布密集化：
                # 调用 add_new_gaussians 函数，该函数接受当前模型参数 params、变量 variables、密集化数据 densify_curr_data，以及一些配置参数，如阈值、时间索引等。
                # 在 add_new_gaussians 函数中，根据输入的深度图，通过阈值 config['mapping']['sil_thres'] 生成一个Silhouette掩码，然后在场景中添加新的高斯分布。这些高斯分布代表了场景中的新结构。
                params, variables = add_new_gaussians(params, variables, densify_curr_data, 
                                                      config['mapping']['sil_thres'], time_idx,
                                                      config['mean_sq_dist_method'])
                # 记录高斯分布数量：
                post_num_pts = params['means3D'].shape[0] #获取密集化后的高斯分布的数量，并将其记录为 post_num_pts。
                if config['use_wandb']: #如果使用了 WandB，则将密集化后的高斯分布数量和当前迭代步数记录到 WandB 中，以便在 WandB 仪表板中进行监控。
                    wandb_run.log({"Mapping/Number of Gaussians": post_num_pts,
                                   "Mapping/step": wandb_time_step})
            
            # 选择用于建图的关键帧
            with torch.no_grad():
                # Get the current estimated rotation & translation
                # 获取当前帧的估计旋转和平移:
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach()) #获取当前帧的相机未归一化旋转信息。(使用 F.normalize 对旋转信息进行归一化。)
                curr_cam_tran = params['cam_trans'][..., time_idx].detach() #获取当前帧的相机平移信息
                # 构建当前帧相机到世界坐标系的变换矩阵:
                curr_w2c = torch.eye(4).cuda().float() #创建一个单位矩阵
                # 利用归一化后的旋转信息和当前帧的相机平移信息，更新变换矩阵的旋转和平移部分。
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Select Keyframes for Mapping (选择关键帧)
                num_keyframes = config['mapping_window_size']-2 #20-2=18,表示用于建图的关键帧数量。
                # 调用 keyframe_selection_overlap 函数，传入当前帧的深度图、相机变换矩阵、内参、以及之前的关键帧列表（keyframe_list[:-1]）和要选择的关键帧数量。
                # 获取被选中的关键帧的索引列表 selected_keyframes。
                selected_keyframes = keyframe_selection_overlap(depth, curr_w2c, intrinsics, keyframe_list[:-1], num_keyframes)
                # 构建关键帧对应的时间索引列表 selected_time_idx，其中包括之前的关键帧和当前帧的时间索引。
                selected_time_idx = [keyframe_list[frame_idx]['id'] for frame_idx in selected_keyframes]
                # 如果已有关键帧列表 keyframe_list 不为空，将最后一个关键帧添加到被选中的关键帧列表，并更新对应的时间索引。
                if len(keyframe_list) > 0:
                    # Add last keyframe to the selected keyframes
                    selected_time_idx.append(keyframe_list[-1]['id'])
                    selected_keyframes.append(len(keyframe_list)-1)
                # Add current frame to the selected keyframes
                selected_time_idx.append(time_idx)
                selected_keyframes.append(-1)
                # Print the selected keyframes
                print(f"\nSelected Keyframes at Frame {time_idx}: {selected_time_idx}") #输出当前帧的时间索引以及被选中的关键帧的时间索引列表。

            # Reset Optimizer & Learning Rates for Full Map Optimization
            # 调用 initialize_optimizer 函数，根据配置和参数信息初始化地图优化所使用的优化器，并设置相关的学习率。
            optimizer = initialize_optimizer(params, config['mapping']['lrs'], tracking=False) 

#********************************************************************************#
            # Mapping
            mapping_start_time = time.time() #记录mapping的时间
            # 使用 tqdm 库创建一个进度条对象 progress_bar，用于在控制台中显示地图优化迭代的进度。这个进度条会在地图优化的主循环中进行迭代，总共迭代 num_iters_mapping 次
            if num_iters_mapping > 0: #num_iters_mapping = config['mapping']['num_iters']，参数文件中定义为30
                progress_bar = tqdm(range(num_iters_mapping), desc=f"Mapping Time Step: {time_idx}")
            
            #num_iters_mapping = config['mapping']['num_iters']，参数文件中定义为30
            # 地图优化的迭代，循环 num_iters_mapping 次。
            for iter in range(num_iters_mapping):
                iter_start_time = time.time()

                # Randomly select a frame until current time step amongst keyframes
                # 随机选择关键帧
                rand_idx = np.random.randint(0, len(selected_keyframes)) # 随机选择一个关键帧。
                selected_rand_keyframe_idx = selected_keyframes[rand_idx] #selected_keyframes 存储了当前帧与之前关键帧之间的选定关键帧。

                # 确定当前迭代使用的数据
                if selected_rand_keyframe_idx == -1: #如果 selected_rand_keyframe_idx 为 -1，表示选择使用当前帧数据，
                    # Use Current Frame Data
                    # 将当前帧的颜色 (iter_color)、深度 (iter_depth)、时间索引 (iter_time_idx) 分配给相应变量。
                    iter_time_idx = time_idx
                    iter_color = color
                    iter_depth = depth
                else: #如果 selected_rand_keyframe_idx 不为 -1，表示选择使用某个关键帧的数据
                    # Use Keyframe Data
                    # 将该关键帧的颜色、深度、以及关键帧的时间索引分配给相应变量。
                    iter_time_idx = keyframe_list[selected_rand_keyframe_idx]['id']
                    iter_color = keyframe_list[selected_rand_keyframe_idx]['color']
                    iter_depth = keyframe_list[selected_rand_keyframe_idx]['depth']
                
                # 构建当前帧的数据字典
                iter_gt_w2c = gt_w2c_all_frames[:iter_time_idx+1] #获取位姿
                # 将迭代过程中使用的数据整理到字典 iter_data 中，包括相机参数、颜色、深度、时间索引等信息。
                iter_data = {'cam': cam, 'im': iter_color, 'depth': iter_depth, 'id': iter_time_idx, 
                             'intrinsics': intrinsics, 'w2c': first_frame_w2c, 'iter_gt_w2c_list': iter_gt_w2c}
                # Loss for current frame
                # 通过调用 get_loss 函数计算当前帧的损失，这里使用了一些配置参数，如损失权重、是否使用轮廓损失、轮廓阈值等。
                loss, variables, losses = get_loss(params, iter_data, variables, iter_time_idx, config['mapping']['loss_weights'],
                                                config['mapping']['use_sil_for_loss'], config['mapping']['sil_thres'],
                                                config['mapping']['use_l1'], config['mapping']['ignore_outlier_depth_loss'], mapping=True)
                if config['use_wandb']:
                    # Report Loss
                    wandb_mapping_step = report_loss(losses, wandb_run, wandb_mapping_step, mapping=True)
                
                # Backprop（反向传播：调用 loss.backward() 进行反向传播，计算梯度。）
                loss.backward()
                with torch.no_grad():
                    # Prune Gaussians
                    if config['mapping']['prune_gaussians']: #如果启用了剪枝（参数文件为true）
                        params, variables = prune_gaussians(params, variables, optimizer, iter, config['mapping']['pruning_dict']) #调用 prune_gaussians 函数对高斯分布进行修剪。
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Pruning": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Gaussian-Splatting's Gradient-based Densification
                    if config['mapping']['use_gaussian_splatting_densification']: #参数文件为false
                        params, variables = densify(params, variables, optimizer, iter, config['mapping']['densify_dict']) #调用 densify 函数进行高斯分布的密集化。
                        if config['use_wandb']:
                            wandb_run.log({"Mapping/Number of Gaussians - Densification": params['means3D'].shape[0],
                                           "Mapping/step": wandb_mapping_step})
                    # Optimizer Update
                    optimizer.step() #调用优化器的 step() 方法更新模型参数。
                    optimizer.zero_grad(set_to_none=True) #调用 optimizer.zero_grad(set_to_none=True) 将梯度清零。
                    # Report Progress （记录训练过程）
                    if config['report_iter_progress']: #如果 config['report_iter_progress'] 为 True，则在控制台上报告地图优化的迭代进度。
                        if config['use_wandb']: #如果启用了 WandB (config['use_wandb'] 为 True)，则调用 report_loss 和 report_progress 函数，将损失和训练进度记录到 WandB 仪表板上。
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_mapping_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx)
                        else:
                            report_progress(params, iter_data, iter+1, progress_bar, iter_time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    else:
                        progress_bar.update(1) #更新 tqdm 进度条
                # Update the runtime numbers （计算并更新地图优化迭代的运行时间和次数。）
                iter_end_time = time.time()
                mapping_iter_time_sum += iter_end_time - iter_start_time
                mapping_iter_time_count += 1
            
            # 关闭之前创建的地图优化迭代的进度条。在使用 tqdm 库时，为了避免在进度条结束后继续显示，应该显式地关闭它。这个操作通常在迭代完成后执行，确保在地图优化的所有迭代结束后，不再在控制台中显示进度条。
            if num_iters_mapping > 0:
                progress_bar.close()
#********************************************************************************#

            # Update the runtime numbers
            mapping_end_time = time.time()
            mapping_frame_time_sum += mapping_end_time - mapping_start_time
            mapping_frame_time_count += 1

            # 如果当前帧索引是第一帧或者符合全局报告进度的条件，则报告建图进度。
            if time_idx == 0 or (time_idx+1) % config['report_global_progress_every'] == 0:
                try:
                    # Report Mapping Progress
                    progress_bar = tqdm(range(1), desc=f"Mapping Result Time Step: {time_idx}")
                    with torch.no_grad():
                        if config['use_wandb']:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            wandb_run=wandb_run, wandb_step=wandb_time_step, wandb_save_qual=config['wandb']['save_qual'],
                                            mapping=True, online_time_idx=time_idx, global_logging=True)
                        else:
                            report_progress(params, curr_data, 1, progress_bar, time_idx, sil_thres=config['mapping']['sil_thres'], 
                                            mapping=True, online_time_idx=time_idx)
                    progress_bar.close()
                except:
                    ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
                    save_params_ckpt(params, ckpt_output_dir, time_idx)
                    print('Failed to evaluate trajectory.')
 #################### ####################
                         
        # Add frame to keyframe list（将当前帧加入关键帧列表，同时根据配置进行定期的保存检查点。）
        if ((time_idx == 0) or ((time_idx+1) % config['keyframe_every'] == 0) or \
                    (time_idx == num_frames-2)) and (not torch.isinf(curr_gt_w2c[-1]).any()) and (not torch.isnan(curr_gt_w2c[-1]).any()):
            with torch.no_grad():
                # Get the current estimated rotation & translation
                curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
                curr_cam_tran = params['cam_trans'][..., time_idx].detach()
                curr_w2c = torch.eye(4).cuda().float()
                curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                curr_w2c[:3, 3] = curr_cam_tran
                # Initialize Keyframe Info
                curr_keyframe = {'id': time_idx, 'est_w2c': curr_w2c, 'color': color, 'depth': depth}
                # Add to keyframe list
                keyframe_list.append(curr_keyframe)
                keyframe_time_indices.append(time_idx)
        
        # Checkpoint every iteration
        if time_idx % config["checkpoint_interval"] == 0 and config['save_checkpoints']:
            ckpt_output_dir = os.path.join(config["workdir"], config["run_name"])
            save_params_ckpt(params, ckpt_output_dir, time_idx)
            np.save(os.path.join(ckpt_output_dir, f"keyframe_time_indices{time_idx}.npy"), np.array(keyframe_time_indices))
        
        # Increment WandB Time Step
        if config['use_wandb']: #如果使用 WandB（Weights & Biases），则更新 WandB 的时间步数。
            wandb_time_step += 1

        # 清理 GPU 内存。
        torch.cuda.empty_cache()

    # Compute Average Runtimes
    if tracking_iter_time_count == 0:
        tracking_iter_time_count = 1
        tracking_frame_time_count = 1
    if mapping_iter_time_count == 0:
        mapping_iter_time_count = 1
        mapping_frame_time_count = 1
    tracking_iter_time_avg = tracking_iter_time_sum / tracking_iter_time_count
    tracking_frame_time_avg = tracking_frame_time_sum / tracking_frame_time_count
    mapping_iter_time_avg = mapping_iter_time_sum / mapping_iter_time_count
    mapping_frame_time_avg = mapping_frame_time_sum / mapping_frame_time_count
    print(f"\nAverage Tracking/Iteration Time: {tracking_iter_time_avg*1000} ms")
    print(f"Average Tracking/Frame Time: {tracking_frame_time_avg} s")
    print(f"Average Mapping/Iteration Time: {mapping_iter_time_avg*1000} ms")
    print(f"Average Mapping/Frame Time: {mapping_frame_time_avg} s")
    if config['use_wandb']:
        wandb_run.log({"Final Stats/Average Tracking Iteration Time (ms)": tracking_iter_time_avg*1000,
                       "Final Stats/Average Tracking Frame Time (s)": tracking_frame_time_avg,
                       "Final Stats/Average Mapping Iteration Time (ms)": mapping_iter_time_avg*1000,
                       "Final Stats/Average Mapping Frame Time (s)": mapping_frame_time_avg,
                       "Final Stats/step": 1})
    
    # Evaluate Final Parameters
    with torch.no_grad():
        if config['use_wandb']:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 wandb_run=wandb_run, wandb_save_qual=config['wandb']['eval_save_qual'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])
        else:
            eval(dataset, params, num_frames, eval_dir, sil_thres=config['mapping']['sil_thres'],
                 mapping_iters=config['mapping']['num_iters'], add_new_gaussians=config['mapping']['add_new_gaussians'],
                 eval_every=config['eval_every'])

    # Add Camera Parameters to Save them
    params['timestep'] = variables['timestep']
    params['intrinsics'] = intrinsics.detach().cpu().numpy()
    params['w2c'] = first_frame_w2c.detach().cpu().numpy()
    params['org_width'] = dataset_config["desired_image_width"]
    params['org_height'] = dataset_config["desired_image_height"]
    params['gt_w2c_all_frames'] = []
    for gt_w2c_tensor in gt_w2c_all_frames:
        params['gt_w2c_all_frames'].append(gt_w2c_tensor.detach().cpu().numpy())
    params['gt_w2c_all_frames'] = np.stack(params['gt_w2c_all_frames'], axis=0)
    params['keyframe_time_indices'] = np.array(keyframe_time_indices)
    
    # Save Parameters
    save_params(params, output_dir)

    # Close WandB Run
    if config['use_wandb']:
        wandb.finish()

if __name__ == "__main__": # 表示以下的代码块将在脚本作为主程序运行时执行，而不是被导入到其他模块中时执行。
    parser = argparse.ArgumentParser() #创建一个命令行解析器，该解析器将帮助您从命令行接收参数。

    parser.add_argument("experiment", type=str, help="Path to experiment file") #添加一个名为 "experiment" 的命令行参数，它是一个字符串类型，用于指定实验文件的路径。(对应就是config文件内的)

    args = parser.parse_args() #解析命令行参数，将其存储在 args 变量中。

    #使用 SourceFileLoader 加载指定路径的实验文件，并将其作为模块加载到 experiment 变量中。
    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed']) #设置实验的随机数种子，种子值来自实验配置文件中的 'seed' 字段。
    
    # Create Results Directory and Copy Config
    # 创建结果目录并复制配置文件：
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"] #存储了实验结果的目录路径，由实验配置文件中的 "workdir" 和 "run_name" 字段组成。
    )
    if not experiment.config['load_checkpoint']: #检查是否需要加载检查点，如果不需要，则执行以下操作：
        os.makedirs(results_dir, exist_ok=True)
        shutil.copy(args.experiment, os.path.join(results_dir, "config.py")) #复制实验配置文件到结果目录下的 "config.py"。

    rgbd_slam(experiment.config) #调用函数rgbd_slam并传递配置文件作为参数