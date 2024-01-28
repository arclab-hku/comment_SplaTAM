"""
Code for Keyframe Selection based on re-projection of points from 
the current frame to the keyframes.
"""

import torch
import numpy as np


def get_pointcloud(depth, intrinsics, w2c, sampled_indices):
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # Compute indices of sampled pixels
    xx = (sampled_indices[:, 1] - CX)/FX
    yy = (sampled_indices[:, 0] - CY)/FY
    depth_z = depth[0, sampled_indices[:, 0], sampled_indices[:, 1]]

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    pts4 = torch.cat([pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1)
    c2w = torch.inverse(w2c)
    pts = (c2w @ pts4.T).T[:, :3]

    # Remove points at camera origin
    A = torch.abs(torch.round(pts, decimals=4))
    B = torch.zeros((1, 3)).cuda().float()
    _, idx, counts = torch.cat([A, B], dim=0).unique(
        dim=0, return_inverse=True, return_counts=True)
    mask = torch.isin(idx, torch.where(counts.gt(1))[0])
    invalid_pt_idx = mask[:len(A)]
    valid_pt_idx = ~invalid_pt_idx
    pts = pts[valid_pt_idx]

    return pts


# 实现了选择与当前相机观测重叠的关键帧,并返回一组重叠程度较高的关键帧
def keyframe_selection_overlap(gt_depth, w2c, intrinsics, keyframe_list, k, pixels=1600):
        """
        Select overlapping keyframes to the current camera observation.

        Args:
            gt_depth (tensor): ground truth depth image of the current frame.
            w2c (tensor): world to camera matrix (4 x 4).
            keyframe_list (list): a list containing info for each keyframe.
            k (int): number of overlapping keyframes to select.
            pixels (int, optional): number of pixels to sparsely sample 
                from the image of the current camera. Defaults to 1600.
        Returns:
            selected_keyframe_list (list): list of selected keyframe id.
        """
        # Radomly Sample Pixel Indices from valid depth pixels
        # 随机采样像素索引：
        # 首先，从当前帧的有效深度像素中（深度大于零的像素）随机选择一定数量（pixels）的像素索引。
        # 这样，就得到了从当前帧中稀疏采样的像素位置。
        width, height = gt_depth.shape[2], gt_depth.shape[1]
        valid_depth_indices = torch.where(gt_depth[0] > 0)
        valid_depth_indices = torch.stack(valid_depth_indices, dim=1)
        indices = torch.randint(valid_depth_indices.shape[0], (pixels,))
        sampled_indices = valid_depth_indices[indices]

        # Back Project the selected pixels to 3D Pointcloud
        # 反投影选定的像素到3D点云：
        # 利用 get_pointcloud 函数，将选定的像素索引反投影到3D点云空间。
        # 得到的 pts 包含了在3D相机坐标系中的稀疏采样点的坐标。
        pts = get_pointcloud(gt_depth, intrinsics, w2c, sampled_indices)

        list_keyframe = []
        # 计算关键帧与3D点云的重叠程度：
        for keyframeid, keyframe in enumerate(keyframe_list):
            # Get the estimated world2cam of the keyframe
            # 获取关键帧的估计世界到相机变换矩阵
            est_w2c = keyframe['est_w2c']

            # Transform the 3D pointcloud to the keyframe's camera space
            # 将3D点云变换到关键帧的相机坐标系下。
            pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
            transformed_pts = (est_w2c @ pts4.T).T[:, :3]

            # Project the 3D pointcloud to the keyframe's image space
            # 将3D点云投影到关键帧的图像坐标系下。
            points_2d = torch.matmul(intrinsics, transformed_pts.transpose(0, 1))
            points_2d = points_2d.transpose(0, 1)
            points_z = points_2d[:, 2:] + 1e-5
            points_2d = points_2d / points_z
            projected_pts = points_2d[:, :2]

            # Filter out the points that are outside the image
            # 过滤掉图像范围之外的点
            edge = 20
            mask = (projected_pts[:, 0] < width-edge)*(projected_pts[:, 0] > edge) * \
                (projected_pts[:, 1] < height-edge)*(projected_pts[:, 1] > edge)
            mask = mask & (points_z[:, 0] > 0)

            # Compute the percentage of points that are inside the image
            # 计算在图像内的点的百分比，即与关键帧的重叠程度。
            percent_inside = mask.sum()/projected_pts.shape[0]

            # 将关键帧的id和重叠百分比加入 list_keyframe 列表。
            list_keyframe.append(
                {'id': keyframeid, 'percent_inside': percent_inside})

        # Sort the keyframes based on the percentage of points that are inside the image
        # 根据重叠百分比对关键帧进行排序，百分比越高的排在前面。
        list_keyframe = sorted(
            list_keyframe, key=lambda i: i['percent_inside'], reverse=True)
        # Select the keyframes with percentage of points inside the image > 0
        # 从排序后的关键帧列表中选择百分比大于零的前 k 个关键帧，即选择重叠程度最高的前 k 个关键帧作为最终选定的关键帧列表。（这里的k就是要参与mapping的关键帧的数据量）
        selected_keyframe_list = [keyframe_dict['id']
                                  for keyframe_dict in list_keyframe if keyframe_dict['percent_inside'] > 0.0]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k])

        return selected_keyframe_list