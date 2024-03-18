import torch
import torch.nn.functional as F
from utils.slam_external import build_rotation

def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    Source: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#matrix_to_quaternion
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


# 将输入的参数 params 转换成一个包含渲染相关变量的字典 rendervar
def params2rendervar(params):
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['rgb_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def transformed_params2rendervar(params, transformed_pts):
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': params['rgb_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def project_points(points_3d, intrinsics):
    """
    Function to project 3D points to image plane.
    params:
    points_3d: [num_gaussians, 3]
    intrinsics: [3, 3]
    out: [num_gaussians, 2]
    """
    points_2d = torch.matmul(intrinsics, points_3d.transpose(0, 1))
    points_2d = points_2d.transpose(0, 1)
    points_2d = points_2d / points_2d[:, 2:]
    points_2d = points_2d[:, :2]
    return points_2d

def params2silhouette(params):
    sil_color = torch.zeros_like(params['rgb_colors'])
    sil_color[:, 0] = 1.0
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': sil_color,
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def transformed_params2silhouette(params, transformed_pts):
    sil_color = torch.zeros_like(params['rgb_colors'])
    sil_color[:, 0] = 1.0
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': sil_color,
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar

# 计算每个高斯分布的深度和轮廓
def get_depth_and_silhouette(pts_3D, w2c):
    """
    Function to compute depth and silhouette for each gaussian.
    These are evaluated at gaussian center.
    """
    # Depth of each gaussian center in camera frame（计算高斯分布中心点在相机坐标系下的深度，并计算深度的平方值。）
    pts4 = torch.cat((pts_3D, torch.ones_like(pts_3D[:, :1])), dim=-1) #将 pts_3D 与一个全为1的列拼接，以便进行仿射变换，得到形状为 (num_gaussians, 4) 的张量 pts4。
    pts_in_cam = (w2c @ pts4.transpose(0, 1)).transpose(0, 1) #获取高斯分布中心点在相机坐标系下的坐标
    # 从 pts_in_cam 中提取了每个高斯分布中心点的 Z 轴坐标，即深度值。最终得到的张量的形状为 (num_gaussians, 1)，其中 num_gaussians 表示高斯分布的数量。
    depth_z = pts_in_cam[:, 2].unsqueeze(-1) # [num_gaussians, 1]
    # 将 depth_z 中每个元素进行平方操作，得到了深度值的平方。得到的张量的形状与 depth_z 相同，为 (num_gaussians, 1)。
    depth_z_sq = torch.square(depth_z) # [num_gaussians, 1]

    # Depth and Silhouette
    # 创建了一个形状为 (pts_3D.shape[0], 3) 的全零张量，并将其转移到 GPU 上并设置为浮点数类型。pts_3D.shape[0] 表示 pts_3D 张量的第一个维度的大小，通常是点的数量。
    depth_silhouette = torch.zeros((pts_3D.shape[0], 3)).cuda().float()
    # depth_z 中存储了每个高斯分布中心点在相机坐标系下的 Z 轴坐标（深度值）。
    # depth_z_sq 中存储了 depth_z 中每个深度值的平方。
    # 通过 squeeze(-1) 操作将形状为 (num_gaussians, 1) 的张量转换为形状为 (num_gaussians,) 的一维张量。
    depth_silhouette[:, 0] = depth_z.squeeze(-1) #（depth_z填充到 depth_silhouette 张量的第一列中。）
    depth_silhouette[:, 1] = 1.0 # 将 depth_silhouette 张量的第二列填充为 1.0，
    depth_silhouette[:, 2] = depth_z_sq.squeeze(-1) #depth_z_sq 中的深度值的平方填充到 depth_silhouette 张量的第三列中。
    
    return depth_silhouette #所谓的深度轮廓其实就是相机坐标系下的（深度值，1，深度的平方）


def params2depthplussilhouette(params, w2c):
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': get_depth_and_silhouette(params['means3D'], w2c), #计算每个高斯分布的深度和轮廓（#所谓的深度轮廓其实就是相机坐标系下的（深度值，1，深度的平方））
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar

# 获取渲染的变量
def transformed_params2depthplussilhouette(params, w2c, transformed_pts):
    rendervar = {
        'means3D': transformed_pts, #在相机坐标系下的高斯中心点
        'colors_precomp': get_depth_and_silhouette(transformed_pts, w2c), #计算每个高斯分布的深度和轮廓（#所谓的深度轮廓其实就是相机坐标系下的（深度值，1，深度的平方））
        'rotations': F.normalize(params['unnorm_rotations']), #归一化旋转
        'opacities': torch.sigmoid(params['logit_opacities']), #高斯分布的不透明度，经过 sigmoid 函数处理
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))), #高斯分布的尺度，经过 exp 函数处理
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0 #在图像坐标系下的高斯中心点，初始化为0
    }
    return rendervar


# 这个函数的目的是将各向同性高斯分布的中心点从世界坐标系转换到相机坐标系中。
    # 注意，此处的params（如下定义，实际上就是高斯函数）
    # params = {
    #     'means3D': means3D,
    #     'rgb_colors': init_pt_cld[:, 3:6],
    #     'unnorm_rotations': unnorm_rots,
    #     'logit_opacities': logit_opacities,
    #     'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    # }
def transform_to_frame(params, time_idx, gaussians_grad, camera_grad):
    """
    Function to transform Isotropic Gaussians from world frame to camera frame.
    
    Args:
        params: dict of parameters 一个包含各种参数的字典（实际上就是3D高斯函数的所有参数，同时也包含pose等）
        time_idx: time index to transform to 表示时间索引，用于指定转换到哪一帧。
        gaussians_grad: enable gradients for Gaussians  一个布尔值，表示是否启用高斯分布的梯度。
        camera_grad: enable gradients for camera pose 一个布尔值，表示是否启用相机位姿的梯度。
    
    Returns:
        transformed_pts: Transformed Centers of Gaussians #返回的在相机坐标系下的高斯中心点
    """
    # Get Frame Camera Pose 获取相机位姿：
    if camera_grad: #如果 camera_grad 为 True，则获取未归一化的相机旋转 cam_rot 和相机平移 cam_tran
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx])
        cam_tran = params['cam_trans'][..., time_idx]
    else: #否则，使用 .detach() 方法获取它们的副本，确保梯度不会在这里传播。
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., time_idx].detach())
        cam_tran = params['cam_trans'][..., time_idx].detach()
    # 构建相机到世界坐标系的变换矩阵 rel_w2c，其中包含旋转矩阵和平移向量。
    rel_w2c = torch.eye(4).cuda().float() #构建一个形状为 (4, 4) 的单位矩阵 rel_w2c。（先初始化为单位矩阵）
    rel_w2c[:3, :3] = build_rotation(cam_rot)
    rel_w2c[:3, 3] = cam_tran

    # Get Centers and norm Rots of Gaussians in World Frame 获取世界坐标系下高斯分布中心和归一化旋转：
    # 获取高斯分布的中心点 pts，如果 gaussians_grad 为 True，则获取高斯分布的中心点 pts(不使用 .detach()，所以 pts 是原始张量，它可能是需要计算梯度的。)
    if gaussians_grad: #如果 gaussians_grad 为 True，则获取高斯分布的中心点 pts(不使用 .detach()，所以 pts 是原始张量，它可能是需要计算梯度的。)
        pts = params['means3D']
    else:#。否则，使用 .detach() 方法获取其副本(通过使用 .detach() 方法，确保返回的张量是不需要计算梯度的。这可以防止梯度在这个张量上进行传播。)。
        pts = params['means3D'].detach()
    
    # Transform Centers and Unnorm Rots of Gaussians to Camera Frame 将中心点和未归一化旋转转换到相机坐标系：
    pts_ones = torch.ones(pts.shape[0], 1).cuda().float() #构建形状为 (N, 4) 的矩阵 pts4，其中 N 是中心点数量，通过在中心点矩阵的最后一列添加全为1的列得到。
    # .cuda() 表示将张量移动到GPU上，如果GPU可用的话。
    # .float() 将张量的数据类型转换为浮点型。
    pts4 = torch.cat((pts, pts_ones), dim=1) #使用 torch.cat 函数在第二维度上拼接 pts 和 pts_ones。(结果是一个形状为 (N, 4) 的张量 pts4，其中最后一列全为1，用于表示齐次坐标。)
    transformed_pts = (rel_w2c @ pts4.T).T[:, :3] #利用相机到世界坐标系的变换矩阵 rel_w2c，将这个矩阵应用于 pts4，并提取结果的前三列，得到转换后的中心点 transformed_pts。
    # 将 pts4 转置（.T）后，利用相机到世界坐标系的变换矩阵 rel_w2c 将其应用于高斯分布的中心点。
    # 将结果再次转置，然后取前三列，得到形状为 (N, 3) 的张量 transformed_pts。
    # 这样得到的 transformed_pts 就是高斯分布中心点在相机坐标系中的转换结果，保留了前三个坐标值。

    return transformed_pts #返回转换后的中心点 transformed_pts。