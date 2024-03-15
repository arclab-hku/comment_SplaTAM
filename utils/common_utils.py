import os

import numpy as np
import random
import torch

# 这段代码是一个用于设置随机种子的函数，目的是使得代码在不同运行环境下生成的随机数保持一致
def seed_everything(seed=42):
    """
        Set the `seed` value for torch and numpy seeds. Also turns on
        deterministic execution for cudnn.
        
        Parameters:
        - seed:     A hashable seed value
    """
    random.seed(seed) #设置 Python 标准库中 random 模块的种子，用于产生伪随机数序列。
    # 设置 Python 进程的环境变量 PYTHONHASHSEED，这是为了确保哈希的稳定性。在某些情况下，Python 字典等数据结构在不同的运行环境下，其哈希值可能会有所变化。通过设置相同的种子，可以确保在不同的环境下哈希值保持一致。
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed) #置 NumPy 库的随机种子，以确保 NumPy 生成的随机数序列也是可重现的。
    torch.manual_seed(seed) #设置 PyTorch 库的随机种子。这样做可以确保 PyTorch 在涉及随机数生成的操作时，例如初始化模型参数时，生成的随机数序列是确定性的。

    # 设置 PyTorch 的 CuDNN 后端为确定性模式。CuDNN 是 NVIDIA 提供的用于深度学习的 GPU 加速库，开启确定性模式可以确保在相同的输入下，网络的计算结果是确定的，这对于实验结果的可重现性非常重要。
    torch.backends.cudnn.deterministic = True

    # 禁用 CuDNN 的自动优化功能。通常情况下，CuDNN 会根据当前硬件和输入数据的大小自动选择最优的算法来进行计算，这样可以提高运行效率。但是为了确保结果的一致性，禁用自动优化是必要的。
    torch.backends.cudnn.benchmark = False

    # 打印出设置的种子值和其类型，以便在程序运行时能够确认种子被正确设置。
    print(f"Seed set to: {seed} (type: {type(seed)})")


def params2cpu(params):
    res = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            res[k] = v.detach().cpu().contiguous().numpy()
        else:
            res[k] = v
    return res


def save_params(output_params, output_dir):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **to_save)


def save_params_ckpt(output_params, output_dir, time_idx):
    # Convert to CPU Numpy Arrays
    to_save = params2cpu(output_params)
    # Save the Parameters containing the Gaussian Trajectories
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **to_save)


def save_seq_params(all_params, output_dir):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params.npz")
    np.savez(save_path, **params_to_save)


def save_seq_params_ckpt(all_params, output_dir,time_idx):
    params_to_save = {}
    for frame_idx, params in enumerate(all_params):
        params_to_save[f"frame_{frame_idx}"] = params2cpu(params)
    # Save the Parameters containing the Sequence of Gaussians
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving parameters to: {output_dir}")
    save_path = os.path.join(output_dir, "params"+str(time_idx)+".npz")
    np.savez(save_path, **params_to_save)