# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Triton point cloud voxeliation."""

from dataclasses import dataclass

import torch
import triton

from conch.kernels.vision.voxelization import generate_dense_voxels_triton_kernel, generate_voxels_triton_kernel


@dataclass
class VoxelizationParameter:
    """Parameters."""

    min_range: tuple[float, float, float]
    max_range: tuple[float, float, float]
    voxel_dim: tuple[float, float, float]
    grid_dim: tuple[int, int, int]
    max_num_points_per_voxel: int
    max_num_voxels: int

    def __init__(
        self,
        min_range: tuple[float, float, float],
        max_range: tuple[float, float, float],
        voxel_dim: tuple[float, float, float],
        max_num_points_per_voxel: int,
    ) -> None:
        """Init parameters."""
        self.min_range = min_range
        self.max_range = max_range
        self.voxel_dim = voxel_dim
        self.max_num_points_per_voxel = max_num_points_per_voxel
        self.grid_dim = self._compute_grid_dim()
        self.max_num_voxels = self.grid_dim[0] * self.grid_dim[1] * self.grid_dim[2]

    def _compute_grid_dim(self) -> tuple[int, int, int]:
        """Compute grid dimensions."""
        grid_x = round((self.max_range[0] - self.min_range[0]) / self.voxel_dim[0])
        grid_y = round((self.max_range[1] - self.min_range[1]) / self.voxel_dim[1])
        grid_z = round((self.max_range[2] - self.min_range[2]) / self.voxel_dim[2])
        return (grid_x, grid_y, grid_z)


def generate_voxels(
    points: torch.Tensor, param: VoxelizationParameter
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates voxels from input points, output voxels and points are randomly ordered due to use of atomics.

    Args:
        points: input points; expected dimensions (num_points, 4), last dimension should have fields of x,y,z,_.
        param: parameters.

    Returns:
        tuple of tensors:
            num_of_filled_voxels, shape [1, cpu]
            point_features, shape [num_filled_voxels, max_num_points_per_voxel, num_features_per_point], empty points are filled with 0.
            voxel_indices, shape [num_filled_voxels, 4], only first 3 fields are used for x,y,z indices.
            capped_num_points_per_voxel, shape [num_filled_voxels], note per voxel point counters are capped with max_num_points_per_voxel.
    """
    device = points.device
    num_points, num_features_per_point = points.shape
    assert num_features_per_point == 4
    # same as original nvidia cuda impl
    num_elements_per_voxel_index = 4

    # dense
    dense_num_points_per_voxel = torch.zeros((param.max_num_voxels), dtype=torch.int32, device=device)
    dense_point_features = torch.empty(
        (param.max_num_voxels, param.max_num_points_per_voxel, num_features_per_point), dtype=torch.float, device=device
    )

    # sparse/contiguous output
    num_filled_voxels = torch.zeros((1), dtype=torch.int32, device=device)
    num_points_per_voxel = torch.empty_like(dense_num_points_per_voxel)
    point_features = torch.empty_like(dense_point_features)
    voxel_indices = torch.empty((param.max_num_voxels, num_elements_per_voxel_index), dtype=torch.int32, device=device)

    # hardcoded for 256 threads per program save as cuda
    block_size = 256
    num_threads_per_warp = 32

    # first generate dense voxels
    grid = (triton.cdiv(num_points, block_size),)
    generate_dense_voxels_triton_kernel[grid](
        points,
        num_points,
        param.min_range[0],
        param.min_range[1],
        param.min_range[2],
        param.max_range[0],
        param.max_range[1],
        param.max_range[2],
        param.voxel_dim[0],
        param.voxel_dim[1],
        param.voxel_dim[2],
        param.grid_dim[0],
        param.grid_dim[1],
        param.grid_dim[2],
        param.max_num_points_per_voxel,
        dense_num_points_per_voxel,
        dense_point_features,
        cxpr_block_size=block_size,
        num_warps=block_size // num_threads_per_warp,
    )

    # compress into contiguous/sparse filled voxels
    grid = (triton.cdiv(param.max_num_voxels, block_size),)
    generate_voxels_triton_kernel[grid](
        dense_point_features,
        dense_num_points_per_voxel,
        param.grid_dim[0],
        param.grid_dim[1],
        param.max_num_points_per_voxel,
        param.max_num_voxels,
        num_filled_voxels,
        num_points_per_voxel,
        point_features,
        voxel_indices,
        cxpr_block_size=block_size,
        num_warps=block_size // num_threads_per_warp,
    )

    total_filled_voxels = num_filled_voxels.cpu()

    return total_filled_voxels, point_features, voxel_indices, num_points_per_voxel
