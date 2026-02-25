# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Quick cumulative sum operator for 3D voxel grids."""

import torch

from conch.kernels.vision.bev_pool import bev_pool_backward_launcher, bev_pool_launcher


def bev_pool(
    image_feats: torch.Tensor,
    geom_feats: torch.Tensor,
    interval_starts: torch.Tensor,
    interval_lengths: torch.Tensor,
    batch_size: int,
    grid_cells_z: int,
    grid_cells_x: int,
    grid_cells_y: int,
) -> torch.Tensor:
    """Cumulative sum pooling operator for 3D voxel grids.

    Args:
        image_feats: input image features, FloatTensor[n, c]
        geom_feats: input coordinates, IntTensor[n, 4]
        interval_starts: starting position for pooled point
        interval_lengths: how many points in each pooled point
        batch_size: batch size
        grid_cells_z: number of z cells of the 3D voxel grid
        grid_cells_x: number of x cells of the 3D voxel grid
        grid_cells_y: number of y cells of the 3D voxel grid.

    Returns:
        output features, FloatTensor[batch_size, grid_z_cells, grid_x_cells, grid_y_cells, c]
    """
    _, num_channels = image_feats.shape

    # Create output tensor
    output = torch.zeros(
        (batch_size, grid_cells_z, grid_cells_x, grid_cells_y, num_channels),
        dtype=image_feats.dtype,
        device=image_feats.device,
    )

    bev_pool_launcher(
        output=output,
        image_feats=image_feats,
        geom_feats=geom_feats,
        interval_starts=interval_starts,
        interval_lengths=interval_lengths,
    )

    return output


def bev_pool_backward(
    grad_output: torch.Tensor,
    geom_feats: torch.Tensor,
    interval_starts: torch.Tensor,
    interval_lengths: torch.Tensor,
) -> torch.Tensor:
    """Backward pass for the BEV pooling operation.

    Args:
        grad_output: gradient of the output, FloatTensor[batch_size, grid_z_cells, grid_x_cells, grid_y_cells, c]
        geom_feats: input coordinates, IntTensor[n, 4]
        interval_starts: starting position for pooled point
        interval_lengths: how many points in each pooled point

    Returns:
        Gradient with respect to image features.
    """
    num_points, _ = geom_feats.shape
    _, _, _, _, num_channels = grad_output.shape

    x_grad = torch.zeros((num_points, num_channels), dtype=grad_output.dtype, device=grad_output.device)

    bev_pool_backward_launcher(
        x_grad=x_grad,
        grad_output=grad_output,
        geom_feats=geom_feats,
        interval_starts=interval_starts,
        interval_lengths=interval_lengths,
    )

    return x_grad
