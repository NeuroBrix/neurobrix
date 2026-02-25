# Copyright 2025 Stack AV Co.
# SPDX-License-Identifier: Apache-2.0

"""Non-Maximum Suppression (NMS) operation."""

import torch

from conch.kernels.vision.nms import nms_launcher


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than `iou_threshold` with another (higher scoring)
    box.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in `(x1, y1, x2, y2)` format with `0 <= x1 < x2` and
            `0 <= y1 < y2`.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept
        by NMS, sorted in decreasing order of scores
    """
    num_boxes = boxes.size(0)
    device = boxes.device

    if num_boxes == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    # Storage for pre-computation of IoU mask
    # This is a boolean matrix indicating whether the IoU between two boxes exceeds the threshold.
    iou_mask = torch.empty((num_boxes, num_boxes), dtype=torch.bool, device=device)
    iou_mask.fill_(False)

    # Initialize keep mask - all boxes are initially kept
    keep_mask = torch.empty(num_boxes, dtype=torch.bool, device=device)
    keep_mask.fill_(True)

    return nms_launcher(
        boxes=boxes,
        scores=scores,
        iou_mask=iou_mask,
        keep_mask=keep_mask,
        iou_threshold=iou_threshold,
    )
