# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F


def flow_warp(
    x, flow, interpolation="bilinear", padding_mode="zeros", align_corners=True
):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    # Fix for MPS compatibility - MPS only supports 'reflection' padding mode
    if x.is_mps and padding_mode in ["zeros", "border"]:
        padding_mode = "reflection"

    # Additional MPS compatibility fixes
    if x.is_mps:
        # Ensure tensors are not empty
        if x.numel() == 0 or flow.numel() == 0:
            raise ValueError(
                f"Empty tensor detected: x.numel={x.numel()}, flow.numel={flow.numel()}"
            )

        # Ensure valid dimensions
        if x.dim() != 4 or flow.dim() != 4:
            raise ValueError(
                f"Invalid tensor dimensions: x.dim={x.dim()}, flow.dim={flow.dim()}"
            )

        # Check for valid spatial dimensions
        h, w = x.size()[-2:]
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid spatial dimensions: h={h}, w={w}")
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(
            f"The spatial sizes of input ({x.size()[-2:]}) and "
            f"flow ({flow.size()[1:3]}) are not the same."
        )
    _, _, h, w = x.size()
    # create mesh grid
    device = flow.device
    # torch.meshgrid has been modified in 1.10.0 (compatibility with previous
    # versions), and will be further modified in 1.12 (Breaking Change)
    if "indexing" in torch.meshgrid.__code__.co_varnames:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype),
            indexing="ij",
        )
    else:
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, h, device=device, dtype=x.dtype),
            torch.arange(0, w, device=device, dtype=x.dtype),
        )
    grid = torch.stack((grid_x, grid_y), 2)  # h, w, 2
    grid.requires_grad_(False)

    grid_flow = grid + flow
    # scale grid_flow to [-1,1]
    grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
    grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
    grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)

    # Additional MPS compatibility fixes
    if x.is_mps:
        # Ensure grid_flow is not empty and has valid values
        if grid_flow.numel() == 0:
            raise ValueError(f"Empty grid_flow tensor detected")

        # Check for NaN or Inf values that might cause issues with MPS
        if torch.isnan(grid_flow).any() or torch.isinf(grid_flow).any():
            print("[WARNING] NaN or Inf values detected in grid_flow, clamping values")
            grid_flow = torch.clamp(grid_flow, min=-2.0, max=2.0)

        # Ensure grid values are within valid range for grid_sample
        grid_flow = torch.clamp(grid_flow, min=-2.0, max=2.0)

    # Fix for MPS compatibility - use dtype instead of type()
    grid_flow = grid_flow.to(dtype=x.dtype)

    # Debug logging for MPS compatibility (commented out for production)
    # if torch.backends.mps.is_available() and x.is_mps:
    #     print(f"[DEBUG] flow_warp: x.shape={x.shape}, x.device={x.device}")
    #     print(f"[DEBUG] flow_warp: flow.shape={flow.shape}, flow.device={flow.device}")
    #     print(
    #         f"[DEBUG] flow_warp: grid_flow.shape={grid_flow.shape}, grid_flow.device={grid_flow.device}"
    #     )
    #     print(
    #         f"[DEBUG] flow_warp: interpolation={interpolation}, padding_mode={padding_mode}"
    #     )
    #
    #     # Check for empty tensors or invalid dimensions
    #     if x.numel() == 0 or flow.numel() == 0 or grid_flow.numel() == 0:
    #         print(
    #             f"[ERROR] flow_warp: Empty tensor detected - x.numel={x.numel()}, flow.numel={flow.numel()}, grid_flow.numel={grid_flow.numel()}"
    #         )
    #
    #     # Check for valid grid_sample parameters
    #     h, w = x.shape[-2:]
    #     if h <= 0 or w <= 0:
    #         print(f"[ERROR] flow_warp: Invalid dimensions - h={h}, w={w}")

    # Try grid_sample with MPS, fallback to CPU if it fails
    try:
        output = F.grid_sample(
            x,
            grid_flow,
            mode=interpolation,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
    except RuntimeError as e:
        if x.is_mps and "Placeholder tensor is empty" in str(e):
            print(f"[WARNING] MPS grid_sample failed with error: {e}")
            print("[INFO] Falling back to CPU for flow_warp operation")

            # Move tensors to CPU and retry
            x_cpu = x.cpu()
            grid_flow_cpu = grid_flow.cpu()

            output = F.grid_sample(
                x_cpu,
                grid_flow_cpu,
                mode=interpolation,
                padding_mode=padding_mode,
                align_corners=align_corners,
            )

            # Move result back to MPS device
            output = output.to(x.device)
        else:
            # Re-raise the exception if it's not the MPS empty tensor error
            raise

    return output
