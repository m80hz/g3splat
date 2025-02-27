import torch

def vis_normal(normal: torch.Tensor) -> torch.Tensor:
    """
    Visualize surface normals in a batched fashion.
    Converts normal values from [-1, 1] to [0, 255].

    Args:
        normal (torch.Tensor): Input tensor of shape [B, H, W, 3] with normal vectors.

    Returns:
        torch.Tensor: Output tensor of shape [B, H, W, 3] in uint8 format.
    """
    # Compute the L2 norm along the channel dimension.
    n_img_L2 = torch.sqrt(torch.sum(normal ** 2, dim=3, keepdim=True))
    # Normalize the normal vectors.
    n_img_norm = normal / (n_img_L2 + 1e-8)
    # Scale and shift the normalized normals to range [0, 255].
    normal_vis = n_img_norm * 127 + 128
    # Convert to uint8.
    normal_vis = normal_vis.to(torch.uint8)
    return normal_vis
