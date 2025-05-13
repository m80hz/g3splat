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



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Build a hemisphere normal map
    H, W = 256, 256
    y = torch.linspace(-1, 1, H)
    x = torch.linspace(-1, 1, W)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    mask = (X**2 + Y**2) <= 1
    Z = torch.zeros_like(X)
    Z[mask] = torch.sqrt(1 - X[mask]**2 - Y[mask]**2)

    normals = torch.stack([X, Y, Z], dim=2).unsqueeze(0)  # [1, H, W, 3]

    vis = vis_normal(normals)[0].numpy()  # (H, W, 3), uint8

    bg = np.ones_like(vis) * 255
    m = mask.numpy()
    bg[m] = vis[m]
    img = bg.astype(np.float32) / 255.0  # for imshow

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, origin='lower', extent=(-1, 1, -1, 1))
    ax.axis('off')
    ax.set_title('Surface Normal Legend (RGB Ball)')

    # ax.quiver(0, 0, 1, 0, scale=1, scale_units='xy', width=0.005, color='k')
    # ax.text(1.05, 0, 'X', va='center', fontsize=12)
    # ax.quiver(0, 0, 0, 1, scale=1, scale_units='xy', width=0.005, color='k')
    # ax.text(0, 1.05, 'Y', ha='center', fontsize=12)
    # # Z is out‐of‐plane: use ⊙ symbol at origin
    # ax.text(0, -0.1, '⊙ Z', ha='center', va='top', fontsize=12)

    plt.tight_layout()
    plt.show()
    