import torch
import re
import numpy as np

from src.visualization.color_map import apply_color_map_to_image


def inverse_normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor.mul(std).add(mean)


# Color-map the result.
def vis_scalar_map(result, norm_min=0.0, norm_max=1.0, colormap="turbo"):
    # Normalize using the provided constant values.
    normalized = (result - norm_min) / (norm_max - norm_min)
    normalized = normalized.clamp(0, 1)
    return apply_color_map_to_image(normalized, colormap)


def vis_depth_map(result, norm_min=None, norm_max=None, colormap="turbo"):
    """
    Color-map the depth map result with constant normalization.
    
    Args:
        result (torch.Tensor): Input tensor of shape [B, H, W] (or [H, W]) with scalar values.
        norm_min (float, optional): Constant value (after log-transform) for the minimum.
        norm_max (float, optional): Constant value (after log-transform) for the maximum.
        colormap (str): Matplotlib colormap to use. Default "bwr" maps low values to blue and high values to red.
    
    Returns:
        torch.Tensor: A color-mapped image tensor (float in [0, 1] with shape [B, 3, H, W]).
    """
    # Ensure contiguous-friendly operations
    result = result.detach()
    # Identify exact-zero depths to paint black later.
    zero_mask = (result == 0)
    # Avoid -inf by substituting 1.0 for zeros during normalization only.
    safe_result = torch.where(zero_mask, torch.ones_like(result), result)
    # Apply log-transform to the input values.
    result_log = safe_result.log()
    
    if norm_min is None or norm_max is None:
        # Fallback: compute quantiles per image.
        far = result.reshape(-1)[:16_000_000].quantile(0.99).log()
        try:
            near = result[result > 0].reshape(-1)[:16_000_000].quantile(0.01).log()
        except Exception as e:
            print("No valid depth values found.", e)
            near = torch.zeros_like(far)
        norm_min = near
        norm_max = far

    # Normalize using constant values.
    normalized = 1 - (result_log - norm_min) / (norm_max - norm_min)
    # Apply colormap
    colored = apply_color_map_to_image(normalized, colormap)

    # Paint zero depths black only, preserve other colors
    if zero_mask.ndim == 2:  # H W -> 3 H W
        mask_c = zero_mask.unsqueeze(0).expand(3, -1, -1)
        colored[mask_c] = 0.0
    elif zero_mask.ndim == 3:  # B H W -> B 3 H W
        mask_c = zero_mask.unsqueeze(1).expand(-1, 3, -1, -1)
        colored[mask_c] = 0.0

    return colored

def confidence_map(result):
    # far = result.reshape(-1)[:16_000_000].quantile(0.99).log()
    # try:
    #     near = result[result > 0].reshape(-1)[:16_000_000].quantile(0.01).log()
    # except:
    #     print("No valid depth values found.")
    #     near = torch.zeros_like(far)
    # result = result.log()
    # result = 1 - (result - near) / (far - near)
    result = result / result.reshape(-1).max()
    return apply_color_map_to_image(result, "magma")


def get_overlap_tag(overlap):
    if 0.05 <= overlap <= 0.3:
        overlap_tag = "small"
    elif overlap <= 0.55:
        overlap_tag = "medium"
    elif overlap <= 0.8:
        overlap_tag = "large"
    else:
        overlap_tag = "ignore"

    return overlap_tag


def inspect_depth_tensor(depth_tensor: torch.Tensor, name: str = "Depth Tensor") -> None:
    """
    Print detailed statistics about a depth tensor.
    
    Args:
        depth_tensor (torch.Tensor): The depth tensor, expected shape can be arbitrary.
        name (str): Name of the tensor (for printing purposes).
    """
    # Ensure the tensor is float and on CPU
    depth_tensor = depth_tensor.float().detach().cpu()
    total_elements = depth_tensor.numel()

    # Create masks for finite values, zeros, negatives, inf, nan
    finite_mask = torch.isfinite(depth_tensor)
    finite_values = depth_tensor[finite_mask]
    
    zero_mask = (depth_tensor == 0)
    neg_mask = (depth_tensor < 0)
    inf_mask = torch.isinf(depth_tensor)
    nan_mask = torch.isnan(depth_tensor)
    
    zero_count = zero_mask.sum().item()
    neg_count = neg_mask.sum().item()
    inf_count = inf_mask.sum().item()
    nan_count = nan_mask.sum().item()
    
    percent_zero = (zero_count / total_elements) * 100
    percent_neg = (neg_count / total_elements) * 100
    percent_inf = (inf_count / total_elements) * 100
    percent_nan = (nan_count / total_elements) * 100

    # Compute basic statistics on finite values (if any)
    if finite_values.numel() > 0:
        min_val = finite_values.min().item()
        max_val = finite_values.max().item()
        mean_val = finite_values.mean().item()
        std_val = finite_values.std().item()
        median_val = finite_values.median().item()
    else:
        min_val = max_val = mean_val = std_val = median_val = float('nan')

    print(f"--- {name} Statistics ---")
    print(f"Total elements: {total_elements}")
    print(f"Finite values: {finite_values.numel()} ({finite_values.numel()/total_elements*100:.2f}%)")
    print(f"Min (finite): {min_val:.6f}")
    print(f"Max (finite): {max_val:.6f}")
    print(f"Mean (finite): {mean_val:.6f}")
    print(f"Std (finite): {std_val:.6f}")
    print(f"Median (finite): {median_val:.6f}")
    print(f"Zero count: {zero_count} ({percent_zero:.2f}%)")
    print(f"Negative count: {neg_count} ({percent_neg:.2f}%)")
    print(f"Inf count: {inf_count} ({percent_inf:.2f}%)")
    print(f"NaN count: {nan_count} ({percent_nan:.2f}%)")
    print(f"---------------------------\n")


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale



if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
 # --- Legend parameters ---
    norm_min, norm_max = 0.0, 1.0
    height = 256
    width  = 20   # legend bar width in pixels

    # 1) Build a [height × width] gradient in [norm_min, norm_max]
    gradient = torch.linspace(norm_min, norm_max, height)      # [height]
    gradient = gradient.unsqueeze(1).repeat(1, width)          # [height, width]

    # 2) Colorize with your function → shape [3, height, width]
    legend_tensor = apply_color_map_to_image(gradient, color_map="turbo_r")

    # 3) Convert to H×W×3 NumPy for plotting
    legend_np = legend_tensor.permute(1, 2, 0).cpu().numpy()

    # 4) Plot & save as an image
    plt.figure(figsize=(1.5, 6), facecolor='white')
    plt.imshow(legend_np, origin='lower', aspect='auto')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('colorbar_legend.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()