import torch

from src.visualization.color_map import apply_color_map_to_image


def inverse_normalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor.mul(std).add(mean)


# Color-map the result.
def vis_depth_map(result):
    far = result.view(-1)[:16_000_000].quantile(0.99).log()
    try:
        near = result[result > 0][:16_000_000].quantile(0.01).log()
    except:
        print("No valid depth values found.")
        near = torch.zeros_like(far)
    result = result.log()
    result = 1 - (result - near) / (far - near)
    return apply_color_map_to_image(result, "turbo")


def confidence_map(result):
    # far = result.view(-1)[:16_000_000].quantile(0.99).log()
    # try:
    #     near = result[result > 0][:16_000_000].quantile(0.01).log()
    # except:
    #     print("No valid depth values found.")
    #     near = torch.zeros_like(far)
    # result = result.log()
    # result = 1 - (result - near) / (far - near)
    result = result / result.view(-1).max()
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
