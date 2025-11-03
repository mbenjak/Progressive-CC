import torch
import torch.nn.functional as F

def generate_lanczos_upsampling_kernel(scale_factor, a=3):
    """
    Generate a 2D Lanczos kernel for upsampling.
    Args:
        scale_factor: Upsampling factor.
        a: Lanczos kernel window size (default is 3).
    Returns:
        2D Lanczos kernel as a PyTorch tensor.
    """
    size = int(2 * a * scale_factor + 1)  # Kernel size
    #size = int(2 * a + 1)  # Kernel size
    x = torch.linspace(-a, a, steps=size)
    y = torch.linspace(-a, a, steps=size)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)

    # Lanczos kernel formula
    kernel = torch.where(
        r == 0,
        torch.tensor(1.0),
        a * torch.sin(torch.pi * r) * torch.sin(torch.pi * r / a) / (torch.pi**2 * r**2),
    )
    kernel = torch.where(r > a, torch.tensor(0.0), kernel)

    # Normalize the kernel
    kernel /= kernel.sum()

    # Expand to match conv2d input format (out_channels, in_channels, height, width)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    return kernel

def lanczos_interpolation(input_tensor, scale_factor, a=3):
    """
    Perform Lanczos upsampling using 2D convolution.
    Args:
        input_tensor: Input tensor of shape (N, C, H, W).
        scale_factor: Upsampling factor.
        a: Lanczos kernel window size (default is 3).
    Returns:
        Upsampled tensor.
    """
    # Generate Lanczos kernel
    kernel = generate_lanczos_upsampling_kernel(scale_factor, a=a)
    kernel = kernel.to(input_tensor.device)

    # Upsample using nearest neighbor interpolation
    n, c, h, w = input_tensor.shape
    upsampled = F.interpolate(input_tensor, scale_factor=scale_factor, mode="nearest")

    # Apply Lanczos kernel using conv2d
    upsampled = F.conv2d(
        upsampled.view(n * c, 1, int(h * scale_factor), int(w * scale_factor)),
        kernel,
        padding=kernel.shape[-1] // 2,
        groups=1,
    )
    return upsampled.view(n, c, int(h * scale_factor), int(w * scale_factor))