import torch
import torch.nn.functional as F


def create_gaussian_window(
    window_size: int, 
    sigma: float,
    device=None, 
    dtype=None
) -> torch.Tensor:
    """Create 2D Gaussian window for SSIM."""
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
    g_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g_1d = g_1d / g_1d.sum()
    g_2d = torch.outer(g_1d, g_1d)
    
    window = g_2d.unsqueeze(0).unsqueeze(0)
    return window


def ssim_2d(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2
) -> torch.Tensor:
    """
    Compute scalar SSIM between two real-valued 2D tensors (H, W).
    """
    assert img1.shape == img2.shape and img1.dim() == 2, "img1/img2 must be 2D and same shape"

    img1 = img1.unsqueeze(0).unsqueeze(0)
    img2 = img2.unsqueeze(0).unsqueeze(0)

    device = img1.device
    dtype = img1.dtype
    window = create_gaussian_window(window_size, sigma, device=device, dtype=dtype)

    padding = window_size // 2

    mu1 = F.conv2d(img1, window, padding=padding)
    mu2 = F.conv2d(img2, window, padding=padding)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding) - mu1_mu2

    numerator1 = 2 * mu1_mu2 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
    return ssim_map.mean()  # scalar


def complex_ssim_2d(
    z1: torch.Tensor,
    z2: torch.Tensor,
    **ssim_kwargs
) -> torch.Tensor:
    """
    'Complex SSIM' between two complex-valued images (H, W).

    Here we define it as the average of SSIM over real and imaginary parts:
        cSSIM = 0.5 * [SSIM(Re(z1), Re(z2)) + SSIM(Im(z1), Im(z2))]
    """
    assert torch.is_complex(z1) and torch.is_complex(z2)
    assert z1.shape == z2.shape and z1.dim() == 2

    ssim_real = ssim_2d(z1.real, z2.real, **ssim_kwargs)
    ssim_imag = ssim_2d(z1.imag, z2.imag, **ssim_kwargs)
    return 0.5 * (ssim_real + ssim_imag)


def build_heatmaps(x: torch.Tensor):
    """
    Given a complex tensor x with shape (D, R, A, E) = (16, 64, 64, 8),
    construct range-doppler, range-azimuth, and range-elevation heatmaps
    by averaging over the remaining axes.

    Returns:
        rd: (D, R)  range-doppler      (doppler x range)
        ra: (R, A)  range-azimuth      (range x azimuth)
        re: (R, E)  range-elevation    (range x elevation)
    """
    assert x.ndim == 4, "x must be (D,R,A,E)"
    D, R, A, E = x.shape
    assert (D, R, A, E) == (16, 64, 64, 8)

    # range-doppler: average over azimuth, elevation
    rd = x.mean(dim=(2, 3))          # (D, R)

    # range-azimuth: average over doppler, elevation
    ra = x.mean(dim=(0, 3))          # (R, A)

    # range-elevation: average over doppler, azimuth
    re = x.mean(dim=(0, 2))          # (R, E)

    return rd, ra, re


def compute_all_ssims(x: torch.Tensor, y: torch.Tensor):
    """
    x, y: complex tensors of shape (16, 64, 64, 8)
    Returns a dict with SSIM and complex SSIM for:
       - range-doppler
       - range-azimuth
       - range-elevation
    """
    assert torch.is_complex(x) and torch.is_complex(y)
    assert x.shape == y.shape == (16, 64, 64, 8)

    rd_x, ra_x, re_x = build_heatmaps(x)
    rd_y, ra_y, re_y = build_heatmaps(y)

    results = {}

    # ----- Range-Doppler (D x R) -----
    rd_mag_x = rd_x.abs()
    rd_mag_y = rd_y.abs()
    results["ssim_range_doppler"] = ssim_2d(rd_mag_x, rd_mag_y)
    results["complex_ssim_range_doppler"] = complex_ssim_2d(rd_x, rd_y)

    # ----- Range-Azimuth (R x A) -----
    ra_mag_x = ra_x.abs()
    ra_mag_y = ra_y.abs()
    results["ssim_range_azimuth"] = ssim_2d(ra_mag_x, ra_mag_y)
    results["complex_ssim_range_azimuth"] = complex_ssim_2d(ra_x, ra_y)

    # ----- Range-Elevation (R x E) -----
    re_mag_x = re_x.abs()
    re_mag_y = re_y.abs()
    results["ssim_range_elevation"] = ssim_2d(re_mag_x, re_mag_y)
    results["complex_ssim_range_elevation"] = complex_ssim_2d(re_x, re_y)

    return results

if __name__ == "__main__":
    # Dummy example
    x = torch.randn(16, 64, 64, 8, dtype=torch.complex64).to("cuda")
    y = torch.randn(16, 64, 64, 8, dtype=torch.complex64).to("cuda")

    metrics = compute_all_ssims(x, y)
    for k, v in metrics.items():
        print(f"{k}: {v.item():.4f}")
