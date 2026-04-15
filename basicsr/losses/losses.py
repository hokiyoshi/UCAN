import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torchvision import models
from torchvision.models import (
    EfficientNet_B7_Weights,
    ResNet101_Weights,
    VGG19_Weights,
    efficientnet_b7,
    resnet101,
    vgg,
)
from torchvision.transforms import GaussianBlur

import pywt
import basicsr.losses.SWT as SWT
from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


# ----------------------------- Basic losses ----------------------------- #

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)
        return x_diff + y_diff


# @LOSS_REGISTRY.register()
# class PerceptualLoss(nn.Module):
#     """Perceptual loss with commonly used style loss.
#
#     Args:
#         layer_weights (dict): The weight for each layer of vgg feature.
#             Here is an example: {'conv5_4': 1.}, which means the conv5_4
#             feature layer (before relu5_4) will be extracted with weight
#             1.0 in calculating losses.
#         vgg_type (str): The type of vgg network used as feature extractor.
#             Default: 'vgg19'.
#         use_input_norm (bool):  If True, normalize the input image in vgg.
#             Default: True.
#         range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
#             Default: False.
#         perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
#             loss will be calculated and the loss will multiplied by the
#             weight. Default: 1.0.
#         style_weight (float): If `style_weight > 0`, the style loss will be
#             calculated and the loss will multiplied by the weight.
#             Default: 0.
#         criterion (str): Criterion used for perceptual loss. Default: 'l1'.
#     """
#     ...


# ----------------------------- SWT loss ----------------------------- #

@LOSS_REGISTRY.register()
class SWTLoss(nn.Module):
    """Stationary Wavelet Transform (SWT) loss.

    Computes L1 loss on each wavelet subband (LL, LH, HL, HH) of the
    Y channel, allowing independent weighting per subband.

    Args:
        loss_weight_ll (float): Weight for the LL subband. Default: 0.01.
        loss_weight_lh (float): Weight for the LH subband. Default: 0.01.
        loss_weight_hl (float): Weight for the HL subband. Default: 0.01.
        loss_weight_hh (float): Weight for the HH subband. Default: 0.01.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight_ll=0.01, loss_weight_lh=0.01,
                 loss_weight_hl=0.01, loss_weight_hh=0.01, reduction='mean'):
        super(SWTLoss, self).__init__()
        self.loss_weight_ll = loss_weight_ll
        self.loss_weight_lh = loss_weight_lh
        self.loss_weight_hl = loss_weight_hl
        self.loss_weight_hh = loss_weight_hh
        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        """
        Args:
            pred (Tensor): of shape (N, 3, H, W). Predicted tensor.
            target (Tensor): of shape (N, 3, H, W). Ground truth tensor.
        """
        wavelet = pywt.Wavelet('sym19')
        dlo = wavelet.dec_lo
        an_lo = np.divide(dlo, sum(dlo))
        an_hi = wavelet.dec_hi
        rlo = wavelet.rec_lo
        syn_lo = 2 * np.divide(rlo, sum(rlo))
        syn_hi = wavelet.rec_hi
        filters = pywt.Wavelet('wavelet_normalized', [an_lo, an_hi, syn_lo, syn_hi])
        sfm = SWT.SWTForward(1, filters, 'periodic').to('cuda')

        sr_y = 16.0 + (pred[:, 0:1] * 65.481 + pred[:, 1:2] * 128.553 + pred[:, 2:] * 24.966)
        hr_y = 16.0 + (target[:, 0:1] * 65.481 + target[:, 1:2] * 128.553 + target[:, 2:] * 24.966)

        wavelet_sr = sfm(sr_y)[0]
        wavelet_hr = sfm(hr_y)[0]

        return (
            self.loss_weight_ll * self.criterion(wavelet_sr[:, 0:1], wavelet_hr[:, 0:1]) +
            self.loss_weight_lh * self.criterion(wavelet_sr[:, 1:2], wavelet_hr[:, 1:2]) +
            self.loss_weight_hl * self.criterion(wavelet_sr[:, 2:3], wavelet_hr[:, 2:3]) +
            self.loss_weight_hh * self.criterion(wavelet_sr[:, 3:],  wavelet_hr[:, 3:])
        )


# ----------------------------- MSSIM loss ----------------------------- #

class GaussianFilter2D(nn.Module):
    """2D Gaussian filter module.

    Args:
        window_size (int): Window size of the Gaussian filter. Must be odd. Default: 11.
        in_channels (int): Number of input channels. Default: 3.
        sigma (float): Sigma of the Gaussian filter. Default: 1.5.
        padding (int, optional): Padding size. If None, uses window_size // 2. Default: None.
    """

    def __init__(self, window_size=11, in_channels=3, sigma=1.5, padding=None):
        super(GaussianFilter2D, self).__init__()
        self.window_size = window_size
        if window_size % 2 != 1:
            raise ValueError('Window size must be odd.')
        self.padding = padding if padding is not None else window_size // 2
        self.sigma = sigma

        kernel = self._get_gaussian_window1d()
        kernel = self._get_gaussian_window2d(kernel)
        self.register_buffer('gaussian_window', kernel.repeat(in_channels, 1, 1, 1))

    def _get_gaussian_window1d(self):
        sigma2 = self.sigma * self.sigma
        x = torch.arange(-(self.window_size // 2), self.window_size // 2 + 1)
        w = torch.exp(-0.5 * x**2 / sigma2)
        w = w / w.sum()
        return w.reshape(1, 1, 1, self.window_size)

    def _get_gaussian_window2d(self, gaussian_window_1d):
        return torch.matmul(gaussian_window_1d.transpose(-1, -2), gaussian_window_1d)

    def forward(self, x):
        return F.conv2d(input=x, weight=self.gaussian_window, stride=1,
                        padding=self.padding, groups=x.shape[1])


@LOSS_REGISTRY.register()
class MssimLoss(nn.Module):
    """Multi-Scale Structural Similarity (MS-SSIM) loss.

    Adapted from 'A better pytorch-based implementation for the mean structural
    similarity. Differentiable simpler SSIM and MS-SSIM':
    https://github.com/lartpang/mssim.pytorch.

    Args:
        window_size (int): Window size of the Gaussian filter. Default: 11.
        in_channels (int): Number of input channels. Default: 3.
        sigma (float): Sigma of the Gaussian filter. Default: 1.5.
        K1 (float): Stability constant K1 of MSSIM. Default: 0.01.
        K2 (float): Stability constant K2 of MSSIM. Default: 0.03.
        L (int): Dynamic range of pixel values (255 for 8-bit images). Default: 1.
        padding (int, optional): Padding of the Gaussian filter. Default: None.
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, window_size=11, in_channels=3, sigma=1.5,
                 K1=0.01, K2=0.03, L=1, padding=None, loss_weight=1.0):
        super(MssimLoss, self).__init__()
        self.C1 = (K1 * L) ** 2
        self.C2 = (K2 * L) ** 2
        self.loss_weight = loss_weight
        self.gaussian_filter = GaussianFilter2D(
            window_size=window_size, in_channels=in_channels,
            sigma=sigma, padding=padding)

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        assert pred.shape == target.shape, f'pred: {pred.shape} and target: {target.shape} must match'
        assert pred.ndim == 4, f'Expected 4D tensors, got {pred.ndim}D'

        if pred.type() != self.gaussian_filter.gaussian_window.type():
            pred = pred.type_as(self.gaussian_filter.gaussian_window)
        if target.type() != self.gaussian_filter.gaussian_window.type():
            target = target.type_as(self.gaussian_filter.gaussian_window)

        return self.loss_weight * (1.0 - self._msssim(pred, target))

    def _ssim(self, x, y):
        mu_x = self.gaussian_filter(x)
        mu_y = self.gaussian_filter(y)
        sigma2_x = self.gaussian_filter(x * x) - mu_x * mu_x
        sigma2_y = self.gaussian_filter(y * y) - mu_y * mu_y
        sigma_xy = self.gaussian_filter(x * y) - mu_x * mu_y

        A1 = 2 * mu_x * mu_y + self.C1
        A2 = 2 * sigma_xy + self.C2
        B1 = mu_x.pow(2) + mu_y.pow(2) + self.C1
        B2 = sigma2_x + sigma2_y + self.C2

        ssim = (A1 / B1) * (A2 / B2)
        cs = A2 / B2
        return ssim, cs

    def _msssim(self, x, y):
        ms_components = []
        for i, w in enumerate((0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
            ssim, cs = self._ssim(x, y)
            if i == 4:
                ms_components.append(ssim.mean() ** w)
            else:
                ms_components.append(cs.mean() ** w)
                padding = [s % 2 for s in x.shape[2:]]
                x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=padding)
                y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=padding)
        return math.prod(ms_components)


# ----------------------------- LDL loss ----------------------------- #

@LOSS_REGISTRY.register()
class LDLLoss(nn.Module):
    """Locally Discriminative Learning (LDL) loss.

    Adapted from 'Details or Artifacts: A Locally Discriminative Learning
    Approach to Realistic Image Super-Resolution':
    https://arxiv.org/abs/2203.09195.

    Args:
        criterion (str): Loss type. Supported choices are 'l1' | 'l2' | 'huber'. Default: 'l1'.
        loss_weight (float): Loss weight. Default: 1.0.
        ksize (int): Size of the local window. Default: 7.
    """

    def __init__(self, criterion='l1', loss_weight=1.0, ksize=7):
        super(LDLLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ksize = ksize
        self.criterion_type = criterion

        if self.criterion_type == 'l1':
            self.criterion = nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = nn.MSELoss()
        elif self.criterion_type == 'huber':
            self.criterion = nn.HuberLoss()
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def _get_local_weights(self, residual):
        """Get local weights for generating the artifact map.

        Args:
            residual (Tensor): Residual between predicted and ground truth images.

        Returns:
            Tensor: Per-pixel artifact weight.
        """
        pad = (self.ksize - 1) // 2
        residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')
        unfolded = residual_pad.unfold(2, self.ksize, 1).unfold(3, self.ksize, 1)
        return torch.var(unfolded, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

    def _get_refined_artifact_map(self, img_gt, img_output):
        """Calculate the refined artifact map.

        Args:
            img_gt (Tensor): Ground truth images.
            img_output (Tensor): Model output images.

        Returns:
            Tensor: Combined patch-level and pixel-level artifact weight.
        """
        residual_sr = torch.sum(torch.abs(img_gt - img_output), 1, keepdim=True)
        patch_level_weight = torch.var(residual_sr.clone(), dim=(-1, -2, -3), keepdim=True) ** (1 / 5)
        pixel_level_weight = self._get_local_weights(residual_sr.clone())
        return patch_level_weight * pixel_level_weight

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        overall_weight = self._get_refined_artifact_map(target, pred)
        return self.criterion(torch.mul(overall_weight, pred),
                              torch.mul(overall_weight, target)) * self.loss_weight


# ----------------------------- CHC loss ----------------------------- #

@LOSS_REGISTRY.register()
class CHCLoss(nn.Module):
    """Clipped pseudo-Huber with Cosine Similarity (CHC) loss.

    For reference see:
    https://github.com/HolmesShuan/AIM2020-Real-Super-Resolution
    https://github.com/dmarnerides/hdr-expandnet

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        criterion (str): Specifies the base loss. Supported choices are 'l1' | 'huber'. Default: 'huber'.
        loss_lambda (float): Scaling factor for the cosine similarity term. Default: 0.
        clip_min (float): Minimum clip threshold (sets well-trained pixel gradients to zero). Default: 0.003921.
        clip_max (float): Maximum clip limit (acts as a noise filter). Default: 0.996078.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', criterion='huber',
                 loss_lambda=0, clip_min=0.003921, clip_max=0.996078):
        super(CHCLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.criterion = criterion
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-20)
        self.loss_lambda = loss_lambda
        self.clip_min = clip_min
        self.clip_max = clip_max

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        cosine_term = (1 - self.similarity(pred, target)).mean()

        if self.criterion == 'l1':
            loss = torch.mean(torch.clamp(
                torch.abs(pred - target) + self.loss_lambda * cosine_term,
                self.clip_min, self.clip_max))
        elif self.criterion == 'huber':
            loss = torch.mean(torch.clamp(
                torch.sqrt((pred - target) ** 2 + 1e-12) + self.loss_lambda * cosine_term,
                self.clip_min, self.clip_max))
        else:
            raise NotImplementedError(f'{self.criterion} not implemented.')

        return self.loss_weight * loss


# ----------------------------- Consistency loss ----------------------------- #

@LOSS_REGISTRY.register()
class ConsistencyLoss(nn.Module):
    """Color and Luma Consistency loss using Oklab and CIE L*.

    Args:
        criterion (str): Base loss type. Supported choices are 'l1' | 'chc'. Default: 'chc'.
        blur (bool): Apply Gaussian blur before computing luma. Default: True.
        cosim (bool): Add cosine similarity regularization. Default: True.
        saturation (float): Saturation scaling factor. Default: 1.0.
        brightness (float): Brightness scaling factor. Default: 1.0.
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, criterion='chc', blur=True, cosim=True,
                 saturation=1.0, brightness=1.0, loss_weight=1.0):
        super(ConsistencyLoss, self).__init__()
        self.use_blur = blur
        self.cosim = cosim
        self.saturation = saturation
        self.brightness = brightness
        self.loss_weight = loss_weight
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-20)
        self.mean = torch.tensor((0.5, 0.5)).view(1, 2, 1, 1)

        if self.use_blur:
            self.blur = GaussianBlur(21, 3)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = nn.L1Loss()
        elif self.criterion_type == 'chc':
            self.criterion = CHCLoss(loss_lambda=0, clip_min=0, clip_max=1)
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def _lin_rgb(self, img):
        """Convert sRGB (gamma 2.4) to linear RGB.

        Args:
            img (Tensor): of shape (N, C, H, W).

        Returns:
            Tensor: Linear RGB tensor.
        """
        return torch.where(img <= 0.04045, img / 12.92, torch.pow((img + 0.055) / 1.055, 2.4))

    def _rgb_to_oklab_chroma(self, img):
        """Convert RGB to Oklab chroma (a, b channels).

        Args:
            img (Tensor): of shape (N, 3, H, W).

        Returns:
            Tensor: of shape (N, 2, H, W).
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError(f'Input type is not a Tensor. Got {type(img)}')
        if img.shape[-3] != 3:
            raise ValueError(f'Input size must have a shape of (*, 3, H, W). Got {img.shape}')

        img = self._lin_rgb(img)
        r, g, b = img[:, 0], img[:, 1], img[:, 2]

        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

        l_ = l.sign() * l.abs().pow(1 / 3)
        m_ = m.sign() * m.abs().pow(1 / 3)
        s_ = s.sign() * s.abs().pow(1 / 3)

        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        return torch.stack([a, b], dim=1)

    def _rgb_to_l_star(self, img):
        """Convert RGB to CIE L* (normalized to [0, 1]).

        Args:
            img (Tensor): of shape (N, C, H, W).

        Returns:
            Tensor: of shape (N, H, W).
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError(f'Input type is not a Tensor. Got {type(img)}')
        img = self._lin_rgb(img.permute(0, 2, 3, 1))
        img = img @ torch.tensor([0.2126, 0.7152, 0.0722]).to(img.device)
        img = torch.where(img <= (216 / 24389), img * (24389 / 27),
                          img.sign() * img.abs().pow(1 / 3) * 116 - 16)
        return torch.clamp(img / 100, 0, 1)

    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        pred = torch.clamp(pred, 1 / 255, 1)
        target = torch.clamp(target, 1 / 255, 1)

        if self.use_blur:
            input_luma = self._rgb_to_l_star(torch.clamp(self.blur(pred), 0, 1))
            target_luma = self._rgb_to_l_star(torch.clamp(self.blur(target), 0, 1)) * self.brightness
        else:
            input_luma = self._rgb_to_l_star(pred)
            target_luma = self._rgb_to_l_star(target) * self.brightness

        input_chroma = self._rgb_to_oklab_chroma(pred)
        target_chroma = self._rgb_to_oklab_chroma(target) * self.saturation

        self.mean = self.mean.to(input_chroma.device)
        input_chroma = torch.clamp(input_chroma + self.mean, 0, 1)
        target_chroma = torch.clamp(target_chroma + self.mean, 0, 1)

        loss = self.criterion(input_luma, target_luma) + self.criterion(input_chroma, target_chroma)

        if self.cosim:
            cosim = (0.5 * (1 - self.similarity(input_chroma, target_chroma).mean()) +
                     0.5 * (1 - self.similarity(input_luma, target_luma).mean()))
            if cosim < 1e-3:
                loss = loss + cosim

        return loss * self.loss_weight


# ----------------------------- DISTS loss ----------------------------- #

class L2Pooling(nn.Module):
    """L2 pooling layer used by DISTS.

    Args:
        filter_size (int): Filter size. Default: 5.
        stride (int): Stride. Default: 2.
        channels (int): Number of channels. Default: 64.
        as_loss (bool): If False, moves filter to CUDA directly. Default: True.
    """

    def __init__(self, filter_size=5, stride=2, channels=64, as_loss=True):
        super(L2Pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))
        if not as_loss:
            self.filter = self.filter.cuda()

    def forward(self, x):
        x = x ** 2
        out = F.conv2d(x, self.filter, stride=self.stride, padding=self.padding, groups=x.shape[1])
        return (out + 1e-12).sqrt()


@LOSS_REGISTRY.register()
class DistsLoss(nn.Module):
    """DISTS loss. Image Quality Assessment: Unifying Structure and Texture Similarity.
    https://arxiv.org/abs/2004.07728.

    Args:
        as_loss (bool): If True, returns scalar loss. If False, returns per-image scores. Default: True.
        loss_weight (float): Loss weight. Default: 1.0.
        load_weights (bool): Load pretrained DISTS alpha/beta weights. Default: True.
    """

    def __init__(self, as_loss=True, loss_weight=1.0, load_weights=True, **kwargs):
        super(DistsLoss, self).__init__()
        self.as_loss = as_loss
        self.loss_weight = loss_weight

        vgg_pretrained_features = models.vgg16(weights='DEFAULT').features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2Pooling(channels=64, as_loss=as_loss))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2Pooling(channels=128, as_loss=as_loss))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2Pooling(channels=256, as_loss=as_loss))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2Pooling(channels=512, as_loss=as_loss))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.chns = [3, 64, 128, 256, 512, 512]
        self.register_parameter('alpha', nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.register_parameter('beta', nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)

        if load_weights:
            model_path = Path(__file__).parent / 'dists_weights.pth'
            try:
                if not model_path.exists():
                    url = 'https://huggingface.co/neosr/models/resolve/main/dists_weights.pth?download=true'
                    raise FileNotFoundError(f'DISTS weights not found at {model_path}. Download from: {url}')
                weights = torch.load(model_path, map_location='cuda', weights_only=True)
                if weights is not None:
                    self.alpha.data = weights['alpha']
                    self.beta.data = weights['beta']
                if not as_loss:
                    self.alpha.data = self.alpha.data.cuda()
                    self.beta.data = self.beta.data.cuda()
            except FileNotFoundError:
                pass

    def _forward_once(self, x):
        h = self.stage1(x)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        feats0 = self._forward_once(pred)
        feats1 = self._forward_once(target)
        dist1 = 0
        dist2 = 0
        c1, c2 = 1e-6, 1e-6

        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)

        if self.as_loss:
            return (1 - (dist1 + dist2).mean()) * self.loss_weight
        return 1 - (dist1 + dist2).squeeze()


# ----------------------------- FDL loss ----------------------------- #

class _VGGBackbone(nn.Module):
    """VGG19 feature extractor backbone for FDL loss."""

    def __init__(self, requires_grad=False, vgg_weights=None):
        super(_VGGBackbone, self).__init__()
        vgg_pretrained_features = vgg.vgg19(weights=VGG19_Weights.DEFAULT).features
        vgg_pretrained_features.eval()
        self.vgg_weights = vgg_weights if vgg_weights is not None else (0.5, 0.5, 1.0, 1.0, 1.0)

        self.stage1 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(4)])
        self.stage2 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(4, 9)])
        self.stage3 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(9, 18)])
        self.stage4 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(18, 27)])
        self.stage5 = nn.Sequential(*[vgg_pretrained_features[x] for x in range(27, 36)])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        self.chns = [64, 128, 256, 512, 512]

    def forward(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        f1 = h * self.vgg_weights[0]
        h = self.stage2(h)
        f2 = h * self.vgg_weights[1]
        h = self.stage3(h)
        f3 = h * self.vgg_weights[2]
        h = self.stage4(h)
        f4 = h * self.vgg_weights[3]
        h = self.stage5(h)
        f5 = h * self.vgg_weights[4]
        return [f1, f2, f3, f4, f5]


class _ResNetBackbone(nn.Module):
    """ResNet101 feature extractor backbone for FDL loss."""

    def __init__(self, requires_grad=False):
        super(_ResNetBackbone, self).__init__()
        model = resnet101(weights=ResNet101_Weights.DEFAULT)
        model.eval()
        self.stage1 = nn.Sequential(model.conv1, model.bn1, model.relu)
        self.stage2 = nn.Sequential(model.maxpool, model.layer1)
        self.stage3 = nn.Sequential(model.layer2)
        self.stage4 = nn.Sequential(model.layer3)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        self.chns = [64, 256, 512, 1024]

    def forward(self, x):
        h = (x - self.mean) / self.std
        f1 = self.stage1(h)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class _EffNetBackbone(nn.Module):
    """EfficientNet-B7 feature extractor backbone for FDL loss."""

    def __init__(self):
        super(_EffNetBackbone, self).__init__()
        model = efficientnet_b7(weights=EfficientNet_B7_Weights.DEFAULT).features
        model.eval()
        self.stage1 = model[0:2]
        self.stage2 = model[2]
        self.stage3 = model[3]
        self.stage4 = model[4]
        self.stage5 = model[5]
        for param in self.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        self.chns = [32, 48, 80, 160, 224]

    def forward(self, x):
        h = (x - self.mean) / self.std
        f1 = self.stage1(h)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        f5 = self.stage5(f4)
        return [f1, f2, f3, f4, f5]


class _DINOv2Backbone(nn.Module):
    """DINOv2 feature extractor backbone for FDL loss.

    Developed by musl from the neosr-project: https://github.com/neosr-project/neosr.
    """

    def __init__(self, layers=None, weights=None, norm=False):
        super(_DINOv2Backbone, self).__init__()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.model = (
                torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14',
                               trust_repo='check', verbose=False)
                .to('cuda', memory_format=torch.channels_last, non_blocking=True)
                .eval()
            )
        if layers is None:
            layers = [0, 1, 2, 3, 4, 5, 6, 7]
        if weights is None:
            weights = (1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.1)
        if len(weights) != len(layers):
            raise ValueError('Number of layer weights must match number of layers.')
        self.layers = layers
        self.chns = [768] * len(self.layers)
        self.register_buffer('layer_weights',
                             torch.tensor(weights, dtype=torch.float32).view(-1, 1, 1))
        self.norm = norm
        if self.norm:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.norm:
            x = (x - self.mean) / self.std
        _, _, H, W = x.shape
        target_h = ((H + 13) // 14) * 14
        target_w = ((W + 13) // 14) * 14
        pad_h, pad_w = target_h - H, target_w - W
        if pad_h != 0 or pad_w != 0:
            x = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), mode='reflect')
        features = self.model.get_intermediate_layers(x, n=self.layers, reshape=True, return_class_token=False)
        return [feat * weight for feat, weight in zip(features, self.layer_weights, strict=True)]


@LOSS_REGISTRY.register()
class FDLLoss(nn.Module):
    """Fourier Distribution Loss (FDL).

    Adapted from: https://github.com/eezkni/FDL.

    Args:
        patch_size (int): Patch size for random projections. Default: 4.
        stride (int): Stride for projection convolution. Default: 1.
        num_proj (int): Number of random projections. Default: 24.
        model (str): Backbone network. Supported choices are 'vgg' | 'resnet' | 'effnet' | 'dinov2'.
            Default: 'dinov2'.
        vgg_weights (tuple, optional): Per-stage VGG weights. Default: None.
        dino_layers (list, optional): DINOv2 layer indices to extract. Default: None.
        dino_weights (tuple, optional): Per-layer DINOv2 weights. Default: None.
        dino_norm (bool): Normalize input for DINOv2. Default: False.
        phase_weight (float): Weight for the phase component. Default: 1.0.
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, patch_size=4, stride=1, num_proj=24, model='dinov2',
                 vgg_weights=None, dino_layers=None, dino_weights=None,
                 dino_norm=False, phase_weight=1.0, loss_weight=1.0):
        super(FDLLoss, self).__init__()
        self.model_name = model
        model = model.lower()

        if model == 'resnet':
            self.model = _ResNetBackbone()
        elif model == 'effnet':
            self.model = _EffNetBackbone()
        elif model == 'vgg':
            self.model = _VGGBackbone(vgg_weights=vgg_weights)
        elif model == 'dinov2':
            self.model = _DINOv2Backbone(dino_layers, dino_weights, dino_norm)
        else:
            raise NotImplementedError('Invalid model type. Supported: vgg | resnet | effnet | dinov2.')

        self.phase_weight = phase_weight
        self.loss_weight = loss_weight
        self.stride = stride

        for i in range(len(self.model.chns)):
            rand = torch.randn(num_proj, self.model.chns[i], patch_size, patch_size, device='cuda')
            rand = rand / rand.view(rand.shape[0], -1).norm(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            self.register_buffer(f'rand_{i}', rand)

    def _forward_once(self, x, y, idx):
        """
        Args:
            x (Tensor): Feature map of shape (N, C, H, W).
            y (Tensor): Feature map of shape (N, C, H, W).
            idx (int): Index of the random projection buffer.
        """
        rand = getattr(self, f'rand_{idx}')
        projx = F.conv2d(x, rand, stride=self.stride).reshape(x.shape[0], rand.shape[0], -1)
        projy = F.conv2d(y, rand, stride=self.stride).reshape(y.shape[0], rand.shape[0], -1)
        projx, _ = torch.sort(projx, dim=-1)
        projy, _ = torch.sort(projy, dim=-1)
        return torch.abs(projx - projy).mean([1, 2])

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(self, pred, target, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
        """
        feats_pred = self.model(pred)
        feats_target = self.model(target)
        score = []
        for i in range(len(feats_pred)):
            fft_x = torch.fft.fftn(feats_pred[i], dim=(-2, -1))
            fft_y = torch.fft.fftn(feats_target[i], dim=(-2, -1))
            s_amplitude = self._forward_once(torch.abs(fft_x), torch.abs(fft_y), i)
            s_phase = self._forward_once(torch.angle(fft_x), torch.angle(fft_y), i)
            score.append(s_amplitude + s_phase * self.phase_weight)
        score = sum(score)
        score = score.mean() * 0.01 if self.model_name != 'dinov2' else score.mean()
        return score * self.loss_weight


# ----------------------------- Focal Frequency loss ----------------------------- #

@LOSS_REGISTRY.register()
class FFLoss(nn.Module):
    """Focal Frequency Loss.

    From: https://github.com/EndlessSora/focal-frequency-loss.

    Args:
        loss_weight (float): Weight for focal frequency loss. Default: 1.0.
        alpha (float): Scaling factor alpha of the spectrum weight matrix. Default: 1.0.
        patch_factor (int): Factor to crop image patches for patch-based loss. Default: 1.
        ave_spectrum (bool): Whether to use minibatch average spectrum. Default: True.
        log_matrix (bool): Whether to adjust spectrum weight matrix by logarithm. Default: False.
        batch_matrix (bool): Whether to calculate spectrum weight matrix using batch statistics. Default: False.
    """

    def __init__(self, loss_weight=1.0, alpha=1.0, patch_factor=1,
                 ave_spectrum=True, log_matrix=False, batch_matrix=False):
        super(FFLoss, self).__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def _tensor2freq(self, x):
        if x.dtype is not torch.float32:
            x = x.to(torch.float32)
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0 and w % patch_factor == 0, \
            'Patch factor should be divisible by image height and width.'
        patch_h, patch_w = h // patch_factor, w // patch_factor
        patch_list = [
            x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w]
            for i in range(patch_factor) for j in range(patch_factor)
        ]
        y = torch.stack(patch_list, 1)
        freq = torch.fft.fft2(y, norm='ortho')
        return torch.stack([freq.real, freq.imag], -1)

    def _loss_formulation(self, recon_freq, real_freq, matrix=None):
        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            matrix_tmp = torch.sqrt((recon_freq - real_freq) ** 2 + 1e-12) ** self.alpha
            if self.log_matrix:
                matrix_tmp = torch.log1p(matrix_tmp)
            if self.batch_matrix:
                matrix_tmp /= matrix_tmp.max()
            else:
                matrix_tmp /= matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]
            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            weight_matrix = torch.clamp(matrix_tmp, min=0.0, max=1.0).clone().detach()

        assert weight_matrix.min().item() >= 0 and weight_matrix.max().item() <= 1, \
            'Spectrum weight matrix values should be in range [0, 1].'

        freq_distance = (recon_freq - real_freq) ** 2
        freq_distance = freq_distance[..., 0] + freq_distance[..., 1]
        return torch.mean(weight_matrix * freq_distance)

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(self, pred, target, matrix=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            matrix (Tensor, optional): Predefined spectrum weight matrix. Default: None.
        """
        pred_freq = self._tensor2freq(pred)
        target_freq = self._tensor2freq(target)

        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        loss = self._loss_formulation(pred_freq, target_freq, matrix) * 50
        return loss * self.loss_weight
