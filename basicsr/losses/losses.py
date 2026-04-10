import math
import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss
from torch import Tensor
import pywt
import numpy as np
from torchvision import models
from pathlib import Path
import basicsr.losses.SWT as SWT

# advance
from torchvision.transforms import GaussianBlur

_reduction_modes = ['none', 'mean', 'sum']


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

        loss = x_diff + y_diff

        return loss


# @LOSS_REGISTRY.register()
# class PerceptualLoss(nn.Module):
#     """Perceptual loss with commonly used style loss.

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

#     def __init__(self,
#                  layer_weights,
#                  vgg_type='vgg19',
#                  use_input_norm=True,
#                  range_norm=False,
#                  perceptual_weight=1.0,
#                  style_weight=0.,
#                  criterion='l1'):
#         super(PerceptualLoss, self).__init__()
#         self.perceptual_weight = perceptual_weight
#         self.style_weight = style_weight
#         self.layer_weights = layer_weights
#         self.vgg = VGGFeatureExtractor(
#             layer_name_list=list(layer_weights.keys()),
#             vgg_type=vgg_type,
#             use_input_norm=use_input_norm,
#             range_norm=range_norm)

#         self.criterion_type = criterion
#         if self.criterion_type == 'l1':
#             self.criterion = torch.nn.L1Loss()
#         elif self.criterion_type == 'l2':
#             self.criterion = torch.nn.L2loss()
#         elif self.criterion_type == 'fro':
#             self.criterion = None
#         else:
#             raise NotImplementedError(f'{criterion} criterion has not been supported.')

#     def forward(self, x, gt):
#         """Forward function.

#         Args:
#             x (Tensor): Input tensor with shape (n, c, h, w).
#             gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

#         Returns:
#             Tensor: Forward results.
#         """
#         # extract vgg features
#         x_features = self.vgg(x)
#         gt_features = self.vgg(gt.detach())

#         # calculate perceptual loss
#         if self.perceptual_weight > 0:
#             percep_loss = 0
#             for k in x_features.keys():
#                 if self.criterion_type == 'fro':
#                     percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
#                 else:
#                     percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
#             percep_loss *= self.perceptual_weight
#         else:
#             percep_loss = None

#         # calculate style loss
#         if self.style_weight > 0:
#             style_loss = 0
#             for k in x_features.keys():
#                 if self.criterion_type == 'fro':
#                     style_loss += torch.norm(
#                         self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
#                 else:
#                     style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
#                         gt_features[k])) * self.layer_weights[k]
#             style_loss *= self.style_weight
#         else:
#             style_loss = None

#         return percep_loss, style_loss

#     def _gram_mat(self, x):
#         """Calculate Gram matrix.

#         Args:
#             x (torch.Tensor): Tensor with shape of (n, c, h, w).

#         Returns:
#             torch.Tensor: Gram matrix.
#         """
#         n, c, h, w = x.size()
#         features = x.view(n, c, w * h)
#         features_t = features.transpose(1, 2)
#         gram = features.bmm(features_t) / (c * h * w)
#         return gram




# --------- Advance loss ----------

@LOSS_REGISTRY.register()
class SWTLoss(nn.Module):
    def __init__(self, loss_weight_ll=0.01, loss_weight_lh=0.01, loss_weight_hl=0.01, loss_weight_hh=0.01, reduction='mean'):
        super(SWTLoss, self).__init__()
        self.loss_weight_ll = loss_weight_ll
        self.loss_weight_lh = loss_weight_lh
        self.loss_weight_hl = loss_weight_hl
        self.loss_weight_hh = loss_weight_hh

        self.criterion = nn.L1Loss(reduction=reduction)

    def forward(self, pred, target):
        wavelet = pywt.Wavelet('sym19')
            
        dlo = wavelet.dec_lo
        an_lo = np.divide(dlo, sum(dlo))
        an_hi = wavelet.dec_hi
        rlo = wavelet.rec_lo
        syn_lo = 2*np.divide(rlo, sum(rlo))
        syn_hi = wavelet.rec_hi

        filters = pywt.Wavelet('wavelet_normalized', [an_lo, an_hi, syn_lo, syn_hi])
        sfm = SWT.SWTForward(1, filters, 'periodic').to("cuda")

        ## wavelet bands of sr image
        sr_img_y       = 16.0 + (pred[:,0:1,:,:]*65.481 + pred[:,1:2,:,:]*128.553 + pred[:,2:,:,:]*24.966)
        # sr_img_cb      = 128 + (-37.797 *pred[:,0:1,:,:] - 74.203 * pred[:,1:2,:,:] + 112.0* pred[:,2:,:,:])
        # sr_img_cr      = 128 + (112.0 *pred[:,0:1,:,:] - 93.786 * pred[:,1:2,:,:] - 18.214 * pred[:,2:,:,:])

        wavelet_sr  = sfm(sr_img_y)[0]

        LL_sr   = wavelet_sr[:,0:1, :, :]
        LH_sr   = wavelet_sr[:,1:2, :, :]
        HL_sr   = wavelet_sr[:,2:3, :, :]
        HH_sr   = wavelet_sr[:,3:, :, :]     

        ## wavelet bands of hr image
        hr_img_y       = 16.0 + (target[:,0:1,:,:]*65.481 + target[:,1:2,:,:]*128.553 + target[:,2:,:,:]*24.966)
        # hr_img_cb      = 128 + (-37.797 *target[:,0:1,:,:] - 74.203 * target[:,1:2,:,:] + 112.0* target[:,2:,:,:])
        # hr_img_cr      = 128 + (112.0 *target[:,0:1,:,:] - 93.786 * target[:,1:2,:,:] - 18.214 * target[:,2:,:,:])
     
        wavelet_hr     = sfm(hr_img_y)[0]

        LL_hr   = wavelet_hr[:,0:1, :, :]
        LH_hr   = wavelet_hr[:,1:2, :, :]
        HL_hr   = wavelet_hr[:,2:3, :, :]
        HH_hr   = wavelet_hr[:,3:, :, :]

        loss_subband_LL = self.loss_weight_ll * self.criterion(LL_sr, LL_hr)
        loss_subband_LH = self.loss_weight_lh * self.criterion(LH_sr, LH_hr)
        loss_subband_HL = self.loss_weight_hl * self.criterion(HL_sr, HL_hr)
        loss_subband_HH = self.loss_weight_hh * self.criterion(HH_sr, HH_hr)

        return loss_subband_LL + loss_subband_LH + loss_subband_HL + loss_subband_HH


# mssim_loss


class GaussianFilter2D(nn.Module):
    def _get_gaussian_window1d(self):
        sigma2 = self.sigma * self.sigma
        x = torch.arange(-(self.window_size // 2), self.window_size // 2 + 1)
        w = torch.exp(-0.5 * x**2 / sigma2)
        w = w / w.sum()
        return w.reshape(1, 1, 1, self.window_size)

    def _get_gaussian_window2d(self, gaussian_window_1d):
        return torch.matmul(
            gaussian_window_1d.transpose(dim0=-1, dim1=-2), gaussian_window_1d
        )

    def __init__(
        self,
        window_size: int = 11,
        in_channels: int = 3,
        sigma: float = 1.5,
        padding: int | None = None,
    ) -> None:
        """2D Gaussian Filer.

        Args:
        ----
            window_size (int, optional): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float, optional): The sigma of the gaussian filter. Defaults to 1.5.
            padding (int, optional): The padding of the gaussian filter. Defaults to None.
                If it is set to None, the filter will use window_size//2 as the padding. Another common setting is 0.

        """
        super().__init__()
        self.window_size = window_size
        if window_size % 2 != 1:
            msg = "Window size must be odd."
            raise ValueError(msg)
        self.padding = padding if padding is not None else window_size // 2
        self.sigma = sigma

        kernel = self._get_gaussian_window1d()
        kernel = self._get_gaussian_window2d(kernel)
        self.register_buffer(
            name="gaussian_window", tensor=kernel.repeat(in_channels, 1, 1, 1)
        )

    def forward(self, x):
        return F.conv2d(
            input=x,
            weight=self.gaussian_window,
            stride=1,
            padding=self.padding,
            groups=x.shape[1],
        )


@LOSS_REGISTRY.register()
class mssim_loss(nn.Module):
    def __init__(
        self,
        window_size: int = 11,
        in_channels: int = 3,
        sigma: float = 1.5,
        K1: float = 0.01,
        K2: float = 0.03,
        L: int = 1,
        padding: int | None = None,
        loss_weight: float = 1.0,
    ) -> None:
        """Adapted from 'A better pytorch-based implementation for the mean structural
            similarity. Differentiable simpler SSIM and MS-SSIM.':
                https://github.com/lartpang/mssim.pytorch.

            Calculate the mean SSIM (MSSIM) between two 4D tensors.

        Args:
        ----
            window_size (int): The window size of the gaussian filter. Defaults to 11.
            in_channels (int, optional): The number of channels of the 4d tensor. Defaults to False.
            sigma (float): The sigma of the gaussian filter. Defaults to 1.5.
            K1 (float): K1 of MSSIM. Defaults to 0.01.
            K2 (float): K2 of MSSIM. Defaults to 0.03.
            L (int): The dynamic range of the pixel values (255 for 8-bit grayscale images). Defaults to 1.
            padding (int, optional): The padding of the gaussian filter. Defaults to None. If it is set to None,
                the filter will use window_size//2 as the padding. Another common setting is 0.
            loss_weight (float): Weight of final loss value.

        """
        super().__init__()

        self.window_size = window_size
        self.C1 = (K1 * L) ** 2  # equ 7 in ref1
        self.C2 = (K2 * L) ** 2  # equ 7 in ref1
        self.loss_weight = loss_weight

        self.gaussian_filter = GaussianFilter2D(
            window_size=window_size,
            in_channels=in_channels,
            sigma=sigma,
            padding=padding,
        )

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, x, y):
        """x, y (Tensor): tensors of shape (N,C,H,W)
        Returns: Tensor.
        """
        assert x.shape == y.shape, f"x: {x.shape} and y: {y.shape} must be the same"
        assert x.ndim == y.ndim == 4, f"x: {x.ndim} and y: {y.ndim} must be 4"

        if x.type() != self.gaussian_filter.gaussian_window.type():
            x = x.type_as(self.gaussian_filter.gaussian_window)
        if y.type() != self.gaussian_filter.gaussian_window.type():
            y = y.type_as(self.gaussian_filter.gaussian_window)

        loss = 1.0 - self.msssim(x, y)

        return self.loss_weight * loss

    def _ssim(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        mu_x = self.gaussian_filter(x)  # equ 14
        mu_y = self.gaussian_filter(y)  # equ 14
        sigma2_x = self.gaussian_filter(x * x) - mu_x * mu_x  # equ 15
        sigma2_y = self.gaussian_filter(y * y) - mu_y * mu_y  # equ 15
        sigma_xy = self.gaussian_filter(x * y) - mu_x * mu_y  # equ 16

        A1 = 2 * mu_x * mu_y + self.C1
        A2 = 2 * sigma_xy + self.C2
        B1 = mu_x.pow(2) + mu_y.pow(2) + self.C1
        B2 = sigma2_x + sigma2_y + self.C2

        # equ 12, 13 in ref1
        l1 = A1 / B1
        cs = A2 / B2
        ssim = l1 * cs

        return ssim, cs

    def msssim(self, x: Tensor, y: Tensor) -> Tensor:
        ms_components = []
        for i, w in enumerate((0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
            ssim, cs = self._ssim(x, y)
            ssim = ssim.mean()
            cs = cs.mean()

            if i == 4:
                ms_components.append(ssim**w)
            else:
                ms_components.append(cs**w)
                padding = [s % 2 for s in x.shape[2:]]  # spatial padding
                x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=padding)
                y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=padding)

        return math.prod(ms_components)  # equ 7 in ref2


#ldl_loss

@LOSS_REGISTRY.register()
class ldl_loss(nn.Module):
    """LDL loss. Adapted from 'Details or Artifacts: A Locally Discriminative
    Learning Approach to Realistic Image Super-Resolution':
    https://arxiv.org/abs/2203.09195.

    Args:
    ----
        criterion (str): loss type. Default: 'huber'
        loss_weight (float): weight for colorloss. Default: 1.0
        ksize (int): size of the local window. Default: 7

    """

    def __init__(
        self, criterion: str = "l1", loss_weight: float = 1.0, ksize: int = 7
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.ksize = ksize
        self.criterion_type = criterion
        self.criterion: nn.L1Loss | nn.MSELoss | nn.HuberLoss

        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "l2":
            self.criterion = nn.MSELoss()
        elif self.criterion_type == "huber":
            self.criterion = nn.HuberLoss()
        else:
            msg = f"{criterion} criterion has not been supported."
            raise NotImplementedError(msg)

    def get_local_weights(self, residual):
        """Get local weights for generating the artifact map of LDL.

        It is only called by the `get_refined_artifact_map` function.

        Args:
        ----
            residual (Tensor): Residual between predicted and ground truth images.

        Returns:
        -------
            Tensor: weight for each pixel to be discriminated as an artifact pixel

        """
        pad = (self.ksize - 1) // 2
        residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode="reflect")

        unfolded_residual = residual_pad.unfold(2, self.ksize, 1).unfold(
            3, self.ksize, 1
        )
        return (
            torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True)
            .squeeze(-1)
            .squeeze(-1)
        )

    def get_refined_artifact_map(self, img_gt, img_output) -> Tensor:
        """Calculate the artifact map of LDL
        (Details or Artifacts: A Locally Discriminative Learning Approach to Realistic Image Super-Resolution. In CVPR 2022).

        Args:
        ----
            img_gt (Tensor): ground truth images.
            img_output (Tensor): output images given by the optimizing model.

        Returns:
        -------
            overall_weight: weight for each pixel to be discriminated as an artifact pixel
            (calculated based on both local and global observations).

        """
        residual_sr = torch.sum(torch.abs(img_gt - img_output), 1, keepdim=True)

        patch_level_weight = torch.var(
            residual_sr.clone(), dim=(-1, -2, -3), keepdim=True
        ) ** (1 / 5)
        pixel_level_weight = self.get_local_weights(residual_sr.clone())
        return patch_level_weight * pixel_level_weight

    def forward(self, net_output, gt):
        overall_weight = self.get_refined_artifact_map(gt, net_output)
        self.output = torch.mul(overall_weight, net_output)
        self.gt = torch.mul(overall_weight, gt)

        return self.criterion(self.output, self.gt) * self.loss_weight



@LOSS_REGISTRY.register()
class chc_loss(nn.Module):
    """Clipped pseudo-Huber with Cosine Similarity Loss.

       For reference on research, see:
       https://github.com/HolmesShuan/AIM2020-Real-Super-Resolution
       https://github.com/dmarnerides/hdr-expandnet

    Args:
    ----
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        criterion (str): Specifies the loss to apply.
            Supported choices are 'l1' and 'huber'. Default: 'huber'.
        loss_lambda (float):  constant factor that adjusts the contribution of the cosine similarity term
        clip_min (float): threshold that sets the gradients of well-trained pixels to zeros
        clip_max (float): max clip limit, can act as a noise filter

    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction: str = "mean",
        criterion: str = "huber",
        loss_lambda: float = 0,
        clip_min: float = 0.003921,
        clip_max: float = 0.996078,
    ) -> None:
        super().__init__()

        if reduction not in {"none", "mean", "sum"}:
            msg = f"Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}"
            raise ValueError(msg)

        # Loss params
        self.loss_weight = loss_weight
        self.criterion = criterion

        # CoSim
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-20)
        self.loss_lambda = loss_lambda  # 5/255 = 0.019607

        # Clip
        self.clip_min = clip_min  # 1/255 = 0.03921
        self.clip_max = clip_max

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        **kwargs,  # noqa: ARG002
    ) -> Tensor:
        """Args:
        ----
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.

        """
        cosine_term = (1 - self.similarity(pred, target)).mean()

        # absolute mean
        if self.criterion == "l1":
            loss = torch.mean(
                torch.clamp(
                    (torch.abs(pred - target) + self.loss_lambda * cosine_term),
                    self.clip_min,
                    self.clip_max,
                )
            )
        # pseudo-huber (charbonnier)
        elif self.criterion == "huber":
            loss = torch.mean(
                torch.clamp(
                    (
                        torch.sqrt((pred - target) ** 2 + 1e-12)
                        + self.loss_lambda * cosine_term
                    ),
                    self.clip_min,
                    self.clip_max,
                )
            )
        else:
            msg = f"{self.criterion} not implemented."
            raise NotImplementedError(msg)

        return self.loss_weight * loss
        
# consistency_loss

@LOSS_REGISTRY.register()
class consistency_loss(nn.Module):
    """Color and Luma Consistency loss using Oklab and CIE L*.

    Args:
    ----
        criterion (str): loss type. Default: 'huber'
        avgpool (bool): apply downscaling after conversion. Default: False
        scale (int): value used by avgpool. Default: 4
        loss_weight (float): weight for colorloss. Default: 1.0

    """

    def __init__(
        self,
        criterion: str = "chc",
        blur: bool = True,
        cosim: bool = True,
        saturation: float = 1.0,
        brightness: float = 1.0,
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_blur = blur
        self.cosim = cosim
        self.saturation = saturation
        self.brightness = brightness
        self.loss_weight = loss_weight
        self.mean = torch.tensor((0.5, 0.5)).view(1, 2, 1, 1)
        self.similarity = nn.CosineSimilarity(dim=1, eps=1e-20)

        if self.use_blur:
            self.blur = GaussianBlur(21, 3)

        self.criterion_type = criterion
        self.criterion: nn.L1Loss | nn.HuberLoss | Callable

        if self.criterion_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.criterion_type == "chc":
            self.criterion = chc_loss(loss_lambda=0, clip_min=0, clip_max=1)  # type: ignore[reportCallIssue]
        else:
            msg = f"{criterion} criterion has not been supported."
            raise NotImplementedError(msg)

    def lin_rgb(self, img):
        """Transforms sRGB gamma 2.4 to linear

        Args:
            img: Tensor (B,C,H,W).
        Returns:
            Tensor (B,C,H,W).

        """
        return torch.where(
            img <= 0.04045, img / 12.92, torch.pow((img + 0.055) / 1.055, 2.4)
        )

    def rgb_to_oklab_chroma(self, img):
        """RGB to Oklab chroma

        Args:
            img: Tensor (B,3,H,W).
        Returns:
            Tensor (B,2,H,W).

        """
        if not isinstance(img, torch.Tensor):
            msg = f"Input type is not a Tensor. Got {type(img)}"
            raise TypeError(msg)
        if len(img.shape) < 3 or img.shape[-3] != 3:
            msg = f"Input size must have a shape of (*, 3, H, W). Got {img.shape}"
            raise ValueError(msg)

        # linearize rgb
        img = self.lin_rgb(img)

        # separate into R, G, B
        r = img[:, 0, :, :]
        g = img[:, 1, :, :]
        b = img[:, 2, :, :]

        # to oklab
        l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
        m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
        s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

        l_ = l.sign() * l.abs().pow(1 / 3)
        m_ = m.sign() * m.abs().pow(1 / 3)
        s_ = s.sign() * s.abs().pow(1 / 3)

        l = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

        # stacked chroma
        return torch.stack([a, b], dim=1)

    def rgb_to_l_star(self, img):
        """RGB to CIELAB L*

        Args:
            img: Tensor (B,C,H,W).
        Returns:
            Tensor (B,H,W)

        """
        if not isinstance(img, torch.Tensor):
            msg = f"Input type is not a Tensor. Got {type(img)}"
            raise TypeError(msg)

        # permute
        img = img.permute(0, 2, 3, 1)

        # linearize rgb
        img = self.lin_rgb(img)

        # convert to luma - Y axis of sRGB > XYZ standard
        img = img @ torch.tensor([0.2126, 0.7152, 0.0722]).to(img.device)

        # convert Y to L* (from CIELAB L*a*b*)
        # NOTE: will convert from range [0, 1] to range [0, 100]
        img = torch.where(
            img <= (216 / 24389),
            img * (img * (24389 / 27)),
            # torch workaround for cube-root in negative numbers
            img.sign() * img.abs().pow(1 / 3) * 116 - 16,
        )

        # normalize to [0, 1] range again and clamp
        return torch.clamp(img / 100, 0, 1)

    def forward(self, net_output, gt):
        """
        Args:
            net_output: Tensor. Generator output.
            gt: Tensor. Generator output.
        Returns:
            float.
        """

        # clamp
        net_output = torch.clamp(net_output, 1 / 255, 1)
        gt = torch.clamp(gt, 1 / 255, 1)

        # luma
        if self.use_blur:
            net_output_blur = torch.clamp(self.blur(net_output), 0, 1)
            gt_blur = torch.clamp(self.blur(gt), 0, 1)
            input_luma = self.rgb_to_l_star(net_output_blur)
            target_luma = self.rgb_to_l_star(gt_blur) * self.brightness
        else:
            input_luma = self.rgb_to_l_star(net_output)
            target_luma = self.rgb_to_l_star(gt) * self.brightness

        # chroma
        input_chroma = self.rgb_to_oklab_chroma(net_output)
        target_chroma = self.rgb_to_oklab_chroma(gt) * self.saturation

        # clip and normalize
        self.mean = self.mean.to(input_chroma.device)
        input_chroma = torch.clamp((input_chroma + self.mean * 1), 0, 1)
        target_chroma = torch.clamp((target_chroma + self.mean * 1), 0, 1)

        # loss formulation
        loss = self.criterion(input_luma, target_luma) + self.criterion(
            input_chroma, target_chroma
        )

        if self.cosim:
            # cosine-similarity
            cosim_chroma = 1 - self.similarity(input_chroma, target_chroma).mean()
            cosim_luma = 1 - self.similarity(input_luma, target_luma).mean()
            # hardcoded lambda for now, as values above 0.5 causes instability
            cosim = (0.5 * cosim_chroma) + (0.5 * cosim_luma)
            # set threshold to avoid instability on early iters
            if cosim < 1e-3:
                loss = loss + cosim

        return loss * self.loss_weight


class L2pooling(nn.Module):
    def __init__(
        self,
        filter_size: int = 5,
        stride: int = 2,
        channels: int = 64,
        as_loss: bool = True,
    ) -> None:
        super().__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer(
            "filter", g[None, None, :, :].repeat((self.channels, 1, 1, 1))
        )

        if as_loss is False:
            # send to cuda
            self.filter = self.filter.cuda()

    def forward(self, x):
        x = x**2
        out = F.conv2d(
            x, self.filter, stride=self.stride, padding=self.padding, groups=x.shape[1]
        )
        return (out + 1e-12).sqrt()


@LOSS_REGISTRY.register()
class dists_loss(nn.Module):
    r"""DISTS. "Image Quality Assessment: Unifying Structure and Texture Similarity":
    https://arxiv.org/abs/2004.07728.

    Args:
    ----
        as_loss (bool): True to use as loss, False for metric.
            Default: True.
        loss_weight (float).
            Default: 1.0.
        load_weights (bool): loads pretrained weights for DISTS.
            Default: False.

    """

    def __init__(
        self,
        as_loss: bool = True,
        loss_weight: float = 1.0,
        load_weights: bool = True,
        **kwargs,  # noqa: ARG002
    ) -> None:
        super().__init__()
        self.as_loss = as_loss
        self.loss_weight = loss_weight

        vgg_pretrained_features = models.vgg16(weights="DEFAULT").features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])  # type: ignore[reportIndexIssue]
        self.stage2.add_module(str(4), L2pooling(channels=64, as_loss=as_loss))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])  # type: ignore[reportIndexIssue]
        self.stage3.add_module(str(9), L2pooling(channels=128, as_loss=as_loss))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])  # type: ignore[reportIndexIssue]
        self.stage4.add_module(str(16), L2pooling(channels=256, as_loss=as_loss))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])  # type: ignore[reportIndexIssue]
        self.stage5.add_module(str(23), L2pooling(channels=512, as_loss=as_loss))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])  # type: ignore[reportIndexIssue]

        for param in self.parameters():
            param.requires_grad = False

        self.chns = [3, 64, 128, 256, 512, 512]
        self.register_parameter(
            "alpha", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1))
        )
        self.register_parameter(
            "beta", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1))
        )
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)

        # load dists weights
        if load_weights:
            model_path = Path(__file__).parent / "dists_weights.pth"
            try:
                if not model_path.exists():
                    url = "https://huggingface.co/neosr/models/resolve/main/dists_weights.pth?download=true"
                    request.urlretrieve(url, model_path)  # noqa: S310
            except:
                msg = "Could not download TOPIQ weights."
                raise ValueError(msg)

            weights = torch.load(
                model_path, map_location=torch.device("cuda"), weights_only=True
            )

            if weights is not None:
                self.alpha.data = weights["alpha"]
                self.beta.data = weights["beta"]

            if as_loss is False:
                # send to cuda
                self.alpha.data = self.alpha.data.cuda()
                self.beta.data = self.beta.data.cuda()

    def forward_once(self, x):
        h = x
        h = self.stage1(h)
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

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, x, y):
        feats0 = self.forward_once(x)
        feats1 = self.forward_once(y)
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6

        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean(
                [2, 3], keepdim=True
            ) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)

        if self.as_loss:
            out = 1 - (dist1 + dist2).mean()  # type: ignore[attr-defined]
            out *= self.loss_weight
        else:
            out = 1 - (dist1 + dist2).squeeze()  # type: ignore[attr-defined]

        return out

#--- fdl loss----
from torchvision.models import (
    EfficientNet_B7_Weights,
    ResNet101_Weights,
    VGG19_Weights,
    efficientnet_b7,
    resnet101,
    vgg,
)


class VGG(nn.Module):
    def __init__(self, requires_grad=False, vgg_weights=None):
        super().__init__()
        vgg_pretrained_features = vgg.vgg19(weights=VGG19_Weights.DEFAULT).features
        vgg_pretrained_features.eval()

        if vgg_weights is None:
            self.vgg_weights = (0.5, 0.5, 1.0, 1.0, 1.0)
        else:
            self.vgg_weights = vgg_weights

        self.stage1 = nn.Sequential()
        self.stage2 = nn.Sequential()
        self.stage3 = nn.Sequential()
        self.stage4 = nn.Sequential()
        self.stage5 = nn.Sequential()

        # vgg19
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

        self.chns = [64, 128, 256, 512, 512]

    def get_features(self, x):
        # normalize the data
        h = (x - self.mean) / self.std

        h = self.stage1(h)
        h_relu1_2 = h * self.vgg_weights[0]

        h = self.stage2(h)
        h_relu2_2 = h * self.vgg_weights[1]

        h = self.stage3(h)
        h_relu3_3 = h * self.vgg_weights[2]

        h = self.stage4(h)
        h_relu4_3 = h * self.vgg_weights[3]

        h = self.stage5(h)
        h_relu5_3 = h * self.vgg_weights[4]

        # get the features of each layer
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x):
        return self.get_features(x)


class ResNet(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()

        model = resnet101(weights=ResNet101_Weights.DEFAULT)
        model.eval()

        self.stage1 = nn.Sequential(model.conv1, model.bn1, model.relu)
        self.stage2 = nn.Sequential(model.maxpool, model.layer1)
        self.stage3 = nn.Sequential(model.layer2)
        self.stage4 = nn.Sequential(model.layer3)
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

        self.chns = [64, 256, 512, 1024]

    def get_features(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]

    def forward(self, x):
        return self.get_features(x)


class EffNet(nn.Module):
    def __init__(self):
        super().__init__()
        model = efficientnet_b7(
            weights=EfficientNet_B7_Weights.DEFAULT
        ).features  # [:6]
        model.eval()

        self.stage1 = model[0:2]
        self.stage2 = model[2]
        self.stage3 = model[3]
        self.stage4 = model[4]
        self.stage5 = model[5]

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        )

        for param in self.parameters():
            param.requires_grad = False
        self.chns = [32, 48, 80, 160, 224]

    def get_features(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x):
        return self.get_features(x)

import warnings
class dinov2(nn.Module):
    """
    DINOv2 backend, developed by musl from the neosr-project: https://github.com/neosr-project/neosr
    """

    def __init__(self, layers=None, weights=None, norm=False):
        super().__init__()

        # load model and suppress xformers dependency warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = (
                torch.hub.load(
                    "facebookresearch/dinov2",
                    "dinov2_vitb14",
                    trust_repo="check",
                    verbose=False,
                )
                .to("cuda", memory_format=torch.channels_last, non_blocking=True)
                .eval()
            )

        if layers is None:
            layers = [0, 1, 2, 3, 4, 5, 6, 7]
        if weights is None:
            weights = (1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 0.1)

        self.layers = layers
        self.chns = [768] * len(self.layers)

        if len(weights) != len(self.layers):
            msg = "Number of layer weights must match number of layers"
            raise ValueError(msg)

        self.register_buffer(
            "layer_weights", torch.tensor(weights, dtype=torch.float32).view(-1, 1, 1)
        )

        self.norm = norm
        if self.norm:
            # imagenet norm values
            self.register_buffer(
                "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
            )
            self.register_buffer(
                "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
            )

        for param in self.parameters():
            param.requires_grad = False

    def adapt_size(self, dim):
        return ((dim + 13) // 14) * 14

    def get_features(self, x):
        if self.norm:
            x = (x - self.mean) / self.std
        # pad because embedded patch expects multiples of 14
        _, _, H, W = x.shape
        target_h = self.adapt_size(H)
        target_w = self.adapt_size(W)
        pad_h = target_h - H
        pad_w = target_w - W

        if pad_h != 0 or pad_w != 0:
            x = F.pad(
                x,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                mode="reflect",
            )

        # extract features
        features = self.model.get_intermediate_layers(
            x, n=self.layers, reshape=True, return_class_token=False
        )
        return [
            feat * weight
            for feat, weight in zip(features, self.layer_weights, strict=True)
        ]

    def forward(self, x):
        return self.get_features(x)


@LOSS_REGISTRY.register()
class fdl_loss(nn.Module):
    """
    Adapted from: https://github.com/eezkni/FDL
    """

    def __init__(
        self,
        patch_size=4,
        stride=1,
        num_proj=24,
        model="dinov2",
        vgg_weights=None,
        dino_layers=None,
        dino_weights=None,
        dino_norm=False,
        phase_weight=1.0,
        loss_weight=1.0,
    ):
        super().__init__()
        self.model_name = model
        model = model.lower()

        if model == "resnet":
            self.model = ResNet()
        elif model == "effnet":
            self.model = EffNet()
        elif model == "vgg":
            self.model = VGG(vgg_weights=vgg_weights)
        elif model == "dinov2":
            self.model = dinov2(dino_layers, dino_weights, dino_norm)
        else:
            msg = "Invalid model type! Valid models: VGG, EffNet, ResNet or DINOv2"
            raise NotImplementedError(msg)

        self.phase_weight = phase_weight
        self.loss_weight = loss_weight
        self.stride = stride

        for i in range(len(self.model.chns)):
            rand = torch.randn(
                num_proj, self.model.chns[i], patch_size, patch_size, device="cuda"
            )
            rand = rand / rand.view(rand.shape[0], -1).norm(dim=1).unsqueeze(
                1
            ).unsqueeze(2).unsqueeze(3)
            self.register_buffer(f"rand_{i}", rand)

    def forward_once(self, x, y, idx):
        """
        x, y: input image tensors with the shape of (N, C, H, W)
        """
        rand = getattr(self, f"rand_{idx}")
        projx = F.conv2d(x, rand, stride=self.stride)
        projx = projx.reshape(projx.shape[0], projx.shape[1], -1)
        projy = F.conv2d(y, rand, stride=self.stride)
        projy = projy.reshape(projy.shape[0], projy.shape[1], -1)

        # sort the convolved input
        projx, _ = torch.sort(projx, dim=-1)
        projy, _ = torch.sort(projy, dim=-1)

        # compute the mean of the sorted convolved input
        return torch.abs(projx - projy).mean([1, 2])

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(self, x, y):
        x = self.model(x)
        y = self.model(y)
        score = []
        for i in range(len(x)):
            # Transform to Fourier Space
            fft_x = torch.fft.fftn(x[i], dim=(-2, -1))
            fft_y = torch.fft.fftn(y[i], dim=(-2, -1))

            # get the magnitude and phase of the extracted features
            x_mag = torch.abs(fft_x)
            x_phase = torch.angle(fft_x)
            y_mag = torch.abs(fft_y)
            y_phase = torch.angle(fft_y)

            s_amplitude = self.forward_once(x_mag, y_mag, i)
            s_phase = self.forward_once(x_phase, y_phase, i)

            score.append(s_amplitude + s_phase * self.phase_weight)

        score = sum(score)
        # decrease magnitude to balance with other losses
        score = score.mean() * 0.01 if self.model_name != "dinov2" else score.mean()
        return score * self.loss_weight


@LOSS_REGISTRY.register()
class ff_loss(nn.Module):
    """Focal Frequency Loss.
       From: https://github.com/EndlessSora/focal-frequency-loss.

    Args:
    ----
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False

    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        alpha: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = True,
        log_matrix: bool = False,
        batch_matrix: bool = False,
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x: Tensor) -> Tensor:
        # for amp dtype
        if x.dtype is not torch.float32:
            x = x.to(torch.float32)

        # crop image patches
        patch_factor = self.patch_factor
        _, _, h, w = x.shape
        assert h % patch_factor == 0, (
            "Patch factor should be divisible by image height and width"
        )
        assert w % patch_factor == 0, (
            "Patch factor should be divisible by image height and width"
        )
        patch_list = []
        patch_h = h // patch_factor
        patch_w = w // patch_factor

        patch_list.extend([
            x[:, :, i * patch_h : (i + 1) * patch_h, j * patch_w : (j + 1) * patch_w]
            for i in range(patch_factor)
            for j in range(patch_factor)
        ])

        # stack to patch tensor
        y = torch.stack(patch_list, 1)

        # perform 2D DFT (real-to-complex, orthonormalization)
        freq = torch.fft.fft2(y, norm="ortho")
        return torch.stack([freq.real, freq.imag], -1)

    def loss_formulation(
        self, recon_freq: Tensor, real_freq: Tensor, matrix: Tensor | None = None
    ) -> Tensor:
        # spectrum weight matrix
        if matrix is not None:
            # if the matrix is predefined
            weight_matrix = matrix.detach()
        else:
            # if the matrix is calculated online: continuous, dynamic, based on current Euclidean distance
            matrix_tmp = (recon_freq - real_freq) ** 2
            matrix_tmp = (
                torch.sqrt(matrix_tmp[..., 0] + matrix_tmp[..., 1]) ** self.alpha
            )

            # whether to adjust the spectrum weight matrix by logarithm
            if self.log_matrix:
                matrix_tmp = torch.log1p(matrix_tmp)

            # whether to calculate the spectrum weight matrix using batch-based statistics
            if self.batch_matrix:
                matrix_tmp /= matrix_tmp.max()
            else:
                matrix_tmp /= (
                    matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None]
                )

            matrix_tmp[torch.isnan(matrix_tmp)] = 0.0
            matrix_tmp = torch.clamp(matrix_tmp, min=0.0, max=1.0)
            weight_matrix = matrix_tmp.clone().detach()

        assert weight_matrix.min().item() >= 0, (
            "The values of spectrum weight matrix should be in the range [0, 1]"
        )
        assert weight_matrix.max().item() <= 1, (
            "The values of spectrum weight matrix should be in the range [0, 1]"
        )

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        freq_distance = tmp[..., 0] + tmp[..., 1]

        # dynamic spectrum weighting (Hadamard product)
        loss = weight_matrix * freq_distance
        return torch.mean(loss)

    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        matrix: Tensor | None = None,
        **kwargs,  # noqa: ARG002
    ) -> Tensor:
        """Forward function to calculate focal frequency loss.

        Args:
        ----
            pred (torch.Tensor): of shape (N, C, H, W). Predicted tensor.
            target (torch.Tensor): of shape (N, C, H, W). Target tensor.
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Default: None (If set to None: calculated online, dynamic).

        """
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)

        # whether to use minibatch average spectrum
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, 0, keepdim=True)
            target_freq = torch.mean(target_freq, 0, keepdim=True)

        # calculate focal frequency loss
        # lambda 50 to increase weight compared to other losses
        loss = self.loss_formulation(pred_freq, target_freq, matrix) * 50

        return loss * self.loss_weight
