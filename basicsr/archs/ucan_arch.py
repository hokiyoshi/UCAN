import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from torch.nn.attention import SDPBackend, sdpa_kernel

def has_nan_or_inf(tensor: torch.Tensor) -> bool:
    """Returns True if tensor contains any NaN or ±Inf."""
    return not torch.isfinite(tensor).all()

# --------------------
class HedgehogFeatureMap(nn.Module):

    r"""
    Hedgehog feature map as introduced in
    `The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry <https://arxiv.org/abs/2402.04347>`_
    """

    def __init__(
        self,
        head_dim: int
    ):
        super().__init__()
        # Trainable map
        self.layer = nn.Linear(head_dim, head_dim)
        self.init_weights_()

    def init_weights_(self):
        """Initialize trainable map as identity"""
        with torch.no_grad():
            identity = torch.eye(*self.layer.weight.shape[-2:], dtype=torch.float)
            self.layer.weight.copy_(identity.to(self.layer.weight))
        nn.init.zeros_(self.layer.bias)

    def forward(self, x: torch.Tensor):
        x = self.layer(x)  # shape b, h, l, d
        return torch.cat([2*x, -2*x], dim=-1).softmax(-1)
    
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        rope_theta: float           = 10_000.0,
        max_sequence_length: int    = 8192,
        fope: bool                  = False,     # filter out too-high freqs if True
    ):
        super().__init__()
        # store hyper-params exactly once
        self.d_model             = d_model
        self.n_heads             = n_heads
        self.rope_theta          = rope_theta
        self.max_sequence_length = max_sequence_length
        self.fope                = fope

        # per-head dimension
        self.dim       = d_model // n_heads
        self.inv_freq  = self.get_inv_freq(self.dim)

    # --------------------------------------------------------------------- helpers
    def get_inv_freq(self, dim: int) -> torch.Tensor:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
        )
        if self.fope:                                   # optional FOPE trimming
            inv_freq[inv_freq < 2 * torch.pi / self.max_sequence_length] = 0
            inv_freq = inv_freq[inv_freq != 0.0]
        return inv_freq[None, :] # shape: (1, dim//2)

    # --------------------------------------------------------------------- rotary core
    def get_rotary_embedding(self, seq_len: int):
        device = self.inv_freq.device
        with torch.autocast(device.type, enabled=False):
            seq   = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("t,hd -> htd", seq, self.inv_freq)     # [L, D/2]

            if self.fope:
                positions = freqs.unsqueeze(0)                        # [1,L,D/2]
            else:                                                     # duplicate
                positions = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
            return positions.sin(), positions.cos()

    @staticmethod
    def rotate_half(x):
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin, pos_cos, t):
        return ((t * pos_cos) - (self.rotate_half(t) * pos_sin)).to(t.dtype)

    # --------------------------------------------------------------------- forward
    def forward(self, x, all_len):
        """
        Parameters
        ----------
        x       : [B, n_heads, T, head_dim]
        all_len : full sequence length in the current context/window
        """
        with torch.autocast(x.device.type, enabled=False):
            x_len = x.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(all_len)
            pos_sin, pos_cos = pos_sin.type_as(x), pos_cos.type_as(x)

            x = self.apply_rotary_pos_embed(
                pos_sin[..., all_len - x_len : all_len, :],
                pos_cos[..., all_len - x_len : all_len, :],
                x,
            )
            return x


# ---------------------------------------------------------------------------
# Fourier Embedding (hyper-parameter version of the sample) -----------------
# ---------------------------------------------------------------------------
class FourierEmbedding(RotaryEmbedding):
    """
    RotaryEmbedding + learnable Fourier mixing.

    Setting `learnable=True` makes the sin/cos coefficient tensors trainable.
    """
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        # rotary hyper-params
        rope_theta: float        = 10_000,
        max_sequence_length: int = 4096,
        fope: bool               = True,
        # fourier-specific
        rope_fourier_init_norm_gain: float = 0.3,
        learnable: bool           = True,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            rope_theta=rope_theta,
            max_sequence_length=max_sequence_length,
            fope=fope,
        )

        self.head_dim = d_model // n_heads
        self.input_dim  = self.inv_freq.size(-1)
        self.output_dim = self.input_dim if self.input_dim <= self.head_dim//4 else self.head_dim//4

        # sin & cos coefficient tensors ( n_heads × in_dim × out_dim )
        self.sin_coef = nn.Parameter(
            torch.randn(n_heads, self.input_dim, self.output_dim),
            requires_grad=learnable,
        )
        self.cos_coef = nn.Parameter(
            torch.randn(n_heads, self.input_dim, self.output_dim),
            requires_grad=learnable,
        )

        # Xavier init with given gain
        torch.nn.init.xavier_normal_(self.sin_coef, gain=rope_fourier_init_norm_gain)
        torch.nn.init.xavier_normal_(self.cos_coef, gain=rope_fourier_init_norm_gain)

        with torch.no_grad():
            # add identity so first iteration reproduces classic RoPE
            if self.input_dim == self.output_dim:    
                self.sin_coef += torch.eye(self.input_dim, device=self.sin_coef.device)
                self.cos_coef += torch.eye(self.input_dim, device=self.cos_coef.device)
            else:
                self.sin_coef += self.get_step_eye(self.sin_coef)
                self.cos_coef += self.get_step_eye(self.cos_coef)

    def get_step_eye(self, _param):
        _param = torch.zeros_like(_param)
        
        step = math.ceil(self.input_dim / self.output_dim)
        for i in range(self.output_dim):
            if i*step < self.input_dim:
                _param[..., i*step, i] = 1.0
        
        return _param
    # --------------------------------------------------------------------- override mix
    def apply_rotary_pos_embed(self, pos_sin, pos_cos, t):
        # Fourier mix (ℓ1-normalized along in_dim, exactly like the sample)
        fourier_sin = torch.einsum(
            "bhtD, hDd -> bhtd",
            pos_sin,
            self.sin_coef / self.sin_coef.sum(dim=-2, keepdim=True),
        )
        fourier_cos = torch.einsum(
            "bhtD, hDd -> bhtd",
            pos_cos,
            self.cos_coef / self.cos_coef.sum(dim=-2, keepdim=True),
        )

        # pad back to head_dim // 2 if we reduced dimensionality
        fourier_sin = F.pad(
            fourier_sin,
            pad=(0, self.dim // 2 - fourier_sin.size(-1)),
            mode="constant",
            value=1,
        )
        fourier_cos = F.pad(
            fourier_cos,
            pad=(0, self.dim // 2 - fourier_cos.size(-1)),
            mode="constant",
            value=1,
        )

        # duplicate to full D and call base apply
        fourier_sin = torch.cat((fourier_sin, fourier_sin), dim=-1)
        fourier_cos = torch.cat((fourier_cos, fourier_cos), dim=-1)
        return super().apply_rotary_pos_emb(fourier_sin, fourier_cos, t)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y
        
class MBConv(nn.Module):
    def __init__(
            self,
        dim_in,
        dim_out,
        expansion_rate = 1,
        shrinkage_rate = 0.25,
        dropout = 0.):

        super().__init__()
        hidden_dim = int(expansion_rate * dim_out)
        stride = 1

        self.net = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, 1),
            # nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
            # nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            ChannelAttention(hidden_dim, squeeze_factor=4),
            nn.Conv2d(hidden_dim, dim_out, 1),
            # nn.BatchNorm2d(dim_out)
        )

    def forward(self, x, x_size):
        shortcut = x
        B, L, C = x.shape
        H, W = x_size
        x = x.permute(0, 2, 1).contiguous().view(B, C, H, W)
        return self.net(x).contiguous().view(B, -1, H*W).permute(0,2,1) + shortcut
        
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # 结构为 [B, num_patches, C]
        if self.norm is not None:
            x = self.norm(x)  # 归一化
        return x


    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops
        
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    输入:
        img_size (int): 图像的大小，默认为 224*224.
        patch_size (int): Patch token 的大小，默认为 4*4.
        in_chans (int): 输入图像的通道数，默认为 3.
        embed_dim (int): 线性 projection 输出的通道数，默认为 96.
        norm_layer (nn.Module, optional): 归一化层， 默认为N None.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)  # 图像的大小，默认为 224*224
        patch_size = to_2tuple(patch_size)  # Patch token 的大小，默认为 4*4
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]  # patch 的分辨率
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]  # patch 的个数，num_patches

        self.in_chans = in_chans  # 输入图像的通道数
        self.embed_dim = embed_dim  # 线性 projection 输出的通道数

    def forward(self, x, x_size):
        B, HW, C = x.shape  # 输入 x 的结构
        x = x.transpose(1, 2).view(B, -1, x_size[0], x_size[1])  # 输出结构为 [B, Ph*Pw, C]
        return x



class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)



class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by 
    `Residual Feature Aggregation Network for Image Super-Resolution`
    Note: `conv_max` and `conv3_` are NOT used here, so the corresponding codes
    are deleted.
    """

    def __init__(self, esa_channels, n_feats, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m
        
class SpatialGate(nn.Module):
    """ Spatial-Gate.
    Args:
        dim (int): Half of input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim) # DW Conv

    def forward(self, x, H, W):
        # Split
        x1, x2 = x.chunk(2, dim = -1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C//2, H, W)).flatten(2).transpose(-1, -2).contiguous()

        return F.gelu(x1) * x2

class SGFN(nn.Module):
    """ Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features//2)
        self.fc2 = nn.Linear(hidden_features//2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
            """
            Input: x: (B, H*W, C), H, W
            Output: x: (B, H*W, C)
            """
            H, W = x_size
            x = self.fc1(x)
            #x = self.act(x)
            x = self.drop(x)

            x = self.sg(x, H, W)
            x = self.drop(x)

            x = self.fc2(x)
            x = self.drop(x)
            return x


class LKSA(nn.Module):
    r"""
    Large-Kernel Spatial Attention enhanced with an attention-based (Global Context)
    dynamic 3x3 core. This version is self-contained and does not use inheritance.

    Args
    ----
    dim  : input / output channels
    k    : total kernel size of the depth-wise spatial filter
           (must be odd and one of {7, 11, 23, 35, 41, 53})
    ratio: bottleneck ratio for the dynamic-kernel MLP
    """
    STATIC_SPECS = {
        7 : dict(core=3, dilate=1),
        11: dict(core=5, dilate=1),
        23: dict(core=5, dilate=2),
        35: dict(core=5, dilate=3, extra=11),
        41: dict(core=5, dilate=3, extra=13),
        53: dict(core=5, dilate=3, extra=17),
    }
    def __init__(self, dim: int, k: int = 35, ratio: int = 4):
        super().__init__()
        assert k in self.STATIC_SPECS, f"Unsupported k={k}"
        spec = self.STATIC_SPECS[k]
        core_k    = spec["core"]
        dil       = spec["dilate"]
        extra_k   = spec.get("extra", 0)

        # ---------- static LKA rim (depth-wise) ----------
        self.conv_h = nn.Conv2d(dim, dim, kernel_size=(1, core_k), padding=(0, (core_k-1)//2), groups=dim)
        self.conv_v = nn.Conv2d(dim, dim, kernel_size=(core_k, 1), padding=((core_k-1)//2, 0), groups=dim)
        self.conv_dh = nn.Conv2d(dim, dim, kernel_size=(1, core_k), padding=(0, dil * (core_k//2)), dilation=dil, groups=dim)
        self.conv_dv = nn.Conv2d(dim, dim, kernel_size=(core_k, 1), padding=(dil * (core_k//2), 0), dilation=dil, groups=dim)

        if extra_k:  # k >= 35
            self.conv_big_h = nn.Conv2d(dim, dim, kernel_size=(1, extra_k), padding=(0, dil * (extra_k//2)), dilation=dil, groups=dim)
            self.conv_big_v = nn.Conv2d(dim, dim, kernel_size=(extra_k, 1), padding=(dil * (extra_k//2), 0), dilation=dil, groups=dim)
        else:
            self.conv_big_h = self.conv_big_v = nn.Identity()

        # ----------  3x3 ----------
        mid = int(dim * ratio)
        self.conv3 = nn.Sequential(nn.Conv2d(dim, dim // ratio, 1, 1, 0),
                        nn.GELU(),
                        nn.Conv2d(dim // ratio, dim // ratio, 3, 1, 1),
                        nn.GELU(),
                        nn.Conv2d(dim // ratio, dim, 1, 1, 0))

        self.k_size = k  # full spatial size (after padding)

        # 1x1 fuse
        self.channel = nn.Conv2d(dim, dim, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LKSA.
        """
        B, C, H, W = x.shape

        # ---------------- static LKA pathway ----------------
        y = self.conv_h(x)
        y = self.conv_v(y)
        y = self.conv_dh(y)
        y = self.conv_dv(y)
        y = self.conv_big_h(y)
        large_kn = self.conv_big_v(y)

        out = self.channel(x) * (large_kn + self.conv3(x))
       
        return out
# -------------------------------------- Dual Fusion Block ---------------------------------------


class SDFL(nn.Module):

    def __init__(self, dim, input_resolution, num_heads,
                 attn_drop=0., qkv_bias=True, proj_drop=0., **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.eps = 1e-6                              
        self.attn_drop = nn.Dropout(attn_drop)

        # ------------------- unchanged -------------------
        self.head_dim = dim // 2 // num_heads
        self.after_taylor_dim = (self.head_dim + 1) * (self.head_dim + 2) // 2
        self.after_taylor_dim += self.after_taylor_dim % 2                # make even

        self.q = nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.k = nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim // 2, bias=qkv_bias)

        self.ln_q = HedgehogFeatureMap(self.head_dim)
        self.ln_k = HedgehogFeatureMap(self.head_dim)

        self.scale = self.head_dim ** -0.5
        self.lepe = nn.Conv2d(dim // 2, dim // 2, 3, padding=1, groups=dim // 2)

        self.proj = nn.Linear(dim, dim)
        self.temperature = nn.Parameter(torch.ones(dim // 2, 1))
        self.proj_drop = nn.Dropout(proj_drop)

        self.pos_emb = FourierEmbedding(
            d_model=self.head_dim * 2 * self.num_heads,
            n_heads=self.num_heads,
            learnable=True,
            rope_fourier_init_norm_gain=0.5
        )
        
    # ---------- helpers (unchanged) ----------
    def spatial_brand(self, q, k, v, x_size):
        b, n, c = v.shape
        h, w = x_size

        shortcut = v.clone()

        num_heads = self.num_heads
        head_dim = c // num_heads
        q, k, v = [t.view(b, n, num_heads, head_dim).transpose(1, 2) for t in (q, k, v)]
        oq, ok = q, k

        q = self.ln_q(q)
        k = self.ln_k(k)

        q_fope = self.pos_emb(q, n)
        k_fope = self.pos_emb(k, n)

        z = 1.0 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + self.eps)
        kv = (k_fope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_fope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v_img = v.transpose(-1, -2).reshape(b, head_dim * num_heads, h, w)
        gate = self.lepe(v_img).reshape(b, c, n).permute(0, 2, 1) + x
        return gate 

    def channel_brand(self, _q, _k, _v):
        B, L, C = _q.shape
        q, k, v = map(lambda t: rearrange(t, 'b l c -> b c l'), (_q, _k, _v))
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        return (attn @ v).transpose(-1, -2)

    # ---------- forward with dispersion gate ----------
    def forward(self, x, x_size):
        """
        x: (B, N, C)  where N = H*W
        """
        b, n, c = x.shape

        # standard projections
        q, k, v = self.q(x), self.k(x), self.v(x)

        # 2. two half-channel branches
        x_spatial = self.spatial_brand(q, k, v, x_size)
        x_channel = self.channel_brand(q, k, v)

        # 3. concatenate halves, project, dropout
        out = torch.cat([x_spatial, x_channel], dim=-1)
        out = self.proj_drop(self.proj(out))

        return out, (q, k)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}'

class DFRL(nn.Module):
    r""" Linear Attention with LePE and RoPE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_heads, attn_drop = 0., qkv_bias=True, proj_drop=0., **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.attn_drop = nn.Dropout(attn_drop)
        self.eps = 1e-6   
        self.head_dim = dim // 2 // num_heads 
        self.after_taylor_dim = (self.head_dim + 1)* (self.head_dim +2) // 2
        self.after_taylor_dim = self.after_taylor_dim if self.after_taylor_dim % 2 == 0 else self.after_taylor_dim + 2

        self.v = nn.Linear(dim, dim // 2)
        
        # self.elu = nn.ELU()
        self.ln_q = HedgehogFeatureMap(self.head_dim)
        self.ln_k = HedgehogFeatureMap(self.head_dim)

        # self.ln_q = TaylorFeatureMap(self.head_dim)
        # self.ln_k = TaylorFeatureMap(self.head_dim)
        
        self.scale = self.head_dim ** (-0.5)
        # self.dwc = nn.Conv2d(in_channels=self.head_dim // 2, out_channels=self.head_dim // 2, kernel_size=3,
        #                     groups=self.head_dim // 2, padding=3 // 2)
        self.lepe = self.dwc = nn.Conv2d(
                    in_channels=dim // 2,
                    out_channels=dim // 2,
                    kernel_size=3, padding=1,
                    groups=dim // 2
                )

        self.proj = nn.Linear(dim, dim)

        self.temperature = nn.Parameter(torch.ones(dim // 2, 1))
        self.proj_drop = nn.Dropout(proj_drop)

        self.pos_emb = FourierEmbedding(d_model=self.head_dim * 2 * self.num_heads,
                                        n_heads=self.num_heads,
                                        learnable=True,                      # make coefficients trainable
                                        rope_fourier_init_norm_gain=0.5)
    
    
    def spatial_brand(self, q, k, v, x_size):
        b, n, c = v.shape
        h, w = x_size

        shortcut = v.clone()

        num_heads = self.num_heads
        head_dim = c // num_heads
        q, k, v = [t.view(b, n, num_heads, head_dim).transpose(1, 2) for t in (q, k, v)]
        oq, ok = q, k

        q = self.ln_q(q)
        k = self.ln_k(k)

        q_fope = self.pos_emb(q, n)
        k_fope = self.pos_emb(k, n)
        
        # Test local monotonicity
        #self.test_local_monotonicity(oq, ok)
        
        z = 1.0 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + self.eps)
        kv = (k_fope.transpose(-2, -1) * (n ** -0.5)) @ (v * (n ** -0.5))
        x = q_fope @ kv * z

        
        x = x.transpose(1, 2).reshape(b, n, c)
        v_img = v.transpose(-1, -2).reshape(b, head_dim * num_heads, h, w)
        gate = self.lepe(v_img).reshape(b, c, n).permute(0, 2, 1) + x
        return gate
    def channel_brand(self, _q, _k, _v):
        B, L, C = _q.shape
        
        q,k,v = map(lambda t: rearrange(t, 'b l c -> b c l'), (_q,_k,_v))
        
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature 
        attn = attn.softmax(dim=-1)
        out =  (attn @ v)

        return out.transpose(-1,-2)

    def forward(self, x, share_qk, x_size):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        h,w = x_size
        q, k = share_qk

        v = self.v(x)

        x_spatial_1 = self.spatial_brand(q,k, v, x_size)

        x_channel_1 = self.channel_brand(q,k,v)

        out = torch.cat([x_spatial_1, x_channel_1], dim = -1)

        out = self.proj_drop(self.proj(out ))

        return out

# ----------------------------------- Multihead Attention Block -----------------------------------

class WindowsAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, attn_drop=0., proj_drop=0., qkv_bias=True, flash=False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attn_drop_value =attn_drop
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.flash = flash
        

        # get pair-wise relative position index for each token inside the window
        pos = torch.arange(window_size[0])
        grid = torch.stack(torch.meshgrid(pos, pos))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size[0] - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size[0] - 1, 1])).sum(dim = -1)

        # compresses the channel dimension of KV
        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Embedding((2 * window_size[0] - 1) ** 2, self.num_heads)
        
        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b_, n, c = x.shape
        
        q,k,v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), (q, k, v))

        q = q * self.scale
        relative_position_bias = self.relative_position_bias_table(self.rel_pos_indices)  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = rearrange(relative_position_bias, 'i j h -> h i j')

        attn = None
        if self.flash == True: 
            with torch.nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.FLASH_ATTENTION]):
                x = F.scaled_dot_product_attention(q,k,v,relative_position_bias, dropout_p=self.attn_drop_value)
        else:
        
            attn = (q @ k.transpose(-2, -1))   # (num_windows*b, num_heads, n, n//4)
            attn = attn + relative_position_bias
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)

            x = (attn @ v)
        x = self.proj(x.transpose(1, 2).reshape(b_, n, c))
        x = self.proj_drop(x)
        return x, attn

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'

class SharedWindowsAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, attn_drop=0., proj_drop=0., qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # compresses the channel dimension of KV
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn, mask=None):
        b_, n, c = x.shape
        
        v = self.v(x)
        v =  rearrange(v, 'b n (h d) -> b h n d', h = self.num_heads)

        x = (attn @ v)
        x = self.proj(x.transpose(1, 2).reshape(b_, n, c))
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'
    

# ------------------------------------------- Main part -------------------------------------------
class HybridBlock(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, window_size, shift_size, mlp_ratio=4, qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, share="N", mhsa_num_heads=2, dfl_num_heads=1, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mhsa_num_heads = mhsa_num_heads
        self.dfl_num_heads = dfl_num_heads
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size
        self.mlp_hidden_dim = int(mlp_ratio * dim) 
        self.shift_size = shift_size
        self.share = share  # N: None, L: Linear, A: Attention, F: Share full

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        if self.share == 'N':
            self.win_mhsa = WindowsAttention(
                self.dim,
                window_size=to_2tuple(self.window_size),
                num_heads=mhsa_num_heads,
                qkv_bias=qkv_bias,
            )
        else:
            self.win_mhsa = SharedWindowsAttention(
                self.dim,
                window_size=to_2tuple(self.window_size),
                num_heads=mhsa_num_heads,
                qkv_bias=qkv_bias,
            )

        if self.share == 'N':
            self.l_attn = SDFL(dim=dim, 
                                  input_resolution=input_resolution, 
                                  num_heads=dfl_num_heads, 
                                  qkv_bias=qkv_bias)
        else:
            self.l_attn = DFRL(dim=dim,
                                  input_resolution=input_resolution,
                                  num_heads=dfl_num_heads,
                                  qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        
        self.mlp1 = SGFN(in_features=dim, hidden_features=self.mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = SGFN(in_features=dim, hidden_features=self.mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        layer_scale = 1e-4
        self.scale1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        self.scale2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
    

    def forward(self, x, x_size, attn = None, share_qk = None):
        h, w = x_size
        b, n, c = x.shape
        nh = h // self.window_size
        nw = w // self.window_size

         # part1: Window-MHSA
        shortcut = x
        x = self.norm1(x)  
        x = x.reshape(b, h, w, c)
        x_windows = rearrange(x, 'b (nh ws1) (nw ws2) c -> (b nh nw) (ws1 ws2) c', nh=nh, ws1=self.window_size, nw=nw, ws2=self.window_size)
        if self.share == 'N':
            attn_windows, attn = self.win_mhsa(x_windows)
        elif self.share == 'F': 
            attn_windows = self.win_mhsa(x_windows, attn=attn)
        attn_x = rearrange(attn_windows, '(b nh nw) (ws1 ws2) c -> b (nh ws1) (nw ws2) c', nh=nh, ws1=self.window_size, nw=nw, ws2=self.window_size)
        x_win = attn_x.view(b, n, c) + shortcut 
        x_win = self.mlp1(self.norm2(x_win), x_size) + x_win
        x = shortcut * self.scale1 + x_win
        
        if has_nan_or_inf(x):
            raise ValueError("Input contains NaN or Inf.")
            
        # part2: Linear Attention
        shortcut = x
        if self.share == 'N':
            x, share_qk = self.l_attn(x, x_size=x_size)
        elif self.share == 'F': 
            x = self.l_attn(x, share_qk=share_qk, x_size=x_size)
        x = self.norm3(x)
        x = shortcut + x
        x = x + self.drop_path(self.norm4(self.mlp2(x, x_size)))
        x = x + self.scale2 * shortcut

        if has_nan_or_inf(x):
            raise ValueError("Input contains NaN or Inf.")
        
        if self.share == 'N':
            return x, attn, share_qk
        else:
            return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, mhsa_num_heads={self.mhsa_num_heads}, dfl_num_heads= {self.dfl_num_heads} " \
               f"mlp_ratio={self.mlp_ratio}"
    

# ---------------- Block ----------------
class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 drop_path=0.,
                 large_window_size=32,
                 window_size=16,
                 mlp_ratio=2.,
                 patch_size=1,
                 depth=4,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=True,
                 mhsa_num_heads = 2,
                 dfl_num_heads = 1,
                 conv_depth= 5,
                 share = 'N'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio=mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.depth = depth
        self.share = share
        self.conv_depth = conv_depth
        self.large_window_size = large_window_size

        # High performace block
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.sgfn1 = SGFN(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU)
        self.attn1 = WindowsAttention(
                self.dim,
                window_size=to_2tuple(large_window_size),
                num_heads=mhsa_num_heads,
                qkv_bias=False,
                flash=True
            )

        self.blocks = nn.ModuleList([
            HybridBlock(
                dim=dim,
                input_resolution=self.input_resolution,
                mhsa_num_heads=mhsa_num_heads,
                dfl_num_heads=dfl_num_heads,
                window_size=window_size,
                shift_size = 0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                share=share)
            for i in range(depth)])
        
        self.mbconv = MBConv(dim,dim)
        self.blocks_2 = nn.ModuleList([
            LKSA(dim=16, k=23)
            for i in range(self.conv_depth)])        
        self.block_conv = nn.ModuleList([
            SGFN(in_features=dim, hidden_features=int(dim * mlp_ratio))
            for i in range(self.conv_depth)])
        self.block_merge = nn.ModuleList([
            nn.Conv2d(dim, dim, 1, padding=0, groups=dim)
            for i in range(self.conv_depth)])

        self.norm_out = norm_layer(dim)
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
    

    def check_image_size(self, x, win_size):
        x = x.permute(0,3,1,2).contiguous()
        _, _, h, w = x.size()
        mod_pad_h = (win_size[0] - h % win_size[0]) % win_size[0]
        mod_pad_w = (win_size[1] - w % win_size[1]) % win_size[1]

        if mod_pad_h >= h or mod_pad_w >= w:
            pad_h, pad_w = h-1, w-1
            x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')
        else:
            pad_h, pad_w = 0, 0
        
        mod_pad_h = mod_pad_h - pad_h
        mod_pad_w = mod_pad_w - pad_w
        
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = x.permute(0,2,3,1).contiguous()
        return x
    
    def forward(self, x, x_size, layer_share_attn=None, layer_share_qk=None):
        
        b, n, c = x.shape
        h, w = x_size

        # Local
        x = self.sgfn1(self.norm1(x), x_size)
        x = x.view(b, h, w, c)

        # padding
        x = self.norm2(x)

        x_win = self.check_image_size(x, (self.large_window_size, self.large_window_size))
        _, h_pad, w_pad, _ = x_win.shape # shape after padding

        x_windows = rearrange(x_win, 'b (nh ws1) (nw ws2) c -> (b nh nw) (ws1 ws2) c', nh=h_pad // self.large_window_size, ws1=self.large_window_size, nw=w_pad // self.large_window_size, ws2=self.large_window_size)
        x_windows, _ = self.attn1(x_windows)
        x_windows = rearrange(x_windows, '(b nh nw) (ws1 ws2) c -> b (nh ws1) (nw ws2) c', nh=h_pad // self.large_window_size, ws1=self.large_window_size, nw=w_pad // self.large_window_size, ws2=self.large_window_size)

        x_win = x_windows[:, :h, :w, :].contiguous()
        x = (x + x_win).view(b, -1, c)


        new_layer_share_attn = []
        new_layer_share_qk = []
        x = self.mbconv(x, x_size)
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                if self.share == 'N':
                    x, attn, share_qk = torch.utils.checkpoint.checkpoint(blk, x, x_size, use_reentrant=False)
                    new_layer_share_attn.append(attn)
                    new_layer_share_qk.append(share_qk)
                else:
                    x = torch.utils.checkpoint.checkpoint(blk, x, x_size, attn=layer_share_attn[i], share_qk=layer_share_qk[i], use_reentrant=False)
            else:
                if self.share == 'N':
                    x, attn, share_qk = blk(x, x_size)
                    new_layer_share_attn.append(attn)
                    new_layer_share_qk.append(share_qk)
                else:
                    x = blk(x, x_size, attn=layer_share_attn[i], share_qk=layer_share_qk[i])
        
        for i, cvl in enumerate(self.blocks_2):
            shortcut = x
            x = self.block_conv[i](x, x_size)
            x = x.transpose(-1, -2).reshape(b, c, h, w)
            x1, x_loc = x.split([16, self.dim-16], 1)
            x1 = cvl(x1)
            x = self.block_merge[i](torch.cat((x1, x_loc), dim=1))
            x = x.reshape(b, c, n).permute(0, 2, 1) + shortcut
        
        x = self.norm_out(x)

        if self.downsample is not None:
            x = self.downsample(x)

        if self.share == 'N':
            return x, new_layer_share_attn, new_layer_share_qk
        else:
            return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'
    

class ResidualGroup(nn.Module):
    """Residual Group (RG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of HybridBlocks per group.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        conv_depth (int): Number of convolutional blocks (LKSA + SGFN) in BasicLayer. Default: 5.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size (int): Input image size.
        patch_size (int): Patch size.
        mhsa_num_heads (int): Number of attention heads in multi-head self-attention. Default: 2.
        dfl_num_heads (int): Number of attention heads in deformable attention. Default: 1.
        resi_connection (str): The convolutional block before residual connection. '1conv'/'3conv'.
        share (str): Whether this group shares attention maps from a previous group. 'N' (no) or 'F' (full).
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 window_size=16,
                 mlp_ratio=2,
                 conv_depth=5,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=60,
                 patch_size=None,
                 mhsa_num_heads=2,
                 dfl_num_heads=1,
                 resi_connection='1conv',
                 share='N'):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution # [64, 64]
        self.share = share

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            conv_depth=conv_depth,
            drop_path=drop_path,
            norm_layer=norm_layer,
            depth=depth,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            mhsa_num_heads=mhsa_num_heads,
            dfl_num_heads=dfl_num_heads,
            share=share)
            

        # build the last conv layer in each residual group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 1, 1, 0)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.esa = ESA(max(dim // 4, 16), dim)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size, layer_share_attn = None, layer_share_qk=None):
        if self.share == 'N':
            out, new_layer_attn_share, new_layer_qk_share  = self.residual_group(x, x_size)  
            out = self.patch_unembed(out, x_size)
            return self.patch_embed(self.esa(self.conv(out)+ self.patch_unembed(x, x_size))), new_layer_attn_share, new_layer_qk_share
        else:
            out = self.residual_group(x, x_size, layer_share_attn=layer_share_attn, layer_share_qk=layer_share_qk)
            out = self.patch_unembed(out, x_size)
            return self.patch_embed(self.esa(self.conv(out) + self.patch_unembed(x, x_size)))

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        return flops
    
@ARCH_REGISTRY.register()
class UCAN(nn.Module):
    r""" UCAN: Unified Convolutional Attention Network for Expansive Receptive Fields in Lightweight Super-Resolution.

       Args:
           img_size (int | tuple(int)): Input image size. Default: 64
           patch_size (int | tuple(int)): Patch size. Default: 1
           in_chans (int): Number of input image channels. Default: 3
           embed_dim (int): Patch embedding dimension. Default: 96
           drop_rate (float): Dropout rate. Default: 0
           window_size (int): Window size for local attention. Default: 8
           mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 1.5
           conv_depth (int): Number of convolutional blocks (LKSA + SGFN) per group. Default: 5
           share (list[str]): Sharing mode per residual group, each 'N' or 'F'. Default: ['N','F','N','F']
           mhsa_num_heads (int): Number of heads in multi-head self-attention. Default: 2
           dfl_num_heads (int): Number of heads in deformable attention. Default: 1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           upscale (int): Upscale factor. 2/3/4 for image SR.
           img_range (float): Image range. 1. or 255.
           upsampler (str): Reconstruction module. 'pixelshuffle' / 'pixelshuffledirect' / None
           resi_connection (str): Convolutional block before residual connection. '1conv' / '3conv'
       """
    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 drop_rate=0.,
                 window_size=8,
                 mlp_ratio=1.5,
                 conv_depth=5,
                 share=None,
                 mhsa_num_heads=2,
                 dfl_num_heads=1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='pixelshuffledirect',
                 resi_connection='1conv',
                 **kwargs):
        super(UCAN, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        self.use_checkpoint = use_checkpoint
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio = mlp_ratio
        if share is None:
            share = ['N', 'F', 'N', 'F']
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = 4
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.window_size = window_size

        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # build Residual Groups (RG)
        self.rg_1 = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=4,
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                conv_depth=conv_depth,
                mhsa_num_heads=mhsa_num_heads,
                dfl_num_heads=dfl_num_heads,
                drop_path=0,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                share=share[0])
        self.rg_2 = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=4,
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                conv_depth=conv_depth,
                mhsa_num_heads=mhsa_num_heads,
                dfl_num_heads=dfl_num_heads,
                drop_path=0,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                share=share[1])
        self.rg_3 = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=4,
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                conv_depth=conv_depth,
                mhsa_num_heads=mhsa_num_heads,
                dfl_num_heads=dfl_num_heads,
                drop_path=0,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                share=share[2])
        self.rg_4 = ResidualGroup(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=4,
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                conv_depth=conv_depth,
                mhsa_num_heads=mhsa_num_heads,
                dfl_num_heads=dfl_num_heads,
                drop_path=0,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                share=share[3])
        
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # -------------------------3. high-quality image reconstruction ------------------------ #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)

        else:
            # for image denoising
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x) # N,L,C

        x = self.pos_drop(x)

        x, new_attn_map, new_qk_map = self.rg_1(x, x_size)
        x = self.rg_2(x, x_size, new_attn_map, new_qk_map)
        x, new_attn_map, new_qk_map = self.rg_3(x, x_size)
        x = self.rg_4(x, x_size, new_attn_map, new_qk_map)          

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        
        # padding
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        params = None
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.upsample(x)

        else:
            # for image denoising
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first, params)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        # unpadding
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]

        return x

if __name__ == '__main__':
    upscale = 2
    model = UCAN(
        upscale=2,
        img_size=64,
        embed_dim=48,
        window_size=16,
        img_range=1.,
        mlp_ratio=1.,
        upsampler='pixelshuffledirect')

    # Model Size
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_num)

    # Test
    _input = torch.randn([2, 3, 64, 64])
    output = model(_input)
    print(output.shape)