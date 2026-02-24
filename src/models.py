import torch
from torch import nn
from typing import Tuple
from torch.nn import functional as F
from .math_utils import (
    norm_tensor, unnorm_tensor, itfftc,
    complex_to_chan_dim, chan_dim_to_complex,
    forward_op, adjoint_op, conj_grad,
    pad, unpad, center_crop, zero_pad
)


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two conv layers followed by
    instance normalization, LeakyReLU activation and dropout (optional).
    If use_deform is True, the first conv layer is deformable.
    """

    def __init__(
        self, in_chans: int, out_chans: int, drop_prob: float
    ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
            use_deform: Whether to use deformable conv2d in the first layer.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_chans, out_chans, kernel_size=3, padding=1, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(
                out_chans, out_chans, kernel_size=3, padding=1, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model from the fastMRI library.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList(
            [ConvBlock(in_chans, chans, drop_prob)]
        )
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(
                ConvBlock(ch, ch * 2, drop_prob)
            )
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/bottom if needed to handle odd input dim
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        return output


class NormUnet(nn.Module):
    """
    A modified version of fastMRI's NormUnet that supports
    complex-valued inputs and outputs, although the U-Net
    itself operates on real-valued tensors.
    """
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def _norm_tensor(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.ndim == 4 and x.is_complex(), \
            "Input tensor must be 4D (b, c, h, w) and cfloat"
        return norm_tensor(x)

    def _unnorm_tensor(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        assert x.ndim == 4 and x.is_complex(), \
            "Input tensor must be 4D (b, c, h, w) and cfloat"
        return unnorm_tensor(x, mean, std)

    def _complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4 and x.is_complex(), \
            "Input tensor must be 4D (b, c, h, w) and cfloat"
        return complex_to_chan_dim(x)

    def _chan_dim_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input tensor must be 4D (b, 2 * c, h, w)"
        return chan_dim_to_complex(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get shapes for unet and normalize
        x, mean, std = self._norm_tensor(x)
        x = self._complex_to_chan_dim(x)
        x, pad_sizes = pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = unpad(x, *pad_sizes)
        x = self._chan_dim_to_complex(x)
        x = self._unnorm_tensor(x, mean, std)
        return x


class DualEncoderUnet(nn.Module):
    """
    Dual-encoder, shared-decoder U-Net.

    - One encoder processes the prior s (Ns channels).
    - One encoder processes the current estimate x (2 channels).
    - At each scale, features from both encoders are fused (concat + 1x1 conv)
      and passed to a single shared decoder, which is structurally identical
      to the decoder in the standard U-Net implementation.
    """

    def __init__(
        self,
        in_chans_s: int,      # e.g. 11 (number of prior slices)
        in_chans_x: int,      # e.g. 2  (current estimate)
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        super().__init__()

        self.in_chans_s = in_chans_s
        self.in_chans_x = in_chans_x
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        #  Encoder for s (prior input)
        self.down_sample_layers_s = nn.ModuleList(
            [ConvBlock(in_chans_s, chans, drop_prob)]
        )
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers_s.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2

        #  Encoder for x (current estimate)
        self.down_sample_layers_x = nn.ModuleList(
            [ConvBlock(in_chans_x, chans, drop_prob)]
        )
        ch_x = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers_x.append(
                ConvBlock(ch_x, ch_x * 2, drop_prob)
            )
            ch_x *= 2

        assert ch == ch_x, "s and x encoders must end with same channel count"

        #  Skip fusion (concat + 1x1 conv)
        #  We fuse s- and x-features at each scale into a single skip tensor
        #  with the same channels as the original U-Net.
        skip_chs = []
        ch_tmp = chans
        skip_chs.append(ch_tmp)
        for _ in range(num_pool_layers - 1):
            ch_tmp *= 2
            skip_chs.append(ch_tmp)

        self.skip_fuse_convs = nn.ModuleList([
            nn.Conv2d(2 * c, c, kernel_size=1)
            for c in skip_chs
        ])

        #  Bottleneck fusion and conv
        self.bottom_fuse = nn.Conv2d(2 * ch, ch, kernel_size=1)
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        #  Decoder (same as original U-net)
        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()

        # ch here is channels before bottleneck (same as original)
        # After self.conv, bottleneck feature has 2*ch channels.
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, s: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: Prior input tensor of shape (N, in_chans_s, H, W).
            x: Current estimate tensor of shape (N, in_chans_x, H, W).

        Returns:
            Output tensor of shape (N, out_chans, H, W).
        """
        stack = []
        out_s = s
        out_x = x

        #  Down-sampling / encoding
        for layer_s, layer_x, fuse_conv in zip(
            self.down_sample_layers_s,
            self.down_sample_layers_x,
            self.skip_fuse_convs,
        ):
            out_s = layer_s(out_s)  # (N, C_l, H_l, W_l)
            out_x = layer_x(out_x)  # (N, C_l, H_l, W_l)

            # Fuse s and x features at this scale for decoder skip
            fused = torch.cat([out_s, out_x], dim=1)  # (N, 2*C_l, H_l, W_l)
            fused = fuse_conv(fused)                  # (N, C_l, H_l, W_l)
            stack.append(fused)

            # Downsample both streams for next level
            out_s = F.avg_pool2d(out_s, kernel_size=2, stride=2, padding=0)
            out_x = F.avg_pool2d(out_x, kernel_size=2, stride=2, padding=0)

        #  Bottleneck
        bottom = torch.cat([out_s, out_x], dim=1)  # (N, 2*ch, H_b, W_b)
        bottom = self.bottom_fuse(bottom)          # (N, ch, H_b, W_b)
        output = self.conv(bottom)                 # (N, 2*ch, H_b, W_b)

        #  Up-sampling / decoding
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            skip = stack.pop()
            output = transpose_conv(output)

            # reflect pad if needed to match skip spatial dims (odd sizes)
            padding = [0, 0, 0, 0]
            if output.shape[-1] != skip.shape[-1]:
                padding[1] = 1  # pad right
            if output.shape[-2] != skip.shape[-2]:
                padding[3] = 1  # pad bottom
            if padding[1] or padding[3]:
                output = F.pad(output, padding, mode="reflect")

            output = torch.cat([output, skip], dim=1)
            output = conv(output)

        return output


class Phi(nn.Module):
    """
    A modified version of fastMRI's NormUnet that supports
    complex-valued inputs and outputs, although the U-Net
    itself operates on real-valued tensors.
    """
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.unet = Unet(
            in_chans=in_chans,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def _norm_tensor(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.ndim == 4 and x.is_complex(), \
            "Input tensor must be 4D (b, c, h, w) and cfloat"
        return norm_tensor(x)

    def _unnorm_tensor(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        assert x.ndim == 4 and x.is_complex(), \
            "Input tensor must be 4D (b, c, h, w) and cfloat"
        return unnorm_tensor(x, mean, std)

    def _complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4 and x.is_complex(), \
            "Input tensor must be 4D (b, c, h, w) and cfloat"
        return complex_to_chan_dim(x)

    def _chan_dim_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input tensor must be 4D (b, 2 * c, h, w)"
        return chan_dim_to_complex(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get shapes for unet and normalize
        x, mean, std = self._norm_tensor(x)
        x = self._complex_to_chan_dim(x)
        x, pad_sizes = pad(x)

        x = self.unet(x)

        # get shapes back and unnormalize
        x = unpad(x, *pad_sizes)
        x = self._chan_dim_to_complex(x)
        x = self._unnorm_tensor(x, mean, std)

        return x


class H(nn.Module):
    """
    A modified version of fastMRI's NormUnet that takes Ns-channel prior
    as input, current image as side information and outputs the correction
    term in the trust-guidance block. Uses the dimensions from the header.
    """
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 11,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.deunet = DualEncoderUnet(
            in_chans_s=in_chans,
            in_chans_x=2,
            out_chans=out_chans,
            chans=chans,
            num_pool_layers=num_pools,
            drop_prob=drop_prob,
        )

    def _norm_tensor(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert x.ndim == 4 and x.is_complex(), \
            "Input tensor must be 4D (b, c, h, w) and cfloat"
        return norm_tensor(x)

    def _unnorm_tensor(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        assert x.ndim == 4 and x.is_complex(), \
            "Input tensor must be 4D (b, c, h, w) and cfloat"
        return unnorm_tensor(x, mean, std)

    def _complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4 and x.is_complex(), \
            "Input tensor must be 4D (b, c, h, w) and cfloat"
        return complex_to_chan_dim(x)

    def _chan_dim_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input tensor must be 4D (b, 2 * c, h, w)"
        return chan_dim_to_complex(x)

    def forward(
        self, s: torch.Tensor,
        x: torch.Tensor, recon_size: Tuple[int, int]
    ) -> torch.Tensor:
        # Crop x to recon_size to ensure s and x have compatible sizes
        # Normalize x and pad both (s is already normalized)
        x = center_crop(x, recon_size)
        x, mean, std = self._norm_tensor(x)
        x = self._complex_to_chan_dim(x)
        x, pad_sizes = pad(x)
        s, _ = pad(s)

        x = self.deunet(s, x)

        # Get shapes back and unnormalize
        x = unpad(x, *pad_sizes)
        x = self._chan_dim_to_complex(x)
        x = self._unnorm_tensor(x, mean, std)

        return x  # recon_size is the final shape


class SensitivityModel(nn.Module):
    """
    Model for finetuning the estimated sensitivity maps.
    This model uses a NormUnet to enhance the coil
    sensitivity maps which were computed from the ref
    scan using standard techniques, e.g., ESPIRiT.
    """
    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def _chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w = x.shape
        return x.view(b * c, 1, h, w), b

    def _batch_chans_to_chan_dim(
        self, x: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        bc, _, h, w = x.shape
        c = bc // batch_size
        return x.view(batch_size, c, h, w)

    def forward(
        self,
        masked_kspace: torch.Tensor,
    ) -> torch.Tensor:
        images, batches = self._chans_to_batch_dim(itfftc(masked_kspace))
        out = self._batch_chans_to_chan_dim(self.norm_unet(images), batches)
        rss = torch.sum(out.abs() ** 2, dim=1, keepdim=True).sqrt()
        return out / rss


class VNBlock(nn.Module):
    """
    Model block for variational network reconstruction.
    """
    def __init__(self, model_ref: nn.Module):
        """
        Args:
            model_ref: Refinement network Phi
        """
        super().__init__()
        self.model_ref = model_ref

        # etas in the paper (DC weights)
        self.dc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Image estimate.
            ref_kspace: Ref k-space data.
            mask: Undersampling mask.
            sens_maps: Coil sensitivity maps.
        Returns:
            Updated image estimates after one DC + refinement step.
        """
        k = forward_op(x, mask, sens_maps)
        dc = adjoint_op(k - ref_kspace, mask, sens_maps)
        dc = dc * torch.abs(self.dc_weight)
        refine = self.model_ref(x)

        return x - dc - refine


class TGVNBlock(nn.Module):
    """
    Model block for trust-guided variational network reconstruction.
    """
    def __init__(self, model_ref: nn.Module, model_tg: nn.Module):
        """
        Args:
            model_ref: Refinement network Phi
            model_tg: Trust-guidance network H
        """
        super().__init__()
        self.model_ref = model_ref
        self.model_tg = model_tg

        # etas in the paper (DC weights)
        self.dc_weight = nn.Parameter(torch.ones(1))

        # mus in the paper (ASC weights)
        self.asc_weight = nn.Parameter(torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        ref_kspace: torch.Tensor,
        mask: torch.Tensor,
        sens_maps: torch.Tensor,
        recon_size: Tuple[int, int],
        delta: torch.Tensor,
        num_iter: int = 10,
    ) -> torch.Tensor:
        """
        Args:
            x: Image estimate.
            s: Prior volume (Ns slices).
            ref_kspace: Ref k-space data.
            mask: Undersampling mask.
            sens_maps: Coil sensitivity maps.
            recon_size: Reconstruction size.
            delta: Projection threshold parameter.
            num_iter: Number of iterations for the CG
        Returns:
            Updated image estimates after one DC + TG + refinement step.
        """
        k = forward_op(x, mask, sens_maps)
        dc = adjoint_op(k - ref_kspace, mask, sens_maps)
        dc = dc * torch.abs(self.dc_weight)

        tg = zero_pad(
            center_crop(x, recon_size) - self.model_tg(s, x, recon_size),
            x.shape[-2:]
        )
        # Apply the projection onto the ambiguous space using CG
        tg = conj_grad(tg, mask, sens_maps, delta, num_iter=num_iter)
        tg = tg * torch.abs(self.asc_weight)
        refine = self.model_ref(x)

        return x - dc - tg - refine


class VN(nn.Module):
    """
    Variational Network model implemented in image domain
    """
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 12,
        sens_pools: int = 4,
        Phi_chans: int = 18,
        pools: int = 4,
    ):
        """
        Args:
            num_cascades: Number of DC + refinement steps.
            sens_chans: Number of channels in the sensitivity model.
            sens_pools: Number of pooling layers in the sensitivity model.
            Phi_chans: Number of channels in the refinement model.
            pools: Number of pooling layers in the refinement model.
        """
        super().__init__()

        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.cascades = nn.ModuleList(
            [
                VNBlock(
                    Phi(Phi_chans, pools)
                ) for _ in range(num_cascades)
            ]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        recon_size: Tuple[int, int],
    ) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace)
        image_pred = adjoint_op(masked_kspace, mask, sens_maps)

        for cascade in self.cascades:
            image_pred = cascade(
                image_pred,
                masked_kspace,
                mask, sens_maps
            )

        return center_crop(image_pred.abs(), recon_size)


class TGVN(nn.Module):
    """
    Trust-Guided Variational Network model implemented in image domain
    """
    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 12,
        sens_pools: int = 4,
        Phi_chans: int = 18,
        H_chans: int = 12,
        pools: int = 4,
    ):
        """
        Args:
            num_cascades: Number of DC + TG + refinement steps.
            sens_chans: Number of channels in the sensitivity model.
            sens_pools: Number of pooling layers in the sensitivity model.
            Phi_chans: Number of channels in the refinement model.
            H_chans: Number of channels in the trust-guidance model.
            pools: Number of pooling layers in the refinement and TG models.
        """
        super().__init__()
        self.delta = nn.Parameter(0.1 * torch.ones(1))
        self.sens_net = SensitivityModel(sens_chans, sens_pools)
        self.cascades = nn.ModuleList(
            [
                TGVNBlock(
                    Phi(Phi_chans, pools),
                    H(H_chans, pools)
                ) for _ in range(num_cascades)
            ]
        )

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        prior: torch.Tensor,
        recon_size: Tuple[int, int],
    ) -> torch.Tensor:
        sens_maps = self.sens_net(masked_kspace)
        image_pred = adjoint_op(masked_kspace, mask, sens_maps)

        for cascade in self.cascades:
            image_pred = cascade(
                image_pred,
                prior,
                masked_kspace,
                mask,
                sens_maps,
                recon_size,
                self.delta,
            )

        return center_crop(image_pred.abs(), recon_size)
