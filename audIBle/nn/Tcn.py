import torch

from torch import nn
from collections import OrderedDict


def z_norm(x, dims, eps: float = 1e-8):
    mean = x.mean(dim=dims, keepdim=True)
    var2 = torch.var(x, dim=dims, keepdim=True, unbiased=False)
    value = (x - mean) / torch.sqrt((var2 + eps))
    return value


def _glob_norm(x, eps: float = 1e-8):
    dims = torch.arange(1, len(x.shape)).tolist()
    return z_norm(x, dims, eps)


class _LayerNorm(nn.Module):
    """Layer Normalization base class from asteroid."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """Assumes input of size `[batch, chanel, *]`."""
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x, EPS: float = 1e-8):
        """Applies forward pass.
        Works for any input size > 2D.
        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`
        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        value = _glob_norm(x, eps=EPS)
        return self.apply_gain_and_bias(value)


class Simple_Norm(nn.Module):
    """Global Layer Normalization (globLN)."""

    def __init__(self, dim_in, dim_out, device):
        super(Simple_Norm, self).__init__()
        self.mat = torch.zeros((dim_in, dim_out)).to(device)
        x = torch.arange(dim_out)
        x_mapped = x / dim_out * dim_in
        x_int = x_mapped.long()
        self.mat[x_int, x] = 1

    def forward(self, x):
        """Applies forward pass."""

        # print(x.shape,self.mat.shape)
        return torch.matmul(x, self.mat)


class Conv1DBlock(nn.Module):
    """One dimensional convolutional block, as proposed in [1].
    Come From the Github of OSDC: https://github.com/popcornell/OSDC
    Args:
        in_chan (int): Number of input channels.
        hid_chan (int): Number of hidden channels in the depth-wise
            convolution.
        skip_out_chan (int): Number of channels in the skip convolution.
        kernel_size (int): Size of the depth-wise convolutional kernel.
        padding (int): Padding of the depth-wise convolution.
        dilation (int): Dilation of the depth-wise convolution.
        norm_type (str, optional): Type of normalization to use. To choose from
            -  ``'gLN'``: global Layernorm
            -  ``'cLN'``: channelwise Layernorm
            -  ``'cgLN'``: cumulative global Layernorm
    References:
        [1] : "Conv-TasNet: Surpassing ideal time-frequency magnitude masking
        for speech separation" TASLP 2019 Yi Luo, Nima Mesgarani
        https://arxiv.org/abs/1809.07454
    """

    def __init__(
        self,
        in_chan,
        hid_chan,
        kernel_size,
        padding,
        dilation,
        norm_type="bN",
        delta=False,
    ):
        super(Conv1DBlock, self).__init__()

        self.delta = delta
        if delta:
            self.linear = nn.Linear(in_chan, in_chan)
            self.linear_norm = GlobLN(
                in_chan * 2
            )  # TODO select between different types of norm

        conv_norm = GlobLN(hid_chan)
        in_bottle = in_chan if not delta else in_chan * 2
        in_conv1d = nn.Conv1d(in_bottle, hid_chan, 1)
        depth_conv1d = nn.Conv1d(
            hid_chan,
            hid_chan,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=hid_chan,
        )
        self.shared_block = nn.Sequential(
            in_conv1d, nn.PReLU(), conv_norm, depth_conv1d, nn.PReLU(), conv_norm
        )
        self.res_conv = nn.Conv1d(hid_chan, in_chan, 1)

    def forward(self, x):
        """Input shape [batch, feats, seq]"""
        if self.delta:
            delta = self.linear(x.transpose(1, -1)).transpose(1, -1)
            x = torch.cat((x, delta), 1)
            x = self.linear_norm(x)

        shared_out = self.shared_block(x)
        res_out = self.res_conv(shared_out)
        return res_out


class TCN(nn.Module):
    """Temporal Convolutional network used in ConvTasnet.
    Come From the Github of OSDC: https://github.com/popcornell/OSDC
    Args:
        in_chan (int): Number of input filters.
        n_src (int): Number of masks to estimate.
        out_chan (int, optional): Number of bins in the estimated masks.
            If ``None``, `out_chan = in_chan`.
        n_blocks (int, optional): Number of convolutional blocks in each
            repeat. Defaults to 8.
        n_repeats (int, optional): Number of repeats. Defaults to 3.
        bn_chan (int, optional): Number of channels after the bottleneck.
        hid_chan (int, optional): Number of channels in the convolutional
            blocks.
        skip_chan (int, optional): Number of channels in the skip connections.
        kernel_size (int, optional): Kernel size in convolutional blocks.
        norm_type (str, optional): To choose from ``'BN'``, ``'gLN'``,
            ``'cLN'``.
        mask_act (str, optional): Which non-linear function to generate mask.
    """

    def __init__(
        self,
        in_chan,
        n_src,
        out_chan=None,
        n_blocks=8,
        n_repeats=3,
        bn_chan=128,
        hid_chan=512,
        kernel_size=3,
        norm_type="gLN",
        representation=False,
    ):
        super(TCN, self).__init__()
        self.in_chan = in_chan
        self.n_src = n_src
        out_chan = out_chan if out_chan else in_chan
        self.out_chan = out_chan
        self.n_blocks = n_blocks
        self.n_repeats = n_repeats
        self.bn_chan = bn_chan
        self.hid_chan = hid_chan
        self.kernel_size = kernel_size
        self.norm_type = norm_type
        self.output_representation = representation
        layer_norm = GlobLN(in_chan)
        bottleneck_conv = nn.Conv1d(in_chan, bn_chan, 1)
        self.bottleneck = nn.Sequential(layer_norm, bottleneck_conv)

        # Succession of Conv1DBlock with exponentially increasing dilation.
        self.TCN = nn.ModuleList()
        for r in range(n_repeats):
            for x in range(n_blocks):
                padding = (kernel_size - 1) * 2**x // 2
                self.TCN.append(
                    Conv1DBlock(
                        bn_chan,
                        hid_chan,
                        kernel_size,
                        padding=padding,
                        dilation=2**x,
                        norm_type=norm_type,
                    )
                )
        out_conv = nn.Conv1d(bn_chan, n_src * out_chan, 1)
        self.out = nn.Sequential(nn.PReLU(), out_conv)
        self.n_blocks = n_blocks
        # Get activation function.

    def forward(self, mixture_w):
        """
        Args:
            mixture_w (:class:`torch.Tensor`): Tensor of shape
                [batch, n_filters, n_frames]
        Returns:
            :class:`torch.Tensor`:
                estimated mask of shape [batch, n_src, n_filters, n_frames]
        """
        intermediate = {}
        output = self.bottleneck(mixture_w)
        for i in range(len(self.TCN)):
            residual = self.TCN[i](output)
            output = output + residual
            if self.output_representation:
                if i % self.n_blocks == 0:
                    intermediate[f"repr_{i//self.n_blocks}"] = residual.detach()
        logits = self.out(output)
        if self.output_representation:
            intermediate["out"] = logits.squeeze(1)
            return intermediate
        else:
            if len(logits.shape) != 3:
                return logits.squeeze(1)
            else:
                return logits
