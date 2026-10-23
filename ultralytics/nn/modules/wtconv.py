import torch
import torch.nn as nn
import torch.nn.functional as F

from . import wavelet


class WTConv2d(nn.Module):
    """
    Core 2D WTConv spatial operator.

    Preserves channel count and does not include
    BatchNorm or activation.    
    """

    def __init__(self, c1, c2=None, k=5, s=1, bias=True, wt_levels=1, wt_type='db1'):

        """
        Initialize WTConv2d

        Args:
            c1 (int): Number of input channels.
            c2 (int | None): Number of output channels.
            k (int): Kernel size.
            s (int): Output stride.
            bias (bool): Whether the base convolution uses bias.
            wt_levels (int): Number of wavelet decomposition levels.
            wt_type (str): Wavelet type, e.g. "db1".
        """
        super().__init__()

        if c2 is None:
            c2 = c1

        assert c1 == c2, (
            f"WTConv2d requires c1 == c2, "
            f"but received c1={c1}, c2={c2}"
        )


        self.in_channels = c1
        self.wt_levels = wt_levels
        self.stride = s
        self.dilation = 1

        # Fixed analysis/synthesis wavelet filters
        wt_filter, iwt_filter = wavelet.create_2d_wavelet_filter(
        wt_type, c1, c1, torch.float,
        )

        self.register_buffer("wt_filter", wt_filter)
        self.register_buffer("iwt_filter", iwt_filter)

        # Base spatial branch
        self.base_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=k, padding='same', stride=1, dilation=self.dilation, groups=self.in_channels, bias=bias)

        self.base_scale = _ScaleModule([1, self.in_channels,1,1])

        # Wavelet-domain depthwise convolutions
        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(self.in_channels*4, self.in_channels*4, kernel_size=k, padding='same', stride=1, dilation=self.dilation, groups=self.in_channels*4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1,self.in_channels*4,1,1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=self.stride)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet.wavelet_2d_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:,:,0,:,:]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:,:,0,:,:])
            x_h_in_levels.append(curr_x_tag[:,:,1:4,:,:])

        next_x_ll = 0

        for i in range(self.wt_levels-1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = wavelet.inverse_2d_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll

        assert len(x_ll_in_levels) == 0
        assert len(x_h_in_levels) == 0
        assert len(shapes_in_levels) == 0


        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0):
        super(_ScaleModule, self).__init__()

        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)

    def forward(self, x):
        return torch.mul(self.weight, x)
