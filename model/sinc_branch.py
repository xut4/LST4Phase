import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv_fast(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels,
        kernel_size,
        sample_rate=100,
        in_channels=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        groups=1,
        min_low_hz=0.5,
        min_band_hz=0.5,
    ):
        super().__init__()

        if in_channels != 1:
            raise ValueError(f"SincConv only supports one input channel, got {in_channels}")

        if kernel_size % 2 == 0:
            kernel_size += 1

        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 1.0
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz = self.to_hz(mel)

        self.low_hz_ = nn.Parameter(torch.tensor(hz[:-1]).view(-1, 1).float())
        self.band_hz_ = nn.Parameter(torch.tensor(np.diff(hz)).view(-1, 1).float())

        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int(self.kernel_size / 2))
        self.register_buffer(
            "window_",
            0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)
        )

        n = (self.kernel_size - 1) / 2.0
        self.register_buffer(
            "n_",
            2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate
        )

    def forward(self, waveforms):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2,
        )
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left = (
            (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (self.n_ / 2)
        ) * self.window_

        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None])

        filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(
            waveforms,
            filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class SincFrequencyBranch(nn.Module):
    """
    raw waveform -> RSS -> Sinc filter bank -> abs/log compression
    -> interpolate -> conv -> projection
    """
    def __init__( 
        self,
        out_ch,
        sinc_filters=12 ,
        sinc_kernel=129,
        sample_rate=100,
        use_log_compression=True,
    ):
        super().__init__()
        self.use_log_compression = use_log_compression

        self.sinc = SincConv_fast(
            out_channels=sinc_filters,
            kernel_size=sinc_kernel,
            sample_rate=sample_rate,
            in_channels=1,
            padding=sinc_kernel // 2,
        )

        self.tf_conv = nn.Sequential(
            nn.Upsample(scale_factor=1.5),
            nn.Conv1d(sinc_filters, 16, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Upsample(scale_factor=1.5),
            nn.Conv1d(16, 24, kernel_size=5, padding="same"),
            nn.ReLU(),
            nn.Conv1d(24, 32, kernel_size=7, padding="same"),
            nn.ReLU(),
        )

        self.tf_proj = nn.Sequential(
            nn.Linear(32, out_ch),
            nn.ReLU()
        )

    def forward(self, wave_3c, target_len):
        # wave_3c: (B, 3, T)
        rss = torch.sqrt(torch.clamp(torch.sum(wave_3c ** 2, dim=1, keepdim=True), min=1e-8))
        feat = self.sinc(rss)  # (B, sinc_filters, T)

        feat = torch.abs(feat)
        if self.use_log_compression:
            feat = torch.log1p(feat)

        feat = self.tf_conv(feat)
        feat = F.interpolate(feat, size=target_len, mode="linear", align_corners=True)
        feat = feat.permute(0, 2, 1)   # (B, L, 32)
        feat = self.tf_proj(feat)      # (B, L, out_ch)
        return feat