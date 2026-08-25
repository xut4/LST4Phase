import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchinfo import summary

class WaveletTFBranch(nn.Module):
    """
    Wavelet-like time-frequency branch.

    Input:
        x: (B, C, T)

    Output:
        feat: (B, out_ch, target_len)

    Logic:
        waveform -> Morlet-like wavelet filter bank
                 -> magnitude response
                 -> Conv1d projection
                 -> resize to target_len
    """
    def __init__(
        self,
        out_ch,
        in_channels=3,
        sample_rate=100,
        n_scales=32,
        kernel_size=129,
        f_min=0.5,
        f_max=45.0,
        use_log_compression=True
    ):
        super().__init__()

        self.out_ch = out_ch
        self.in_channels = in_channels
        self.sample_rate = sample_rate
        self.n_scales = n_scales
        self.kernel_size = kernel_size
        self.use_log_compression = use_log_compression

        # Initialize center frequencies for wavelet filters.
        freqs = torch.linspace(f_min, f_max, n_scales)
        self.log_freqs = nn.Parameter(torch.log(freqs))

        # Learnable wavelet width.
        self.log_sigma = nn.Parameter(torch.zeros(n_scales))

        # Frequency/scale importance gate.
        self.scale_gate = nn.Parameter(torch.zeros(n_scales))

        self.tf_conv = nn.Sequential(
            nn.Conv1d(n_scales, 16, kernel_size=5, padding='same', bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Conv1d(16, 24, kernel_size=5, padding='same', bias=False),
            nn.BatchNorm1d(24),
            nn.ReLU(),

            nn.Conv1d(24, 32, kernel_size=7, padding='same', bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.tf_proj = nn.Sequential(
            nn.Conv1d(32, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def _build_wavelet_kernels(self, device, dtype):
        """
        Build Morlet-like wavelet kernels.

        Return:
            real_kernel: (n_scales, 1, K)
            imag_kernel: (n_scales, 1, K)
        """
        K = self.kernel_size
        center = K // 2

        t = torch.arange(K, device=device, dtype=dtype) - center
        t = t / self.sample_rate

        freqs = torch.exp(self.log_freqs).to(device=device, dtype=dtype)
        sigma = torch.exp(self.log_sigma).to(device=device, dtype=dtype)

        # Morlet-like wavelet:
        # gaussian envelope * complex sinusoid
        #
        # Compared with the S-transform-like branch,
        # here the Gaussian width is treated as a learnable wavelet scale.
        gaussian = torch.exp(
            -0.5 * (t[None, :] / (sigma[:, None] + 1e-6)) ** 2
        )

        phase = 2 * torch.pi * freqs[:, None] * t[None, :]

        real = gaussian * torch.cos(phase)
        imag = gaussian * torch.sin(phase)

        # Remove DC component to make it more wavelet-like.
        real = real - real.mean(dim=1, keepdim=True)
        imag = imag - imag.mean(dim=1, keepdim=True)

        # Normalize each wavelet filter.
        real = real / (real.norm(dim=1, keepdim=True) + 1e-6)
        imag = imag / (imag.norm(dim=1, keepdim=True) + 1e-6)

        real = real.unsqueeze(1)  # (n_scales, 1, K)
        imag = imag.unsqueeze(1)  # (n_scales, 1, K)

        return real, imag

    def forward(self, x, target_len):
        """
        x: (B, C, T)
        target_len: target temporal length, usually x1.shape[-1]
        """
        B, C, T = x.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, but got {C} channels."
            )

        real_kernel, imag_kernel = self._build_wavelet_kernels(
            device=x.device,
            dtype=x.dtype
        )

        # Apply the same wavelet filter bank to each channel.
        # (B, C, T) -> (B*C, 1, T)
        x_ = x.reshape(B * C, 1, T)

        real = F.conv1d(
            x_,
            real_kernel,
            padding=self.kernel_size // 2
        )

        imag = F.conv1d(
            x_,
            imag_kernel,
            padding=self.kernel_size // 2
        )

        # Magnitude response: (B*C, n_scales, T)
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

        # Reshape: (B, C, n_scales, T)
        mag = mag.reshape(B, C, self.n_scales, T)

        # Combine input channels by RMS over channel dimension.
        # Result: (B, n_scales, T)
        mag = torch.sqrt(torch.clamp(torch.mean(mag ** 2, dim=1), min=1e-8))

        if self.use_log_compression:
            mag = torch.log1p(mag)

        # Learnable scale gate.
        gate = torch.sigmoid(self.scale_gate).view(1, self.n_scales, 1)
        mag = mag * gate

        # Resize to match SegPhase first-stage feature length.
        mag = F.interpolate(
            mag,
            size=target_len,
            mode='linear',
            align_corners=True
        )

        feat = self.tf_conv(mag)      # (B, 32, target_len)
        feat = self.tf_proj(feat)     # (B, out_ch, target_len)

        return feat
class SeismoDualTFBranch(nn.Module):
    def __init__(self, out_ch, n_fft=64, win_length=20, hop_length=16):
        super().__init__()
        self.out_ch = out_ch
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.max_freq = 12

        self.tf_conv = nn.Sequential(
            nn.Conv1d(self.max_freq, 16, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv1d(16, 24, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.Conv1d(24, 32, kernel_size=7, padding='same'),
            nn.ReLU(),
        )

        self.tf_proj = nn.Linear(32, out_ch)

    def forward(self, x, target_len):
        rss = torch.sqrt(torch.clamp(torch.sum(x ** 2, dim=1), min=1e-8))

        window = torch.hamming_window(self.win_length, device=x.device)

        spec = torch.stft(
            rss,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=False
        )

        spec = spec.real
        spec = spec[:, :self.max_freq, :]
        spec = F.interpolate(spec, size=target_len, mode='linear', align_corners=True)

        feat = self.tf_conv(spec)
        feat = feat.permute(0, 2, 1)
        feat = self.tf_proj(feat)
        feat = feat.permute(0, 2, 1)
        return feat

class FeatureFusion(nn.Module):
    """
    time-domain feature + tf feature -> concat -> 1x1 conv
    """
    def __init__(self, ch):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv1d(ch * 2, ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU()
        )

    def forward(self, x_time, x_tf):
        x = torch.cat([x_time, x_tf], dim=1)
        return self.fuse(x)


class ModelTF(nn.Module):
    def __init__(self, 
                 in_length,
                 in_channels,
                 class_num,
                 strides,
                 kernel_size,
                 expantion_ratio: int = 4):
        super().__init__()

        self.ch1 = 16
        self.ch2 = 32
        self.ch3 = 64

        self.ks1 = strides[0]*2-1
        self.ks2 = strides[1]*2-1
        self.ks3 = strides[2]*2-1

        self.st1 = strides[0]
        self.st2 = strides[1]
        self.st3 = strides[2]

        self.hn1 = 2
        self.hn2 = 4
        self.hn3 = 8

        self.rr1 = 3
        self.rr2 = 2
        self.rr3 = 1

        self.bn1 = 3
        self.bn2 = 3
        self.bn3 = 3

        self.tf_branch = SeismoDualTFBranch(
            out_ch=self.ch1,
            n_fft=64,
            win_length=20,
            hop_length=16
        )
        #self.fusion1 = FeatureFusion(self.ch1)

        # SegPhase block
        self.seg_block2 = SegPhaseBlock(
            in_length=in_length // self.st1,
            in_channels=self.ch1,
            emb_dim=self.ch2,
            patch_size=self.ks2,
            stride=self.st2,
            head_num=self.hn2,
            reduction_ratio=self.rr2,
            expantion_ratio=expantion_ratio,
            block_num=self.bn2
        )

        self.seg_block3 = SegPhaseBlock(
            in_length=in_length // (self.st1 * self.st2),
            in_channels=self.ch2,
            emb_dim=self.ch3,
            patch_size=self.ks3,
            stride=self.st3,
            head_num=self.hn3,
            reduction_ratio=self.rr3,
            expantion_ratio=expantion_ratio,
            block_num=self.bn3
        )

        self.output = SegPhaseOutput(
            self.ch3, self.ch2, self.ch1,
            self.st3, self.st2, self.st1,
            kernel_size,
            class_num
        )
        
    def forward(self, x):
        # t branch
        #x1 = self.seg_block1(x)       # [B, 16, L1]

        # tf branch
        tf1 = self.tf_branch(x, target_len=1000)     # [B, 16, L1]

        # merge
        #x1 = self.fusion1(x1, tf1)    # [B, 16, L1]

        # keep SegPhase
        x2 = self.seg_block2(tf1)      # [B, 32, L2]
        x3 = self.seg_block3(x2)      # [B, 64, L3]

        out = self.output(x3, x2, tf1)
        return out

class ModelWavelet(nn.Module):
    def __init__(self, 
                 in_length,
                 in_channels,
                 class_num,
                 strides,
                 kernel_size,
                 expantion_ratio: int = 4):
        super().__init__()

        self.ch1 = 16
        self.ch2 = 32
        self.ch3 = 64

        self.ks1 = strides[0]*2-1
        self.ks2 = strides[1]*2-1
        self.ks3 = strides[2]*2-1

        self.st1 = strides[0]
        self.st2 = strides[1]
        self.st3 = strides[2]

        self.hn1 = 2
        self.hn2 = 4
        self.hn3 = 8

        self.rr1 = 3
        self.rr2 = 2
        self.rr3 = 1

        self.bn1 = 3
        self.bn2 = 3
        self.bn3 = 3

        #  SeismoDual TF branch
        self.tf_branch = WaveletTFBranch(
            out_ch=self.ch1,
            in_channels=in_channels,
            sample_rate=100,
            n_scales=32,
            kernel_size=129,
            f_min=0.5,
            f_max=45.0,
            use_log_compression=True
        )
        #self.fusion1 = FeatureFusion(self.ch1)

        # SegPhase block
        self.seg_block2 = SegPhaseBlock(
            in_length=in_length // self.st1,
            in_channels=self.ch1,
            emb_dim=self.ch2,
            patch_size=self.ks2,
            stride=self.st2,
            head_num=self.hn2,
            reduction_ratio=self.rr2,
            expantion_ratio=expantion_ratio,
            block_num=self.bn2
        )

        self.seg_block3 = SegPhaseBlock(
            in_length=in_length // (self.st1 * self.st2),
            in_channels=self.ch2,
            emb_dim=self.ch3,
            patch_size=self.ks3,
            stride=self.st3,
            head_num=self.hn3,
            reduction_ratio=self.rr3,
            expantion_ratio=expantion_ratio,
            block_num=self.bn3
        )

        self.output = SegPhaseOutput(
            self.ch3, self.ch2, self.ch1,
            self.st3, self.st2, self.st1,
            kernel_size,
            class_num
        )
        
    def forward(self, x):
        # t branch
        #x1 = self.seg_block1(x)       # [B, 16, L1]

        # tf branch
        tf1 = self.tf_branch(x, target_len=1000)     # [B, 16, L1]

        # merge
        #x1 = self.fusion1(x1, tf1)    # [B, 16, L1]

        # keep SegPhase
        x2 = self.seg_block2(tf1)      # [B, 32, L2]
        x3 = self.seg_block3(x2)      # [B, 64, L3]

        out = self.output(x3, x2, tf1)
        return out
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TraditionalSTransformBranch(nn.Module):
    """
    Traditional S-transform branch.

    Input:
        x: (B, C, T), usually (B, 3, 3000)

    Output:
        feat: (B, out_ch, target_len)

    Notes:
        - S-transform kernels are fixed and non-learnable.
        - Frequency-dependent Gaussian window follows standard S-transform form.
        - Only the projection CNN is learnable.
    """

    def __init__(
        self,
        out_ch,
        in_channels=3,
        sample_rate=100,
        n_freq=64,
        kernel_size=257,
        f_min=1.0,
        f_max=45.0,
        use_log_compression=True,
        hidden_ch=16,
        merge_components="rms",
        eps=1e-8,
    ):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd, e.g., 129, 257, 501.")

        if f_max > sample_rate / 2:
            raise ValueError(
                f"f_max={f_max} exceeds Nyquist frequency {sample_rate / 2}."
            )

        self.out_ch = out_ch
        self.in_channels = in_channels
        self.sample_rate = sample_rate
        self.n_freq = n_freq
        self.kernel_size = kernel_size
        self.f_min = f_min
        self.f_max = f_max
        self.use_log_compression = use_log_compression
        self.merge_components = merge_components
        self.eps = eps

        freqs = torch.linspace(f_min, f_max, n_freq)
        self.register_buffer("freqs", freqs)

        # Encode dense S-transform magnitude map: (B, 1, F, T)
        self.tf_encoder = nn.Sequential(
            nn.Conv2d(1, hidden_ch, kernel_size=(5, 7), padding=(2, 3)),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=(5, 7), padding=(2, 3)),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(inplace=True),
        )

        # Collapse frequency dimension and project to out_ch
        self.proj = nn.Sequential(
            nn.Conv1d(hidden_ch, out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _build_st_kernels(self, device, dtype):
        """
        Build fixed traditional S-transform kernels.

        real_kernel, imag_kernel:
            shape = (F, 1, K)
        """
        K = self.kernel_size
        center = K // 2

        # Time axis in seconds
        t = torch.arange(K, device=device, dtype=dtype) - center
        t = t / self.sample_rate

        freqs = self.freqs.to(device=device, dtype=dtype)

        # Standard S-transform Gaussian window:
        # g(t, f) = |f| / sqrt(2*pi) * exp(-0.5 * f^2 * t^2)
        norm = freqs[:, None].abs() / math.sqrt(2.0 * math.pi)

        gaussian = norm * torch.exp(
            -0.5 * (freqs[:, None] * t[None, :]) ** 2
        )

        phase = 2.0 * math.pi * freqs[:, None] * t[None, :]

        real = gaussian * torch.cos(phase)
        imag = -gaussian * torch.sin(phase)

        # Discrete approximation of continuous integral
        dt = 1.0 / self.sample_rate
        real = real * dt
        imag = imag * dt

        real_kernel = real.unsqueeze(1)  # (F, 1, K)
        imag_kernel = imag.unsqueeze(1)  # (F, 1, K)

        return real_kernel, imag_kernel

    def compute_st_magnitude(self, x):
        """
        Compute traditional S-transform magnitude.

        Input:
            x: (B, C, T)

        Output:
            mag: (B, 1, F, T)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B, C, T), but got {x.shape}")

        B, C, T = x.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, but got {C}."
            )

        real_kernel, imag_kernel = self._build_st_kernels(
            device=x.device,
            dtype=x.dtype,
        )

        # Apply same ST kernels to each component
        x_ = x.reshape(B * C, 1, T)

        real = F.conv1d(
            x_,
            real_kernel,
            padding=self.kernel_size // 2,
        )

        imag = F.conv1d(
            x_,
            imag_kernel,
            padding=self.kernel_size // 2,
        )

        mag = torch.sqrt(real ** 2 + imag ** 2 + self.eps)
        mag = mag.reshape(B, C, self.n_freq, T)

        if self.merge_components == "rms":
            # Merge Z/N/E components by RMS
            mag = torch.sqrt(
                torch.clamp(torch.mean(mag ** 2, dim=1), min=self.eps)
            )
        elif self.merge_components == "mean":
            mag = torch.mean(mag, dim=1)
        elif self.merge_components == "max":
            mag = torch.max(mag, dim=1).values
        else:
            raise ValueError(
                "merge_components should be one of: 'rms', 'mean', 'max'."
            )

        mag = mag.unsqueeze(1)  # (B, 1, F, T)

        if self.use_log_compression:
            mag = torch.log1p(mag)

        return mag

    def forward(self, x, target_len):
        """
        Input:
            x: (B, C, T)
            target_len: temporal length of the feature map to fuse with

        Output:
            feat: (B, out_ch, target_len)
        """
        mag = self.compute_st_magnitude(x)     # (B, 1, F, T)

        feat = self.tf_encoder(mag)            # (B, hidden_ch, F, T)

        # Collapse frequency dimension
        feat = feat.mean(dim=2)                # (B, hidden_ch, T)

        # Align temporal length with target feature
        if feat.shape[-1] != target_len:
            feat = F.interpolate(
                feat,
                size=target_len,
                mode="linear",
                align_corners=False,
            )

        feat = self.proj(feat)                 # (B, out_ch, target_len)

        return feat       
class ModelST(nn.Module):
    def __init__(self, 
                 in_length,
                 in_channels,
                 class_num,
                 strides,
                 kernel_size,
                 expantion_ratio: int = 4):
        super().__init__()

        self.ch1 = 32##
        self.ch2 = 32
        self.ch3 = 64

        self.ks1 = strides[0]*2-1
        self.ks2 = strides[1]*2-1
        self.ks3 = strides[2]*2-1

        self.st1 = strides[0]
        self.st2 = strides[1]
        self.st3 = strides[2]

        self.hn1 = 2
        self.hn2 = 4
        self.hn3 = 8

        self.rr1 = 3
        self.rr2 = 2
        self.rr3 = 1

        self.bn1 = 3
        self.bn2 = 3
        self.bn3 = 3


        #  SeismoDual TF branch
        self.tf_branch = TraditionalSTransformBranch(
                out_ch=self.ch1,
                in_channels=in_channels,
                sample_rate=100,
                n_freq=10,
                kernel_size=129,
                f_min=1.0,
                f_max=20.0,
                use_log_compression=True,
            )
       # self.fusion1 = FeatureFusion(self.ch1)

        # SegPhase block
        self.seg_block2 = SegPhaseBlock(
            in_length=in_length // self.st1,
            in_channels=self.ch1,
            emb_dim=self.ch2,
            patch_size=self.ks2,
            stride=self.st2,
            head_num=self.hn2,
            reduction_ratio=self.rr2,
            expantion_ratio=expantion_ratio,
            block_num=self.bn2
        )

        self.seg_block3 = SegPhaseBlock(
            in_length=in_length // (self.st1 * self.st2),
            in_channels=self.ch2,
            emb_dim=self.ch3,
            patch_size=self.ks3,
            stride=self.st3,
            head_num=self.hn3,
            reduction_ratio=self.rr3,
            expantion_ratio=expantion_ratio,
            block_num=self.bn3
        )

        self.output = SegPhaseOutput(
            self.ch3, self.ch2, self.ch1,
            self.st3, self.st2, self.st1,
            kernel_size,
            class_num
        )
        
    def forward(self, x):
        # t branch
        #x1 = self.seg_block1(x)       # [B, 16, L1]

        # tf branch
        tf1 = self.tf_branch(x, target_len=1000)     # [B, 16, L1]

        # merge
        #x1 = self.fusion1(x1, tf1)    # [B, 16, L1]

        # keep SegPhase
        x2 = self.seg_block2(tf1)      # [B, 32, L2]
        x3 = self.seg_block3(x2)      # [B, 64, L3]

        out = self.output(x3, x2, tf1)
        return out
class OverLapPatchMerging(nn.Module):
  def __init__(self, 
               in_channels, 
               emb_dim, 
               patch_size,
               stride,
               in_length):
    super().__init__()
    self.conv = nn.Conv1d(in_channels = in_channels,
                          out_channels = emb_dim,
                          kernel_size = patch_size,
                          stride = stride,
                          padding = patch_size//2,
                          bias=False)
    
    self.pos_emb = nn.Parameter(torch.randn(1, in_length//stride, emb_dim))
  
  def forward(self, x):
    x = self.conv(x)
    x = rearrange(x, 'B C L -> B L C')
    x = x + self.pos_emb
    return x 
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, 
                  channels, 
                  reduction_ratio,
                  num_heads):
        super().__init__()
        
        self.rr = reduction_ratio
        
        dropout = 0
        
        if reduction_ratio > 1:
        
            self.reducer = nn.Conv1d(in_channels = channels, 
                                     out_channels = channels, 
                                     kernel_size=reduction_ratio*2-1, 
                                     stride=reduction_ratio,
                                     padding = (reduction_ratio*2-1)//2,
                                     bias=False)
            
            self.ln = nn.LayerNorm(channels)
        
        self.linear_q = nn.Linear(channels, channels, bias=False)
        self.linear_k = nn.Linear(channels, channels, bias=False)
        self.linear_v = nn.Linear(channels, channels, bias=False)
        
        self.ln_q = nn.LayerNorm(channels)
        self.ln_k = nn.LayerNorm(channels)
        self.ln_v = nn.LayerNorm(channels)
        
        self.head = num_heads
        self.head_ch = channels // num_heads
        self.sqrt_dh = self.head_ch**0.5 
        
        self.attn_drop = nn.Dropout(dropout)

        self.w_o = nn.Linear(channels, channels, bias=False)
        self.w_drop = nn.Dropout(dropout)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        
        if self.rr > 1:
            xr = rearrange(x, 'B L C -> B C L')
            
            reduced = self.reducer(xr)
            reduced = rearrange(reduced, 'B C L -> B L C')
            reduced = self.ln(reduced)
        
            q = self.linear_q(x)
            k = self.linear_k(reduced)
            v = self.linear_v(reduced)
            
        else:
            q = self.linear_q(x)
            k = self.linear_k(x)
            v = self.linear_v(x)
            
        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)
            
        q = rearrange(q, 'B L (h C) -> B h L C', h=self.head)
        k = rearrange(k, 'B L (h C) -> B h L C', h=self.head)
        v = rearrange(v, 'B L (h C) -> B h L C', h=self.head)
        
        k_T = k.transpose(2, 3)
        
        dots = (q @ k_T) / self.sqrt_dh
        attn = self.softmax(dots)
        attn = self.attn_drop(attn)
        out = attn @ v
        
        out = rearrange(out, 'B h L C -> B L (h C)')
        
        out = self.w_o(out) 
        out = self.w_drop(out)
        
        return out, attn
    
class MixFFN(nn.Module):
    def __init__(self,
                 emb_dim,
                 kernel_size,
                 expantion_ratio):
        super().__init__()
        self.linear1 = nn.Conv1d(emb_dim, 
                                 emb_dim, 
                                 kernel_size = 1)
        
        self.linear2 = nn.Conv1d(emb_dim * expantion_ratio, 
                                 emb_dim, 
                                 kernel_size = 1)
        
        self.conv = nn.Conv1d(in_channels=emb_dim, 
                              out_channels=emb_dim * expantion_ratio, 
                              kernel_size=3, 
                              groups=emb_dim,
                              padding='same')
        
        self.gelu = nn.GELU()

    def forward(self,x):
        x = rearrange(x, 'B L C -> B C L')
        x = self.linear1(x)
        x = self.conv(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = rearrange(x, 'B C L -> B L C')
        return x
        
class ViTEncoderMixFFN(nn.Module):
    def __init__(self,
                 emb_dim,
                 kernel_size,
                 reduction_ratio,
                 head_num,
                 expantion_ratio):
        
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(emb_dim, 
                                           reduction_ratio,
                                           head_num)
        
        self.ffn = MixFFN(emb_dim, 
                          kernel_size,
                          expantion_ratio)
        
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
       
    def forward(self, x):
        
        residual_mhsa = x
        mhsa_input = self.ln1(x)
        mhsa_output, attn = self.mhsa(mhsa_input)
        mhsa_output2 = mhsa_output + residual_mhsa
        
        residual_ffn = mhsa_output2
        ffn_input = self.ln2(mhsa_output2)
        ffn_output = self.ffn(ffn_input) + residual_ffn
        
        return ffn_output

class EncoderBlock(nn.Module):
    def __init__(self, 
                 emb_dim,
                 kernel_size,
                 reduction_ratio,
                 head_num,
                 expantion_ratio,
                 block_num):
        super().__init__()
       
        self.Encoder = nn.Sequential(*[ViTEncoderMixFFN(emb_dim,
                                                        kernel_size,
                                                        reduction_ratio,
                                                        head_num,
                                                        expantion_ratio)
                                       for _ in range(block_num)])
        
    def forward(self, x):
        x = self.Encoder(x)
        return x
    
class SegPhaseBlock(nn.Module):
    def __init__(self, 
                 in_length,
                 in_channels, 
                 emb_dim, 
                 patch_size,
                 stride, 
                 head_num, 
                 reduction_ratio,
                 expantion_ratio, 
                 block_num):
        super().__init__() 
        self.OLPM = OverLapPatchMerging(in_channels, 
                                        emb_dim, 
                                        patch_size, 
                                        stride,
                                        in_length)
        
        self.ENCB = EncoderBlock(emb_dim = emb_dim,
                                 kernel_size = patch_size,
                                 reduction_ratio = reduction_ratio,
                                 head_num = head_num,
                                 expantion_ratio = expantion_ratio,
                                 block_num = block_num)
        
    def forward(self,x):
        x = self.OLPM(x)
        x = self.ENCB(x)
        x = rearrange(x, 'B L C -> B C L')
        return x
    
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=kernel_size,
                              bias=False, 
                              padding='same')
        
        self.bn = nn.BatchNorm1d(out_channels)
        
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SegPhaseOutput(nn.Module):
    def __init__(self, 
                 ch1, ch2, ch3, 
                 st1, st2, st3,
                 ks,
                 class_num):
        super().__init__()
        
        self.kernel_size = ks
        self.ch = 64
        
        self.conv1 = ConvBNReLU(in_channels=ch1, out_channels=self.ch, kernel_size=self.kernel_size)
        self.conv2 = ConvBNReLU(in_channels=ch2, out_channels=self.ch, kernel_size=self.kernel_size)
        self.conv3 = ConvBNReLU(in_channels=ch3, out_channels=self.ch, kernel_size=self.kernel_size)
        
        self.conv4 = ConvBNReLU(in_channels=self.ch*3, out_channels=self.ch, kernel_size=self.kernel_size)
        
        self.conv5 = nn.Conv1d(in_channels=self.ch,
                               out_channels=class_num,
                               kernel_size=1,
                               padding='same')
        
        self.Up1 = nn.Upsample(scale_factor=st1*st2*st3, mode='linear', align_corners=True)
        self.Up2 = nn.Upsample(scale_factor=st2*st3, mode='linear', align_corners=True)
        self.Up3 = nn.Upsample(scale_factor=st3, mode='linear', align_corners=True)
        
        self.relu = nn.ReLU()
        
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self, x1, x2, x3):
        out1 = self.Up1(x1)
        out1 = self.conv1(out1)
       
        out2 = self.Up2(x2)
        out2 = self.conv2(out2)

        out3 = self.Up3(x3)
        out3 = self.conv3(out3)
       
        out = torch.concat([out1, out2, out3], dim = 1)
        
        out = self.conv4(out)
        
        out = self.conv5(out)
        out = self.softmax(out)
        return out
    
class Model(nn.Module):
    def __init__(self, 
                 in_length,
                 in_channels,
                 class_num,
                 strides,
                 kernel_size,
                 expantion_ratio: int = 4
                 ):
        super().__init__()
        
        
        self.ch1 = 16
        self.ch2 = 32
        self.ch3 = 64
        
        self.ks1 = strides[0]*2-1
        self.ks2 = strides[1]*2-1
        self.ks3 = strides[2]*2-1
        
        self.st1 = strides[0]
        self.st2 = strides[1]
        self.st3 = strides[2]
        
        self.hn1 = 2
        self.hn2 = 4
        self.hn3 = 8
        
        self.rr1 = 3
        self.rr2 = 2
        self.rr3 = 1
        
        self.bn1 = 3
        self.bn2 = 3
        self.bn3 = 3
      
        self.seg_block1 = SegPhaseBlock(in_length = in_length,
                                         in_channels = in_channels, 
                                         emb_dim = self.ch1, 
                                         patch_size = self.ks1, 
                                         stride=self.st1,
                                         head_num = self.hn1,
                                         reduction_ratio = self.rr1,
                                         expantion_ratio = expantion_ratio, 
                                         block_num = self.bn1)
        
        self.seg_block2 = SegPhaseBlock(in_length = in_length//self.st1,
                                         in_channels = self.ch1, 
                                         emb_dim = self.ch2, 
                                         patch_size = self.ks2, 
                                         stride = self.st2,
                                         head_num = self.hn2, 
                                         reduction_ratio = self.rr2,
                                         expantion_ratio = expantion_ratio, 
                                         block_num = self.bn2)
        
        self.seg_block3 = SegPhaseBlock(in_length = in_length//(self.st1*self.st2),
                                         in_channels = self.ch2, 
                                         emb_dim = self.ch3, 
                                         patch_size = self.ks3, 
                                         stride = self.st3,
                                         head_num = self.hn3, 
                                         reduction_ratio = self.rr3,
                                         expantion_ratio = expantion_ratio, 
                                         block_num = self.bn3)
        
        self.output = SegPhaseOutput(self.ch3, self.ch2, self.ch1, 
                                      self.st3, self.st2, self.st1,
                                      kernel_size,
                                      class_num)
        
    def forward(self, x):
        x1 = self.seg_block1(x)
        x2 = self.seg_block2(x1)
        x3 = self.seg_block3(x2)
        out = self.output(x3, x2, x1)
        return out
    
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model = ModelTF(    in_length=100*30,    in_channels=3,    class_num=3,    strides=[3,2,2],    kernel_size=3).to('cpu')
    #model = Model(in_length=100*30, in_channels=3, class_num=3, strides=[3,2,2], kernel_size=3).to('cpu')
    summary(model, input_size=(32, 3, 100*30))
    
    # model = Model(in_length=250*30, in_channels=3, class_num=3, strides=[5,3,2]).to('cpu')
    # summary(model, input_size=(32, 3, 250*30))
    
    # model = Model(in_length=100*30, in_channels=1, class_num=2, strides=[3,2,2]).to('cpu')
    # summary(model, input_size=(32, 1, 100*30))
