import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchinfo import summary
import math
class LightSSTNetLikeBranch(nn.Module):#只用某個維度來輕量
    def __init__(
        self,
        out_ch,
        in_channels=3,
        channel_type=1,#0mix1Z2N3E
        sample_rate=100,
        n_freq=32,
        kernel_size=65,
        f_min=1,
        f_max=45.0,
        sparse_gate=True,
        use_log_compression=True,
    ):
        super().__init__()

        self.channel_mix = nn.Conv1d(in_channels, 1, kernel_size=1, bias=False)
        
        self.branch = SSTNetLikeBranch(
            out_ch=out_ch,
            in_channels=1,
            sample_rate=sample_rate,
            n_freq=n_freq,
            kernel_size=kernel_size,
            f_min=f_min,
            f_max=f_max,
            sparse_gate=sparse_gate,
            use_log_compression=use_log_compression,
        )
        self.channel_type = channel_type
    def forward(self, x, target_len, return_tfmap=False):
        match self.channel_type:
            case 0:
                x = self.channel_mix(x)  # (B, 1, T)
                #x= torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + 1e-8)##RSS
            case 1:
                x = x[:, 0:1, :]##Z
            case 2:
                x = x[:, 1:2, :]##N
            case 3:
                x = x[:, 2:3, :]##E
                
        return self.branch(x, target_len, return_tfmap)

class SSTNetLikeBranch2(nn.Module):
    """
    Learnable GST-like / S-transform-like branch with ablation switches.

    Input:
        x: (B, C, T), e.g. (B, 3, 3000)

    Output:
        feat: (B, out_ch, target_len)

    Learnable / fixed parameters:
        f_i : center frequency
        p_i : frequency-dependent window scaling exponent
        K_i : Gaussian window scale
        w_i : optional frequency gate

    Gaussian window:
        sigma_t,i = K_i / f_i ^ p_i

    Kernel:
        real_i(t) = g_i(t) * cos(2*pi*f_i*t)
        imag_i(t) = -g_i(t) * sin(2*pi*f_i*t)

    Ablation switches:
        learn_freq=False  -> fixed center frequencies
        learn_p=False     -> fixed p_i = 1
        learn_sigma=False -> fixed K_i = 1
        learn_gate=False  -> fixed gate = 1
    """

    def __init__(
        self,
        out_ch,
        in_channels=3,
        sample_rate=100,
        n_freq=10,
        kernel_size=129,
        f_min=1.0,
        f_max=20.0,
        p_min=0.3,
        p_max=1.7,
        sparse_gate=True,
        use_log_compression=True,
        gate_init=2.0,
        learn_freq=True,
        learn_p=False,
        learn_sigma=True,
        learn_gate=True,
    ):
        super().__init__()

        self.out_ch = out_ch
        self.in_channels = in_channels
        self.sample_rate = sample_rate
        self.n_freq = n_freq
        self.kernel_size = kernel_size
        self.sparse_gate = sparse_gate
        self.use_log_compression = use_log_compression

        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.p_min = float(p_min)
        self.p_max = float(p_max)

        self.learn_freq = learn_freq
        self.learn_p = learn_p
        self.learn_sigma = learn_sigma
        self.learn_gate = learn_gate

        nyquist = sample_rate / 2.0
        if self.f_max >= nyquist:
            raise ValueError(
                f"f_max must be smaller than Nyquist frequency. "
                f"Got f_max={self.f_max}, Nyquist={nyquist}"
            )

        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size should be odd for a symmetric kernel. Got {kernel_size}."
            )

        def inv_sigmoid(x, eps=1e-4):
            x = torch.clamp(x, eps, 1.0 - eps)
            return torch.log(x / (1.0 - x))

        # --------------------------------------------------
        # Center frequencies f_i
        # If learnable:
        #   f_i = f_min + (f_max - f_min) * sigmoid(raw_freqs)
        # If fixed:
        #   f_i = linearly spaced frequencies
        # --------------------------------------------------
        freqs = torch.linspace(self.f_min, self.f_max, n_freq)

        if self.learn_freq:
            freqs_norm = (freqs - self.f_min) / (self.f_max - self.f_min)
            self.raw_freqs = nn.Parameter(inv_sigmoid(freqs_norm))
        else:
            self.register_buffer("fixed_freqs", freqs)

        # --------------------------------------------------
        # Frequency exponent p_i
        # If learnable:
        #   p_i = p_min + (p_max - p_min) * sigmoid(raw_p)
        # If fixed:
        #   p_i = 1
        # --------------------------------------------------
        p_init = torch.ones(n_freq)

        if self.learn_p:
            p_norm = (p_init - self.p_min) / (self.p_max - self.p_min)
            self.raw_p = nn.Parameter(inv_sigmoid(p_norm))
        else:
            self.register_buffer("fixed_p", p_init)

        # --------------------------------------------------
        # Gaussian window scale K_i
        # If learnable:
        #   K_i = exp(log_sigma_scale_i)
        # If fixed:
        #   K_i = 1
        # --------------------------------------------------
        if self.learn_sigma:
            self.log_sigma_scale = nn.Parameter(torch.zeros(n_freq))
        else:
            self.register_buffer("fixed_sigma_scale", torch.ones(n_freq))

        # --------------------------------------------------
        # Optional frequency gate w_i
        # If learnable:
        #   w_i = sigmoid(freq_gate_i)
        # If fixed:
        #   w_i = 1
        # --------------------------------------------------
        if self.sparse_gate:
            if self.learn_gate:
                self.freq_gate = nn.Parameter(torch.full((n_freq,), float(gate_init)))
            else:
                self.register_buffer("fixed_gate", torch.ones(n_freq))

        # --------------------------------------------------
        # Project the GST-like time-frequency map into 1D features
        # Input shape after transform: (B, C, F, T)
        # --------------------------------------------------
            '''
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=(3, 7),
                stride=(1, 3),
                padding=(1, 3),
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            '''
        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                16,
                kernel_size=(3, 7),
                padding=(1, 3),
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
    
            nn.Conv2d(
                16,
                32,
                kernel_size=(3, 5),
                padding=(1, 2),
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # (B, 32, F, T) -> (B, 32, 1, T)
            nn.AdaptiveAvgPool2d((1, None)),
        )

        self.out_proj = nn.Sequential(
            nn.Conv1d(32, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )

    def _get_freqs(self, device, dtype):
        """
        Return center frequencies.
        Shape: (F,)
        """
        if self.learn_freq:
            freqs = self.f_min + (self.f_max - self.f_min) * torch.sigmoid(self.raw_freqs)
        else:
            freqs = self.fixed_freqs

        return freqs.to(device=device, dtype=dtype)

    def _get_p(self, device, dtype):
        """
        Return p values.
        Shape: (F,)
        """
        if self.learn_p:
            p = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(self.raw_p)
        else:
            p = self.fixed_p

        return p.to(device=device, dtype=dtype)

    def _get_sigma_scale(self, device, dtype):
        """
        Return K_i values.
        Shape: (F,)
        """
        if self.learn_sigma:
            sigma_scale = torch.exp(self.log_sigma_scale)
        else:
            sigma_scale = self.fixed_sigma_scale

        return sigma_scale.to(device=device, dtype=dtype)

    def _get_gate(self, device, dtype):
        """
        Return frequency gate values.
        Shape: (F,)
        """
        if not self.sparse_gate:
            return None

        if self.learn_gate:
            gate = torch.sigmoid(self.freq_gate)
        else:
            gate = self.fixed_gate

        return gate.to(device=device, dtype=dtype)

    def _build_complex_kernels(self, device, dtype):
        """
        Build GST-like complex Gaussian-modulated sinusoidal kernels.

        Return:
            real_kernel: (F, 1, K)
            imag_kernel: (F, 1, K)
        """
        K = self.kernel_size
        center = K // 2

        # Time axis in seconds: shape (K,)
        t = torch.arange(K, device=device, dtype=dtype) - center
        t = t / self.sample_rate

        # f_i
        freqs = self._get_freqs(device=device, dtype=dtype)

        # p_i
        p = self._get_p(device=device, dtype=dtype)

        # K_i
        sigma_scale = self._get_sigma_scale(device=device, dtype=dtype)

        # Generalized S-transform-like window:
        #   sigma_t,i = K_i / f_i^p_i
        # Equivalent:
        #   exp(-0.5 * ((t * f_i^p_i) / K_i)^2)
        freq_power = torch.pow(freqs + 1e-6, p)

        gaussian = torch.exp(
            -0.5 * (
                (t[None, :] * freq_power[:, None])
                / (sigma_scale[:, None] + 1e-6)
            ) ** 2
        )

        # Center frequency modulation
        phase = 2.0 * math.pi * freqs[:, None] * t[None, :]

        real = gaussian * torch.cos(phase)
        imag = -gaussian * torch.sin(phase)

        # Normalize each filter to avoid scale domination
        real = real / (real.norm(dim=1, keepdim=True) + 1e-6)
        imag = imag / (imag.norm(dim=1, keepdim=True) + 1e-6)

        real = real.unsqueeze(1)  # (F, 1, K)
        imag = imag.unsqueeze(1)  # (F, 1, K)

        return real, imag

    def forward(self, x, target_len, return_tfmap=False):
        """
        x:
            (B, C, T)

        return:
            feat: (B, out_ch, target_len)

        if return_tfmap=True:
            return feat, tf_map
            tf_map: (B, C, F, T)
        """
        B, C, T = x.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, "
                f"but got {C} channels."
            )

        real_kernel, imag_kernel = self._build_complex_kernels(
            device=x.device,
            dtype=x.dtype
        )

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

        # Magnitude spectrum: (B*C, F, T)
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

        # Time-frequency map: (B, C, F, T)
        mag = mag.reshape(B, C, self.n_freq, T)

        if self.use_log_compression:
            mag = torch.log1p(mag)

        if self.sparse_gate:
            gate = self._get_gate(device=x.device, dtype=x.dtype)
            gate = gate.view(1, 1, self.n_freq, 1)
            mag = mag * gate

        tf_map = mag

        # CNN projection
        feat = self.proj(mag)     # (B, 32, 1, T)
        feat = feat.squeeze(2)    # (B, 32, T)

        feat = F.interpolate(
            feat,
            size=target_len,
            mode="linear",
            align_corners=True
        )

        feat = self.out_proj(feat)  # (B, out_ch, target_len)

        if return_tfmap:
            return feat, tf_map

        return feat

    def get_learned_tf_params(self):
        """
        Use this after or during training to inspect current TF parameters.
        Even fixed parameters will also be returned.
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype

            freqs = self._get_freqs(device=device, dtype=dtype)
            p = self._get_p(device=device, dtype=dtype)
            sigma_scale = self._get_sigma_scale(device=device, dtype=dtype)

            if self.sparse_gate:
                gate = self._get_gate(device=device, dtype=dtype)
            else:
                gate = None

        return {
            "freqs": freqs.detach().cpu(),
            "p": p.detach().cpu(),
            "sigma_scale": sigma_scale.detach().cpu(),
            "gate": None if gate is None else gate.detach().cpu(),
            "learn_freq": self.learn_freq,
            "learn_p": self.learn_p,
            "learn_sigma": self.learn_sigma,
            "learn_gate": self.learn_gate,
        }
class SSTNetLikeBranch(nn.Module):
    """
    SSTNet-like S-transform branch.

    Input:
        x: (B, C, T), e.g. (B, 3, 3000)

    Output:
        feat: (B, out_ch, target_len)

    Concept:
        waveform -> learnable S-transform-like filter bank
                 -> sparse frequency gating
                 -> CNN projection
                 -> temporal resize
    """
    def __init__(
        self,
        out_ch,
        in_channels=3,
        sample_rate=100,
        n_freq=64,
        kernel_size=129,
        f_min=1,#0.5
        f_max=45.0,
        sparse_gate=True,
        use_log_compression=True,
    ):
        super().__init__()

        self.out_ch = out_ch
        self.in_channels = in_channels
        self.sample_rate = sample_rate
        self.n_freq = n_freq
        self.kernel_size = kernel_size
        self.sparse_gate = sparse_gate
        self.use_log_compression = use_log_compression

        # Frequency initialization
        freqs = torch.linspace(f_min, f_max, n_freq)
        self.log_freqs = nn.Parameter(torch.log(freqs))

        # Controls Gaussian window width.
        # A larger value gives a wider window.
        self.log_sigma_scale = nn.Parameter(torch.zeros(n_freq))

        if sparse_gate:
            # Learnable frequency importance
            self.freq_gate = nn.Parameter(torch.zeros(n_freq))##0.5
            #self.freq_gate = nn.Parameter(torch.full((n_freq,), 3.0))##0.95

        # Project the S-transform-like spectrum into a 1D feature sequence.
        # Input shape after transform: (B, C, F, T)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 7),padding=(1, 3), bias=False), #stride=(1, 3),    3000->1000
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=(3, 5),padding=(1, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Compress the frequency dimension.
            nn.AdaptiveAvgPool2d((1, None)),
        )

        self.out_proj = nn.Sequential(
            nn.Conv1d(32, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU()
        )
    def _build_complex_kernels(self, device, dtype):
        """
        Build S-transform-like complex Gaussian-modulated sinusoidal kernels.

        Return:
            real_kernel: (F, 1, K)
            imag_kernel: (F, 1, K)
        """
        K = self.kernel_size
        center = K // 2

        t = torch.arange(K, device=device, dtype=dtype) - center
        t = t / self.sample_rate

        freqs = torch.exp(self.log_freqs).to(device=device, dtype=dtype)
        sigma_scale = torch.exp(self.log_sigma_scale).to(device=device, dtype=dtype)

        # S-transform property:
        # higher frequency uses a narrower window,
        # lower frequency uses a wider window.
        gaussian = torch.exp(
            -0.5 * ((t[None, :] * freqs[:, None]) / (sigma_scale[:, None] + 1e-6)) ** 2
        )

        phase = 2 * math.pi * freqs[:, None] * t[None, :]

        real = gaussian * torch.cos(phase)
        imag = -gaussian * torch.sin(phase)

        # Normalize each filter.
        real = real / (real.norm(dim=1, keepdim=True) + 1e-6)
        imag = imag / (imag.norm(dim=1, keepdim=True) + 1e-6)

        real = real.unsqueeze(1)
        imag = imag.unsqueeze(1)

        return real, imag
    def forward(self, x, target_len, return_tfmap=False):
        B, C, T = x.shape
    
        if C != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, but got {C} channels."
            )
    
        real_kernel, imag_kernel = self._build_complex_kernels(
            device=x.device,
            dtype=x.dtype
        )
    
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
    
        # Magnitude spectrum: (B*C, F, T)
        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
    
        # Time-frequency map: (B, C, F, T)
        mag = mag.reshape(B, C, self.n_freq, T)
    
        if self.use_log_compression:
            mag = torch.log1p(mag)
    
        if self.sparse_gate:
            gate = torch.sigmoid(self.freq_gate).view(1, 1, self.n_freq, 1)
            mag = mag * gate
    
    
        tf_map = mag
    
        # CNN projection
        feat = self.proj(mag)        # (B, 32, 1, T)
        feat = feat.squeeze(2)       # (B, 32, T)
    
        feat = F.interpolate(
            feat,
            size=target_len,
            mode="linear",
            align_corners=True
        )
        feat = self.out_proj(feat)   # (B, out_ch, target_len)
        #alignment
    
        if return_tfmap:
            return feat, tf_map
    
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
class LSTPatchBranch(nn.Module):
    def __init__(
        self,
        out_ch,
        in_channels=3,
        sample_rate=100,
        n_freq=10,
        kernel_size=129,
        f_min=1.0,
        f_max=20.0,
        sparse_gate=True,
        use_log_compression=True,
    ):
        super().__init__()

        self.out_ch = out_ch
        self.in_channels = in_channels
        self.sample_rate = sample_rate
        self.n_freq = n_freq
        self.kernel_size = kernel_size
        self.sparse_gate = sparse_gate
        self.use_log_compression = use_log_compression

        freqs = torch.linspace(f_min, f_max, n_freq)
        self.log_freqs = nn.Parameter(torch.log(freqs))

        self.log_sigma_scale = nn.Parameter(torch.zeros(n_freq))

        if sparse_gate:
            self.freq_gate = nn.Parameter(torch.zeros(n_freq))

        self.patch_embed = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_ch,
                kernel_size=(5, 7),
                stride=(1, 3),
                padding=(2, 3),
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def _build_complex_kernels(self, device, dtype):
        K = self.kernel_size
        center = K // 2

        t = torch.arange(K, device=device, dtype=dtype) - center
        t = t / self.sample_rate

        freqs = torch.exp(self.log_freqs).to(device=device, dtype=dtype)
        sigma_scale = torch.exp(self.log_sigma_scale).to(device=device, dtype=dtype)

        gaussian = torch.exp(
            -0.5 * ((t[None, :] * freqs[:, None]) / (sigma_scale[:, None] + 1e-6)) ** 2
        )

        phase = 2 * math.pi * freqs[:, None] * t[None, :]

        real = gaussian * torch.cos(phase)
        imag = -gaussian * torch.sin(phase)

        real = real / (real.norm(dim=1, keepdim=True) + 1e-6)
        imag = imag / (imag.norm(dim=1, keepdim=True) + 1e-6)

        return real.unsqueeze(1), imag.unsqueeze(1)

    def forward(self, x, target_len, return_tfmap=False):
        # x: (B, C, T)
        B, C, T = x.shape

        if C != self.in_channels:
            raise ValueError(
                f"Expected input with {self.in_channels} channels, but got {C} channels."
            )

        real_kernel, imag_kernel = self._build_complex_kernels(
            device=x.device,
            dtype=x.dtype
        )

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

        mag = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)

        # (B*C, Freq, T) -> (B, C, Freq, T)
        mag = mag.reshape(B, C, self.n_freq, T)

        if self.use_log_compression:
            mag = torch.log1p(mag)

        if self.sparse_gate:
            gate = torch.sigmoid(self.freq_gate).view(1, 1, self.n_freq, 1)
            mag = mag * gate

        tf_map = mag

        # Patch embedding
        feat = self.patch_embed(mag)     # (B, out_ch, Freq, T/3)

        # Compress frequency dimension
        feat = feat.mean(dim=2)          # (B, out_ch, T/3)

        if feat.shape[-1] != target_len:
            feat = F.interpolate(
                feat,
                size=target_len,
                mode="linear",
                align_corners=True
            )

        if return_tfmap:
            return feat, tf_map

        return feat                       # (B, out_ch, target_len)
class ModelLST2D(nn.Module):
    def __init__(
        self,
        in_length,
        in_channels,
        class_num,
        strides,
        kernel_size,
        expantion_ratio: int = 4
    ):
        super().__init__()

        self.in_length = in_length

        self.ch1 = 16
        self.ch2 = 32
        self.ch3 = 64

        self.ks1 = strides[0] * 2 - 1
        self.ks2 = strides[1] * 2 - 1
        self.ks3 = strides[2] * 2 - 1

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

        # ---------------------------------------------------------
        # LST frontend
        # Raw waveform -> learnable time-frequency feature
        # Output shape: (B, 16, in_length / st1)
        # ---------------------------------------------------------
        self.tf_branch = LSTPatchBranch(
            out_ch=self.ch1,
            in_channels=in_channels,
            sample_rate=100,
            n_freq=10,
            kernel_size=129,
            f_min=1,
            f_max=20.0,
            sparse_gate=True,
            use_log_compression=True
        )

        # ---------------------------------------------------------
        # ViT block 1
        # Input:  (B, 16, 1000)
        # Output: (B, 32, 500)
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # ViT block 2
        # Input:  (B, 32, 500)
        # Output: (B, 64, 250)
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Decoder
        # x3, x2, x1 -> output probability sequence
        # ---------------------------------------------------------
        self.output = SegPhaseOutput(
            self.ch3,
            self.ch2,
            self.ch1,
            self.st3,
            self.st2,
            self.st1,
            kernel_size,
            class_num
        )

    def forward(self, x, return_tfmap=False):
        # x: (B, 3, 3000)

        target_len = self.in_length // self.st1

        if return_tfmap:
            x1, tf_map = self.tf_branch(
                x,
                target_len=target_len,
                return_tfmap=True
            )
        else:
            x1 = self.tf_branch(
                x,
                target_len=target_len,
                return_tfmap=False
            )
            tf_map = None

        # x1: (B, 16, 1000)
        x2 = self.seg_block2(x1)      # (B, 32, 500)
        x3 = self.seg_block3(x2)      # (B, 64, 250)

        out = self.output(x3, x2, x1) # (B, 3, 3000)

        if return_tfmap:
            return out, tf_map

        return out
class ModelLSTCNNViT(nn.Module):#Final
    def __init__(
        self,
        in_length,
        in_channels,
        class_num,
        strides,
        kernel_size,
        expantion_ratio: int = 4
    ):
        super().__init__()

        self.in_length = in_length

        self.ch1 = 16##16
        self.ch2 = 32
        self.ch3 = 64

        self.ks1 = strides[0] * 2 - 1
        self.ks2 = strides[1] * 2 - 1
        self.ks3 = strides[2] * 2 - 1

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

        # ---------------------------------------------------------
        # LST frontend
        # Raw waveform -> learnable time-frequency feature
        # Output shape: (B, 32, in_length)
        # ---------------------------------------------------------
        self.tf_branch  = SSTNetLikeBranch2(
            out_ch=self.ch1, 
            in_channels=in_channels,
            sample_rate=100,
            n_freq=10,#64,
            kernel_size=129,#129
            f_min=0.5,#0.5
            f_max=20,
            sparse_gate=True,
            use_log_compression=True,
            learn_freq=True,#F
            learn_p=True,#P
            learn_sigma=True,#K
            learn_gate=True,#W
        )

        # ---------------------------------------------------------
        # ViT block 1
        # Input:  (B, 32, 3000)
        # Output: (B, 32, 1000)
        # ---------------------------------------------------------
        self.seg_block2 = SegPhaseBlock(
            in_length=in_length// (self.st1),  
            in_channels=self.ch1,
            emb_dim=self.ch2,
            patch_size=self.ks2,#2
            stride=self.st2,#2
            head_num=self.hn2,
            reduction_ratio=self.rr2,#2
            expantion_ratio=expantion_ratio,
            block_num=self.bn2
        )

        # ---------------------------------------------------------
        # ViT block 2
        # Input:  (B, 32, 1000)
        # Output: (B, 64, 500)
        # ---------------------------------------------------------
        self.seg_block3 = SegPhaseBlock(
            in_length=in_length // (self.st1* self.st2),
            in_channels=self.ch2,
            emb_dim=self.ch3,
            patch_size=self.ks3,#3
            stride=self.st3,#3
            head_num=self.hn3,
            reduction_ratio=self.rr3,#3
            expantion_ratio=expantion_ratio,
            block_num=self.bn3
        )

        # ---------------------------------------------------------
        # Decoder
        # x3, x2, x1 -> output probability sequence
        # ---------------------------------------------------------
        self.output = SegPhaseOutput(
            self.ch3,
            self.ch2,
            self.ch1, 
            self.st3,   # 2
            self.st2,
            self.st1,
            #1,
            kernel_size,
            class_num
        )

    def forward(self, x, return_tfmap=False):
        # x: (B, 3, 3000)

        target_len = self.in_length // self.st1

        if return_tfmap:
            x1, tf_map = self.tf_branch(
                x,
                target_len=target_len,
                return_tfmap=True
            )
        else:
            x1 = self.tf_branch(
                x,
                target_len=target_len,
                return_tfmap=False
            )
            tf_map = None

        # x1: (B, 32, 3000)
        x2 = self.seg_block2(x1)      # (B, 32, 1000)
        x3 = self.seg_block3(x2)      # (B, 64, 500)

        out = self.output(x3, x2, x1) # (B, 3, 3000)

        if return_tfmap:
            return out, tf_map

        return out
class ModelLST(nn.Module):
    def __init__(
        self,
        in_length,
        in_channels,
        Lightmode,
        class_num,
        strides,
        kernel_size,
        expantion_ratio: int = 4
    ):
        super().__init__()

        self.ch1 = 16
        self.ch2 = 32
        self.ch3 = 64

        self.ks1 = strides[0] * 2 - 1
        self.ks2 = strides[1] * 2 - 1
        self.ks3 = strides[2] * 2 - 1

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

        # SegPhase block 1
        self.seg_block1 = SegPhaseBlock(
            in_length=in_length,
            in_channels=in_channels,
            emb_dim=self.ch1,
            patch_size=self.ks1,
            stride=self.st1,
            head_num=self.hn1,
            reduction_ratio=self.rr1,
            expantion_ratio=expantion_ratio,
            block_num=self.bn1
        )

        # SegPhase block 2
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

        # SegPhase block 3
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

        # SSTNet-like time-frequency branch.
        # This branch is fused with the third SegPhase feature.
        if Lightmode:
            self.tf_branch = LightSSTNetLikeBranch(
                out_ch=self.ch1,
                in_channels=in_channels,
                channel_type=0,#0mix1Z2N3E
                sample_rate=100,
                n_freq=64,
                kernel_size=129, 
                f_min=1,
                f_max=45.0,
                sparse_gate=True,
                use_log_compression=True
            )
        else:
            
            self.tf_branch  = SSTNetLikeBranch2(
                out_ch=self.ch1,
                in_channels=in_channels,
                sample_rate=100,
                n_freq=64,#64,
                kernel_size=129,#129
                f_min=1,#0.5
                f_max=45,
                sparse_gate=True,
                use_log_compression=True,
                learn_freq=True,#F
                learn_p=True,#P
                learn_sigma=True,#K
                learn_gate=True,#W
            )

        # Feature fusion at the third SegPhase stage.
        self.fusion3 = FeatureFusion(self.ch1)

        self.output = SegPhaseOutput(
            self.ch3, self.ch2, self.ch1,
            self.st3, self.st2, self.st1,
            kernel_size,
            class_num
        )
        self.Foutput = FSegPhaseOutput(self.ch1, self.st1,
            kernel_size,
            class_num
        )
    def forward(self, x, ablation=False,return_tfmap=False):
        # Main SegPhase branch
        x1 = self.seg_block1(x)       # (B, 16, L1)   
        
            # Fusion
        if ablation:#freq #if time then just segphase
            # SSTNet-like time-frequency branch 
            if return_tfmap:
                tf1, tf_map = self.tf_branch(
                    x,
                    target_len=x1.shape[-1],
                    return_tfmap=True
                )
            else:
                tf1 = self.tf_branch(
                    x,
                    target_len=x1.shape[-1],
                    return_tfmap=False
                )
                tf_map = None
            out = self.Foutput(tf1)
        else:#all
            # SSTNet-like time-frequency branch
            if return_tfmap:
                tf1, tf_map = self.tf_branch(
                    x,
                    target_len=x1.shape[-1],
                    return_tfmap=True
                )
            else:
                tf1 = self.tf_branch(
                    x,
                    target_len=x1.shape[-1],
                    return_tfmap=False
                )
                tf_map = None
        
            x1 = self.fusion3(x1, tf1)
            x2 = self.seg_block2(x1)
            x3 = self.seg_block3(x2)  
            out = self.output(x3, x2, x1)
        
        if return_tfmap:
            return out, tf_map
    
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
    
class FSegPhaseOutput(nn.Module):
    def __init__(self, 
                 ch1, st1, ks,
                 class_num):
        super().__init__()
        
        self.kernel_size = ks
        self.ch = 64
        
        self.conv1 = ConvBNReLU(in_channels=ch1, out_channels=self.ch, kernel_size=self.kernel_size)
        
        
        self.conv5 = nn.Conv1d(in_channels=self.ch,
                               out_channels=class_num,
                               kernel_size=1,
                               padding='same')
        
        self.Up1 = nn.Upsample(scale_factor=st1, mode='linear', align_corners=True)
        
        self.relu = nn.ReLU()
        
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self, x1):
        out1 = self.Up1(x1)
        out = self.conv1(out1)
        
        out = self.conv5(out)
        out = self.softmax(out)
        return out
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    model = ModelSST(    in_length=100*30,    in_channels=3,    class_num=3,    strides=[3,2,2],    kernel_size=3).to('cpu')
    #model = Model(in_length=100*30, in_channels=3, class_num=3, strides=[3,2,2], kernel_size=3).to('cpu')
    summary(model, input_size=(32, 3, 100*30))
    
    # model = Model(in_length=250*30, in_channels=3, class_num=3, strides=[5,3,2]).to('cpu')
    # summary(model, input_size=(32, 3, 250*30))
    
    # model = Model(in_length=100*30, in_channels=1, class_num=2, strides=[3,2,2]).to('cpu')
    # summary(model, input_size=(32, 1, 100*30))
