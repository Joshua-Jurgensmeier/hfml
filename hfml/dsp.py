import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.functional import avg_pool2d

class EnergyDetector(nn.Module):
    def __init__(self, n_bins: int, pwr_thresh_dB: float, win_kernel_size=(4,4)):
        super().__init__()
        self.n_bins = n_bins
        self.pwr_thresh_dB = pwr_thresh_dB
        self.win_kernel_size = win_kernel_size
        
    def forward(self, spectrograms: torch.Tensor):      
        # Estimate noise floor with time avg of avg of bin avg
        noise_floor_dB = rearrange(spectrograms, "M T (B F) -> M T B F", B=self.n_bins).mean(dim=-1).mean(dim=-1).mean(dim=-1)
        
        spectrogram_avg_dB = avg_pool2d(spectrograms, self.win_kernel_size)

        detections = spectrogram_avg_dB > (noise_floor_dB + self.pwr_thresh_dB).unsqueeze(-1).unsqueeze(-1)
        chan_detects = detections.any(dim=-1).any(dim=-1)

        return chan_detects, noise_floor_dB

class Spectrogrammer(nn.Module):
    def __init__(self, nfft, hop_overlap, num_channels):
        self.nfft = nfft
        self.hop_length = int(nfft * hop_overlap)
        self.window = torch.hann_window(nfft, periodic=True)
        overlap = nfft - self.hop_length
        self.pad = torch.zeros(overlap)
        self.m = num_channels

    def forward(self, x):
        self.pad = self.pad.to(x)
        self.window = self.window.to(x)
        
        x = torch.cat([self.pad, x], -1)
        sg = 20*torch.stft(x, self.nfft, self.hop_length, window=self.window, center=False).abs().log10().T
        sgs = rearrange(sg, "T (M F) -> M T F", M=self.m)
        
        return sgs