"""
AASIST (Attention-based Audio Spoofing and Injection Spectrogram Transformer) model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer.
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # Weight matrix for linear transformation
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        
        # Attention parameters
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)
        
        # Leaky ReLU for attention mechanism
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj):
        """
        h: Input features (batch_size, N, in_features)
        adj: Adjacency matrix (batch_size, N, N)
        """
        batch_size, N, _ = h.size()
        
        # Linear transformation
        Wh = torch.matmul(h, self.W)  # (batch_size, N, out_features)
        
        # Self-attention mechanism
        a_input = torch.cat([Wh.repeat(1, 1, N).view(batch_size, N * N, -1), 
                            Wh.repeat(1, N, 1)], dim=2).view(batch_size, N, N, 2 * self.out_features)
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # (batch_size, N, N)
        
        # Mask attention coefficients for non-existing edges
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Normalize attention coefficients
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention
        h_prime = torch.matmul(attention, Wh)  # (batch_size, N, out_features)
        
        return h_prime

class SincConv(nn.Module):
    """
    Sinc-based convolution layer for processing raw waveforms.
    """
    def __init__(self, out_channels, kernel_size, sample_rate=16000):
        super(SincConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        
        # Initialize filterbanks
        self.freq_low = nn.Parameter(torch.Tensor(out_channels, 1))
        self.freq_band = nn.Parameter(torch.Tensor(out_channels, 1))
        
        # Initialize parameters
        self.freq_low.data.uniform_(0.0, self.sample_rate / 2 - 1)
        self.freq_band.data.uniform_(0.0, self.sample_rate / 2)
        
    def forward(self, x):
        """
        x: Input signal (batch_size, 1, signal_length)
        """
        # Get filterbanks
        filters = self.get_filters()  # (out_channels, 1, kernel_size)
        
        # Convolve input with filterbanks
        output = F.conv1d(x, filters, padding=(self.kernel_size - 1) // 2)
        
        return output
    
    def get_filters(self):
        """
        Generate bandpass filter bank.
        """
        # Ensure frequencies are positive and ordered
        freq_low = torch.abs(self.freq_low)
        freq_band = torch.abs(self.freq_band)
        freq_high = torch.clamp(freq_low + freq_band, 0, self.sample_rate / 2)
        
        # Normalize frequencies
        freq_low_normalized = freq_low / (self.sample_rate / 2)
        freq_high_normalized = freq_high / (self.sample_rate / 2)
        
        # Create filter grid
        t = torch.linspace(-0.5, 0.5, self.kernel_size).view(1, 1, -1)
        t = t.expand(self.out_channels, 1, self.kernel_size)
        
        # Compute sinc filters
        low_pass1 = 2 * freq_low_normalized * torch.sinc(2 * freq_low_normalized * t * math.pi)
        low_pass2 = 2 * freq_high_normalized * torch.sinc(2 * freq_high_normalized * t * math.pi)
        band_pass = low_pass2 - low_pass1
        
        # Apply Hamming window
        window = 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(self.kernel_size) / (self.kernel_size - 1))
        window = window.view(1, 1, -1).expand(self.out_channels, 1, self.kernel_size)
        filters = band_pass * window
        
        # Normalize filters
        energy = torch.sum(filters**2, dim=2, keepdim=True)
        filters = filters / torch.sqrt(energy)
        
        return filters

class ResidualBlock(nn.Module):
    """
    Residual block with convolution layers.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class AASIST(nn.Module):
    """
    AASIST (Attention-based Audio Spoofing and Injection Spectrogram Transformer) model.
    """
    def __init__(self, raw_input=True, sinc_filter=True, graph_attention=True, n_class=2):
        super(AASIST, self).__init__()
        self.raw_input = raw_input
        self.sinc_filter = sinc_filter
        self.graph_attention = graph_attention
        
        # Frontend processing for raw waveform
        if raw_input:
            if sinc_filter:
                self.frontend = SincConv(out_channels=64, kernel_size=251, sample_rate=16000)
            else:
                self.frontend = nn.Conv1d(1, 64, kernel_size=251, stride=1, padding=125)
            
            self.bn_frontend = nn.BatchNorm1d(64)
            self.frontend_activation = nn.ReLU()
            self.frontend_pooling = nn.MaxPool1d(kernel_size=3)
        
        # Feature extraction (CNN)
        self.conv_block1 = ResidualBlock(in_channels=1 if not raw_input else 64, out_channels=32, stride=1)
        self.conv_block2 = ResidualBlock(in_channels=32, out_channels=64, stride=2)
        self.conv_block3 = ResidualBlock(in_channels=64, out_channels=128, stride=2)
        self.conv_block4 = ResidualBlock(in_channels=128, out_channels=256, stride=2)
        
        # Graph attention layers
        if graph_attention:
            self.gat1 = GraphAttentionLayer(in_features=256, out_features=256)
            self.gat2 = GraphAttentionLayer(in_features=256, out_features=256)
        
        # Classification layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, n_class)
    
    def forward(self, x):
        """
        x: Input signal (batch_size, 1, signal_length) or spectrogram (batch_size, 1, time_steps, freq_bins)
        """
        batch_size = x.size(0)
        
        # Process raw waveform if needed
        if self.raw_input:
            # Frontend processing
            x = self.frontend(x)
            x = self.bn_frontend(x)
            x = self.frontend_activation(x)
            x = self.frontend_pooling(x)
            
            # Reshape for 2D convolution
            x = x.unsqueeze(1)  # (batch_size, 1, 64, time_steps)
        
        # Feature extraction (CNN)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        # Reshape for graph attention
        if self.graph_attention:
            # Create a fully connected graph
            n_nodes = x.size(2) * x.size(3)
            x = x.view(batch_size, 256, -1).transpose(1, 2)  # (batch_size, n_nodes, 256)
            
            # Create adjacency matrix for fully connected graph
            adj = torch.ones(batch_size, n_nodes, n_nodes, device=x.device)
            
            # Apply graph attention
            x = self.gat1(x, adj)
            x = F.relu(x)
            x = self.gat2(x, adj)
            x = F.relu(x)
            
            # Global pooling
            x = torch.mean(x, dim=1)  # (batch_size, 256)
        else:
            # Global average pooling
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.view(batch_size, -1)
        
        # Classification
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x 