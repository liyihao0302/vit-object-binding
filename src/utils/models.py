from torch import nn
from torch.nn import functional as F
import torch
import math

class LinearProbe(nn.Module):
    '''IsSameObject(x,y) = sigmoid(W[x;y] + b)'''
    def __init__(self, cfg):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(cfg.probe.in_dim, 1)
        
    def forward(self, x, y):
        #[B, 2C] -> [B,1]
        out = self.linear(x) + self.linear(y)
        return out
    def forward_pairwise(self, x, y):
        #[B, C], [B,C] -> [B,B]
        B, C = x.shape
        # Expand x and y to form pairwise combinations: [B, 1, C] and [1, B, C]
        x_exp = x.unsqueeze(1).expand(B, B, C)  # Shape: [B, B, C]
        y_exp = y.unsqueeze(0).expand(B, B, C)  # Shape: [B, B, C]

        out = self.linear(x_exp) + self.linear(y_exp)  # Shape: [B, B, 1]
        #[B, B, 1] -> [B, B]
        return out.squeeze(-1)

class DiagonalQuadraticProbe(nn.Module):
    '''IsSameObject(x,y) = sigmoid(x^T W y + b)'''
    def __init__(self, cfg):
        super(DiagonalQuadraticProbe, self).__init__()
        
        self.diag = nn.Linear(cfg.probe.in_dim, 1)
        
        
    def forward(self, x, y):
        # linear([B,C]) -> [B,1]
        # Element-wise product: equivalent to diag(W) in matrix form
        out = self.diag(x * y)
        return out
    def forward_pairwise(self, x, y): 
        B, C = x.shape
        x_exp = x.unsqueeze(1).expand(B, B, C)  # Shape: [B, B, C]
        y_exp = y.unsqueeze(0).expand(B, B, C)  # Shape: [B, B, C]
        pairwise_product = x_exp * y_exp  # Shape: [B, B, C]
        out = self.diag(pairwise_product).squeeze(-1)  # Shape: [B, B, 1] -> [B, B]
        return out

class QuadraticProbe(nn.Module):
    '''IsSameObject(x,y) = sigmoid(x^T W y + b). Ideally, rank(W) < C'''
    def __init__(self, cfg):
        super(QuadraticProbe, self).__init__()
        self.W = nn.Linear(cfg.probe.in_dim, cfg.probe.in_dim)
        
        
    def forward(self, x, y):
        W_sym = (self.W.weight + self.W.weight.T) / 2 / math.sqrt(x.shape[1])
        # [B, C] @ [C, C] * [B,C] ->  [B,C]
        out = torch.sum((x @ W_sym) * y, dim=1) + self.W.bias[0]
        return out
    def forward_pairwise(self, x, y):
        W_sym = (self.W.weight + self.W.weight.T) / 2 / math.sqrt(x.shape[1])
        out = x @ W_sym @ y.T + self.W.bias[0]
        return out

class LinearClassifier(nn.Module):
    def __init__(self, cfg):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(cfg.probe.in_dim, cfg.probe.out_dim)
        
    def forward(self, x):
        out = self.linear(x)
        return out

class LinearInstanceSeg(nn.Module):
    def __init__(self, cfg):
        super(LinearInstanceSeg, self).__init__()
        self.linear = nn.Linear(cfg.probe.in_dim, cfg.seg.num_queries)
    def forward(self, x):
        out = self.linear(x)
        return out

def get_model(cfg):
    
    if cfg.probe.mode == 'linear':
        model = LinearProbe(cfg)
    elif cfg.probe.mode == 'diag_quadratic':
        model = DiagonalQuadraticProbe(cfg)
    elif cfg.probe.mode == 'quadratic':
        model = QuadraticProbe(cfg)
    elif cfg.probe.mode == 'linear_class':
        model = LinearClassifier(cfg)
    elif cfg.probe.mode == 'linear_instance':
        model = LinearInstanceSeg(cfg)
    
    return model