from torch import nn
from torch.nn import functional as F
import torch
import math
from transformers import AutoModel

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

class SoftmaxProbe(nn.Module):
    '''IsSameObject(x,y) = softmax(Wx)^T @ softmax(Wy)'''
    def __init__(self, cfg):
        super(SoftmaxProbe, self).__init__()
        self.linear = nn.Linear(cfg.probe.in_dim, cfg.probe.out_dim)
        
        
    def forward(self, x, y):
        out = torch.sum(F.softmax(self.linear(x), dim=1) * F.softmax(self.linear(y), dim=1), dim=1)
        return out
    def forward_pairwise(self, x, y):
        out = F.softmax(self.linear(x), dim=1) @ F.softmax(self.linear(y), dim=1).T
        return out


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


class DotProductProbe(nn.Module):
    '''IsSameObject(x,y) = sigmoid(x^T W y + b)'''
    def __init__(self, cfg):
        super(DotProductProbe, self).__init__()
        
        self.diag = nn.Linear(cfg.probe.in_dim, 1)
        self.diag.weight.data = torch.ones_like(self.diag.weight.data)
        self.diag.weight.requires_grad = False
        
        
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


class CosineSimilarityProbe(nn.Module):
    '''IsSameObject(x,y) = sigmoid(x^T W y + b). Ideally, rank(W) < C'''
    def __init__(self, cfg):
        super(CosineSimilarityProbe, self).__init__()
        self.W = nn.Linear(cfg.probe.in_dim, cfg.probe.in_dim)
        self.W.weight.data = torch.eye(cfg.probe.in_dim)
        self.W.weight.requires_grad = False

        
    def forward(self, x, y):
        """
        x: (B, D)
        y: (B, D)
        returns: (B,)
        """
        # cosine similarity
        x_norm = x.norm(dim=1, keepdim=True)
        y_norm = y.norm(dim=1, keepdim=True)
        denom = (x_norm * y_norm).clamp(min=1e-8)

        cos = (x * y).sum(dim=1) / denom.squeeze(1)

        # add bias
        out = cos + self.W.bias[0]
        return out


    def forward_pairwise(self, x, y):
        """
        x: (N, D)
        y: (M, D)
        returns: (N, M)
        """
        # (N,1) and (1,M)
        x_norm = x.norm(dim=1, keepdim=True)     # (N,1)
        y_norm = y.norm(dim=1, keepdim=True).T   # (1,M)
        denom = (x_norm * y_norm).clamp(min=1e-8)

        # cosine similarity matrix (N,M)
        cos = (x @ y.T) / denom

        # bias is scalar
        out = cos + self.W.bias[0]
        return out


class SelfAttentionProbe(nn.Module):
    """
    IsSameObject(x, y) = (query(x) · key(y)) / sqrt(C) + b
    """
    def __init__(self, cfg):
        super().__init__()

        model = AutoModel.from_pretrained(
            cfg.model.name,
            cache_dir=cfg.data_extractor.cache_dir
        )

        # Reuse transformer attention projections
        self.Wq = model.encoder.layer[cfg.trainer.layer+1].attention.attention.query
        self.Wk = model.encoder.layer[cfg.trainer.layer+1].attention.attention.key
        for p in self.Wq.parameters():
            p.requires_grad = False
        for p in self.Wk.parameters():
            p.requires_grad = False

        self.scale = 1.0 / math.sqrt(cfg.probe.in_dim)

        # New learnable bias (scalar)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        """
        x: [B, C]
        y: [B, C]
        returns: [B] logits (pre-sigmoid)
        """
        q = self.Wq(x)  # [B, C]
        k = self.Wk(y)  # [B, C]
        out = (q * k).sum(dim=-1) * self.scale + self.bias
        return out

    def forward_pairwise(self, x, y):
        """
        x: [Bx, C]
        y: [By, C]
        returns: [Bx, By] logits (pre-sigmoid)
        """
        q = self.Wq(x)                  # [Bx, C]
        k = self.Wk(y)                  # [By, C]
        out = (q @ k.transpose(-1, -2)) * self.scale + self.bias
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

class QuadraticFixedRankProbe(nn.Module):
    '''IsSameObject(x,y) = sigmoid(x^T W y + b). Ideally, rank(W) < C'''
    def __init__(self, cfg):
        super(QuadraticFixedRankProbe, self).__init__()
        self.W1 = nn.Linear(cfg.probe.in_dim, cfg.probe.rank)
        self.W2 = nn.Linear(cfg.probe.in_dim, cfg.probe.rank)
        
    def forward(self, x, y):
        # / math.sqrt(x.shape[1])
        W_sym = (self.W1.weight.T @ self.W2.weight + self.W2.weight.T @ self.W1.weight) / 2 / math.sqrt(x.shape[1])
        # [B, C] @ [C, C] * [B,C] ->  [B,C]
        out = torch.sum((x @ W_sym) * y, dim=1) + self.W1.bias[0]
        return out
    def forward_pairwise(self, x, y):
        W_sym = (self.W1.weight.T @ self.W2.weight + self.W2.weight.T @ self.W1.weight) / 2 / math.sqrt(x.shape[1])
        out = x @ W_sym @ y.T + self.W1.bias[0]
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
        self.linear = nn.Linear(cfg.probe.in_dim, cfg.probe.instance_seg.num_queries+1)
    def forward(self, x):
        out = self.linear(x)
        return out

def get_model(cfg):
    if cfg.probe.mode == 'linear_class' or cfg.trainer.train_mode == 'pointwise_class':
        model = LinearClassifier(cfg)
    
    elif cfg.probe.mode == 'linear':
        model = LinearProbe(cfg)
    elif cfg.probe.mode == 'diag_quadratic':
        model = DiagonalQuadraticProbe(cfg)
    elif cfg.probe.mode == 'quadratic':
        model = QuadraticProbe(cfg)
    elif cfg.probe.mode == 'quadratic_fixed_rank':
        model = QuadraticFixedRankProbe(cfg)
    elif cfg.probe.mode == 'cosine_similarity':
        model = CosineSimilarityProbe(cfg)
    elif cfg.probe.mode == 'dot_product':
        model = DotProductProbe(cfg)
    
    elif cfg.probe.mode == 'linear_instance':
        model = LinearInstanceSeg(cfg)
    elif cfg.probe.mode == 'softmax':
        model = SoftmaxProbe(cfg)
    elif cfg.probe.mode == 'self_attention':
        model = SelfAttentionProbe(cfg)
    
    
    return model