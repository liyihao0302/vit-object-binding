import torch
import torch.nn.functional as F
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import autocast
def compute_attention(qkv, num_heads):
    # Return attn [B, n_heads, N, N]
    B, N, C = qkv.shape
    C = C//3
    head_dim = C// num_heads
    scale = head_dim**-0.5

    qkv = qkv.reshape(B, N, 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)

    q, k, v = qkv[0] * scale, qkv[1], qkv[2] # q, k, v [B, n_heads, N, C//n_heads]
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1) # [B, n_heads, N, N]

    return attn

def compute_issameobject(probe, act, n):
    
    _, N, C = act.shape #[1, N, C]
    n_act = act[:, n].unsqueeze(1).expand(-1, N, -1) # [1, N, C]
    
    scores = probe.forward(n_act.squeeze(0), act.squeeze(0)) #[N, 1]
    scores = F.sigmoid(scores)
    #import pdb; pdb.set_trace()
    #scores = (scores>0.5).float()
    #import pdb; pdb.set_trace()
    return scores.reshape(1, N)


class InfoNCELossWithLabels(nn.Module):
    def __init__(self, temperature=0.5):
        """
        Args:
            temperature (float): Scaling factor for softmax.
        """
        super(InfoNCELossWithLabels, self).__init__()
        self.temperature = temperature

    def forward(self, sim_matrix, labels):
        """
        Args:
            sim_matrix (Tensor): Tensor of shape (B, N, N), pairwise similarities (symmetric).
            labels (Tensor): Tensor of shape (B, N), representing class labels for each sample.
        
        Returns:
            loss (Tensor): InfoNCE loss value.
        """
        B, N, _ = sim_matrix.shape  # Batch size, number of samples per batch
        
        # Create a mask for the diagonal (self-similarities)
        diag_mask = torch.eye(N, dtype=torch.bool, device=sim_matrix.device).unsqueeze(0)  # (1, N, N)

        # Invalidate self-similarities by setting them to -inf
        sim_matrix = sim_matrix.masked_fill(diag_mask, -1e9)

        logits = sim_matrix / self.temperature  # (B, N, N)
        logits = logits.reshape(B * N, N)  # (B * N, N)

        label_matrix = labels.unsqueeze(1) == labels.unsqueeze(2)  # (B, N, N)
        label_matrix = label_matrix.masked_fill(diag_mask, 0)  # Invalidate self-similarities
        label_matrix = label_matrix.reshape(B * N, N).float()  # (B * N, N)

        mask = label_matrix.sum(dim=1) > 0
        label_matrix = label_matrix[mask]
        logits = logits[mask]
        label_matrix /= label_matrix.sum(dim=1, keepdim=True)  # Normalize
        loss = F.cross_entropy(logits, label_matrix)

        num_updates = label_matrix.shape[0]
        return loss, num_updates
       
        
        

def compute_batch_pairwise_similarity(probe, x, y):
    
    '''
    Input: x,y[B, N, C]; probe: nn.Module
    Output: pairwise_similarity [B, N, N]
    '''
    B, N, C = x.shape  # Batch size, number of elements, feature dimension
    pairwise_matrices = []
    for b in range(B):
        pairwise_b = probe.forward_pairwise(x[b], y[b])  # [N, C] * [N, C] -> [N, N]
        pairwise_matrices.append(pairwise_b)

    
    return torch.stack(pairwise_matrices, dim=0)


# Adapted from https://github.com/facebookresearch/Mask2Former/blob/9b0651c6c1d5b3af2e6da0589b719c514ec0d69a/mask2former/modeling/criterion.py#L21

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks

def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks

def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw

batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
) 

batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)

class InstanceSegmentationLoss():
    def __init__(self, cfg):
        
        self.class_weight = cfg.seg.class_weight
        self.mask_weight = cfg.seg.mask_weight
        self.dice_weight = cfg.seg.dice_weight

    def __call__(self, inputs, targets):
        '''
        inputs: [B, n_points, n_queries]
        targets: [B, n_points]

        OUT:
        losses: scalar
        inputs: [B, n_queries, n_points]
        targets: list[n_targets, n_points]
        '''
        targets = self.process_targets(targets)
        B = inputs.shape[0]
        inputs = inputs.permute(0, 2, 1)
        num_masks = B
        losses = 0.0
        num_masks = sum([target.shape[0] for target in targets])
        for b in range(B):
            pred_indices, tgt_indices = self.hungarian_matching(inputs[b], targets[b])

            loss = self.mask_weight * sigmoid_ce_loss(inputs[b][pred_indices], targets[b][tgt_indices], num_masks) + \
                self.dice_weight * dice_loss(inputs[b][pred_indices], targets[b][tgt_indices], num_masks)
            
            losses += loss
            
        return losses
    

    def process_targets(self, targets):
        '''
        targets: [B, n_points]  (each value represents an instance ID, 0 is background)
        
        Returns:
        target_list: List of tensors, each with shape [n_targets, n_points] where n_targets is unique foreground instances.
        '''
        target_list = []
        
        for b in range(targets.shape[0]):  # Iterate over batches
            unique_labels = torch.unique(targets[b])  # Get unique instance IDs
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background (0)

            # Create one-hot encoding for the instance masks
            one_hot_mask = torch.zeros((len(unique_labels), targets.shape[1]), dtype=torch.float32, device=targets.device)

            for i, label in enumerate(unique_labels):
                one_hot_mask[i] = (targets[b] == label).float()  # Assign 1 where the instance matches
            
            target_list.append(one_hot_mask)

        return target_list

    
    def hungarian_matching(self, out_mask, tgt_mask):

        '''
        out_mask [num_queries, n_points]
        tgt_mask [num_targets, n_points]
        '''
        num_queries = out_mask.shape[0]
        with autocast(enabled=False):
            out_mask = out_mask.float()
            tgt_mask = tgt_mask.float()
            # Compute the focal loss between masks
            cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)

            # Compute the dice loss betwen masks
            cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
        
        # Final cost matrix
        C = (
            self.mask_weight * cost_mask
            + self.dice_weight * cost_dice
        )
        C = C.reshape(num_queries, -1).cpu().detach().numpy()

        i, j = linear_sum_assignment(C)

        return torch.as_tensor(i), torch.as_tensor(j)