import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification.
        
        Args:
            alpha (float): Weighting factor for the rare class (label 1). 
                           If None, no weighting is applied.
            gamma (float): Focusing parameter. Higher gamma reduces loss for easy examples.
            reduction (str): 'mean' or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: Logits (before sigmoid)
        targets: Binary labels (0 or 1)
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss) # prob of correctness
        
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        
        if self.alpha is not None:
            # Apply alpha to class 1, and (1-alpha) to class 0
            alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_factor * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
