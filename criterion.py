import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
"""
Adapted from https://github.com/won-bae/rethinkingCAM/blob/master/src/core/criterions.py 
"""

class CrossEntropyLoss(_Loss):
    def __init__(self, weight=None, gamma=1., temp=1., reduction='mean', eps=1e-6, num_classes=200):
        super(_Loss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.temp = temp
        self.reduction = reduction
        self.eps = eps
        print(f'CrossEntropyLoss is built')

    def forward(self, preds, labels):
        preds = preds / self.temp
        if self.gamma >= 1.:
            loss = F.cross_entropy(
                preds, labels, weight=self.weight, reduction=self.reduction)
        else:
            log_prob = preds - torch.logsumexp(preds, dim=1, keepdim=True)
            log_prob = log_prob * self.gamma
            loss = F.nll_loss(
                log_prob, labels, weight=self.weight, reduction=self.reduction)

        return loss

class BinaryCrossEntropyLoss(_Loss):
    def __init__(self, weight=None, gamma=1.0, temp=1.0, reduction='mean', eps=1e-6, num_classes=200, pos_weight=True):
        super(_Loss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.temp = temp
        self.reduction = reduction
        self.eps = eps
        self.num_classes = num_classes
        print(f'BinaryCrossEntropyLoss is built')

        if pos_weight:
            weight = float(num_classes - 1)
            self.pos_weight = torch.tensor(weight) 
            print(f'Positive weight is given as {weight}')

    def forward(self, preds, labels):        
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float() 

        preds = preds / self.temp  

        
        if self.gamma >= 1.0:
            loss = F.binary_cross_entropy_with_logits(
                preds, labels_one_hot, weight=self.weight, reduction=self.reduction, pos_weight= self.pos_weight
            )
            
        else: 
            probs = torch.sigmoid(preds)
            
            log_prob_pos = torch.log(probs + self.eps) * self.gamma
            log_prob_neg = torch.log(1 - probs + self.eps) * self.gamma

            loss = -labels_one_hot * log_prob_pos - (1 - labels_one_hot) * log_prob_neg

            if self.weight is not None:
                loss = loss * self.weight

            if self.reduction == 'mean':
                loss = loss.mean()
            elif self.reduction == 'sum':
                loss = loss.sum()

        return loss
    
