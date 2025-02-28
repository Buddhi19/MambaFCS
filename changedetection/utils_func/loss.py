import os
import sys

main_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_dir)

import torch
import torch.nn as nn
import numpy as np
from torch import Tensor, einsum
import torch.nn .functional as F
from typing import Iterable, Set, Tuple
from scipy.ndimage import distance_transform_edt
import torchvision.models as models

from RemoteSensing.changedetection.utils_func.utils import simplex, class2one_hot, uniq

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def boundary_loss(pred, target):
    target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()  # [B, C, H, W]

    boundary_distances = []
    for b in range(target.shape[0]):  # Iterate over batch
        for c in range(target_onehot.shape[1]):  # Iterate over classes
            mask = target_onehot[b, c].cpu().numpy()  # Get binary mask for class c
            if np.sum(mask) == 0:  # Skip if no pixels for this class
                boundary_distances.append(np.zeros_like(mask))
                continue
            # Compute the distance transform for the foreground (class c)
            pos_dist = distance_transform_edt(mask)
            neg_dist = distance_transform_edt(1 - mask)
            boundary_dist = pos_dist + neg_dist
            boundary_distances.append(boundary_dist)
    
    boundary_distances = np.stack(boundary_distances, axis=0)  # [B*C, H, W]
    boundary_distances = torch.from_numpy(boundary_distances).float().to(pred.device)  # Move to GPU if needed

    boundary_distances = boundary_distances.view(pred.shape)  # [B, C, H, W]

    # Compute the boundary loss
    pred_softmax = F.softmax(pred, dim=1)  # Convert logits to probabilities
    loss = torch.mean(pred_softmax * boundary_distances)  # Weighted sum

    return loss


def weighted_BCE_logits(logit_pixel, truth_pixel, weight_pos=0.25, weight_neg=0.75):
    logit = logit_pixel.reshape(-1)
    truth = truth_pixel.reshape(-1)
    assert(logit.shape==truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
    
    pos = (truth>0.5).float()
    neg = (truth<0.5).float()
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos*pos*loss/pos_num + weight_neg*neg*loss/neg_num).sum()

    return loss

def dice_loss(predicts,target,weight=None):
    idc= [0, 1]
    probs = torch.softmax(predicts, dim=1)
    # target = target.unsqueeze(1)
    target = class2one_hot(target, 7)
    assert simplex(probs) and simplex(target)

    pc = probs[:, idc, ...].type(torch.float32)
    tc = target[:, idc, ...].type(torch.float32)
    intersection: Tensor = einsum("bcwh,bcwh->bc", pc, tc)
    union: Tensor = (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

    divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)

    loss = divided.mean()
    return loss


def dice_loss_multiclass(pred, target, smooth=1e-6, ignore_index=255):
    # Mask out invalid pixels
    valid_mask = (target != ignore_index).float()
    target = target.clone()
    target[target == ignore_index] = 0
    
    pred = F.softmax(pred, dim=1)
    num_classes = pred.shape[1]
    target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Apply valid_mask
    pred = pred * valid_mask.unsqueeze(1)
    target_one_hot = target_one_hot * valid_mask.unsqueeze(1)
    
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def ce_dice(input, target, weight=None):
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    loss = 0.5 * ce_loss + 0.5 * dice_loss_
    return loss

def dice(input, target, weight=None):
    dice_loss_ = dice_loss(input, target)
    return dice_loss_

def ce2_dice1(input, target, ignore_index=255):
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    labels_bn = (target > 0).float()  # Binary labels (0 or 1)

    logits_positive = input[:, 1, :, :]  # Shape: [N, H, W]

    bce_loss = weighted_BCE_logits(logits_positive, labels_bn)
    loss = 1 * ce_loss + 0.15* dice_loss_ + 0.35 * bce_loss
    return loss

def ce2_dice1_multiclass(input, target, weight=None):
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    target2 = target.clone()
    dice_loss_ = dice_loss_multiclass(input, target2)
    loss = ce_loss + 0.25 * dice_loss_ 
    return loss


def ce1_dice2(input, target, weight=None):
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    loss = 0.5 * ce_loss +  dice_loss_
    return loss

def ce_scl(input, target, weight=None):
    ce_loss = F.cross_entropy(input, target, ignore_index=255)
    dice_loss_ = dice_loss(input, target)
    loss = 0.5 * ce_loss + 0.5 * dice_loss_
    return loss

def contrastive_loss(features_1, features_2, label, margin=1.0):
    similarity = F.cosine_similarity(features_1, features_2, dim=1)

    # Loss for unchanged regions (maximize similarity)
    unchanged_loss = (1 - similarity) * (1 - label)  # Mask for unchanged regions

    # Loss for changed regions (minimize similarity)
    changed_loss = torch.clamp(similarity - margin, min=0) * label  # Mask for changed regions

    # Combine losses
    loss = unchanged_loss.mean() + changed_loss.mean()
    return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', ignore_index=255)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg16(pretrained=True).features[:16].cuda().eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        input_features = self.vgg(input)
        target_features = self.vgg(target)
        return F.mse_loss(input_features, target_features)
