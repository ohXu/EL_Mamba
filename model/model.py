import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import os
import open_clip
import libpysal
import numpy as np


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum() / size

        return dice_loss


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


class MSE(_Loss):
    def __init__(self, reduction='mean'):
        super().__init__()

    def forward(self, inputs, targets):
        loss = F.mse_loss(inputs, targets, reduction='reduction')
        return loss


### desin adaptive params
class MSE_adapt(_Loss):
    def __init__(self, log_var=0.0):
        super().__init__()
        self.log_var = torch.nn.Parameter(torch.tensor(log_var, device="cuda"))

    def forward(self, inputs, targets):
        loss = F.mse_loss(inputs, targets, reduction='mean')
        precision = torch.exp(-self.log_var)
        loss = loss * precision + self.log_var
        return loss


class MSE_adapt_weight(_Loss):
    def __init__(self, log_var=0.0):
        super().__init__()
        self.log_var = torch.nn.Parameter(torch.tensor(log_var, device="cuda"))

    def forward(self, inputs, targets, weight):
        loss = F.mse_loss(inputs, targets, reduction='none')
        loss = (loss * weight).mean()
        precision = torch.exp(-self.log_var)
        loss = loss * precision + self.log_var
        return loss


def normal(tensor, min_val=-1):
    t_min = torch.min(tensor)
    t_max = torch.max(tensor)
    if t_min == 0 and t_max == 0:
        return tensor.clone().detach()
    if min_val == -1:
        tensor_norm = 2 * ((tensor - t_min) / (t_max - t_min)) - 1
    if min_val == 0:
        tensor_norm = ((tensor - t_min) / (t_max - t_min))
    return tensor_norm.clone().detach()


# Light-weight Local Moran's I for tensor data, requiring a sparse weight matrix input.
# This can be used when there is no need to re-compute the weight matrix at each step
def lw_tensor_local_moran(y, w_sparse, na_to_zero=True, norm=True, norm_min_val=-1):
    y = y.reshape(-1)
    n = len(y)
    n_1 = n - 1
    z = y - y.mean()
    sy = y.std()
    z /= sy
    den = (z * z).sum()
    zl = torch.tensor(w_sparse * z)
    mi = n_1 * z * zl / den
    if na_to_zero == True:
        mi[torch.isnan(mi)] = 0
    if norm == True:
        mi = normal(mi, min_val=norm_min_val)
    return torch.tensor(mi)


# Batch version of lw_tensor_local_moran
def batch_lw_tensor_local_moran(y_batch, w_sparse, na_to_zero=True, norm=True, norm_min_val=-1):
    batch_size = y_batch.shape[0]
    N = y_batch.shape[2]
    mi_y_batch = torch.zeros(y_batch.shape)
    for i in range(batch_size):
        y = y_batch[i, :, :, :].reshape(N, N)
        y = y.reshape(-1)
        n = len(y)
        n_1 = n - 1
        z = y - y.mean()
        sy = y.std()
        z /= sy
        den = (z * z).sum()
        zl = w_sparse * z
        mi = torch.tensor(n_1 * z * zl / den)
        if na_to_zero == True:
            mi[torch.isnan(mi)] = 0
        if norm == True:
            mi = normal(mi, min_val=norm_min_val)
        # if sy == 0:
        #     print(y, sy, np.max(y), y.shape)
        mi_y_batch[i, 0, :, :] = mi.reshape(N, N)
    return mi_y_batch


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 正样本的权重（默认 0.25）
        self.gamma = gamma  # 难样本调节因子（默认 2.0）
        self.reduction = reduction  # 'mean' | 'sum' | 'none'

    def forward(self, outputs, targets):
        """
        参数：
            outputs: (B, 2, H, W)，未归一化的 logits
            targets: (B, 2, H, W)，One-Hot 编码（每个像素的两个通道仅有一个为 1）
        返回：
            Focal Loss
        """
        # 检查输入形状是否匹配
        if outputs.shape != targets.shape:
            raise ValueError(f"Shape mismatch: outputs {outputs.shape}, targets {targets.shape}")

        # 计算 Softmax 概率
        probs = F.softmax(outputs, dim=1)  # (B, 2, H, W)

        # 提取真实类别的概率 p_t
        pt = (probs * targets).sum(dim=1)  # (B, H, W)
        pt = torch.clamp(pt, 1e-7, 1 - 1e-7)  # 避免 log(0)

        # Focal Loss 计算
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt)

        # 聚合方式
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # 'none'


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, dir, name, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.name = name
        self.dir = dir

    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        # score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif epoch >= 300:
            print(f'Epoch already equals Max')
            self.early_stop = True
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(self.dir, self.name + '_checkpoint.pt'))
        # torch.save(model, os.path.join(self.dir, self.name + '_checkpoint.pt'))
        self.val_loss_min = val_loss


def exp_schedule(epoch, total_epochs, start_weight, end_weight):
    return start_weight * (end_weight / start_weight) ** (epoch / total_epochs)


if __name__ == '__main__':
    # model = xqyModel2(s1_band=3, s2_band=2, uis_class=1)
    # s1 = torch.randn([8, 5, 256, 256]).to('cpu')
    # bh = model(s1[:, :3, :, :], s1[:, 3:, :, :])

    # model = SBHE(in_channels2=4, out_channels2=2, in_channels1=2, out_channels1=1)
    # s1 = torch.randn([1, 5, 256, 256]).to('cpu')
    # s2 = torch.randn([1, 3, 256, 256]).to('cpu')
    # bh = model(s1[:, :3, :, :], s1[:, 3:, :, :], s2)

    w = libpysal.weights.lat2W(256, 256, rook=False)
    print(w.weights)
