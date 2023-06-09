import torch
import torch.nn as nn
loss_names = ['Ucertl2','l1', 'l2']



class UcertRELossL2(nn.Module):
    def __init__(self):
        super(UcertRELossL2, self).__init__()

    def forward(self, pred, predm, Ucert, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred - predm
        diff = diff[valid_mask]
        Ucert = Ucert[valid_mask] + 1.

        self.loss = ((diff ** 2) * Ucert).mean()
        return self.loss

class UcertRELossL1(nn.Module):
    def __init__(self):
        super(UcertRELossL1, self).__init__()

    def forward(self, pred, predm, Ucert, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred - predm
        diff = diff[valid_mask]
        Ucert = Ucert[valid_mask] + 1.

        self.loss = (diff * Ucert).abs().mean()
        return self.loss

class Ucertl2MaskedMSELoss(nn.Module):
    def __init__(self):
        super(Ucertl2MaskedMSELoss, self).__init__()

    def forward(self, pred, Ucert, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        #print(pred.size(), Ucert.size())
        self.loss = (((-1) * Ucert[valid_mask]).exp()*(diff**2)).mean() + 2*Ucert[valid_mask].mean()

        return self.loss

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, depth):
        def second_derivative(x):
            assert x.dim(
            ) == 4, "expected 4-dimensional data, but instead got {}".format(
                x.dim())
            horizontal = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, 1:-1, :
                                                     -2] - x[:, :, 1:-1, 2:]
            vertical = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, :-2, 1:
                                                   -1] - x[:, :, 2:, 1:-1]
            der_2nd = horizontal.abs() + vertical.abs()
            return der_2nd.mean()

        self.loss = second_derivative(depth)
        return self.loss
