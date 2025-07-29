import torch
import torch.nn as nn
import numpy as np
import pandas as pd


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class HeightMetric(nn.Module):
    def __init__(self, numClass=7, device='cpu'):
        super().__init__()
        self.numClass = numClass
        self.device = device
        self.reset()

    def calc_rmse(self, pred, ref):
        rmse = torch.sqrt(((pred - ref) ** 2).mean())
        return rmse

    #
    # def calc_mse(self, pred, ref):
    #     mse = ((pred-ref)**2).mean()
    #     return mse

    def calc_mae(self, pred, ref):
        mae = (torch.abs(pred - ref)).mean()
        return mae

    def calc_me(self, pred, ref):
        me = (pred - ref).mean()
        return me

    def addBatch(self, pred, ref, buildhir):
        for i in range(self.numClass):
            mask = (buildhir == i)
            count = mask.sum().float()
            if int(count.item()) == 0:
                continue
            rmse = self.calc_rmse(pred[mask], ref[mask])
            mae = self.calc_mae(pred[mask], ref[mask])
            me = self.calc_me(pred[mask], ref[mask])

            self.stats[i, 0] += rmse * count
            self.stats[i, 1] += mae * count
            self.stats[i, 2] += me * count
            self.count[i] += count

    def getAvgEach(self):
        res = self.stats / (self.count + 1e-10)
        return res

    def getAvgBalance(self):
        res = self.getAvgEach().mean(dim=0)
        return res

    def getAvgAll(self):
        res = self.stats.sum(dim=0) / (self.count.sum())
        return res

    def getCount(self):
        return self.count

    # def getAvgEach(self):
    #     res = self.__AvgEach_()
    #     # res[:, 0] = torch.sqrt(res[:, 0])
    #     return res
    #
    # def getAvgBalance(self):
    #     res = self.__AvgBalance_()
    #     # res[0] = torch.sqrt(res[0])
    #     return res
    #
    # def getAvgAll(self):
    #     res = self.__AvgAll_()
    #     # res[0] = torch.sqrt(res[0])
    #     return res

    def reset(self):
        self.count = torch.zeros((self.numClass, 1), dtype=torch.float64).to(self.device)
        self.stats = torch.zeros((self.numClass, 3), dtype=torch.float64).to(self.device)  # rmse, mae, me
        self.balance_stats = torch.zeros((self.numClass, 3), dtype=torch.float64).to(self.device)  # balanced version


class SegmentationMetric(nn.Module):
    def __init__(self, numClass, device='cpu'):
        super().__init__()
        self.numClass = numClass
        self.device = device
        self.reset()
        self.count = 0

    # OA
    def OverallAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = torch.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    # UA
    def Precision(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(0)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    # PA
    def Recall(self):
        # acc = (TP) / TP + FN
        classAcc = torch.diag(self.confusionMatrix) / self.confusionMatrix.sum(1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    # F1-score
    def F1score(self):
        # 2*Recall*Precision/(Recall+Precision)
        p = self.Precision()
        r = self.Recall()
        return 2 * p * r / (p + r)

    # MIOU
    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        IoU = self.IntersectionOverUnion()
        mIoU = torch.mean(IoU)
        return mIoU

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = torch.sum(self.confusionMatrix, dim=1) + torch.sum(self.confusionMatrix, dim=0) - torch.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        # mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return IoU

    # FWIOU
    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = torch.sum(self.confusionMatrix, dim=1) / (torch.sum(self.confusionMatrix) + 1e-8)
        iu = torch.diag(self.confusionMatrix) / (
                torch.sum(self.confusionMatrix, dim=1) + torch.sum(self.confusionMatrix, dim=0) -
                torch.diag(self.confusionMatrix) + 1e-8)
        # FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        FWIoU = freq * iu
        return FWIoU

    def mFWIoU(self):
        return self.Frequency_Weighted_Intersection_over_Union().sum()

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        # mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        # label = self.numClass * imgLabel[mask] + imgPredict[mask]
        label = self.numClass * imgLabel.flatten() + imgPredict.flatten()
        count = torch.bincount(label, minlength=self.numClass ** 2)
        cm = count.reshape(self.numClass, self.numClass)
        return cm

    def getConfusionMatrix(self):  # 同FCN中score.py的fast_hist()函数
        # cfM = self.confusionMatrix / np.sum(self.confusionMatrix, axis=0)
        cfM = self.confusionMatrix
        return cfM

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = torch.zeros((self.numClass, self.numClass), dtype=torch.float64).to(
            self.device)  # float, dtype=torch.int64) # int 64 is important


def acc2fileHeight(acc, txtpath):
    acceach = acc.getAvgEach().cpu().numpy()
    accbalance = acc.getAvgBalance().cpu().numpy()
    accall = acc.getAvgAll().cpu().numpy()
    counts = acc.getCount().cpu().numpy()

    acceach = np.concatenate([acceach, counts], axis=1)
    accbalance0 = np.zeros((1, 4))
    accbalance0[0, :3] = accbalance

    accall0 = np.zeros((1, 4))
    accall0[0, :3] = accall

    data = np.concatenate([acceach, accbalance0, accall0], axis=0)
    print(data)
    df = pd.DataFrame(data=data,
                      columns=['rmse', 'mae', 'me', 'count'])
    # df.to_csv(txtpath[:-4] + '.csv', index=False)
