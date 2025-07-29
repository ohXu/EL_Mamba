from torch.utils.data import DataLoader
import argparse
from dataset.dataLoader import sentinelData
from model.model import *
from model.EL_Mamba import EL_Mamba
from tqdm import tqdm
from utils.metrics import *
import numpy as np
import open_clip
from sklearn.metrics import r2_score
from monai.losses import DiceCELoss
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainDataPath', default=r'./dataset/train_data.csv')
    parser.add_argument('--valDataPath', default=r'./dataset/val_data.csv')
    parser.add_argument('--testDataPath', default=r'./dataset/test_data.csv')
    parser.add_argument('--batchsize', default=16, type=int)
    parser.add_argument('--maxepoch', default=100, type=int)
    args = parser.parse_args()
    return args


def train_epoch(net, criterion, dataloader, optimizer, device, epoch):
    net.train()
    losses = AverageMeter()
    acc = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)

    for idx, (img, height, build, name) in enumerate(dataloader):
        img = img.to(device, non_blocking=True)
        height = height.to(device, non_blocking=True)
        build = build.float().to(device, non_blocking=True)

        build_pred, height_pred, height_pred2 = net(img[:, :4, :, :], img[:, 4:, :, :])
        height_pred = height_pred.squeeze(1)
        height_pred2 = height_pred2.squeeze(1)

        mask_bh_gt3 = torch.where(height >= 3, torch.ones_like(height), torch.zeros_like(height))
        pred_bh_mask = torch.mul(mask_bh_gt3, height_pred)

        loss_mse_local = criterion[1](pred_bh_mask, height)
        loss_mse = criterion[1](height_pred, height)

        loss_uis = criterion[0](build_pred, build)
        loss = loss_uis + loss_mse + loss_mse_local
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rmse = torch.sqrt(((height_pred - height) ** 2).mean())
        bsize = img.size(0)
        acc.update(rmse, bsize)
        losses.update(loss.item(), bsize)
        pbar.set_description(
            'Train Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. Loss {loss:.3f}. RMSE {rmse:.3f}'.format(
                epoch=epoch, batch=idx, iter=num, loss=losses.avg, rmse=acc.avg))
        pbar.update()
    pbar.close()
    return losses.avg


def vtest_epoch(model, dataloader, device, epoch):
    model.eval()
    acc = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    acc_seg = SegmentationMetric(2, device)
    with torch.no_grad():
        for idx, (x, y_true, build_true, _) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)
            build_true = build_true.to(device, non_blocking=True)

            build_pred, _, ypred_Fine = model.forward(x[:, :4, :, :], x[:, 4:, :, :])
            ypred_Fine = ypred_Fine.squeeze(1)

            build_pred = torch.argmax(build_pred, dim=1)
            build_true = torch.argmax(build_true, dim=1)
            build_pred = build_pred.squeeze(1)

            acc_seg.addBatch(build_pred.to(torch.int), build_true.to(torch.int))
            r2 = r2_score(torch.flatten(y_true).cpu(), torch.flatten(ypred_Fine).cpu())
            rmse = torch.sqrt(((ypred_Fine - y_true) ** 2).mean())
            mae = torch.mean(torch.abs(y_true - ypred_Fine))
            acc.update(rmse, x.size(0))
            acc2.update(r2, x.size(0))
            acc3.update(mae, x.size(0))

            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. RMSE {rmse:.3f}. R2 {r2:.3f}. MAE {mae:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, rmse=acc.avg, r2=acc2.avg, mae=acc3.avg))
            pbar.update()
        oa = acc_seg.OverallAccuracy().cpu().numpy()
        miou = acc_seg.meanIntersectionOverUnion().cpu().numpy()
        iou = acc_seg.IntersectionOverUnion().cpu().numpy()
        f1 = acc_seg.F1score().cpu().numpy()
        print(oa, miou, iou, f1)
        pbar.close()
    return acc.avg, acc.avg


def calculate_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    计算 DICE 损失，用于衡量预测掩码和真实掩码的重叠程度。
    """
    assert inputs.size(0) == targets.size(0)  # 确保输入和目标的第一个维度一致
    inputs = inputs.sigmoid()  # 对输入应用 sigmoid，将值映射到 [0, 1]
    inputs, targets = inputs.flatten(1), targets.flatten(1)  # 将输入和目标展开为二维

    numerator = 2 * (inputs * targets).sum(-1)  # 计算分子，2 * (预测值 * 真实值) 之和
    denominator = inputs.sum(-1) + targets.sum(-1)  # 计算分母，预测值和真实值之和
    loss = 1 - (numerator + 1) / (denominator + 1)  # 计算 DICE 损失
    return loss.mean()  # 返回平均损失值


if __name__ == "__main__":
    start_time = time.time()
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = get_args()
    _, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-B-32", pretrained="mscoco_finetuned_laion2b_s13b_b90k"
    )
    trainLoader = DataLoader(sentinelData(datalist=args.trainDataPath, transform=transform), batch_size=args.batchsize,
                             shuffle=True, num_workers=8, pin_memory=True)
    valLoader = DataLoader(sentinelData(datalist=args.valDataPath, transform=transform), batch_size=args.batchsize,
                           shuffle=False, num_workers=8, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = EL_Mamba().to(device)
    name = "EL-Mamba"

    epochs = args.maxepoch
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    criterion = [DiceCELoss(softmax=True), nn.MSELoss(), DiceCELoss(), BinaryFocalLoss(alpha=0.25, gamma=2.0)]
    best_acc = 10000
    early_stopping = EarlyStopping("./checkpoints/", name, patience=20, verbose=True)

    for epoch in range(epochs):
        epoch = epoch + 1
        train_loss = train_epoch(net, criterion, trainLoader, optimizer, device, epoch)
        val_loss, val_rmse = vtest_epoch(net, valLoader, device, epoch)
        # save every epoch
        if (epoch - 1) % 10 == 0:
            torch.save(net.state_dict(), f"./checkpoints/model_epoch_{epoch}.pth")
            print(f"Saved model at epoch {epoch}")

        early_stopping(val_loss, net, epoch)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    print((time.time() - start_time) / 3600, time.time() - start_time)
