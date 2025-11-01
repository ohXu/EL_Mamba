from train import *
from osgeo import gdal, gdalconst
import numpy as np

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


def read_geotiff(filename):
    ds = gdal.Open(filename)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    return arr, ds


def write_geotiff(filename, arr, in_ds, zero_convert_NAN=False, recolor=False):
    driver = gdal.GetDriverByName("GTiff")
    # 判断传入的是单波段数据或多波段数据
    if isinstance(arr, list):
        band_nums = len(arr)
    else:
        if arr.shape[0] < 10:
            band_nums = arr.shape[0]
        else:
            band_nums = 1

    if band_nums == 1:
        if arr.dtype == np.float32:
            arr_type = gdal.GDT_Float32
        else:
            arr_type = gdal.GDT_Int32

        out_ds = driver.Create(filename, arr.shape[1], arr.shape[0], 1, arr_type)
        out_ds.SetProjection(in_ds.GetProjection())
        out_ds.SetGeoTransform(in_ds.GetGeoTransform())

        band = out_ds.GetRasterBand(1)
        # 是否处理图像中的NAN值
        if zero_convert_NAN:
            band.SetNoDataValue(0)
        # 重新渲染单波段图像
        if recolor:
            # create color table
            colors = gdal.ColorTable()

            # # set color for each value
            # colors.SetColorEntry(0, (69, 117, 181))
            # colors.SetColorEntry(100, (214, 47, 39))

            # set color table and color interpretation
            band.SetRasterColorTable(colors)
            band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

        band.WriteArray(arr)
        band.FlushCache()
        band.ComputeStatistics(False)

        del out_ds
        del band

    else:
        if arr[0].dtype == np.float32:
            arr_type = gdal.GDT_Float32
        else:
            arr_type = gdal.GDT_Int32

        # 三波段数据
        out_ds = driver.Create(
            filename, arr[0].shape[1], arr[0].shape[0], band_nums, arr_type
        )
        out_ds.SetProjection(in_ds.GetProjection())
        out_ds.SetGeoTransform(in_ds.GetGeoTransform())

        # 4. 写入数据
        for i in range(band_nums):
            single_band_data = arr[i]
            band = out_ds.GetRasterBand(i + 1)

            # 是否处理图像中的NAN值
            if zero_convert_NAN:
                band.SetNoDataValue(0)

            band.WriteArray(single_band_data)
            band.FlushCache()

        del out_ds
        del band


def inference(model, dataloader, device):
    with torch.no_grad():
        for idx, (x, y_true1, y_true2, img_name) in enumerate(dataloader):
            x = x.to(device, non_blocking=True)
            build, height, _ = model(x[:, :4, :, :], x[:, 4:, :, :])
            build = torch.argmax(build, dim=1)
            img_name = str(img_name[0].item())
            tif_arr, tiff_ds = read_geotiff(
                "./dataset/data/" +
                "height_patch_" + img_name + ".tif")

            bh_save_path = os.path.join("./output/height", img_name + ".tif")
            bh_save_path2 = os.path.join("./output/footprint", img_name + ".tif")
            os.makedirs(os.path.dirname(bh_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(bh_save_path2), exist_ok=True)

            img = (build.squeeze().cpu().numpy()).astype(np.float32)
            img2 = height.squeeze().cpu().numpy()

            write_geotiff(bh_save_path, img, tiff_ds, zero_convert_NAN=True)
            write_geotiff(bh_save_path2, img2, tiff_ds, zero_convert_NAN=True)


def compute_metrics(pred, target):
    """
    计算建筑物分割的 IoU, mIoU, Dice, PA
    :param pred: 预测结果 (batch, H, W)，值范围 0-1
    :param target: 真实标签 (batch, H, W)，值范围 0-1
    :return: IoU, mIoU, Dice, PA
    """
    pred = (pred > 0.5).float()  # 阈值化
    target = target.float()

    TP = (pred * target).sum()  # 真阳性（建筑物区域正确预测）
    FP = (pred * (1 - target)).sum()  # 假阳性（错误预测为建筑）
    FN = ((1 - pred) * target).sum()  # 假阴性（漏掉的建筑）
    TN = ((1 - pred) * (1 - target)).sum()  # 真阴性（背景正确预测）

    IoU = TP / (TP + FP + FN + 1e-6)  # 避免除 0
    Dice = (2 * TP) / (2 * TP + FP + FN + 1e-6)
    PA = (TP + TN) / (TP + TN + FP + FN + 1e-6)

    IoU_bg = TN / (TN + FP + FN + 1e-6)  # 背景 IoU
    mIoU = (IoU + IoU_bg) / 2  # 计算 mIoU

    # return {"IoU": IoU.item(), "mIoU": mIoU.item(), "Dice": Dice.item(), "PA": PA.item()}
    return IoU.item(), mIoU.item(), Dice.item(), PA.item()


def vtest_epoch2(model, dataloader, device, epoch):
    model.eval()

    acc = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    num = len(dataloader)
    pbar = tqdm(range(num), disable=False)
    y_p = []
    y_t = []
    r2_xqy = []
    name_xqy = []
    acc_seg = SegmentationMetric(2, device)
    names = []
    with torch.no_grad():
        for idx, (x, y_true, build_true, iname) in enumerate(dataloader):
            img_name = str(iname[0].item()).zfill(5)
            names.append(img_name)
            x = x.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)
            build_true = build_true.to(device, non_blocking=True)
            build_pred, _, ypred_Fine = model.forward(x[:, :4, :, :], x[:, 4:, :, :])

            build_pred = torch.argmax(build_pred, dim=1)
            build_true = torch.argmax(build_true, dim=1)

            ypred_Fine = ypred_Fine.squeeze(1)
            build_pred = build_pred.squeeze(1)

            # IoU, mIoU, Dice, PA = compute_metrics(build_pred, build_true)
            acc_seg.addBatch(build_pred.to(torch.int), build_true.to(torch.int))

            mask = build_true == 1
            y_true = y_true[mask]
            ypred_Fine = ypred_Fine[mask]
            rmse = torch.sqrt(((ypred_Fine - y_true) ** 2).mean())

            y_p = y_p + list(torch.flatten(ypred_Fine).cpu().numpy())
            y_t = y_t + list(torch.flatten(y_true).cpu().numpy())

            r2 = r2_score(torch.flatten(y_true).cpu(), torch.flatten(ypred_Fine).cpu())
            mae = torch.mean(torch.abs(y_true - ypred_Fine))
            acc.update(rmse, x.size(0))
            acc2.update(r2, x.size(0))
            acc3.update(mae, x.size(0))
            r2_xqy.append(rmse)
            name_xqy.append(iname)
            pbar.set_description(
                'Test Epoch:{epoch:4}. Iter:{batch:4}|{iter:4}. RMSE {rmse:.3f}. R2 {r2:.3f}. MAE {mae:.3f}'.format(
                    epoch=epoch, batch=idx, iter=num, rmse=acc.avg, r2=acc2.avg, mae=acc3.avg))
            pbar.update()

        pbar.close()
        y_p = np.array(y_p)
        y_t = np.array(y_t)

        rmse = np.sqrt(((y_p - y_t) ** 2).mean())
        r2 = r2_score(y_p, y_t)
        mae = np.mean(np.abs(y_t - y_p))
        print(r2, rmse, mae, y_p.shape)
        oa = acc_seg.OverallAccuracy().cpu().numpy()
        miou = acc_seg.meanIntersectionOverUnion().cpu().numpy()
        iou = acc_seg.IntersectionOverUnion().cpu().numpy()
        f1 = acc_seg.F1score().cpu().numpy()
        print(oa, miou, iou, f1)

    return acc.avg, acc.avg


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-B-32", pretrained="mscoco_finetuned_laion2b_s13b_b90k"
    )

    model = EL_Mamba().to(device)
    checkpoint = torch.load(r'./checkpoints/EL-Mamba_checkpoint.pt')
    model.load_state_dict(checkpoint)
    model.eval()

    testLoader = DataLoader(sentinelData(datalist=args.testDataPath, transform=transform), batch_size=1, shuffle=False,
                            pin_memory=True)

    vtest_epoch2(model, testLoader, device, 0)
    inference(model, testLoader, device)


if __name__ == '__main__':
    main()
