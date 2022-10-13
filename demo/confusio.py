import numpy as np
import cv2
__all__ = ['SegmentationMetric']
class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩陣(空)
    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN

        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取對角元素的值，返回列表

        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(

        self.confusionMatrix)  # axis = 1表示混淆矩陣行的值，返回列表； axis = 0表示取混淆矩陣列的值，返回列表

        IoU = intersection / union  # 返回列表，其值為各個類別的IoU
        return IoU
    def genConfusionMatrix(self, imgPredict, imgLabel):
        """

        同FCN中score.py的fast_hist()函式,計算混淆矩陣

        :param imgPredict:

        :param imgLabel:

        :return: 混淆矩陣

        """

        # remove classes from unlabeled pixels in gt image and predict

        mask = (imgLabel >= 0) & (imgLabel < self.numClass)

        label = self.numClass * imgLabel[mask] + imgPredict[mask]

        count = np.bincount(label, minlength=self.numClass ** 2)

        confusionMatrix = count.reshape(self.numClass, self.numClass)

        # print(confusionMatrix)

        return confusionMatrix



    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape

        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩陣

        return self.confusionMatrix



    def con_matrix(ans, test):
        imgPredict = np.array(cv2.cvtColor(ans, cv2.COLOR_BGR2GRAY) / 255., dtype=np.uint8)
        imgLabel = np.array(cv2.cvtColor(test, cv2.COLOR_BGR2GRAY) / 255., dtype=np.uint8)
        metric = SegmentationMetric(2)  # 2表示有2個分類，有幾個分類就填幾
        hist = metric.addBatch(imgPredict, imgLabel)
        fp = hist.sum(axis=0) - np.diag(hist)
        fn = hist.sum(axis=1) - np.diag(hist)
        tp = np.diag(hist)
        tn = hist.sum() - (fp + fn + tp)
        fp = fp.astype(float)
        fn = fn.astype(float)
        tp = tp.astype(float)
        tn = tn.astype(float)
        IoU = metric.IntersectionOverUnion()
        Accuracy = (tp + tn) / (tp + tn + fn + fp)
        Precision = tp / (tp + fp)
        Recall = tp / (tp + fn)
        dice = 2 * ((Precision * Recall) / (Precision + Recall))
        for i in range(len(IoU)):
            if np.isnan(IoU[i]):
                IoU[i] = 0
        for i in range(len(Accuracy)):
            if np.isnan(Accuracy[i]):
                Accuracy[i] = 0
        for i in range(len(IoU)):
            if np.isnan(Precision[i]):
                Precision[i] = 0
        for i in range(len(Recall)):
            if np.isnan(Recall[i]):
                Recall[i] = 0
        for i in range(len(dice)):
            if np.isnan(dice[i]):
                dice[i] = 0
        print('IoU : ',IoU[1])
        print('Accuracy : ', Accuracy[1])
        print('Precision : ', Precision[1])
        print('Recall : ', Recall[1])
        print('dice : ', dice[1])
        return round(IoU[1], 4), round(dice[1], 4), round(Accuracy[1], 4), round(Recall[1], 4), round(Precision[1], 4)