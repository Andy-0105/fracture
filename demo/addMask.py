import cv2
import numpy as np
import matplotlib.pyplot as plt
import iou
import os
from PIL import Image
import confusio as cm
def combineMask(test_path, ans_path,xray_path,count,path,rate):
    test = cv2.imread(test_path)
    ans = cv2.imread(ans_path)
    xray = cv2.imread(xray_path)

    np_ans = np.copy(ans)
    np_xray = np.copy(xray)
    np_test=np.copy(test)
    # boolean indexing and assignment based on mask
    _, correct = cv2.threshold(test, thresh=50, maxval=255, type=cv2.THRESH_BINARY)
    _, predicted = cv2.threshold(ans, thresh=50, maxval=255, type=cv2.THRESH_BINARY)

    # copy where we'll assign the new values
    np_ans[(correct == 255).all(-1)] = [0, 0, 255]
    np_test[(predicted == 255).all(-1)] = [0, 255, 0]
    img_ccorrect_predicted = cv2.addWeighted(np_test, 0.5, np_ans, 0.5, 0)
    img=cv2.addWeighted(img_ccorrect_predicted,0.3,np_xray,0.7,0)
    cv2.imwrite(f'./save/new48time/{count}/{rate}/{path}.png',img)

    # correct_with_predicted2[(predicted == 255).all(-1)] = [0, 255, 0]
    # correct_with_predicted[(correct == 255).all(-1)] = [0, 0, 255]
    # img = cv2.addWeighted(correct_with_predicted2, 0.5, correct_with_predicted, 0.5, 0)
    # cv2.imwrite(f'./save/123/{path}.png',img)
    # fig, ax = plt.subplots(1, figsize=(12, 6))
    # cv2.imshow('predicted_label', predicted)
    # cv2.imshow('correct_with_predicted', correct_with_predicted)
    # ax[0].imshow(cv2.cvtColor(correct_with_predicted, cv2.COLOR_BGR2RGB))
    # ax.imshow(cv2.cvtColor(correct_with_predicted2, cv2.COLOR_BGR2RGB))

    # plt.show()
    # plt.title(f'{path}')
    # plt.savefig(f'./save/{rate}/{count}')
i=0
k_fold=0
# ans_path = os.path.join(os.getcwd(), './label/Chimei')
# test_path = os.path.join(os.getcwd(), './predict/Chimei')
# xray_path = os.path.join(os.getcwd(), './image/Chimei')
# ans_path = os.path.join(os.getcwd(), './label/ChestX')
# test_path = os.path.join(os.getcwd(), './predict/ChestX')
# xray_path = os.path.join(os.getcwd(), './image/ChestX')
# ans_path = os.path.join(os.getcwd(), './label/k_fold')
# test_path = os.path.join(os.getcwd(), './predict/k_fold')
# xray_path = os.path.join(os.getcwd(), './image/k_fold')
ans_path = os.path.join(os.getcwd(), f'./label/new48time/{k_fold}')
test_path = os.path.join(os.getcwd(), f'./predict/new48time/{k_fold}')
xray_path = os.path.join(os.getcwd(), f'./image/new48time/{k_fold}')
ans = os.listdir(ans_path)
test = os.listdir(test_path)
xray = os.listdir(xray_path)
count_iou=0
count_dice=0
count_Accuracy=0
count_recall=0
count_precision=0
for a, t ,x in zip(ans, test, xray):
    ans_allpath = os.path.join(ans_path, a)
    test_allpath = os.path.join(test_path, t)
    xray_allpath = os.path.join(xray_path, x)
    basename=os.path.basename(ans_allpath)
    ans_name=os.path.splitext(basename)
    print(i)
    print(ans_name[0])
    print(ans_allpath)
    print(test_allpath)
    print(xray_allpath)
    image_ans = cv2.imread(ans_allpath)
    image_test = cv2.imread(test_allpath)
    IOU=cm.SegmentationMetric.con_matrix(image_ans,image_test)
    if IOU[0] <= 0.1:
        combineMask(test_allpath, ans_allpath, xray_allpath, k_fold, ans_name[0], "01")
    elif IOU[0] > 0.1 and IOU[0] <= 0.2:
        combineMask(test_allpath, ans_allpath, xray_allpath, k_fold, ans_name[0], "02")
    elif IOU[0] > 0.2 and IOU[0] <= 0.3:
        combineMask(test_allpath, ans_allpath, xray_allpath, k_fold, ans_name[0], "03")
    elif IOU[0] > 0.3 and IOU[0] <= 0.4:
        combineMask(test_allpath, ans_allpath, xray_allpath, k_fold, ans_name[0], "04")
    elif IOU[0] > 0.4 and IOU[0] <= 0.5:
        combineMask(test_allpath, ans_allpath, xray_allpath, k_fold, ans_name[0], "05")
    elif IOU[0] > 0.5 and IOU[0] <= 0.6:
        combineMask(test_allpath, ans_allpath, xray_allpath, k_fold, ans_name[0], "06")
    elif IOU[0] > 0.6 and IOU[0] <= 0.7:
        combineMask(test_allpath, ans_allpath, xray_allpath, k_fold, ans_name[0], "07")
    elif IOU[0] > 0.7 and IOU[0] <= 0.8:
        combineMask(test_allpath, ans_allpath, xray_allpath, k_fold, ans_name[0], "08")
    elif IOU[0] > 0.8 and IOU[0] <= 0.9:
        combineMask(test_allpath, ans_allpath, xray_allpath, k_fold, ans_name[0], "09")
    else:
        combineMask(test_allpath, ans_allpath, xray_allpath, k_fold, ans_name[0], "10")
    # combineMask(t, a, x, i, ans_name[0], "10")

    count_iou=count_iou+IOU[0]
    count_dice = count_dice + IOU[1]
    count_Accuracy=count_Accuracy+IOU[2]
    count_recall=count_recall+IOU[3]
    count_precision=count_precision+IOU[4]
    Aver_iou = count_iou / (i + 1)
    Aver_dice = count_dice / (i + 1)
    Aver_Accuracy = count_Accuracy / (i + 1)
    Aver_recall = count_recall / (i + 1)
    Aver_precision = count_precision / (i + 1)
    print("____________________________")
    print('Aver_iou : ',Aver_iou)
    print('Aver_dice : ',Aver_dice)
    print('Aver_Accuracy : ',Aver_Accuracy)
    print('Aver_recall : ',Aver_recall)
    print('Aver_precision : ',Aver_precision)
    print("____________________________")
    i+=1
Average_iou=count_iou/(i+1)
Average_dice=count_dice/(i+1)
Average_Accuracy=count_Accuracy/(i+1)
Average_recall=count_recall/(i+1)
Average_precision=count_precision/(i+1)
print(f'Average_iou:{Average_iou}')
print(f'Average_dice:{Average_dice}')
print(f'Average_Accuracy:{Average_Accuracy}')
print(f'Average_recall:{Average_recall}')
print(f'Average_precision:{Average_precision}')