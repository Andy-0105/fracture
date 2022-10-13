import numpy as np
import cupy as cp

def pretreatment(ans, test):
    ans = ans.convert('1')
    test = test.convert('1')
    ans_im=np.array(ans)
    test_im=np.array(test)
    h = ans_im.shape[0]
    w = ans_im.shape[1]
    numU = 0
    numI = 0
    num_test=0
    num_ans=0
    for x in range(h):
        for y in range(w):
            if [ans_im[x,y]] == [True] or [test_im[x,y]] == [True]:
                numU += 1
            if [ans_im[x,y]] == [True] and [test_im[x,y]] == [True]:
                numI += 1
            if test_im[x,y] == [True]:
                num_test += 1
            if ans_im[x,y] == [True]:
                num_ans += 1
    print(f"Union {numU}")
    print(f"Intersects {numI}")
    if numI==0:
        numIOU=0
        dice=0
        dice_loss=1
        recall=0
    else:
        numIOU=numI/numU
        dice = (2 * numI) / (numU + numI)
        dice_loss = 1 - dice
        recall = numI / num_ans
    # if numIOU>=0.6:
    #     numIOU=1
    #     recall = 1
    #     dice = 1
    #     dice_loss = 0
    print(f"iou:{numIOU}")
    print(f"dice:{dice}")
    print(f"dice_loss:{dice_loss}")
    print(f"recall:{recall}")
    if num_test==0:
        precision=0
    else:
        precision=numI/num_test
    # if numIOU>=0.6:
    #     precision=1
    print(f"precision:{precision}")
    return round(numIOU,4),round(dice,4),round(dice_loss,4),round(recall,4),round(precision,4)

