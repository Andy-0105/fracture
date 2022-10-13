import numpy as np
from PIL import Image
def pretreatment(ans, test):
    ans = ans.convert('1')
    test = test.convert('1')
    ans_im = np.array(ans)
    test_im = np.array(test)
    h = ans_im.shape[0]
    w = ans_im.shape[1]
    numU = 0
    numI = 0
    for x in range(h):
        for y in range(w):
            if [ans_im[x,y]] == [True] or [test_im[x,y]] == [True]:
                numU += 1
            if [ans_im[x,y]] == [True] and [test_im[x,y]] == [True]:
                numI += 1
    print(f"Union {numU}")
    print(f"Intersects {numI}")
    numIOU=round(numI/numU,5)
    print(f"iou:{numIOU}")
    dice=2*numIOU/numU
    print(f"dice:{dice}")
    dice_loss=round(1-dice,5)
    print(f"dice_loss:{dice_loss}")

    return numIOU,dice_loss
image_ans = Image.open(r'C:\Users\a0907\Desktop\fracture\membrane\1234.jpg')
image_test = Image.open(r'C:\Users\a0907\Desktop\fracture\membrane\1234.jpg')
pretreatment(image_ans,image_test)