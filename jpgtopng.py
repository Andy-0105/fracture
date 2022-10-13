from glob import glob
import cv2
import os
pngs = glob('./demo/image/new48time/0/*.jpg')

for j in pngs:
    img = cv2.imread(j)
    cv2.imwrite(j[:-3] + 'png', img)
    os.remove(j)