import os
from PIL import Image
path= "F:/48times_augmentation/4"
save_path= 'F:/unet_48time/image/4'
files=os.listdir(path)
for pic in files:
    print(pic)
    img=Image.open(os.path.join(path,pic)).convert('L')
    print(img.getbands())
    pic_new=os.path.join(save_path,pic)
    img.save(pic_new[:-3] + 'png')
