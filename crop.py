import numpy as np
import matplotlib.pyplot as plt

def crop_img_center(img,cropx,cropy,cropz):
    z,x,y = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    startz = z//2-(cropz//2)

    return img[startz:startz+cropz,starty:starty+cropy,startx:startx+cropx]

def crop_seg_center(seg,cropx,cropy,cropz):
    z,x,y = seg.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    startz = z//2-(cropz//2)

    return seg[startz:startz+cropz,starty:starty+cropy,startx:startx+cropx]

img_crop_arr = np.zeros((10,16,128,128))
seg_crop_arr = np.zeros((10,16,128,128))
for i in range(0,10):
    case = np.load('./valid_data/case'+str(i)+'.npz') # To see stthe different cases
    img = case['img']
    seg = case['seg']
    img_crop = crop_img_center(img,128,128,16)
    seg_crop = crop_seg_center(seg,128,128,16)
    img_crop_arr[i] = img_crop
    seg_crop_arr[i] = seg_crop
np.savez('crop_test.npz', img_crop = img_crop_arr, seg_crop = seg_crop_arr)
