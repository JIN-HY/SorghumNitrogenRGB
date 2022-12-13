# coding: utf-8
import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

from scipy.signal import medfilt2d

#folder_names = pd.read_csv("folder_names.csv")
#folder_names['vis0'] = [x+"/Vis_SV_0/0_0_0.png" for x in folder_names.d]

# where is convert_time
#folder_names = folder_names.sort_values(by="img_date",key = lambda date: convert_time(date, "%Y-%m-%d"))

folder_names = pd.read_csv("folder_names1.csv")




def read_rgb(fname):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32')/255
    return img


Io = read_rgb("subtractor.png")

def subtract_bg(img, Io):
    img_s_bg = 1 - (Io - img)
    img_s_bg[img_s_bg<0] = 0
    return img_s_bg
    

def remove_white(i): #denoise
    im_gray = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY)
    #im_gray = medfilt2d(im_gray) # can try contour something later
    im_bool = im_gray > 0.8
    im_bin = 1- im_bool * 1
    i_seg = i*im_bin[...,None]
    i_seg = i_seg.astype('float32')
    i_seg[0,:,:] = 0
    i_seg[-1,:,:] = 0
    i_seg[:,0,:] = 0
    i_seg[:,-1:,0] = 0
    return i_seg

def denoise(i, kernel):
    img = cv2.morphologyEx(i, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

# resize
def resize(img, percent):
    width = int(img.shape[1] * percent/100)
    height = int(img.shape[0] * percent/100)
    dim = (width, height)
    img_resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img_resized


machine = ['Vis_SV_0', 'Vis_SV_36', 'Vis_SV_72','Vis_SV_90', 'Vis_SV_108','Vis_SV_144','Vis_SV_216', 'Vis_SV_252','Vis_SV_288','Vis_SV_324']

treatment = ["LN", "HN"]


m = machine[int(sys.argv[1])]
t = treatment[int(sys.argv[2])]
folder_names['png'] = [x+f"/{m}/0_0_0.png" for x in folder_names.d]
folder_names = folder_names[[Path(x).exists() for x in folder_names.png]]
# need to separate by treatment, genotype and time and then parallel

folder_names2 = folder_names[folder_names.trt == t]

img_subtracted = (subtract_bg(read_rgb(x), Io) for x in folder_names2.png)

#rect = [1180, 885, 3500, 2800] # indices obtained from matlab
img_cropped = (x[2000:5456, 885:3685] for x in img_subtracted)
kernel = np.ones((5,5),np.uint8)
img_dark = (remove_white(x) for x in img_cropped)# final segmented images
img_seg = (denoise(x, kernel) for x in img_dark)
scale_percent = 10
img_seg_resized = (resize(img, scale_percent) for img in img_seg)
for isr, smallimg in zip(img_seg_resized, folder_names2.png):
    isr_bgr = cv2.cvtColor(np.uint8(isr*255), cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{smallimg}.sml.png', isr_bgr)
