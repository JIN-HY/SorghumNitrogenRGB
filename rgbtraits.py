# coding: utf-8
# calculate the SS of EXG and will get SD in the next step or just avg

import cv2
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import sys

from scipy.signal import medfilt2d

#folder_names = pd.read_csv("folder_names.csv")
#folder_names['vis0'] = [x+"/Vis_SV_0/0_0_0.png" for x in folder_names.d]

# where is convert_time
#folder_names = folder_names.sort_values(by="img_date",key = lambda date: convert_time(date, "%Y-%m-%d"))

folder_names = pd.read_csv("folder_names1.csv")

machines = ['Vis_SV_0', 'Vis_SV_36', 'Vis_SV_72','Vis_SV_90', 'Vis_SV_108','Vis_SV_144','Vis_SV_216', 'Vis_SV_252','Vis_SV_288','Vis_SV_324']
treatments = ["LN", "HN"]
plantnumbers = ["J"+str(x)[1:] for x in range(1001, 1037)]

plant = sys.argv[1]
#m = machine[int(sys.argv[1])]
#t = treatment[int(sys.argv[2])]

machine = machines[int(sys.argv[2])]

outcsv = f"{plant}_{machine}_rgbtraits.csv"

folder_names['png'] = [x+f"/{machine}/0_0_0.png.sml.png" for x in folder_names.d]

plantnumbers = ["J"+str(x)[1:] for x in range(1001, 1037)]
#folder_names = folder_names[folder_names.genotype == genotype]
#folder_names = folder_names[folder_names.trt == treatment]
results_df = folder_names[folder_names.plant == plant].copy()


def read_rgb(fname):
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

    
smlimg = (read_rgb(x) for x in results_df.png)


class I_seg:
    def __init__(self, img):
        self.i = img
    def get_area(self):
        im_gray = cv2.cvtColor(self.i, cv2.COLOR_RGB2GRAY)
        self.ids = np.where(im_gray>0)    
    def rgb(self):
        R, G, B = cv2.split(self.i)
        sumrgb = R + G + B
        r = np.divide(R, sumrgb)
        g = np.divide(G, sumrgb)
        b = np.divide(B, sumrgb)
        self.r = r
        self.g = g
        self.b = b    
    def plant_height(self):
        x1 = self.ids[0][0]
        x2 = self.ids[0][-1]
        return x2-x1    
    def plant_width(self):
        x1 = min(self.ids[1])
        x2 = max(self.ids[1])
        return x2-x1
    def exg(self):
        exg = 2*self.g-self.r-self.b
        exg = exg[self.ids]
        self.exg = exg
        return np.mean(exg[np.isfinite(exg)])
    def veg(self):
        veg = self.g/self.r**0.667/self.b**0.333
        veg = veg[self.ids]
        return np.mean(veg[np.isfinite(veg)])
    def exgSS(self):
        exg_fin = self.exg[np.isfinite(self.exg)]
        exgSS = (exg_fin - np.mean(exg_fin))**2
        return np.sum(exgSS)/len(self.ids)


PHs = []
PWs = []
AREAs = []
EXGs = []
ExGSS = []
VEGs = []

for i in smlimg:
    i = I_seg(i)
    i.get_area()
    i.rgb()
    AREAs.append(len(i.ids[0]))
    PHs.append(i.plant_height())
    PWs.append(i.plant_width())
    EXGs.append(i.exg())
    VEGs.append(i.veg())
    ExGSS.append(i.exgSS()) 


results_df['PlantHeight'] = PHs
results_df['PlantWidth'] = PWs
results_df['PixelCount'] = AREAs
results_df['ExG'] = EXGs
results_df['VEG'] = VEGs
results_df['ExGSS'] = ExGSS

results_df['machine'] = f"{machine}"

results_df.to_csv(outcsv, sep = ",", index = False, doublequote = False)
