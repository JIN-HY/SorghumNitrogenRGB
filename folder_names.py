# coding: utf-8
import pandas as pd
import numpy as np
import os
import glob


ds = []
plantnumbers = []
trts = []
img_dates = []
for d in glob.iglob("Sorghumnutrient/*"):
    ds.append(d)
    d = d.split("_")
    print(d)
    plant = d[1]
    img_date = d[2]
    pre = plant.split("-")
    plantnumber = pre[2]
    trt = pre[3]
    img_dates.append(img_date)
    plantnumbers.append(plantnumber)
    trts.append(trt)
    

df = pd.DataFrame({'d':ds, 'trt':trts, 'plant':plantnumbers, 'date':img_dates})
df
df.to_csv("folder_names.csv", index=None)
