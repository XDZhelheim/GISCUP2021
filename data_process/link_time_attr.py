import pandas as pd
from tqdm import tqdm
import numpy as np
import csv
import os
import time

start=time.time()

TRAIN_FILES = ['202008'+str(i).zfill(2)+'.csv' for i in range(1,32)]
PATH = '../data/train/train_path/'
roads = {}

for i,fn in enumerate(TRAIN_FILES):
    with open(PATH+fn, 'r') as f:
        next(f)
        for line in tqdm(f.readlines()):
            lid = line.split(',')[1]

            if "_" in lid:
                continue
            else:
                lid=int(lid)

            if (lid in roads.keys()):
                roads[lid].append(float(line.split(',')[2]))
            else:
                roads[lid] = [float(line.split(',')[2])]

df=pd.DataFrame(columns=["std", "avg", "min", "max", "count"])

for key, value in roads.items():
    df.loc[key]=[np.std(value), np.mean(value), np.min(value), np.max(value), len(value)]

df.to_csv("./link_attr2.csv")

end=time.time()
print("Time cost = {:.2f}min".format((end-start))/60)