import pandas as pd
from tqdm import tqdm
import numpy as np
import time

start=time.time()

TRAIN_FILES = ['202008'+str(i).zfill(2)+'.csv' for i in range(1,32)]
PATH = '../data/train/train_path/'
roads = {}

for i,fn in enumerate(TRAIN_FILES):
    with open(PATH+fn, 'r') as f:
        next(f)
        for line in tqdm(f.readlines()):
            line_list = line.split(',')

            lid = line_list[1]

            if "_" in lid:
                continue
            else:
                lid=int(lid)

            link_time = float(line_list[2].strip())

            if link_time <= 0:
                continue

            if (lid in roads.keys()):
                roads[lid].append(link_time)
            else:
                roads[lid] = [link_time]

df=pd.DataFrame(columns=["std", "avg", "min", "max", "count"])

for key, value in roads.items():
    df.loc[key]=[np.std(value), np.mean(value), np.min(value), np.max(value), len(value)]

df.to_csv("./link_time_attr.csv")

end=time.time()
print("Time cost = {:.2f}min".format((end-start)/60))