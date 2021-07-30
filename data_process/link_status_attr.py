import pandas as pd
import numpy as np
from tqdm import tqdm
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

            if not line_list[0].isdigit():
                continue

            lid = line_list[1]

            if "_" in lid:
                continue
            else:
                lid=int(lid)

            link_cur_status = int(line_list[4])
            link_arr_status = int(line_list[5])

            if link_cur_status < 0 or link_arr_status < 0 or link_cur_status > 5 or link_arr_status > 5:
                continue

            if lid not in roads.keys():
                roads[lid] = [0]*10

            roads[lid][link_cur_status]+=1
            roads[lid][link_arr_status+5]+=1

df=pd.DataFrame(columns=["current_status_0", "current_status_1", "current_status_2", "current_status_3", "current_status_4",
    "arrival_status_0", "arrival_status_1", "arrival_status_2", "arrival_status_3", "arrival_status_4"])

for key, value in roads.items():
    df.loc[key]=value

df.to_csv("./link_status_attr.csv")

end=time.time()
print("Time cost = {:.2f}min".format((end-start)/60))