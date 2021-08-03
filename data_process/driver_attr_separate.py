import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import scipy.stats as st
import gc

start=time.time()

TRAIN_FILES = ['202008'+str(i).zfill(2)+'.csv' for i in range(1,32)]
PATH = '../data/train/train_head/'

drivers = {}
for i, fn in tqdm(enumerate(TRAIN_FILES)):
    with open(PATH+fn, 'r') as f:
        next(f)
        for line in f.readlines():
            line_list = line.split(',')

            if not line_list[0].isdigit():
                continue

            lid = line_list[1]

            if "_" in lid:
                continue

            did = line_list[4]

            if did not in drivers.keys():
                drivers[did] = []

            drivers[did].append(int(line_list[1])-int(line_list[3])) # ata-simple_eta

df=pd.DataFrame(columns=[
    "ata_minus_simple_eta_std", "ata_minus_simple_eta_avg", "ata_minus_simple_eta_min", "ata_minus_simple_eta_max", "ata_minus_simple_eta_skew", "ata_minus_simple_eta_kurt",
    "count"
                ])

for key, value in drivers.items():
    df.loc[key]=[
        np.std(value), np.mean(value), np.min(value), np.max(value), st.skew(value), st.kurtosis(value),
        len(value)
    ]
df.to_csv("./driver_attr_0.csv")

del df, drivers
gc.collect()

drivers = {}
for i, fn in tqdm(enumerate(TRAIN_FILES)):
    with open(PATH+fn, 'r') as f:
        next(f)
        for line in f.readlines():
            line_list = line.split(',')

            if not line_list[0].isdigit():
                continue

            lid = line_list[1]

            if "_" in lid:
                continue

            did = line_list[4]

            if did not in drivers.keys():
                drivers[did] = []

            drivers[did].append(int(line_list[1])) # ata

df=pd.DataFrame(columns=[
    "ata_std", "ata_avg", "ata_min", "ata_max", "ata_skew", "ata_kurt"
                ])

for key, value in drivers.items():
    df.loc[key]=[
        np.std(value), np.mean(value), np.min(value), np.max(value), st.skew(value), st.kurtosis(value)
    ]
df.to_csv("./driver_attr_1.csv")

del df, drivers
gc.collect()

drivers = {}
for i, fn in tqdm(enumerate(TRAIN_FILES)):
    with open(PATH+fn, 'r') as f:
        next(f)
        for line in f.readlines():
            line_list = line.split(',')

            if not line_list[0].isdigit():
                continue

            lid = line_list[1]

            if "_" in lid:
                continue

            did = line_list[4]

            if did not in drivers.keys():
                drivers[did] = []

            drivers[did].append(float(line_list[2])) # distance

df=pd.DataFrame(columns=[
    "distance_std", "distance_avg", "distance_min", "distance_max", "distance_skew", "distance_kurt"
                ])

for key, value in drivers.items():
    df.loc[key]=[
        np.std(value), np.mean(value), np.min(value), np.max(value), st.skew(value), st.kurtosis(value)
    ]
df.to_csv("./driver_attr_2.csv")

del df, drivers
gc.collect()

drivers = {}
for i, fn in tqdm(enumerate(TRAIN_FILES)):
    with open(PATH+fn, 'r') as f:
        next(f)
        for line in f.readlines():
            line_list = line.split(',')

            if not line_list[0].isdigit():
                continue

            lid = line_list[1]

            if "_" in lid:
                continue

            did = line_list[4]

            if did not in drivers.keys():
                drivers[did] = []

            drivers[did].append(int(line_list[3])) # simple_eta

df=pd.DataFrame(columns=[
    "simple_eta_std", "simple_eta_avg", "simple_eta_min", "simple_eta_max", "simple_eta_skew", "simple_eta_kurt"
                ])

for key, value in drivers.items():
    df.loc[key]=[
        np.std(value), np.mean(value), np.min(value), np.max(value), st.skew(value), st.kurtosis(value)
    ]
df.to_csv("./driver_attr_3.csv")

del df, drivers
gc.collect()

drivers = {}
for i, fn in tqdm(enumerate(TRAIN_FILES)):
    with open(PATH+fn, 'r') as f:
        next(f)
        for line in f.readlines():
            line_list = line.split(',')

            if not line_list[0].isdigit():
                continue

            lid = line_list[1]

            if "_" in lid:
                continue

            did = line_list[4]

            if did not in drivers.keys():
                drivers[did] = []

            drivers[did].append(int(line_list[5])) # slice_id

df=pd.DataFrame(columns=[
    "slice_id_std", "slice_id_avg", "slice_id_min", "slice_id_max", "slice_id_skew", "slice_id_kurt"
                ])

for key, value in drivers.items():
    df.loc[key]=[
        np.std(value), np.mean(value), np.min(value), np.max(value), st.skew(value), st.kurtosis(value)
    ]
df.to_csv("./driver_attr_4.csv")

del df, drivers
gc.collect()

end=time.time()
print("Time cost = {:.2f}min".format((end-start)/60))