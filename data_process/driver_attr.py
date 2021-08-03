import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import scipy.stats as st

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

            did = int(line_list[4])

            if did not in drivers.keys():
                drivers[did] = [[] for _ in range(5)]

            drivers[did][0].append(int(line_list[1])-int(line_list[3])) # ata-simple_eta
            drivers[did][1].append(int(line_list[1])) # ata
            drivers[did][2].append(float(line_list[2])) # distance
            drivers[did][3].append(int(line_list[3])) # simple_eta
            drivers[did][4].append(int(line_list[5])) # slice_id

df=pd.DataFrame(columns=[
    "ata_minus_simple_eta_std", "ata_minus_simple_eta_avg", "ata_minus_simple_eta_min", "ata_minus_simple_eta_max", "ata_minus_simple_eta_skew", "ata_minus_simple_eta_kurt",
    "ata_std", "ata_avg", "ata_min", "ata_max", "ata_skew", "ata_kurt",
    "distance_std", "distance_avg", "distance_min", "distance_max", "distance_skew", "distance_kurt",
    "simple_eta_std", "simple_eta_avg", "simple_eta_min", "simple_eta_max", "simple_eta_skew", "simple_eta_kurt",
    "slice_id_std", "slice_id_avg", "slice_id_min", "slice_id_max", "slice_id_skew", "slice_id_kurt",
    "count"
                ])

for key, value in drivers.items():
    df.loc[key]=[
        np.std(value[0]), np.mean(value[0]), np.min(value[0]), np.max(value[0]), st.skew(value[0]), st.kurtosis(value[0]),
        np.std(value[1]), np.mean(value[1]), np.min(value[1]), np.max(value[1]), st.skew(value[1]), st.kurtosis(value[1]),
        np.std(value[2]), np.mean(value[2]), np.min(value[2]), np.max(value[2]), st.skew(value[2]), st.kurtosis(value[2]),
        np.std(value[3]), np.mean(value[3]), np.min(value[3]), np.max(value[3]), st.skew(value[3]), st.kurtosis(value[3]),
        np.std(value[4]), np.mean(value[4]), np.min(value[4]), np.max(value[4]), st.skew(value[4]), st.kurtosis(value[4]),
        len(value[0])
    ]

df.to_csv("./driver_attr.csv")

end=time.time()
print("Time cost = {:.2f}min".format((end-start)/60))