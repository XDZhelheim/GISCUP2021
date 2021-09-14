import pandas as pd
import numpy as np
from tqdm import tqdm
import time

start=time.time()

PATH = '../data/train/train_path/'
PATH_FILES = ['202008'+str(i).zfill(2)+'.csv' for i in range(1,32)]

link_dict = {}
for i, f in tqdm(enumerate(PATH_FILES)):    
    df_path = pd.read_csv(PATH+f)

    o_count = df_path.groupby("order_id").first()["link_id"].value_counts()
    d_count = df_path.groupby("order_id").last()["link_id"].value_counts()

    for index, value in o_count.items():
        if index not in link_dict:
            link_dict[index] = [0, 0]
        link_dict[index][0] += value

    for index, value in d_count.items():
        if index not in link_dict:
            link_dict[index] = [0, 0]
        link_dict[index][1] += value

df=pd.DataFrame(columns=["o_count", "d_count"])

for key, value in link_dict.items():
    df.loc[key]=value

df.to_csv("./OD_attr.csv")

end=time.time()
print("Time cost = {:.2f}min".format((end-start)/60))