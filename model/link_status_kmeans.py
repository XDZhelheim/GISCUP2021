import matplotlib
matplotlib.use('agg')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import KMeans

link_id_dict={}

def add_to_dict(row):
    global link_id_dict
    
    link_id=int(row["link_id"])
    link_time=row["link_time"]
    link_cur_status=int(row["link_current_status"])
    link_arr_status=int(row["link_arrival_status"])

    if not link_id in link_id_dict:
        link_id_dict[link_id]=[[link_time, [link_cur_status, link_arr_status]]]
    else:
        link_id_dict[link_id].append([link_time, [link_cur_status, link_arr_status]])

if __name__ == "__main__":
    file_number=1
    count=0
    filenames=os.listdir("../data/train/train_link")[1:] # skip 20200803
    for f in filenames:
        df=pd.read_csv(f"../data/train/train_link/{f}")
        df=df.dropna()
        df.apply(add_to_dict, axis=1)

        count+=1
        if count>=file_number:
            break

    x_time_all=[]
    y_cur_status_all=[]
    y_arr_status_all=[]

    for value in link_id_dict.values():
        for entry in value:
            if entry[0]<400:
                x_time_all.append(entry[0])
                y_cur_status_all.append(entry[1][0])
                y_arr_status_all.append(entry[1][1])

    x_time_all=np.array(x_time_all)
    y_cur_status_all=np.array(y_cur_status_all)
    y_arr_status_all=np.array(y_arr_status_all)

    kmeans=KMeans(n_clusters=5)

    kmeans_input=np.vstack((x_time_all, y_cur_status_all*10)).T
    kmeans.fit(kmeans_input)
    print("kmeans cur")
    print(kmeans.cluster_centers_)
    labels=kmeans.labels_
    plt.scatter(kmeans_input[:, 0], kmeans_input[:, 1], c=labels, s=4)
    plt.savefig("./km_cur.png")

    kmeans_input=np.vstack((x_time_all, y_arr_status_all*10)).T
    kmeans.fit(kmeans_input)
    print("kmeans arr")
    print(kmeans.cluster_centers_)
    labels=kmeans.labels_
    plt.scatter(kmeans_input[:, 0], kmeans_input[:, 1], c=labels, s=4)
    plt.savefig("./km_arr.png")

    plt.figure(figsize=(20, 12))
    plt.plot(y_cur_status_all, x_time_all, "bo", ms=2)
    plt.savefig("./cur-time.png")

    plt.figure(figsize=(20, 12))
    plt.plot(y_arr_status_all, x_time_all, "bo", ms=2)
    plt.savefig("./arr-time.png")

    print("cur: {}".format(np.bincount(y_cur_status_all)))
    print("arr: {}".format(np.bincount(y_arr_status_all)))