import matplotlib
matplotlib.use('agg')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn import tree

link_id_dict={}

def add_to_dict(row):
    global link_id_dict
    
    link_id=row["link_id"]
    link_time=row["link_time"]
    link_cur_status=row["link_current_status"]
    link_arr_status=row["link_arrival_status"]

    if not link_id in link_id_dict:
        link_id_dict[link_id]=[[link_time, [link_cur_status, link_arr_status]]]
    else:
        link_id_dict[link_id].append([link_time, [link_cur_status, link_arr_status]])

if __name__ == "__main__":
    file_number=5
    count=0
    filenames=os.listdir("../data/train/train_link")[1:] # skip 20200803
    for f in filenames:
        df=pd.read_csv(f"../data/train/train_link/{f}")
        df=df.dropna()
        df.apply(add_to_dict, axis=1)

        count+=1
        if count>=file_number:
            break

    max_key=-1
    max_len=-1

    for key, value in link_id_dict.items():
        if len(value)>max_len:
            max_len=len(value)
            max_key=key

    x_time=[]
    y_cur_status=[]
    y_arr_status=[]

    for entry in link_id_dict[max_key]:
        x_time.append(entry[0])
        y_cur_status.append(entry[1][0])
        y_arr_status.append(entry[1][1])

    x_time=np.array(x_time)
    x_time=x_time.reshape(-1, 1)

    clf=tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf=clf.fit(x_time, y_cur_status)

    plt.figure(figsize=(20, 12))
    tree.plot_tree(clf, fontsize=10)
    plt.savefig("./1-1.png")

    clf=tree.DecisionTreeClassifier(criterion="gini", max_depth=3)
    clf=clf.fit(x_time, y_cur_status)

    plt.figure(figsize=(20, 12))
    tree.plot_tree(clf, fontsize=10)
    plt.savefig("./1-2.png")

    clf=tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
    clf=clf.fit(x_time, y_arr_status)

    plt.figure(figsize=(20, 12))
    tree.plot_tree(clf, fontsize=10)
    plt.savefig("./2-1.png")

    clf=tree.DecisionTreeClassifier(criterion="gini", max_depth=3)
    clf=clf.fit(x_time, y_arr_status)

    plt.figure(figsize=(20, 12))
    tree.plot_tree(clf, fontsize=10)
    plt.savefig("./2-2.png")
