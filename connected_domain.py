import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import os#"/home/guojushuai/下载/vibration2000/no/ano_heng_2-2021-04-12-18-47-00/1/2021-04-12-18.47.07.409215.png"
img=cv2.imread('/home/guojushuai/下载/vibration2000/no/ano_heng_2-2021-04-12-18-47-00/1/2021-04-12-18.47.07.409215.mask.png')#[:,:,1]
file_list=os.listdir('/home/guojushuai/下载/vibration2000/list')
img_path=[]
for i in file_list:
    with open('/home/guojushuai/下载/vibration2000/list'+'/'+i,"r") as f:
        lines = f.readlines()
        for p in lines:
            img_path.append('/home/guojushuai/下载/vibration2000'+'/'+p.split(' ')[0][:-4]+'.mask.png')
def connect_domain(img):
    img=img[:,:,1]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    k_list=[]
    for i in range(1,num_labels):
        k_list.append(stats[i][3]/stats[i][2])
        labels = cv2.rectangle(labels, (stats[i][0], stats[i][1]),(stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]), (1), 2)
    plt.imshow(labels)
    plt.show()
    return k_list
k_list=connect_domain(img)

K_list_all=[]
for i in img_path:
    img=cv2.imread(i)
    k_list=connect_domain(img)
    K_list_all+=k_list
print(len(K_list_all))
K_array=np.array(K_list_all).reshape(-1, 1)
#n_clusters=3
#cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(K_array)
#centroid=cluster.cluster_centers_
#print(centroid)
'''
[[0.44380901]
 [6.39415395]
 [1.52039426]]'''