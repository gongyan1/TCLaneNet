import time

import cv2
import json
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
def connect_domain(img):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    labels[labels>0]=1
    k_list=[]
    #plt.imshow(labels)
    #plt.show()
    for i in range(1,num_labels):
        #plt.imshow(labels[stats[i][1]:stats[i][1] + stats[i][3],stats[i][0]:stats[i][0] + stats[i][2]])
        #plt.show()
        #print(np.sum(np.diag(labels[stats[i][0]:stats[i][0] + stats[i][2],stats[i][1]:stats[i][1] + stats[i][3]])))
        #print(np.sum(np.diag(np.fliplr(labels)[stats[i][0]:stats[i][0] + stats[i][2],stats[i][1]:stats[i][1] + stats[i][3]])))
        #if np.sum(np.diag(labels[stats[i][1]:stats[i][1] + stats[i][3],stats[i][0]:stats[i][0] + stats[i][2]]))>np.sum(np.diag(np.fliplr(labels[stats[i][1]:stats[i][1] + stats[i][3],stats[i][0]:stats[i][0] + stats[i][2]]))):
        #print(labels[stats[i][1], stats[i][0]],labels[stats[i][1] + stats[i][3] , stats[i][0] + stats[i][2]],labels[stats[i][1],stats[i][0] + stats[i][2]],labels[stats[i][1] + stats[i][3],stats[i][0]])
        #if labels[stats[i][1], stats[i][0]] != 0 or labels[stats[i][1] + stats[i][3] , stats[i][0] + stats[i][2]]!=0:
        #    labels = cv2.line(labels, (stats[i][0], stats[i][1]),(stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]), (2), 2)
        #else:
        #    labels = cv2.line(labels, (stats[i][0],  stats[i][1] + stats[i][3]),(stats[i][0] + stats[i][2],stats[i][1]), (2), 2)
        k_list.append(stats[i][3]/stats[i][2])
        #labels = cv2.rectangle(labels, (stats[i][0], stats[i][1]),(stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]), (1), 2)
    #plt.imshow(labels)
    #plt.show()
    return k_list
base_path = "/home/guojushuai/下载/tusimple/"
filelist=['/label_data_0601.json','/label_data_0531.json','/label_data_0313.json']
K_list_all=[]
image_num = 0
for i in filelist:
    file = open(base_path + i, 'r')
    for line in file.readlines():
        k_list=[]
        data = json.loads(line)
        print(data['raw_file'])
        image = cv2.imread(os.path.join(base_path, data['raw_file']))#720, 1280
        # 二进制图像数组初始化
        binaryimage = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
        # 实例图像数组初始化
        instanceimage = binaryimage.copy()
        arr_width = data['lanes']
        arr_height = data['h_samples']
        width_num = len(arr_width)  # 标注的道路条数
        height_num = len(arr_height)
        # print width_num
        # print height_num
        # 遍历纵坐标
        for j in range(width_num):
            lane_l=[]
            lane_hist = 40
            # 遍历各个车道的横坐标
            for i in range(height_num):
                # 端点坐标赋值
                if arr_width[j][i - 1] > 0 and arr_width[j][i] > 0:
                    binaryimage[int(arr_height[i]), int(arr_width[j][i])] = 255  # 255白色，0是黑色
                    instanceimage[int(arr_height[i]), int(arr_width[j][i])] = lane_hist
                    if i > 0:
                        # 画线，线宽10像素
                        cv2.line(binaryimage, (int(arr_width[j][i - 1]), int(arr_height[i - 1])),
                                 (int(arr_width[j][i]), int(arr_height[i])), 255, 5)
                        cv2.line(instanceimage, (int(arr_width[j][i - 1]), int(arr_height[i - 1])),
                                 (int(arr_width[j][i]), int(arr_height[i])), lane_hist, 5)
                        lane_l.append([arr_width[j][i - 1],arr_height[i - 1]])
                        lane_l.append([arr_width[j][i], arr_height[i]])
                # lane_hist += 50

            #print(len(lane_l))
            #cv2.imshow('instanceimage.jpg', binaryimage)
            #cv2.waitKey()
            if len(lane_l)>0:
                #print((lane_l[-1][1]-lane_l[0][1])/(lane_l[-1][0]-lane_l[0][0])*image.shape[1]/image.shape[0])
                #print(lane_l[-1][0]-lane_l[0][0])
                K=abs((lane_l[-1][1]-lane_l[0][1])/(lane_l[0][0]-lane_l[-1][0])*image.shape[1]/image.shape[0])
                k_list.append(K)#(lane_l[-1][1]-lane_l[0][1])/(lane_l[0][0]-lane_l[-1][0])*image.shape[1]/image.shape[0]
                #print((lane_l[-1][1]-lane_l[0][1])/(lane_l[0][0]-lane_l[-1][0])*image.shape[1]/image.shape[0])
        instanceimage=cv2.resize(instanceimage,[720,720])
        #print(len(k_list))
        #k_list = connect_domain(instanceimage)
        #print(len(k_list))
        #time.sleep(1)
        if len(k_list)>0:
            K_list_all += k_list
        #cv2.imshow('image.jpg', image)
        #cv2.waitKey()
        #cv2.imshow('binaryimage.jpg', binaryimage)
        #cv2.waitKey()
        #cv2.imshow('instanceimage.jpg', instanceimage)
        #cv2.waitKey()
        # string1 = base_path + "\\" + str(image_num + 10) + ".png"
        # string2 = base_path + "\\" + str(image_num + 11) + ".png"
        # string3 = base_path + "\\" + str(image_num + 12) + ".png"
        # cv2.imwrite(string1, binaryimage)
        # cv2.imwrite(string2, instanceimage)
        # cv2.imwrite(string3, image)
        image_num = image_num + 1
    file.close()

print("total image_num:" + str(image_num))
print(len(K_list_all))
print(np.mean(np.array(K_list_all)))
K_array=np.array(K_list_all).reshape(-1, 1)

n_clusters=4
cluster = KMeans(n_clusters=n_clusters,random_state=0).fit(K_array)
centroid=cluster.cluster_centers_
print(centroid)
#13671
