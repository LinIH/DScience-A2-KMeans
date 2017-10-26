"""KMeans"""

import random
import matplotlib.pyplot as plt
import numpy as np
from test.test_decimal import Coverage

def Normalize(input_data, max, min):
    data = np.array([])
    for d in input_data:
        data = np.append(data, (d-min)/(max-min))
    return data

def GetDistance(p1_x, p1_y, p2_x, p2_y):
    d = ((p1_x-p2_x)**2+(p1_y-p2_y)**2)**0.5
    return d

def UpdateCentroid(input_len, input_wid, group_list, input_cen_x, input_cen_y):
    converge = False
    cutoff = 0.001
    new_centroids_x = np.array([])
    new_centroids_y = np.array([])
    
    for i in range(len(input_cen_x)):
        temp_cluster_x = np.array([])
        temp_cluster_y = np.array([])
        for (i_g, g) in enumerate(group_list):
            if g==i:
                temp_cluster_x = np.append(temp_cluster_x, input_len[i_g])
                temp_cluster_y = np.append(temp_cluster_y, input_wid[i_g])
        new_centroids_x = np.append(new_centroids_x, np.mean(temp_cluster_x))
        new_centroids_y = np.append(new_centroids_y, np.mean(temp_cluster_y))

    for i in range(len(input_cen_x)):
        if GetDistance(input_cen_x[i], input_cen_y[i], new_centroids_x[i], new_centroids_y[i]) < cutoff:
            converge = True

    return new_centroids_x, new_centroids_y, converge

def KMeans(input_len, input_wid, k):
    ####                 initial random centroids                #####
    centroids_x, centroids_y = RandomCen(input_len, input_wid, k)
    ##################################################################

    ####                         classify                        #####
    group_list = np.zeros(150)

    while True:
        for i_data in range(len(input_len)):
            min_distance = 9999.0
            min_i_cen = 0
            for i_cen in range(k):
                distance = GetDistance(centroids_x[i_cen], centroids_y[i_cen], input_len[i_data], input_wid[i_data])
                if distance < min_distance:
                    min_distance = distance
                    min_i_cen = i_cen             #min distance centroid
            group_list[i_data] = min_i_cen
        centroids_x, centroids_y, converge = UpdateCentroid(input_len, input_wid, group_list, centroids_x, centroids_y)
        if converge:
            break
    ##################################################################

    ####                           plot                          #####
 
    LABEL_COLOR_MAP = {0 : 'green',
                       1 : 'red',
                       2 : 'blue',
                       3 : 'gray',
                       4 : 'm',
                       5 : 'lightcoral',
                       6 : 'k'}
    label_color = [LABEL_COLOR_MAP[l] for l in group_list]
    plt.scatter(input_len, input_wid, c = label_color)
    plt.scatter(centroids_x, centroids_y, c = 'yellow', marker = '*')   #centroids
    plt.show()
  
    return 0

def RandomCen(input_len, input_wid, k):
    centroids_x = np.array([])
    centroids_y = np.array([])
    temp_a = np.hstack((input_len, input_wid))
    temp_b = np.reshape(temp_a, (150,2), order='F')
    ran_num = random.sample(range(150), k)
    for i in range(k):
        centroids_x = np.append(centroids_x, temp_b[ran_num[i]][0])
        centroids_y = np.append(centroids_y, temp_b[ran_num[i]][1])
    
    return centroids_x, centroids_y

def main():
    ####                        load data                        #####
    data = np.array([])
    petal_len = np.array([])
    petal_wid = np.array([])
    f = open('iris.data', 'r')
    for row in f:
        if not row.split():
            continue
        data = np.append(data, row.split(','))
    for i in range(len(data)):
        if i%5==2:
            petal_len = np.append(petal_len, float(data[i]))
        elif i%5==3:
            petal_wid = np.append(petal_wid, float(data[i]))
        else:
            continue
    ##################################################################
    ####                     find max and min                    #####
    petal_len_max = np.amax(petal_len)
    petal_len_min = np.amin(petal_len)
    petal_wid_max = np.amax(petal_wid)
    petal_wid_min = np.amin(petal_wid)
    ##################################################################
    ####                      normalize data                     #####
    normed_petal_len = Normalize(petal_len, petal_len_max, petal_len_min)
    normed_petal_wid = Normalize(petal_wid, petal_wid_max, petal_wid_min)
    
    KMeans(normed_petal_len, normed_petal_wid, 2)
    KMeans(normed_petal_len, normed_petal_wid, 3)
    KMeans(normed_petal_len, normed_petal_wid, 4)
    KMeans(normed_petal_len, normed_petal_wid, 5)
    

main()