import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics.cluster import adjusted_rand_score as ARI
from sklearn.cluster import KMeans
from scipy.optimize import minimize

def MAAD(data,b):
    temp = []

    for index in range(len(data)):
        if((data[index]==b[0]).all() or (data[index]==b[1]).all()):
            continue
        else:
            temp.append(data[index])

    z = np.array(temp)
    #print(temp)

    dist_v = np.linalg.norm(z - b[0],axis=1)
    #print('dist_v:',dist_v)
    dist_u = np.linalg.norm(z - b[1],axis=1)
    #print('dist_u:',dist_u)
    dist = abs(dist_v - dist_u)
    #print('dist:',dist)
    MAAD = (dist.sum()/math.sqrt(data.shape[1]))**2/(data.shape[0]-2)
    #print('MAAD:',MAAD)
    return MAAD

def function(cs_old,load,cluster_assigned):
    total = 0
    k = len(np.unique(load[:,load.shape[1]-1]))
    data = load[:,:load.shape[1]-1]
    cs_old = np.reshape(cs_old,(k,data.shape[1]))
    for j in range(k):
        data_cluster = data[cluster_assigned == j]
        inside_sqdist = 0
        for i in range(len(data[cluster_assigned == j])):
            t = []
            
            for index in range(k):
                if((cs_old[index]==cs_old[j]).all()):
                    continue
                else:
                    t.append(cs_old[index])
            t = np.array(t)
            d = 0
            for i1 in range(t.shape[0]):
                d = d + (np.linalg.norm(data_cluster[i] - t[i1])**2 - np.linalg.norm(cs_old[j] - t[i1])**2)**2
            inside_sqdist = inside_sqdist + d
        total = total + inside_sqdist
    return total
        
def grad_fun(data,cluster_assigned,k,cs_old):
    cs = cs_old[j]
    total = 0
    data_cluster = data[cluster_assigned == j]
    for i in range(len(data[cluster_assigned == j])):
        t = []
        for index in range(k):
            if(cs_old[index]==cs.all()):
                continue
            else:
                t.append(cs_old[index])
        t = np.array(t)
        delta = 0       
        for i1 in range(t.shape[0]):
            delta = delta + (np.linalg.norm(data_cluster[i] - t[i1],axis=1)**2 - np.linalg.norm(cs - t[i1],axis=1)**2)*(cs - t[i1])*(-4)
        total = total + delta
    return total
    

load = np.loadtxt('4clusts.txt')
k = len(np.unique(load[:,load.shape[1]-1]))
data = load[:,:load.shape[1]-1]
print('k:',k)
cs_old = np.zeros((k,data.shape[1]))
cs_new = data[np.random.permutation(data.shape[0])[0:k]]

er = np.linalg.norm(cs_new - cs_old)
#print(er)
distance = np.zeros((data.shape[0],k))
cluster_assigned = np.zeros(data.shape[0])
cs_init = np.array(cs_new)
#print(cs_new)
b = np.zeros((2,data.shape[1]))
for iter1 in range(10):
    print('Iteration:', iter1)
    for i in range(k):
        b[0] = cs_new[i]
        #print('b[0]:',b[0])
        MAAD_dist = np.zeros(data.shape[0])
        for j in range(data.shape[0]):
            b[1] = data[j]
            MAAD_dist[j] = MAAD(data,b)*(data.shape[0]-2)
            #print(MAAD_dist[j])
        distance[:,i] = MAAD_dist
        #print('MAAD_dist:',MAAD_dist)

    cluster_assigned = np.argmin(np.array(distance),axis = 1)
    print(cluster_assigned)
    cs_old = np.array(cs_new)
    #for j in range(k):
    res = minimize(fun = function, x0 = cs_old, args=(load,cluster_assigned))
    cs_new = np.reshape(res.x,(k,data.shape[1]))
    er = np.linalg.norm(cs_new - cs_old)
    #print('er:',er)

    if er < 1e-9:
        print('#iter', iter1)
        break

print(cluster_assigned)

print('ARI_MAAD:',ARI(cluster_assigned, load[:,load.shape[1]-1]))
kmeans = KMeans(n_clusters=k,init = 'random', max_iter=100, n_init=1).fit(data)
print('ARI_Kmeans:',ARI(kmeans.labels_, load[:,load.shape[1]-1]))
print()
