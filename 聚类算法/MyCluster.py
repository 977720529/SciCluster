from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
import math

fileName = input("请输入文件名: ")
label = []
data = []
tag = int(input("标签位置："))
for line in open(fileName, "r"):
    items = line.strip("\n").split(",")
    # label.append(items.pop())
    tmp = []
    for i in range(0, len(items), 1):
        if i == tag:
            continue
        tmp.append(float(items[i]))
    data.append(tmp)
data = np.array(data)
# 3 points in dataset
# data = np.array([[1, 1, 1, 1],
#                 [2, 2, 2, 2],
#                 [10, 10, 10, 10],
#                 [3, 3, 3, 3],
#                  [1, 1, 1, 1]])

# distance matrix
D = pairwise_distances(data, metric='euclidean')
m, n = D.shape
d = np.reshape(D, -1)
d = np.sort(d)
# print(d)

distall = []
for i in range(0, n, 1):
    for j in range(i + 1, n, 1):
        distall.append(D[i][j])

newdis = sorted(distall)
num = int(len(distall)*0.02)
dc = newdis[num]     # 截止距离，全部点的距离从小到大的第2%位置的距离

def density_i():
    rho = np.zeros((n, 1))
    for i in range(n):
        for j in range(n):
            if D[i][j] < dc:
                rho[i] = rho[i] + 1
                rho[j] = rho[j] + 1
    return rho


def getRhoDescIndex(rho):
    preOrder = []
    for i in range(n):
        preOrder.append([rho[i], i])
    preOrder = sorted(preOrder, key=lambda x : x[0], reverse=True)
    newOrder = []
    for i in range(n):
        newOrder.append(int(preOrder[i][1]))
    return newOrder

def desityGause():
    rho = np.zeros((n, 1))
    for i in range(0, n, 1):
        for j in range(i + 1, n, 1):
            rho[i] = rho[i] + math.exp(-(D[i][j] / dc) ** 2)
            rho[j] = rho[j] + math.exp(-(D[i][j] / dc) ** 2)
    return rho


def getsigma(order):
    distMax = max(d)
    print(distMax)
    sigma_i = np.ones(n, dtype=np.float) * (-1)
    print(sigma_i)
    neighbor = np.zeros(n, dtype=np.int)
    # delta(ordrho(1)) = -1.;
    # nneigh(ordrho(1)) = 0;
    sigma_i[order[0]] = -1
    neighbor[order[0]] = 0
    print(neighbor)
    for i in range(1, n, 1):
        sigma_i[order[i]] = distMax
        for j in range(0, i, 1):
            if D[order[i]][order[j]] < sigma_i[order[i]]:
                sigma_i[order[i]] = D[order[i]][order[j]]
                neighbor[order[i]] = order[j]
    sigma_i[order[0]] = max(sigma_i)
    return sigma_i, neighbor


# 根据阈值确定聚类中心
def findClusterCenter(rhoset, sigmaset):
    center = 0
    resCluster = np.ones(n, dtype=np.int) * (-1)
    Clustcnt = []
    for i in range(n):
        if rhoset[i] > rhomin and sigmaset[i] > sigmaMin:
            resCluster[i] = center
            center += 1
            Clustcnt.append(i)
    return resCluster, center, Clustcnt


# 根据聚类中心分配非中心点
def clusterAll(res, ClusCnt):
    for i in range(n):
        if res[i] == -1:
            dismal = max(d) + 1
            for j in range(len(ClusCnt)):
                if D[i][ClusCnt[j]] < dismal:
                    dismal = D[i][ClusCnt[j]]
                    res[i] = res[ClusCnt[j]]
    return res
# def clusterAll(res, order, neighbor):
#     for i in range(n):
#         if res[order[i]] == -1:
#             res[order[i]] = res[neighbor[order[i]]]
#     return res


rho = desityGause()
order = getRhoDescIndex(rho)
[sigma, neighbor] = getsigma(order)
plt.plot(rho, sigma, ".")
plt.xlabel("rho")
plt.ylabel("sigma")
plt.show()

rhomin = float(input("rho的阈值: "))
sigmaMin = float(input("sigma的阈值: "))

[resC, centers, Cluster] = findClusterCenter(rho, sigma)
# resC = clusterAll(resC, order, neighbor)
resC = clusterAll(resC, Cluster)
for i in range(centers):
    print("cluster " + str(i) + ":")
    lengths = 0
    for j in range(n):
        if resC[j] == i:
            lengths += 1
            print(data[j])
    print(lengths)
    print("\n")


