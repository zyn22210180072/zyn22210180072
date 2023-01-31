# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from tqdm import tqdm

max = sys.float_info.min
min = sys.float_info.max
list = []
max_x = 0
max_y = 0


def readData():
    global max
    global min
    global list
    global max_x,max_y
    # list = np.array([[1304, 2312], [3639, 1315], [4177, 2244], [712, 1399], [3488, 1535], [3326, 1556],
    #                  [3238, 1229], [4196, 1004], [4312, 790], [4386, 570], [3007, 1970], [2562, 1756],
    #                  [2788, 1491], [2381, 1676], [1332, 695], [3715, 1678], [3918, 2179], [4061, 2370], [3780, 2212],
    #                  [3676, 2578], [4029, 2838], [4263, 2931], [3429, 1908], [3507, 2367], [3394, 2643], [3439, 3201],
    #                  [2935, 3240], [3140, 3550], [2545, 2357], [2778, 2826], [2370, 2975], ])

    list = np.array([[116.46, 39.92], [117.2, 39.13], [121.48, 31.22], [106.54, 29.59], [91.11, 29.97], [87.68, 43.77],
                     [106.27, 38.47], [111.65, 40.82], [108.33, 22.84], [126.63, 45.7], [125.35, 43.8], [123.38, 41.8],
                     [114.48, 38.03], [112.53, 37.87], [101.74, 36.56], [117, 36.65], [113.6, 34.76], [118.78, 32.04],
                     [117.27, 31.86],
                     [120.19, 30.26], [119.3, 26.08], [115.89, 28.68], [113, 28.21], [114.31, 30.52], [113.23, 23.16],
                     [121.5, 25.05],
                     [110.35, 20.02], [103.73, 36.03], [108.95, 34.27], [104.06, 30.67], [106.71, 26.57],
                     [102.73, 25.04], [114.1, 22.2], [113.33, 22.13]])
    # list = np.array([[2, 6], [2, 4], [1, 3], [4, 6], [5, 5], [4, 4], [6, 4], [3, 2]],
    #     )

    n, _ = list.shape
    # with open('citys.txt', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         line = line.replace('\n', '').split('\t')
    #         # x,y=float(line[1]),float(line[2])
    #         print(line[0])
    #         # list.append((x,y))
    #         count += 1

    dist = np.zeros((n, n), dtype=np.float)
    for i in range(n):
        for j in range(n):
            # x0 = list[i - 1],0]
            # y0 = list[i - 1,1]
            # x1 = list[j - 1],0]
            # y1 = list[j - 1],1]
            x0 = list[i][0]
            y0 = list[i][1]
            x1 = list[j][0]
            y1 = list[j][1]
            if x0 > max_x:
                max_x = x0
            if y0 > max_y:
                max_y = y0
            distance = math.sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1))
            if i != j and distance < min:
                min = distance
            dist[i, j] = distance
    max = dist.max()

    # for i in range(n):
    #     for j in range(n):
    #         if i!=j:
    #             dist[i,j]=(dist[i,j]-min)/max
    # else:
    #     dist[i,j]=1
    return dist / max


N = len(readData())

A = N * N
D = N * 10
U0 = 0.0003
step = 0.0001
num_iter = 30000
K = num_iter // 50


# A = N*N
# D = N*15
# U0 = 0.0005
# step = 0.0001
# num_iter = 30000
# K=num_iter//50

def calc_du(V, dist):
    a = np.sum(V, axis=0) - 1  # 每列和
    b = np.sum(V, axis=1) - 1  # 每行和
    t1 = np.zeros((N, N))
    t2 = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            t1[i, j] = a[j]

    for i in range(N):
        for j in range(N):
            t2[i, j] = b[i]
    c1 = V[:, 1:N]
    c2 = np.zeros((N, 1))
    c2[:, 0] = V[:, 0]
    c = np.concatenate((c1, c2), axis=1)
    c = np.dot(dist, c)
    return -A * (t1 + t2) - D * c


def calc_U(U, du, step):
    return U + du * step


def calc_V(U, U0):
    return 1 / 2 * (1 + np.tanh(U / U0))


def calc_energy(V, dist):
    t1 = np.sum(np.power(np.sum(V, axis=0), 2))
    t2 = np.sum(np.power(np.sum(V, axis=1), 2))
    idx = [i for i in range(1, N)]
    idx = idx + [0]
    Vt = V[:, idx]
    t3 = dist * Vt
    t3 = np.sum(np.sum(np.multiply(V, t3)))
    return 1 / 2 * (A * (t1 + t2) + D * t3)


def check_path(V):
    newV = np.zeros((N, N))
    route = []
    for i in range(N):
        mm = np.max(V[:, i])
        for j in range(N):
            if V[j, i] == mm:
                newV[j, i] = 1
                route += [j]
                break
    return route, newV


def calc_dist(route, dist):
    length = 0.0
    for i in range(len(route) - 1):
        x = route[i]
        y = route[i + 1]
        length += dist[x, y]
    return length


def ANNTsp():
    global max
    global min
    global step
    global D

    tq = tqdm(total=100)
    value = num_iter // 100
    dist = readData()
    rand = np.random.rand(N, N) * (-2) + 1
    U = 1 / 2 * U0 * np.log2(N - 1) / np.log2(np.e) + rand * 0
    # print(U.shape)
    V = calc_V(U, U0)
    energy = [0.0 for _ in range(num_iter)]
    ans_length = 200000
    ans_path = []
    H_path = []
    for n in range(num_iter):
        if n % value == 0 and n > 0:
            tq.update(1)
        # if n>num_iter*0.7 and n%1000==0:
        #     D=D+2
        du = calc_du(V, dist)
        U = calc_U(U, du, step)
        V = calc_V(U, U0)
        energy[n] = calc_energy(V, dist)
        route, newV = check_path(V)
        if len(np.unique(route)) == N:
            H_path = []
            route.append(route[0])
            length = calc_dist(route, dist)
            if length < ans_length:
                ans_length = length
                ans_path[:] = route[:]
                [H_path.append((route[i], route[i + 1])) for i in range(len(route) - 1)]
        if n % K == 0 and n > 0:
            print('第%d次迭代最优距离为: %f,能量为: %f,路径为:' % (n, max * ans_length, energy[n]))
            print(ans_path)

    print('最佳路径距离为: %f' % (max * calc_dist(ans_path, dist)))
    print('最佳路径为: ')
    print(ans_path)
    draw(list, ans_path, energy)
    tq.close()


def draw(list, ans_path,energy):
    fig = plt.figure()

    # 绘制哈密顿回路

    ax1 = fig.add_subplot(121)

    plt.xlim(0, max_x)

    plt.ylim(0,max_y)

    for x in range(len(ans_path)-1):
        p1 = plt.Circle(list[ans_path[x]], 0.4, color='red')

        p2 = plt.Circle(list[ans_path[x+1]], 0.4, color='red')

        ax1.add_patch(p1)

        ax1.add_patch(p2)

        ax1.plot((list[ans_path[x]][0], list[ans_path[x+1]][0]), (list[ans_path[x]][1], list[ans_path[x+1]][1]), color='red')

        # ax1.annotate(s=chr(97 + to_), xy=list[to_], xytext=(-8, -4), textcoords='offset points', fontsize=20)

    ax1.axis('equal')

    ax1.grid()
    ax2=fig.add_subplot(122)
    ax2.plot(np.arange(0, len(energy), 1), energy, color='red')
    plt.xlabel('num_iter')
    plt.ylabel('value')
    plt.show()


if __name__ == '__main__':
    ANNTsp()
    # draw(ans_path,energy)
    # rand = np.random.rand(N,N)*(-2)+1

    # print(1 / 2 * U0 * np.loge(N - 1, np.e))
    # print(rand.shape)
