#coding:utf-8
"""
------------------------------------------------
@File Name    : TEW_06_hypothesis_testing_p_value_CI
@Function     : 
@Author       : Minux
@Date         : 2018/10/16
@Revised Date : 2018/10/16
------------------------------------------------
"""
import math
import io
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import scipy.stats
import scipy.optimize
import scipy.spatial

cholera = pd.read_csv('cholera.csv') # 霍乱数据
pumps = pd.read_csv('pumps.csv')     # 水泵数据

def Plot_Cholera_func():
    fig = plt.figure(figsize=(10, 10))
    img = plt.imread('london.png')
    plt.imshow(img, extent=[-0.38, 0.38, -0.38, 0.38])
    plt.scatter(pumps.x, pumps.y, color='b')
    plt.scatter(cholera.x, cholera.y, color='r', s=3)
    plt.show()

def Data_stat_info():
    print(cholera.closest.value_counts())
    print('-'*10,'GroupBy_Closest','-'*10)
    print(cholera.groupby('closest').deaths.sum())

def simulate(n):
    return pd.DataFrame({'closest':np.random.choice([0,1,4,5], size=n, p=[0.65, 0.15, 0.10, 0.10])})

def sampling_function():
    sampling = pd.DataFrame({'counts':[simulate(489).closest.value_counts()[0] for _ in range(10000)]})
    # sampling.counts.hist(histtype='step')
    # plt.show()
    # 计算p-value
    # the smaller p-value the more strongly we can reject the null hypothesis
    p_value = 100.0 - scipy.stats.percentileofscore(sampling.counts, score=340)
    print(p_value)


if __name__ == '__main__':
    sampling_function()



