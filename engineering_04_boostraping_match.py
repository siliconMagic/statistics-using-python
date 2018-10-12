#coding:utf-8
"""
------------------------------------------------
@File Name    : engineering_04_boostraping_match
@Function     : 
@Author       : Minux
@Date         : 2018/10/12
@Revised Date : 2018/10/12
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


def generate_random_grades():
    df_grades = pd.DataFrame()
    df_grades['grades'] = np.random.random_sample(100)*10
    df_grades.to_csv('grades.csv', index=False)

pop = pd.read_csv(r'./grades.csv')
def data_peek():
    print(pop.head())
    print(pop.describe())
    pop.grades.hist(histtype='step')
    plt.show()


def bootstrap_sample(stats_flag=True):
    bootstrap = pd.DataFrame({'sample_mean':[pop.sample(100, replace=True).grades.mean() for _ in range(10000)]})
    # print(bootstrap.head(10))
    if stats_flag:
        print('quantile(0.025) is {}'.format(bootstrap.sample_mean.quantile(0.025)))
        print('quantile(0.975) is {}'.format(bootstrap.sample_mean.quantile(0.975)))
    else:
        bootstrap.sample_mean.hist(histtype='step')
        plt.axvline(pop.grades.mean(), color='C1')
        plt.savefig('bootstrap_sample.png')
        plt.show()
n1 = scipy.stats.norm(7.5,1)
n2 = scipy.stats.norm(4,1)

def bimodal_distribution():
    x = np.linspace(0, 10, 100)
    plt.plot(x, 0.5*n1.pdf(x)+0.5*n2.pdf(x))
    plt.show()

def draw():
    while True:
        v = n1.rvs() if np.random.rand() < 0.5 else n2.rvs()
        if 0<=v<=10:
            return v

def data_set(n=100):
    return pd.DataFrame({'grade':[draw() for _ in range(n)]})

def plot_sample_distribution():
    mean = pd.DataFrame({'mean_grade':[data_set().grade.mean() for _ in range(1000)]})
    mean.mean_grade.hist(histtype='step')
    bootstrap = pd.DataFrame({'sample_mean': [pop.sample(100, replace=True).grades.mean() for _ in range(1000)]})
    bootstrap.sample_mean.hist(histtype='step')
    plt.show()

if __name__ == '__main__':
    # generate_random_grades()
    # data_peek()
    # bootstrap_sample()
    # bimodal_distribution()
    plot_sample_distribution()


