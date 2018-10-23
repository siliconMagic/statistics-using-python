#coding:utf-8
"""
------------------------------------------------
@File Name    : TEW_07_Anova_Fitting_Models
@Function     : 
@Author       : Minux
@Date         : 2018/10/23
@Revised Date : 2018/10/23
------------------------------------------------
"""
import math
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np

gap_minder = pd.read_csv('gapminder.csv')
g_data = gap_minder.query('year==1985')

size = g_data.population * 1e-6
colors = g_data.region.map({'Africa':'skyblue','Europe':'gold','America':'palegreen','Asia':'red'})

def plot_data():
    g_data.plot.scatter('age5_surviving','babies_per_woman',c=colors, s=size, linewidths=0.5, edgecolor='k', alpha=0.5)

model = smf.ols(formula='babies_per_woman ~ 1', data=g_data)
grand_mean = model.fit()

def plot_fit(fit_model):
    plot_data()
    plt.scatter(g_data.age5_surviving, fit_model.predict(g_data), c=colors, s=30, linewidths=0.5,
                edgecolors='k', marker='D')
    plt.show()

# plot_fit(grand_mean)
print(np.char.center('mean', 30, '-'))
'''mean'''
print(grand_mean.params)
print(g_data.babies_per_woman.mean())

print(np.char.center('group mean', 30, '-'))

'''group means'''
group_means = smf.ols(formula='babies_per_woman ~ -1+region', data=g_data).fit()
# plot_fit(group_means)

print(group_means.params)
print(g_data.groupby('region').babies_per_woman.mean())

print(np.char.center('surviving', 30, '-'))
surviving = smf.ols(formula='babies_per_woman ~ -1 + region + age5_surviving', data=g_data).fit()
# plot_fit(surviving)

'''add intersection term'''
surviving_by_region_population = smf.ols(formula='babies_per_woman ~ -1+region+age5_surviving:region'
                                      '-age5_surviving + population', data=g_data).fit()
# plot_fit(surviving_by_region)
print(surviving_by_region_population.params)

'''
Measure of Godness of Fit
Mean Squared Error of Residuals
R^2 = (Explained Variance)/(Total Variance)
F-statistics : explanatory power of fit parameters compared to random fit vectors
'''
print(np.char.center('Statistics_Indicator',30,'-'))
def statistics_indicator(*args):
    for arg in args:
        print(np.char.center(arg,30,'-'))
        if arg is 'resid':
            for model in [group_means, surviving, surviving_by_region_population]:
                print(model.mse_resid)
        elif arg is 'rsquared':
            for model in [group_means, surviving, surviving_by_region_population]:
                print(model.rsquared)
        elif arg is 'f_value':
            for model in [group_means, surviving, surviving_by_region_population]:
                print(model.fvalue)
        else:
            continue

statistics_indicator('resid','rsquared','f_value','xx')

print(surviving.summary())

print(sm.stats.anova_lm(group_means))












