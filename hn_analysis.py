#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:30:59 2021

@author: Emily
Performs RSI/GARD analysis on 2 datasets, one of which should be entered
by the user (default repeats analysis on the same dataset)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import WeibullFitter
from lifelines import CoxPHFitter
from scipy.optimize import curve_fit
from matplotlib import collections as matcoll
from scipy import stats 
# import rpy2.rinterface as rinterface


df = pd.read_csv('GARD_HPVpos3.csv')
df['Source'] = 'Loris'
df = df.sort_values(by='GARD').reset_index().drop(columns=['index'])
df2 = pd.read_csv('NKI_HN2.csv')
df2['Source'] = 'NKI'



# =============================================================================
# # TD vs GARD boxplot
# a = list(df['TD']) + list(df['GARD'])
# b = list(['TD']*len(df)) + list(['GARD']*len(df))
# df2 = np.transpose(pd.DataFrame(data=[a,b]))
# df2 = df2.rename(columns={0: "value", 1: "type"})
# f2 = plt.figure(figsize=(7,5))
# ax2 = f2.add_subplot(1,1,1)
# sns.set_style('white')
# sns.boxplot(data=df2, ax=ax2, x="value", color='white', y='type', showfliers= False)
# sns.stripplot(data=df2, ax=ax2, x="value", palette="deep", y="type")
# # sns.boxplot(data=df, ax=ax2, x ="GARD", color='white', showfliers= False)
# # sns.stripplot(data=df, ax=ax2, x ="GARD", palette="deep", hue="Source")
# plt.ylabel('')
# plt.show()
# =============================================================================


# =============================================================================
# # GARD histogram
# f2 = plt.figure(figsize=(7,5))
# ax2 = f2.add_subplot(1,1,1)
# sns.histplot(data=df, ax=ax2, stat="count", 
#              x="GARD", kde=True,
#              palette="deep", 
#              element="bars", legend=False)
# ax2.set_title("OPSCC GARD Distribution")
# ax2.set_xlabel("GARD")
# ax2.set_ylabel("Count")
# # f2.savefig('Figures/GARD_distribution')
# =============================================================================


# =============================================================================
# # comparing RSI distr from sources
# temp = pd.concat([df,df2]).reset_index(drop=True)
# test = stats.ks_2samp(temp[temp['Source']=='Loris']['RSI'],temp[temp['Source']=='NKI']['RSI'])
# 
# # hist version
# f3 = plt.figure(figsize=(7,5))
# ax3 = f3.add_subplot(1,1,1)
# sns.histplot(data=temp, ax=ax3, stat="count", 
#               x="RSI", kde=True,
#               palette="deep", hue="Source",
#               element="bars", legend=True)
# ax3.set_title("RSI distribution comparison")
# ax3.set_xlabel("RSI")
# ax3.set_ylabel("")
# plt.text(0.30,10,'KS p='+str(round(test[1],2)))
# # kde version
# fig = sns.kdeplot(data=temp, x='RSI', hue='Source', palette='Blues', fill=True, common_norm=False, alpha=.5, linewidth=0.1)
# fig.set_yticklabels([])
# fig.set_ylabel('')
# plt.title('RSI distribution comparison')
# =============================================================================


# comparing RSI distr from sources
temp = pd.concat([df,df2]).reset_index(drop=True)
test = stats.ks_2samp(temp[temp['Source']=='Loris']['GARD'],temp[temp['Source']=='NKI']['GARD'])

# # kde version
# fig = sns.kdeplot(data=temp, x='GARD', hue='Source', palette='Blues', fill=True, common_norm=False, alpha=.5, linewidth=0.1)
# fig.set_yticklabels([])
# fig.set_ylabel('')
# plt.title('GARD distribution comparison')
# hist version
f3 = plt.figure(figsize=(7,5))
ax3 = f3.add_subplot(1,1,1)
sns.histplot(data=temp, ax=ax3, stat="count", 
              x="GARD", kde=True,
              palette="deep", hue="Source",
              element="bars", legend=True)
plt.title("GARD distribution comparison")
ax3.set_ylabel("")
ax3.set_yticklabels([])
plt.text(100,10,'KS p='+str(round(test[1],2)))


# =============================================================================
# # joint plot
# f5 = plt.figure(figsize=(7,5))
# sns.jointplot(data=df, x=df['RSI'], y=df['GARD'])
# =============================================================================


# =============================================================================
# # KM curve
# km = KaplanMeierFitter()
# km.fit(df['Time'],df['Event'])
# km2 = KaplanMeierFitter()
# km2.fit(df2['Time'],df2['Event'])
# km3 = KaplanMeierFitter()
# km3.fit(combdf['Time'],combdf['Event'])
# plt.figure()
# f3 = km.plot(color='black', linestyle='dashed', ci_show=False, label='NKI')
# km2.plot(ax=f3, linestyle='dashed', ci_show=False, label='Other')
# km3.plot(ax=f3, linestyle='dashed', ci_show=False, label='Combined')
# plt.ylim([0,1])
# plt.xlabel('Time (years)')
# plt.title('Event-free Survival')
# # plt.savefig('Figures/KM')
# =============================================================================


# =============================================================================
# # KDEs of GARD by TNM
# temp = df[(~df['TNM8'].isnull()) & (df['TNM8']!=0)]
# fig = sns.kdeplot(data=temp, x='GARD', hue='TNM8', palette='Blues', fill=True, common_norm=False, alpha=.5, linewidth=0.1)
# fig.set_yticklabels([])
# fig.set_ylabel('')
# plt.title('GARD distribution grouped by TNM stage')
# 
# # hist version
# fig=sns.histplot(data=temp, x='GARD', kde=True, hue="TNM8", multiple="layer", palette="deep") # 
# fig.set_yticklabels([])
# fig.set_yticklabels([])
# fig.set_ylabel('')
# plt.title('GARD distribution grouped by TNM stage')
# 
# one = df[df['TNM8']=='I']
# two = df[df['TNM8']=='II']
# three = df[df['TNM8']=='III']
# stats.ks_2samp(one['GARD'],two['GARD'])
# stats.ks_2samp(three['GARD'],two['GARD'])
# stats.ks_2samp(three['GARD'],one['GARD'])
# =============================================================================


# run comparison KM for dataset based on a GARD cut-point
# returns log-rank stats
def KMbyGARD(time, event, sort, cut, show = False):
   
    # throw error if cut-point is out of the range
    if cut < sort.min() or cut > sort.max():
        print("Cut-point out of range")
        return
   
    temp = pd.DataFrame()
    temp['Time'] = time
    temp['Event'] = event
    temp['sort'] = sort
    temp = temp.sort_values(by='sort')
    
    above = temp.loc[temp['sort'] > cut]
    below = temp.loc[temp['sort'] <= cut]
    
    km_above = KaplanMeierFitter()
    km_above.fit(above['Time'],above['Event'],label='GARD > '+str(cut))
    km_below = KaplanMeierFitter()
    km_below.fit(below['Time'],below['Event'],label='GARD < '+str(cut))
    
    results = logrank_test(above['Time'],below['Time'],event_observed_A=above['Event'], event_observed_B=below['Event'])
    
    # optional plot
    if show == True:
        
        a2 = km_above.plot(ci_show=False)
        km_below.plot(ax=a2,ci_show=False)
        
    return results, km_above, km_below



# iterates thtrough GARD to minimize p-value
# returns a 1-smaller list of p-values, ordered by GARD
def findCut(time, event, gard, show = False):
    
    p = []
    
    for cut_val in gard:
        
        if cut_val == gard.max():
            break
        
        results, _, _ = KMbyGARD(time, event, gard, cut_val)
        p.append(results.p_value)
        
    if show == True:
        
        a1 = sns.scatterplot(x=gard[:-1], y=p)
        a1.set_yscale('log')
        plt.title("p-value vs GARD cut-point")
    
    return p, gard[:-1].tolist()


# =============================================================================
# # finding gard_t (cut) for cohorts
# # first calcluate p for each cut
# p_vals, gard = findCut(df['Time_OS'], df['Event_OS'], df['GARD'])
# 
# # use this section to fit a polynomial to estimate cut then manually enter value
# coeff = np.polyfit(gard, p_vals, 6)
# p = np.poly1d(coeff)
# x = np.linspace(df['GARD'].min(), df['GARD'].max(), num=100)
# y = p(x)
# f4 = plt.figure()
# plt.plot(x,y)
# plt.scatter(gard,p_vals)
# plt.xlabel('GARD cut-point')
# plt.ylabel('p-value')
# plt.ylim([-0.2,1.1])
# # f4.savefig('Figures/p_cut')
# # cut = 
# 
# =============================================================================


# =============================================================================
# cut = 65 # median
# results, a, b = KMbyGARD(df['Time_OS'], df['Event_OS'], df['GARD'], cut)
# p = round(results.p_value,3)
# 
# plt.figure()
# f5 = a.plot(color='blue', ci_show=True, label='Above median GARD')
# b.plot(ax=f5, color='blue', linestyle='dashed', ci_show=True, label='Below median GARD')
# plt.title('KM comparison for GARD cut')
# label1 = 'p = '+str(p)+'\nGARD cut: '+str(cut)
# plt.text(0.1,0.1,label1)
# plt.ylim([0,1.1])
# plt.xlabel('Time (months)')
# plt.ylabel('Event-free Survival')
# # plt.savefig('Figures/KM_cut')
# 
# =============================================================================


# =============================================================================
# # plot stratified by TNM instead of GARD
# temp = df[:]
# 
# tnmHigh = temp.loc[temp['TNM8'] == 'III']
# tnmLow = temp.loc[temp['TNM8'] != 'III']
# 
# above = KaplanMeierFitter()
# above.fit(tnmHigh['Time_OS'],tnmHigh['Event_OS'],label='Stage III')
# below = KaplanMeierFitter()
# below.fit(tnmLow['Time_OS'],tnmLow['Event_OS'],label='Stage I and II')
# 
# results = logrank_test(tnmHigh['Time_OS'],tnmLow['Time_OS'],event_observed_A=tnmHigh['Event_OS'], event_observed_B=tnmLow['Event_OS'])
# p = round(results.p_value, 3)
# 
# fig4 = above.plot(ci_show=True)
# below.plot(ax=fig4, ci_show=True)
# plt.title('KM stratified by TNM')
# label1 = 'p = '+str(p)
# plt.text(0.1,0.1,label1)
# plt.ylim([0,1.1])
# plt.xlabel('Time (months)')
# plt.ylabel('Event-free Survival')
# =============================================================================


# =============================================================================
# # this section is not necessary for analysis, so it only runs on original data
# # weibull fits for event above and below cut
# # overall event, not survival
# h = df.loc[df['GARD'] > cut]
# h['Time'].replace(0,0.001,inplace=True) # this gives a minor error but seems OK
# s1 = WeibullFitter()
# s1.fit(h['Time'],h['Event'])
# l = df.loc[df['GARD'] <= cut]
# l['Time'].replace(0,0.001,inplace=True)
# s2 = WeibullFitter()
# s2.fit(l['Time'],l['Event'])
# # save fit parameters
# s1_lambda = s1.lambda_ # adequate dose
# s1_rho = s1.rho_
# s2_lambda = s2.lambda_ # inadequate
# s2_rho = s2.rho_
# # plot weibull fit
# plt.figure()
# f6 = s1.plot_survival_function(label='above cut')
# s2.plot_survival_function(ax=f6, label='below cut')
# plt.ylim([0,1])
# plt.xlabel('Time (years)')
# plt.ylabel('Event-free Survival')
# label2 = 'GARD cut: '+str(cut)
# plt.text(0.1,0.2,label2)
# plt.title('KM fit comparison for GARD cut')
# # plt.savefig('Figures/Weibull_cut')
# 
# 
# # evaluate S1 fit at a value t
# # S1 is TD > GARD_T
# # rho<1 => event likelihood decreases w/time
# def s1(t):
# 
#     return np.exp(-np.power(t/s1_lambda, s1_rho))
# 
# # evaluate S2 fit at a value t
# # below GARD_T
# def s2(t):
# 
#     return np.exp(-np.power(t/s2_lambda, s2_rho))
# =============================================================================


# =============================================================================
# # RxRSI calc and sort for waterfall plot
# # replace df to get that specific plot
# df['RxRSI'] = cut/(df['alpha_g']+beta*d)
# temp = df.sort_values(by='RxRSI').reset_index().drop(columns=['index'])
# 
# # group relative to the SOC range - CHECK RANGE
# low = 66
# high = 70
# hlines = []
# llines = []
# for i in range(len(temp)):
#     y = temp['RxRSI'].loc[i] 
#     if y < low:
#         llines.append([(i+1,y),(i+1,low)])
#     if y > high:
#         hlines.append([(i+1,high),(i+1,y)])
# # percentages for legend
# lperc = round(100*len(llines)/len(temp))
# hperc = round(100*len(hlines)/len(temp))
# mperc = 100 - lperc - hperc
# # below here actually makes the plot
# hlinecoll = matcoll.LineCollection(hlines, colors='tomato')
# llinecoll = matcoll.LineCollection(llines, colors='royalblue')
# plt.figure()
# f7, ax = plt.subplots()
# ax.add_collection(hlinecoll)
# ax.add_collection(llinecoll)
# plt.scatter(np.linspace(1,len(temp),len(temp)),temp['RxRSI'],c=temp['RxRSI'],cmap='coolwarm')
# plt.axhline(y=low,color='gray')
# plt.axhline(y=high,color='gray')
# plt.xlim([0,len(temp)])
# plt.ylim([10,110])
# plt.xlabel('Patient ID')
# plt.ylabel('RxRSI (Gy) for GARD_T = '+str(cut))
# plt.title('RxRSI compared to standard-of-care range')
# plt.text(3, 80, str(lperc)+'% of patients require <'+str(low)+'Gy')
# plt.text(3, 88, str(mperc)+'% of patients receive RxRSI \n within SOC range')
# plt.text(3, 100, str(hperc)+'% of patients require >'+str(high)+'Gy')
# plt.show()
# # plt.savefig('Figures/RxRSI_waterfall')
# 
# 
# # RxRSI histogram, colored
# # replace df to get that specific plot
# # CAUTION the color cutoff depends on the bin arrangement
# # the PDF is also scaled manually
# plt.figure()
# f8, ax = plt.subplots()
# xmax = round(max(df['RxRSI']/20))*20 - 2
# xint = round(max(df['RxRSI']/20))*10
# x = np.linspace(0,xmax,xint)
# array = df['RxRSI']
# N, bins, patches = ax.hist(array,bins=x)
# bw = 1.2*array.std()*np.power(array.size,-1/5)
# kde = stats.gaussian_kde(array)
# scale = 350 # idk if this is the right scale but it's eyeballed CHECK
# curve = scale*kde(x)
# for i in range(0,int(low/2)):
#     patches[i].set_facecolor('royalblue')
# for i in range(int(low/2),int(high/2)):    
#     patches[i].set_facecolor('gray')
# for i in range(int(high/2),xint-1):
#     patches[i].set_facecolor('tomato')
# plt.xlabel("RxRSI")
# plt.title('RxRSI distribution')
# plt.plot(x, curve, linestyle="dashed", color='black')
# # plt.savefig('Figures/RxRSI_distribution')
# =============================================================================

  
# =============================================================================
# # print cox analysis
# temp = df[['Event','Time','GARD']]
# temp['Time'].replace(0,0.001,inplace=True)
# model = CoxPHFitter()
# model.fit(df=temp, duration_col='Time', event_col='Event')
# # print('NKI data, Cox model summary')
# cox = model.summary
# print(cox)
# 
# temp = df2[['Event','Time','GARD']]
# model2 = CoxPHFitter()
# model2.fit(df=temp, duration_col='Time', event_col='Event')
# # print('Other data, Cox model summary')
# cox2 = model2.summary
# print(cox2)
# 
# temp = pd.concat([cox,cox2]).reset_index(drop=True)
# =============================================================================


# =============================================================================
# # TNM AUC(t)
# temp = df[:]
# tmax = round(max(df['Time_OS']))
# times = []
# sens_tnm = []
# spec_tnm = []
# for i in range(10, tmax):
#    high_die = len(temp[(temp['TNM8'] == 'III') & (temp['Event_OS'] == 1) & (temp['Time_OS']<=i)]) 
#    high_live = len(temp[(temp['TNM8'] == 'III')]) - high_die
#    low_die = len(temp[(temp['TNM8'] != 'III') & (temp['Event_OS'] == 1) & (temp['Time_OS']<=i)]) 
#    low_live = len(temp[(temp['TNM8'] != 'III')]) - low_die
#    sens_tnm.append(high_die/(high_die + low_die))
#    spec_tnm.append(low_live/(low_live + high_live))
#    times.append(i)
# auc_tnm = np.array(sens_tnm)+np.array(spec_tnm)
# auc_tnm /= 2
# 
# plt.plot(times, auc_tnm)
# plt.ylabel('AUC')
# plt.xlabel('time (months)')
# plt.ylim([0,1])
# =============================================================================


