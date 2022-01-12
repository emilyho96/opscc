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


# replace next lines to load in new data
# should have RSI, n, d, Time, and Event columns
df2 = pd.read_csv('NKI_HN.csv')
df2['Source'] = "other"
df2 = df2[:50]
# load data
df = pd.read_csv('NKI_HN.csv')
combdf = pd.concat([df,df2]).reset_index(drop=True)

# calculate RxRSI
d = 2
beta = 0.05
n = 1

ag = -np.log(df['RSI'])/(n*d)-beta*d
df['alpha_g'] = ag
gard = df['n']*df['d']*(ag+beta*df['d'])
df['GARD'] = gard


df = df.sort_values(by='GARD').reset_index(drop=True)

# =============================================================================
# # RSI histogram
# f1 = plt.figure(figsize=(7,5))
# ax = f1.add_subplot(1,1,1)
# sns.histplot(data=combdf, ax=ax, stat="count", 
#              x="RSI", kde=True,
#              palette="deep", hue="Source",
#              element="bars", legend=False)
# ax.set_title("RSI Distribution")
# ax.set_xlabel("RSI")
# ax.set_ylabel("Count")
# # f1.savefig('Figures/RSI_distribution')
# 
# =============================================================================

# =============================================================================
# # GARD histogram
# f2 = plt.figure(figsize=(7,5))
# ax2 = f2.add_subplot(1,1,1)
# sns.histplot(data=combdf, ax=ax2, stat="count", 
#              x="GARD", kde=True,
#              palette="deep", hue="Source",
#              element="bars", legend=False)
# ax2.set_title("GARD Distribution")
# ax2.set_xlabel("GARD")
# ax2.set_ylabel("Count")
# # f2.savefig('Figures/GARD_distribution')
# =============================================================================


# =============================================================================
# # GARD/RT distributions, box
# f2 = plt.figure(figsize=(7,5))
# ax2 = f2.add_subplot(1,1,1)
# sns.set_style('white')
# sns.boxplot(data=combdf, ax=ax2, x ="GARD", y="Source", color='white', showfliers= False)
# sns.stripplot(data=combdf, ax=ax2, x ="GARD", y="Source", palette="deep", hue="Source")
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


# joint plot
fig = plt.figure(figsize=(7,5))
sns.jointplot(data=combdf, x=combdf['RSI'], y=combdf['GARD'], hue=combdf['Source'])
# sns.jointplot(data=combdf, x=combdf['TD'], y=combdf['GARD'], hue=combdf['Source'])


# =============================================================================
# # run comparison KM for dataset based on a GARD cut-point
# # returns log-rank stats
# def KMbyGARD(time, event, sort, cut, show = False):
#    
#     # throw error if cut-point is out of the range
#     if cut < sort.min() or cut > sort.max():
#         print("Cut-point out of range")
#         return
#    
#     temp = pd.DataFrame()
#     temp['Time'] = time
#     temp['Event'] = event
#     temp['sort'] = sort
#     temp = temp.sort_values(by='sort')
#     
#     above = temp.loc[temp['sort'] > cut]
#     below = temp.loc[temp['sort'] <= cut]
#     
#     km_above = KaplanMeierFitter()
#     km_above.fit(above['Time'],above['Event'],label='GARD > '+str(cut))
#     km_below = KaplanMeierFitter()
#     km_below.fit(below['Time'],below['Event'],label='GARD < '+str(cut))
#     
#     results = logrank_test(above['Time'],below['Time'],event_observed_A=above['Event'], event_observed_B=below['Event'])
#     
#     # optional plot
#     if show == True:
#         
#         a2 = km_above.plot(ci_show=False)
#         km_below.plot(ax=a2,ci_show=False)
#         
#     return results, km_above, km_below
# 
# 
# # iterates thtrough GARD to minimize p-value
# # returns a 1-smaller list of p-values, ordered by GARD
# def findCut(time, event, gard, show = False):
#     
#     p = []
#     
#     for cut_val in gard:
#         
#         if cut_val == gard.max():
#             break
#         
#         results, _, _ = KMbyGARD(time, event, gard, cut_val)
#         p.append(results.p_value)
#         
#     if show == True:
#         
#         a1 = sns.scatterplot(x=gard[:-1], y=p)
#         a1.set_yscale('log')
#         plt.title("p-value vs GARD cut-point")
#     
#     return p, gard[:-1].tolist()
# 
# 
# # finding gard_t (cut) for cohorts
# # first calcluate p for each cut
# p_vals, gard = findCut(df['Time'], df['Event'], df['GARD'])
# p_vals2, gard2 = findCut(df2['Time'], df2['Event'], df2['GARD'])
# 
# # =============================================================================
# # # use this section to fit a polynomial to estimate cut then manually enter value
# # coeff = np.polyfit(gard, p_vals, 6)
# # p = np.poly1d(coeff)
# # x = np.linspace(df['GARD'].min(), df['GARD'].max(), num=100)
# # y = p(x)
# # coeff2 = np.polyfit(gard2, p_vals2, 6)
# # p2 = np.poly1d(coeff2)
# # x2 = np.linspace(df2['GARD'].min(), df2['GARD'].max(), num=100)
# # y2 = p2(x)
# # f4 = plt.figure()
# # plt.plot(x,y)
# # plt.scatter(gard,p_vals)
# # plt.plot(x2,y2)
# # plt.scatter(gard2,p_vals2)
# # plt.xlabel('GARD cut-point')
# # plt.ylabel('p-value')
# # plt.ylim([-0.2,1.1])
# # f4.savefig('Figures/p_cut')
# # # cut = 
# # =============================================================================
# 
# # alternatively, simply use the cut with minimized p 
# cut = gard[p_vals.index(min(p_vals))]
# _, a, b = KMbyGARD(df['Time'], df['Event'], df['GARD'], cut)
# cut2 = gard2[p_vals2.index(min(p_vals2))]
# _, a2, b2 = KMbyGARD(df2['Time'], df2['Event'], df2['GARD'], cut2)
# plt.figure()
# f5 = a.plot(color='blue', ci_show=False, label='NKI above cut')
# b.plot(ax=f5, color='blue', linestyle='dashed', ci_show=False, label='NKI below cut')
# a2.plot(ax=f5, color='orange', ci_show=False, label='Other above cut')
# b2.plot(ax=f5, color='orange', linestyle='dashed', ci_show=False, label='Other below cut')
# plt.title('KM comparison for GARD cut')
# label1 = 'NKI: p ='+str(min(p_vals))+'\nGARD cut: '+str(cut)
# plt.text(0.1,0.3,label1)
# label2 = 'Other: p ='+str(min(p_vals2))+'\nGARD cut: '+str(cut2)
# plt.text(0.1,0.1,label2)
# plt.ylim([0,1])
# plt.xlabel('Time (years)')
# plt.ylabel('Event-free Survival')
# # plt.savefig('Figures/KM_cut')
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

  
# print cox analysis
temp = df[['Event','Time','GARD']]
temp['Time'].replace(0,0.001,inplace=True)
model = CoxPHFitter()
model.fit(df=temp, duration_col='Time', event_col='Event')
# print('NKI data, Cox model summary')
cox = model.summary
print(cox)

temp = df2[['Event','Time','GARD']]
model2 = CoxPHFitter()
model2.fit(df=temp, duration_col='Time', event_col='Event')
# print('Other data, Cox model summary')
cox2 = model2.summary
print(cox2)

temp = pd.concat([cox,cox2]).reset_index(drop=True)


