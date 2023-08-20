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
from matplotlib.ticker import FuncFormatter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import WeibullFitter
from lifelines import CoxPHFitter
from scipy.optimize import curve_fit
from matplotlib import collections as matcoll
from scipy import stats 

df = pd.read_csv('GARD_HPVpos8.csv')
temp = df.loc[df['RT']=='definitive']

# tertile GARD cutoffs
tert_one = 46
tert_two = 65.015

# constants/calculations
d = 2
beta = 0.05 
n = 1

ag = -np.log(df['RSI'])/(n*d)-beta*d
df['alpha_g'] = ag
gard = df['n']*df['d_c']*(ag+beta*df['d_c'])
df['GARD'] = gard
df['EQD2'] = df['TD'] * (df['d_c']+10) / (d+10) # alpha/beta = 10
# df['RxRSI'] = cut/(df['alpha_g']+beta*d)


# df.to_csv('/Users/Emily/Library/CloudStorage/Box-Box/CWRU/Research/MathOnc/opscc/formike.csv',index=False)


# =============================================================================
# # TD vs GARD boxplot
# a = list(df['TD']) + list(gard)
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
# # GARD vs EQD boxplot
# fig, (ax1, ax2) = plt.subplots(2, figsize=(7,5))
# sns.boxplot(x=df['EQD2'], ax=ax1, color='white', showfliers= False)
# sns.stripplot(x=df['EQD2'], ax=ax1, color=(0.2980392156862745, 0.4470588235294118, 0.6901960784313725))
# ax1.set(xlim=(0,120))
# sns.boxplot(x=df['GARD'], ax=ax2, color='white', showfliers= False)
# sns.stripplot(x=df['GARD'], ax=ax2, color=(0.8666666666666667, 0.5176470588235295, 0.3215686274509804))
# ax2.set(xlim=(0,120))
# ax1.set(xlabel="EQD2 (Gy)")
# ax2.set(xlabel="GARD")
# fig.tight_layout()
# 
# 
# # a = list(df['EQD2']) + list(df['GARD'])
# # b = list(['EQD2']*len(df)) + list(['GARD']*len(df))
# # df2 = np.transpose(pd.DataFrame(data=[a,b]))
# # df2 = df2.rename(columns={0: "value", 1: "type"})
# # f = plt.figure(figsize=(10,7))
# # ax1 = f.add_subplot(1,1,1)
# # ax1.set_xlim([0,110])
# # ax2 = ax1.twiny()
# # ax2.set_xlim([0,110])
# 
# # sns.set_style('white')
# # sns.boxplot(data=df2, ax=ax1, x="value", color='white', y='type', showfliers= False)
# # sns.stripplot(data=df2, ax=ax2, x="value", palette="deep", y="type")
# # plt.xlim([0,110])
# # plt.ylabel('')
# # ax1.set_xlabel('Gy')
# # ax2.set_xlabel('GARD')
# # f.tight_layout()
# # plt.show()
# =============================================================================


# =============================================================================
# # RSI histogram
# f2 = plt.figure(figsize=(7,5))
# ax2 = f2.add_subplot(1,1,1)
# sns.histplot(data=df, ax=ax2, stat="count", 
#              x="RSI", kde=True,
#              palette="deep", 
#              element="bars", legend=False)
# ax2.set_title("RSI Distribution")
# ax2.set_xlabel("RSI")
# ax2.set_ylabel("Count")
# # f2.savefig('Figures/GARD_distribution')
# 
# 
# # len(df[(df['RSI']<np.exp(-46.1/60*n*d)) & (df['RSI']>np.exp(-65/70*n*d))])/len(df)
# =============================================================================


# =============================================================================
# # # joint plot
# # 2 colors 
# df['color'] = np.where(((df['EQD2'] >= 69) & (df['EQD2'] <= 71)), True, False)
# f = sns.jointplot(data=df, x=df['EQD2'], y=df['GARD'], xlim=(30,80), ylim = (0,130), hue="color", legend=False)
# f.set_axis_labels('EQD2 (Gy)', 'GARD', fontsize=14)
# 
# # one color
# df["_"]=""
# f = sns.jointplot(data=df, x=df['EQD2'], y=df['GARD'], xlim=(30,80), ylim=(0,130), hue="_")
# f.set_axis_labels('EQD2 (Gy)', 'GARD', fontsize=14)
# 
# # GARD boxplot for EQD in 69-71
# df_filt = df[((df['EQD2'] >= 69) & (df['EQD2'] <= 71))]
# sns.set(rc={'figure.figsize':(1,7)})
# sns.set_style(style='white')
# ax1 = sns.boxplot(y=df_filt['GARD'], color='white', showfliers= False)
# sns.stripplot(y=df_filt['GARD'], ax=ax1, color=(0.8666666666666667, 0.5176470588235295, 0.3215686274509804))
# ax1.set(ylim=(0,130))
# ax1.set(ylabel="")
# ax1.set(title="EQD ~70 Gy \n")
# =============================================================================


# a = list(df['EQD2']) + list(df['GARD'])
# b = list(['EQD2']*len(df)) + list(['GARD']*len(df))
# df2 = np.transpose(pd.DataFrame(data=[a,b]))
# df2 = df2.rename(columns={0: "value", 1: "type"})
# f = plt.figure(figsize=(10,7))
# ax1 = f.add_subplot(1,1,1)
# ax1.set_xlim([0,110])
# ax2 = ax1.twiny()
# ax2.set_xlim([0,110])

# sns.set_style('white')
# sns.boxplot(data=df2, ax=ax1, x="value", color='white', y='type', showfliers= False)
# sns.stripplot(data=df2, ax=ax2, x="value", palette="deep", y="type")
# plt.xlim([0,110])
# plt.ylabel('')
# ax1.set_xlabel('Gy')
# ax2.set_xlabel('GARD')
# f.tight_layout()
# plt.show()

# =============================================================================
# # KDEs of GARD by TNM
# temp = df[(~df['TNM8'].isnull()) & (df['TNM8']!=0)]
# temp = temp.sort_values(by='TNM8').reset_index().drop(columns=['index'])
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



# not sure what this chunk is for
# =============================================================================
# # KDEs of GARD by TNM
# temp = df[(~df['TNM8'].isnull()) & (df['TNM8']!=0)]
# # fig = sns.kdeplot(data=temp, x='GARD', hue='TNM8', palette='Blues', fill=True, common_norm=False, alpha=.5, linewidth=0.1)
# # fig.set_yticklabels([])
# # fig.set_ylabel('')
# # plt.title('GARD distribution grouped by TNM stage')
# 
# # hist version
# fig=sns.histplot(data=temp, x='GARD', kde=False, hue="TNM8", multiple="layer", palette="deep", alpha=0.3) # 
# fig.set_yticklabels([])
# fig.set_ylabel('')
# # fig.set_xticklabels([1,2,3])
# plt.title('TNM distribution grouped by optimal GARD tertile')
# 
# one = df[df['TNM8']=='I']
# two = df[df['TNM8']=='II']
# three = df[df['TNM8']=='III']
# stats.ks_2samp(one['GARD'],two['GARD'])
# stats.ks_2samp(three['GARD'],two['GARD'])
# stats.ks_2samp(three['GARD'],one['GARD'])
# 
# =============================================================================


# fig=sns.boxplot(data=temp, x='GARD',y='pred_3grp_optimal', hue='TNM8') # 
# sns.histplot(data=temp, x="pred_3grp_optimal", stat='density', shrink=0.8, hue='TNM8',multiple="dodge")
# plt.title('TNM distribution grouped by GARD tertile')



# =============================================================================
# # KM plot stratified by TNM instead of GARD
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
# temp = pd.read_csv('os_tert_opt.csv')
# # temp = temp.replace('1_Low','GARD_low')
# # temp = temp.replace('2_Middle','GARD_medium')
# # temp = temp.replace('3_High','GARD_high')
# temp = temp.rename(columns={"pred_3grp_optimal": "GARD tertile"})
# x,y = 'GARD tertile','TNM8'
# 
# # (df1
# # .groupby(x)[y]
# # .value_counts(normalize=True)
# # .mul(100)
# # .rename('percent')
# # .reset_index()
# # .pipe((sns.catplot,'data'), x=x,y='percent',hue=y,kind='bar'))
# 
# # for adding percentages
# df1 = temp.groupby(x)[y].value_counts(normalize=True)
# df1 = df1.mul(100)
# df1 = df1.rename('Percent').reset_index()
# 
# sns.set(rc={'figure.figsize':(16,12)})
# sns.set(rc={"figure.dpi":150})
# sns.set_context(rc={"font.size":8,"axes.titlesize":14,"axes.labelsize":11})   
# sns.set_theme(style='white')
# 
# g = sns.catplot(x=x,y='Percent',hue=y,kind='bar',data=df1,aspect=1.3)
# g.ax.set_ylim(0,100)
# g.ax.set(xlabel=None)
# g.ax.set_xticklabels(['GARD_low','GARD_medium','GARD_high'])
# g.ax.set(title='TNM distribution grouped by GARD tertile')
# 
# for p in g.ax.patches:
#     txt = str(p.get_height().round(2)) + '%'
#     txt_x = p.get_x() 
#     txt_y = p.get_height()
#     g.ax.text(txt_x,txt_y,txt)
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


# =============================================================================
# results, a, b = KMbyGARD(temp['Time_OS'], temp['Event_OS'], temp['GARD'], cut)
# p = round(results.p_value,3)
# 
# plt.figure()
# f5 = a.plot(color='blue', ci_show=True, label='Above '+str(cut))
# b.plot(ax=f5, color='blue', linestyle='dashed', ci_show=True, label='Below '+str(cut))
# plt.title('KM comparison for GARD cut')
# label1 = 'p = '+str(p)+'\nGARD cut: '+str(cut)
# plt.text(0.1,0.1,label1)
# plt.ylim([0,1.1])
# plt.xlabel('Time (months)')
# plt.ylabel('Event-free Survival')
# # plt.savefig('Figures/KM_cut')
# =============================================================================



#  weibull fits for tertiles based on definitive
h = temp.loc[temp['GARD'] > tert_two]
s1 = WeibullFitter()
s1.fit(h['Time_OS'],h['Event_OS'])
m = temp.loc[(temp['GARD'] <= tert_two) & (temp['GARD'] > tert_one)]
s2 = WeibullFitter()
s2.fit(m['Time_OS'],m['Event_OS'])
l = temp.loc[temp['GARD'] <= tert_one]
s3 = WeibullFitter()
s3.fit(l['Time_OS'],l['Event_OS'])
# save fit parameters
s1_lambda = s1.lambda_ # high GARD
s1_rho = s1.rho_
s2_lambda = s2.lambda_ # med GARD
s2_rho = s2.rho_
s3_lambda = s3.lambda_ # low GARD
s3_rho = s3.rho_
# plot weibull fit
fig2 = s1.plot_survival_function(label='high GARD', ci_show=False)
s2.plot_survival_function(ax=fig2, label='medium GARD', ci_show=False)
s3.plot_survival_function(ax=fig2, label='low GARD', ci_show=False)

# evaluate surv_high fit at a value t
def s_high(t):
    
    s_high_lambda = s1.lambda_
    s_high_rho = s1.rho_

    return np.exp(-np.power(t/s_high_lambda, s_high_rho))

# evaluate surv_med fit at a value t
def s_med(t):
    
    s_med_lambda = s2.lambda_
    s_med_rho = s2.rho_

    return np.exp(-np.power(t/s_med_lambda, s_med_rho))

# evaluate surv_low fit at a value t
def s_low(t):
    
    s_low_lambda = s3.lambda_
    s_low_rho = s3.rho_

    return np.exp(-np.power(t/s_low_lambda, s_low_rho))


# for 2N patients, draw from RSI distribution
def rsi_sample(N, distr):

    # for some reason this was originally giving identical samples but seems fine now
    kde = stats.gaussian_kde(distr)
    patients = kde.resample(2*N).flatten()
    patients[patients<0] = 0.001
    # rsi_sample = np.random.normal(loc=0.4267245088495575, scale=0.11221246412456044, size=2*N)
    return patients
  
rt_low = 60.
rt_high = 70.
cut = 64.185   # df['GARD'].median() # 
tert_one = 46
tert_two = 65.015

rsi_low = np.exp(-n*d*cut/rt_low) # max RSI for de-escalation


# returns penalized survival curves for 2 treatment groups (control and boosted)
# calls s1, s2
def trial_three(temp, t, style):
    
    N = int(len(temp)/2)
    
    # survival curves
    surv_low = s_low(t)
    surv_med = s_med(t)
    surv_high = s_high(t)
    
    # count how many in each arm
    l1 = m1 = h1 = 0
    l2 = m2 = h2 = 0
    
    # count how many de-escalate
    deesc = 0
    
    # initialize settings
    temp['trt'] = '70 Gy'
    temp['GARD60'] = rt_low * (-np.log(temp['RSI'])/2)
    temp['GARD70'] = rt_high * (-np.log(temp['RSI'])/2)    
    
    grp1 = temp[:N].copy()
    grp1['GARD'] = grp1['GARD70']
    grp2 = temp[N:].copy()  # grp1[:]
    
    
    if style == 'random': # randomized trial

        grp2['trt'] = '60 Gy'
        # grp2['TD'] = low
        grp2['GARD'] = grp2['GARD60']
      
    for index, patient in grp1.iterrows():
        
        if patient['GARD'] >= tert_two: h1 += 1
        elif patient['GARD'] >= tert_one: m1 += 1
        else: l1 += 1
            
    for index, patient in grp2.iterrows():
        
        # de-escalate only in selected patients
        if style == 'sorted': 
            
            x = patient['GARD70']
            y = patient['GARD60']
            if (((x > tert_two) & (y < tert_two)) or ((x > tert_one) & (y < tert_one))):
                patient['GARD'] = x
            else: 
                patient['GARD'] = y
                deesc += 1
                
        if patient['GARD'] >= tert_two: h2 += 1
        elif patient['GARD'] >= tert_one: m2 += 1
        else: l2 += 1
            
    surv1 = (l1*surv_low + m1*surv_med + h1*surv_high)/N
    surv2 = (l2*surv_low + m2*surv_med + h2*surv_high)/N
    
    count1 = np.array([l1, m1, h1])
    count2 = np.array([l2, m2, h2])
    
    return surv1, surv2, count1, count2, deesc


# =============================================================================
# # GARD high or low trial (cut 64.2)
# # check what cut is before running
# N = 150
# rsi_distr = df['RSI']
# tmin = 0
# tmax = 72
# t = np.linspace(tmin, tmax) # time axis in months
# style = 'random'
# repeats = 20
# 
# curve1 = []
# curve2 = []
# var1 = []
# var2 = []
# for i in range(repeats):
#     
#     patients = pd.DataFrame(rsi_sample(N, rsi_distr), columns=['RSI'])
#     surv1, surv2 = trial_two(patients, t, style)
#     curve1.append(np.mean(surv1, axis=0))
#     curve2.append(np.mean(surv2, axis=0))
#     
#     
# os1 = np.mean(curve1, axis=0)
# lower1 = np.percentile(curve1, 2.5, axis=0)
# upper1 = np.percentile(curve1, 97.5, axis=0)
# os2 = np.mean(curve2, axis=0)
# lower2 = np.percentile(curve2, 2.5, axis=0)
# upper2 = np.percentile(curve2, 97.5, axis=0)
# 
# fig, ax = plt.subplots(figsize=(7,5),dpi=100) # 
# plt.fill_between(t, lower1, upper1, color='#d95f02', alpha=.3) 
# plt.fill_between(t, lower2, upper2, color='peachpuff', alpha=.3) 
# plt.plot(t, os1, color='coral', label='Standard dose (70Gy)')
# plt.plot(t, os2, color='peachpuff', label='De-intensified dose (60Gy)')
# plt.legend(loc='lower left')
# plt.xlabel('Months')
# plt.ylabel('Percent free of local recurrence')
# ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
# plt.ylim(0,1)
# plt.xlim(0,tmax)
# 
# print(N, repeats)
# print(lower1[-1], os1[-1], upper1[-1])
# print(lower2[-1], os2[-1], upper2[-1])
# print(lower1[18], os1[18], upper1[18])
# print(lower2[18], os2[18], upper2[18])
# =============================================================================


# trial with 3-curve fits
N = 200 # patients per arm
rsi_distr = df['RSI']
tmin = 0
tmax = 72
t = np.linspace(tmin, tmax, 37) # time axis in months
style = 'random'
repeats = 20

curve1 = []
curve2 = []
var1 = []
var2 = []
counts1 = []
counts2 = []
down = []
for i in range(repeats):
    
    patients = pd.DataFrame(rsi_sample(N, rsi_distr), columns=['RSI'])
    surv1, surv2, count1, count2, deesc = trial_three(patients, t, style)
    curve1.append(surv1) #np.mean(surv1, axis=0)
    curve2.append(surv2) #np.mean(surv2, axis=0)
    counts1.append(count1)
    counts2.append(count2)
    down.append(deesc)
    
os1 = np.mean(curve1, axis=0)
lower1 = np.percentile(curve1, 2.5, axis=0)
upper1 = np.percentile(curve1, 97.5, axis=0)
os2 = np.mean(curve2, axis=0)
lower2 = np.percentile(curve2, 2.5, axis=0)
upper2 = np.percentile(curve2, 97.5, axis=0)

fig, ax = plt.subplots(figsize=(7,5),dpi=100) # 
plt.fill_between(t, lower1, upper1, color='#d95f02', alpha=.3) 
plt.fill_between(t, lower2, upper2, color='peachpuff', alpha=.3) 
plt.plot(t, os1, color='coral', label='Standard dose (70Gy)')
plt.plot(t, os2, color='peachpuff', label='De-intensified dose (60Gy)')
plt.legend(loc='lower left')
plt.xlabel('Months')
plt.ylabel('Percent free of local recurrence')
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.ylim(0,1)
plt.xlim(0,60)

print(style,'de-escalation,',str(N),'patients per arm,',str(repeats),'repeats')

# print('2 years')
# print(lower1[12], os1[12], upper1[12])
# print(lower2[12], os2[12], upper2[12])

print('3 years')
print(lower1[18], os1[18], upper1[18])
print(lower2[18], os2[18], upper2[18])

# print('6 years')
# print(lower1[-1], os1[-1], upper1[-1])
# print(lower2[-1], os2[-1], upper2[-1])

print('Average # in l/m/h GARD, group 1:'+str(np.mean(counts1, axis=0)))
print('Average # in l/m/h GARD, group 2:'+str(np.mean(counts2, axis=0)))
print('# de-escalated in group 2:'+str(np.mean(deesc)))




# # print cox analysis
# temp = df[['Event_OS','Time_OS','GARD']]
# temp['Time_OS'].replace(0,0.001,inplace=True)
# model = CoxPHFitter()
# model.fit(df=temp, duration_col='Time_OS', event_col='Event_OS')
# # print('NKI data, Cox model summary')
# cox = model.summary
# print(cox)


