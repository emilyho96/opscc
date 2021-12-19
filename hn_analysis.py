#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:30:59 2021

@author: Emily
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import plot_lifetimes
from lifelines import WeibullFitter
from lifelines.datasets import load_waltons
from scipy.optimize import curve_fit
from matplotlib import collections as matcoll
from scipy import stats 
from scipy.special import ndtr
import scipy
import plotly.graph_objs as go
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


# load data; replace with new dataset or merge datasets as df
# should have RSI, n, d, Time, and Event columns
df = pd.read_csv('NKI_HN.csv')


# calculate RxRSI
d = 2
beta = 0.05
n = 1

ag = -np.log(df['RSI'])/(n*d)-beta*d
df['alpha_g'] = ag
gard = df['n']*df['d']*(ag+beta*df['d'])
df['GARD'] = gard


df = df.sort_values(by='GARD').reset_index(drop=True)
'''
# RSI histogram
f1 = plt.figure(figsize=(7,5))
ax = f1.add_subplot(1,1,1)
sns.histplot(data=df, ax=ax, stat="count", 
             x="RSI", kde=True,
             palette="deep", hue="Source",
             element="bars", legend=False)
ax.set_title("RSI Distribution")
ax.set_xlabel("RSI")
ax.set_ylabel("Count")
f1.savefig('Figures/RSI_distribution')


# GARD histogram
f2 = plt.figure(figsize=(7,5))
ax2 = f2.add_subplot(1,1,1)
sns.histplot(data=df, ax=ax2, stat="count", 
             x="GARD", kde=True,
             palette="deep", hue="Source",
             element="bars", legend=False)
ax2.set_title("GARD Distribution")
ax2.set_xlabel("GARD")
ax2.set_ylabel("Count")
f2.savefig('Figures/GARD_distribution')


# KM curve
km_event_all = KaplanMeierFitter()
km_event_all.fit(df['Time'],df['Event'])
plt.figure()
f3 = km_event_all.plot(color='black', ci_show=False)
plt.ylim([0,1])
plt.xlabel('Time (years)')
plt.title('Event-free Survival')
plt.savefig('Figures/KM')


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
    # print(results.p_value)
    
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


# finding gard_t (cut) for cohorts
# first calcluate p for each cut
p_vals, gard = findCut(df['Time'], df['Event'], df['GARD'])
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
# f4.savefig('Figures/p_cut')
# cut = 
# alternatively, simply use the cut with minimized p 
cut = gard[p_vals.index(min(p_vals))]
_, a, b = KMbyGARD(df['Time'], df['Event'], df['GARD'], cut)
plt.figure()
f5 = a.plot(color='blue', ci_show=False, label='above cut')
b.plot(ax=f5, color='blue', linestyle='dashed', ci_show=False, label='below cut')
plt.title('KM comparison for GARD cut')
label = 'p ='+str(min(p_vals))
plt.text(0.1,0.1,label)
plt.ylim([0,1])
plt.xlabel('Time (years)')
plt.ylabel('Event-free Survival')
plt.savefig('Figures/KM_cut')


# weibull fits for event above and below cut
# overall event, not survival

h = df.loc[df['GARD'] > cut]
h['Time'].replace(0,0.001,inplace=True) # this gives a minor error but seems OK
s1 = WeibullFitter()
s1.fit(h['Time'],h['Event'])
l = df.loc[df['GARD'] <= cut]
l['Time'].replace(0,0.001,inplace=True)
s2 = WeibullFitter()
s2.fit(l['Time'],l['Event'])
# save fit parameters
s1_lambda = s1.lambda_ # adequate dose
s1_rho = s1.rho_
s2_lambda = s2.lambda_ # inadequate
s2_rho = s2.rho_
# plot weibull fit
plt.figure()
f6 = s1.plot_survival_function(label='above cut')
s2.plot_survival_function(ax=f6, label='below cut')
plt.ylim([0,1])
plt.xlabel('Time (years)')
plt.ylabel('Event-free Survival')
plt.title('KM fit comparison for GARD cut')
plt.savefig('Figures/Weibull_cut')


# evaluate S1 fit at a value t
# S1 is TD > GARD_T
# rho<1 => event likelihood decreases w/time
def s1(t):

    return np.exp(-np.power(t/s1_lambda, s1_rho))

# evaluate S2 fit at a value t
# below GARD_T
def s2(t):

    return np.exp(-np.power(t/s2_lambda, s2_rho))
'''

# RxRSI calc and sort for waterfall plot
df['RxRSI'] = cut/(df['alpha_g']+beta*d)
df2 = df.sort_values(by='RxRSI').reset_index().drop(columns=['index'])

# group relative to the SOC range - WHAT TO USE HERE
low = 66
high = 70
hlines = []
llines = []
for i in range(len(df2)):
    y = df2['RxRSI'].loc[i] 
    if y < low:
        llines.append([(i+1,y),(i+1,low)])
    if y > high:
        hlines.append([(i+1,high),(i+1,y)])
# percentages for legend
lperc = round(100*len(llines)/len(df2))
hperc = round(100*len(hlines)/len(df2))
mperc = 100 - lperc - hperc
# below here actually makes the plot
hlinecoll = matcoll.LineCollection(hlines, colors='tomato')
llinecoll = matcoll.LineCollection(llines, colors='royalblue')
plt.figure()
f7, ax = plt.subplots()
ax.add_collection(hlinecoll)
ax.add_collection(llinecoll)
plt.scatter(np.linspace(1,len(df2),len(df2)),df2['RxRSI'],c=df2['RxRSI'],cmap='coolwarm')
plt.axhline(y=low,color='gray')
plt.axhline(y=high,color='gray')
plt.xlim([0,len(df2)])
plt.ylim([10,110])
plt.xlabel('Patient ID')
plt.ylabel('RxRSI (Gy) for GARD_T = '+str(cut))
plt.title('RxRSI compared to standard-of-care range')
plt.text(3, 80, str(lperc)+'% of patients require <'+str(low)+'Gy')
plt.text(3, 88, str(mperc)+'% of patients receive RxRSI \n within SOC range')
plt.text(3, 100, str(hperc)+'% of patients require >'+str(high)+'Gy')
plt.savefig('Figures/RxRSI_waterfall')
plt.show()

# RxRSI histogram, colored
# CAUTION the color cutoff depends on the bin arrangement
# the PDF is also scaled manually
plt.figure()
f8, ax = plt.subplots()
xmax = round(max(df['RxRSI']/20))*20 - 2
xint = round(max(df['RxRSI']/20))*10
x = np.linspace(0,xmax,xint)
array = df['RxRSI']
N, bins, patches = ax.hist(array,bins=x)
bw = 1.2*array.std()*np.power(array.size,-1/5)
kde = stats.gaussian_kde(array)
scale = 350 # idk if this is the right scale but it's eyeballed CHECK
curve = scale*kde(x)
for i in range(0,int(low/2)):
    patches[i].set_facecolor('royalblue')
for i in range(int(low/2),int(high/2)):    
    patches[i].set_facecolor('gray')
for i in range(int(high/2),xint-1):
    patches[i].set_facecolor('tomato')
plt.xlabel("RxRSI")
plt.title('RxRSI_distribution')
plt.plot(x, curve, linestyle="dashed", color='black')
plt.savefig('Figures/RxRSI_distribution')
    

'''
# plot 2a
# the different 'x' may get confusing here
pdf_scaled = kde(x)/max(kde(x))
plt.fill_between(x, y1=pdf_scaled, y2=0, alpha=0.3) #, label="PDF"
pdf = kde.evaluate(x)/sum(kde.evaluate(x))
cdf = np.cumsum(pdf)
plt.plot(x, cdf, label="TCP")
plt.axvline(x=low, color='gray', linestyle='dashed')
plt.axvline(x=high, color='gray', linestyle='dashed')
plt.xlabel("Dose (Gy)")
plt.ylabel("TCP")
# plt.legend()
plt.show()
'''

# NTCP calcs
# this fits to dosimetry data! need to rerun if new 
def ntcp(td, side, dosi):
        
    coeffL = np.polyfit(dosi['Total Dose'], dosi['MHD_L'], 1) # force y-int 0?
    coeffR = np.polyfit(dosi['Total Dose'], dosi['MHD_R'], 1)
    coeffLung = np.polyfit(dosi['Total Dose'], dosi['MLD'], 1)
    mhdL = np.poly1d(coeffL)
    mhdR = np.poly1d(coeffR)
    mld = np.poly1d(coeffLung)
    
    # constants
    b0 = -3.87 # from QUANTEC lung
    b1 = 0.126 # from QUANTEC lung
    
    card_base = 0. # what should this baseline be? 0?
    card_slope = 0.074 
    
    # ntcp from OAR dose
    ntcp_cardL = card_base + card_slope * (mhdL(td) - mhdL(0))
    ntcp_cardR = card_base + card_slope * (mhdR(td) - mhdR(0))
    ntcp_pulm = np.exp(b0+b1*mld(td))/(1+np.exp(b0+b1*mld(td))) - np.exp(b0)/(1+np.exp(b0))    

    if side == 'L': 
        return ntcp_cardL + ntcp_pulm
    
    if side == 'R': 
        return ntcp_cardR + ntcp_pulm
    
    if side == 'plot':
        return ntcp_cardL, ntcp_cardR, ntcp_pulm
    
# fig 2b
'''td = np.linspace(0,90,91)
dosi = pd.read_csv('/Users/Emily/tnbc/dosi_summ.csv')
ntcp_cardL, ntcp_cardR, ntcp_pulm = ntcp(td, 'plot', dosi)
fig, ax = plt.subplots()
plt.plot(td, ntcp_cardL, label="Major Cardiac Event (L)")
plt.plot(td, ntcp_cardR, label="Major Cardiac Event (R)")
plt.plot(td, ntcp_pulm, label="Pneumonitis")
plt.axvline(x=low, color='gray', linestyle='dashed')
plt.axvline(x=high, color='gray', linestyle='dashed')
plt.xlabel("Dose (Gy)")
plt.ylabel("NTCP")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.legend()'''

# fig 2c; need to run code for other parts of plot 2 first
'''fig = plt.plot()
x = np.linspace(0,115,116)
scale = 38 # idk if this is the right scale but it's eyeballed

kde = stats.gaussian_kde(df['RxRSI'])
curve1 = kde(x) # idk if this is the right scale but it's eyeballed
plt.fill_between(x, y1=scale*curve1, y2=0, color='tab:blue', alpha=0.3) 
pdf1 = kde.evaluate(x)/sum(kde.evaluate(x))
cdf1 = np.cumsum(pdf1)
plt.plot(x, cdf1, color='tab:blue', label="TCP")

cdf2 = cdf1 - ntcp_card - ntcp_card
plt.plot(x, cdf2, color='tab:orange', label="Adjusted TCP")
pdf2 = np.diff(cdf2) #THIS ASSUMES Y-VALUES ARE 1 APART
plt.fill_between(x[1:], y1=scale*pdf2, y2=0, color='tab:orange', alpha=0.3)

plt.axvline(x=low, color='gray', linestyle='dashed')
plt.axvline(x=high, color='gray', linestyle='dashed')
plt.xlabel("Dose (Gy)")
plt.ylabel("TCP")
plt.legend()
plt.show()'''


# supp 2 Gaussian mixture modeling to compare modes of different cohorts?
# copied from Geoff's code
# unclear whaat to do with this rn
'''def construct_kde(array, bandwidth=None):
    
    if bandwidth == None:
        bw = 1.2*array.std()*np.power(array.size,-1/5) # this appears to be from wiki for bimodal KDE
    else:
        bw = bandwidth
    kde = KernelDensity(kernel='gaussian', bandwidth=bw)
    kde.fit(array.reshape(-1,1))
    x = np.linspace(array.min(),array.max(),200)
    log_dens=kde.score_samples(x.reshape(-1,1))
    kdens=np.exp(log_dens)

    total_dens=np.sum(kdens)
    cdf_array=np.zeros(shape=len(x))
    delta=x[1]-x[0]
    
    for i in range(len(x)):
        cdf_array[i] = np.sum(kdens[:i])*delta

    return x, kdens, cdf_array

nki_kde = construct_kde(nki['RSI'].to_numpy())
plt.scatter(x=nki_kde[0],y=nki_kde[1])

mcc_kde = construct_kde(mcc['RSI'].to_numpy())
plt.scatter(x=mcc_kde[0],y=mcc_kde[1])

tcc_kde = construct_kde(tcc['RSI'].to_numpy())
plt.scatter(x=tcc_kde[0],y=tcc_kde[1])
'''

'''
# time to simulate boost/no boost
N = 80
rsi_distr = tcc['RSI']
low = 50
high = 66
rsi_l = np.exp(-n*d*cut/low) 
rsi_h = np.exp(-n*d*cut/high) 
t = np.linspace(0,10) # time axis in years
dosi = pd.read_csv('/Users/Emily/tnbc/dosi_summ.csv') #dosimetey data for fits

# for 2N patients, draw from RSI distribution
def rsi_sample(N, distr):

    # for some reason this was originally giving identical samples but seems fine now
    kde = stats.gaussian_kde(distr)
    patients = kde.resample(2*N).flatten()
    patients[patients<0] = 0.001
    # rsi_sample = np.random.normal(loc=0.4267245088495575, scale=0.11221246412456044, size=2*N)
    return patients

# returns dataframe
# calls ntcp, rsi_sample
def trial(N, distr, t, style):
    
    # RSI sample
    temp = pd.DataFrame(rsi_sample(N, distr), columns=['RSI'])
    # calculate GARD, RxRSI 
    # assumes 2Gy dose
    # temp['GARD'] = -temp['TD']/(n*d)*np.log(temp['RSI']) ACTUALLY THIS LINE WON'T WORK AFTER MOVING THIS CHUNK
    temp['RxRSI'] = -n*d*cut/np.log(temp['RSI'])
    # assign sides
    temp['side'] = list('LR'*N)
        
    
    if style == 'random': # randomized trial

        temp['trt'] = 'no boost'
        temp.loc[N:,'trt'] = 'boost'
        
        temp['TD'] = low
        temp.loc[N:,'TD'] = high
        
    if style == 'sorted': # for boost grp, only RSI in middle range get boost

        temp['trt'] = 'no boost'
        temp.loc[N:,'trt'] = 'boost'
        
        # temp = temp.sort_values(by='RSI').reset_index().drop(columns=['index'])
    
        temp['TD'] = low
        temp.loc[(temp['trt']=='boost') & (temp['RSI']>rsi_l) & (temp['RSI']<rsi_h),'TD'] = high # might be problematic but seems to work
        
    # judging by results this may be glitching
    if style == 'custom': # for boost grp, TD = RxRSI within range

        temp['trt'] = 'no boost'
        temp.loc[N:,'trt'] = 'boost'
        
        temp['TD'] = low
        # THESE MIN/MAX VALUES SHOULD BE CHECKD
        temp.loc[N:,'TD'] = list(temp.loc[N:,'RxRSI'].clip(45, 80))
        
        
    # calc NTCP based on dose and side
    tox = []
    for index, patient in temp.iterrows():   
        
        tox.append(ntcp(patient['TD'],patient['side'],dosi))   
        
    temp['NTCP'] = tox   
    
    # get survival curve for each patient
    surv = []
    # noboost_count = temp[temp['TD']==low].count()
    # boost_count = temp[temp['TD']==high].count()
    for index, patient in temp.iterrows():
        
        # select based on whether or not RxRSI is met
        if patient['TD']>=patient['RxRSI']: 
            
            curve = s1(t)
        else: 
            
            curve = s2(t)
            
        # adjust for tox
        surv.append((1-patient['NTCP'])*curve)
        # surv.append(curve)
        
    temp['surv'] = surv # unnecessary line
    
    return temp # boost_surv, noboost_surv


N = 2700
style = 'random'
results = trial(N, rsi_distr, t, style)
    
# average survival for each group
noboost_surv = np.mean(results.loc[results['trt']=='no boost']['surv'], axis=0)
noboost_err = np.std(list(results.loc[results['trt']=='no boost']['surv']), axis=0)
boost_surv = np.mean(results.loc[results['trt']=='boost']['surv'], axis=0)
boost_err = np.std(list(results.loc[results['trt']=='boost']['surv']), axis=0)

fig, ax = plt.subplots()
# should this be 2stdev?
plt.fill_between(t, boost_surv-boost_err, boost_surv+boost_err, alpha=.3) 
plt.fill_between(t, noboost_surv-noboost_err, noboost_surv+noboost_err, alpha=.3) 
plt.plot(t, boost_surv, label='boost')
plt.plot(t, noboost_surv, label='no boost')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Percent event-free')
plt.title(style+' survival comparison, n='+str(2*N))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 
plt.ylim(0,1)

'''






