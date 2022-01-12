#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 11:43:19 2022

@author: Emily

Performs RSI/GARD analysis on 2 datasets, one of which should be entered
by the user (default repeats analysis on the same dataset)
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter

# replace next 3 lines to load in new data;
# should have RSI, Time, and Event columns
df2 = pd.read_csv('NKI_HN.csv')
df2['Source'] = "other"
df2 = df2[:50]
# load original data
df = pd.read_csv('NKI_HN.csv')
combdf = pd.concat([df,df2]).reset_index(drop=True)

df = df.sort_values(by='GARD').reset_index(drop=True)


# RSI histogram
f1 = plt.figure(figsize=(7,5))
ax = f1.add_subplot(1,1,1)
sns.histplot(data=combdf, ax=ax, stat="count", 
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
sns.histplot(data=combdf, ax=ax2, stat="count", 
             x="GARD", kde=True,
             palette="deep", hue="Source",
             element="bars", legend=False)
ax2.set_title("GARD Distribution")
ax2.set_xlabel("GARD")
ax2.set_ylabel("Count")
f2.savefig('Figures/GARD_distribution')


# GARD/RT distributions, box
f3 = plt.figure(figsize=(7,5))
ax2 = f3.add_subplot(1,1,1)
sns.set_style('white')
sns.boxplot(data=combdf, ax=ax2, x ="GARD", y="Source", color='white', showfliers= False)
sns.stripplot(data=combdf, ax=ax2, x ="GARD", y="Source", palette="deep", hue="Source")
f3.savefig('Figures/Box_whisker')


# KM curve
km = KaplanMeierFitter()
km.fit(df['Time'],df['Event'])
km2 = KaplanMeierFitter()
km2.fit(df2['Time'],df2['Event'])
km3 = KaplanMeierFitter()
km3.fit(combdf['Time'],combdf['Event'])
plt.figure()
f4 = km.plot(color='black', linestyle='dashed', ci_show=False, label='NKI')
km2.plot(ax=f4, linestyle='dashed', ci_show=False, label='Other')
km3.plot(ax=f4, linestyle='dashed', ci_show=False, label='Combined')
plt.ylim([0,1])
plt.xlabel('Time (years)')
plt.title('Event-free Survival')
f4.savefig('Figures/KM')


# joint plot
f5 = plt.figure(figsize=(7,5))
sns.jointplot(data=combdf, x=combdf['RSI'], y=combdf['GARD'], hue=combdf['Source'])
plt.savefig('Figures/joint')


plt.show()

# print cox analysis
temp = df[['Event','Time','GARD']]
temp['Time'].replace(0,0.001,inplace=True)
model = CoxPHFitter()
model.fit(df=temp, duration_col='Time', event_col='Event')
cox = model.summary
print('NKI data, Cox model summary')
print(cox)

temp = df2[['Event','Time','GARD']]
model2 = CoxPHFitter()
model2.fit(df=temp, duration_col='Time', event_col='Event')
cox2 = model2.summary
print('Other data, Cox model summary')
print(cox2)

temp = pd.concat([cox,cox2]).reset_index(drop=True)


