#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 10:42:42 2020

@author: yangzhang
"""

import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')










##Combined Feature Relations: more than two feature relations in a single graph
pal = {1:"seagreen", 0:"gray"}
g = sns.FacetGrid(df_train,size=5, col="Sex", row="Survived", margin_titles=True, hue = "Survived",
                  palette=pal)
g = g.map(plt.hist, "Age", edgecolor = 'white');
g.fig.suptitle("Survived by Sex and Age", size = 15)
plt.subplots_adjust(top=0.90)

g = sns.FacetGrid(df_train,size=5, col="Sex", row="Embarked", margin_titles=True, hue = "Survived",
                  palette = pal)
g = g.map(plt.hist, "Age", edgecolor = 'white').add_legend();
g.fig.suptitle("Survived by Sex and Age", size = 15)
plt.subplots_adjust(top=0.90)

g = sns.FacetGrid(df_train, size=5,hue="Survived", col ="Sex", margin_titles=True,
                palette=pal,)
g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
g.fig.suptitle("Survived by Sex, Fare and Age", size = 15)
plt.subplots_adjust(top=0.85)


##Correlation Matrix and Heatmap
pd.DataFrame(abs(df_train.corr()['Survived']).sort_values(ascending = False))

## heatmeap to see the correlation between features. 
# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(df_train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.subplots(figsize = (15,12))
sns.heatmap(df_train.corr(), 
            annot=True,
            mask = mask,
            cmap = 'RdBu_r', ## in order to reverse the bar replace "RdBu" with "RdBu_r"
            linewidths=.9, 
            linecolor='white',
            fmt='.2g',
            center = 0,
            square=True)
plt.title("Correlations Among Features", y = 1.03,fontsize = 20, pad = 40);