#!/usr/bin/env python
# coding: utf-8

# ### CASA0002_Urban simulation
# # London tube network analysis
# ---
# 
# Elsa Arcaute, Carlos Molinero, Valentina Marin, Mateo Neira 
# 
# February 2022
# 

# This code will enable you to convert the tube network into a graph and then we will compute some measures of centrality.

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from operator import itemgetter
import copy
import numpy as np
import scipy.stats
from math import sqrt
#import folium
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.colors import ListedColormap  

if __name__ == '__main__':
    path_out = "../result/IV.3-new/"
    cdata = pd.read_csv("../result/III.2/production_constrained/cdata_new.csv")  
    cdatamat = pd.read_csv("../result/III.2/production_constrained/cdatamat.csv",index_col=0)
    cdatamat2 = pd.read_csv("../result/IV.1/cdatamat2_2.csv",index_col=0)
    cdatamat3 = pd.read_csv("../result/IV.1/cdatamat3.csv",index_col=0)
    cdatamat4 = pd.read_csv("../result/IV.2-new/cdatamat4_4.csv",index_col=0)
    cdatamat5 = pd.read_csv("../result/IV.2-new/cdatamat5_5.csv",index_col=0)
    
    new_data = pd.DataFrame(cdatamat['All'])
    new_data = new_data.merge(cdatamat2['All'], left_on="station_origin", right_on="station_origin", how = "left")
    new_data['difference_from_original2'] =  cdatamat2['All']-cdatamat['All'] 
    new_data.rename(columns={"All_x":"original_flows"}, inplace = True)
    new_data.rename(columns={"All_y":"prodsimest2_2_flows"}, inplace = True)

    new_data = new_data.merge(cdatamat3['All'], left_on="station_origin", right_on="station_origin", how = "left")
    new_data['difference_from_original3'] =  cdatamat3['All']-cdatamat['All']  
    new_data.rename(columns={"All":"prosimfitted3_flows"}, inplace = True)
    new_data = new_data.merge(cdatamat4['All'], left_on="station_origin", right_on="station_origin", how = "left")
    new_data.rename(columns={"All":"prodsimest4_4_flows"}, inplace = True)
    new_data['difference_from_original4'] =  cdatamat4['All']-cdatamat['All']  
    new_data = new_data.merge(cdatamat5['All'], left_on="station_origin", right_on="station_origin", how = "left")
    new_data.rename(columns={"All":"prodsimest5_5_flows"}, inplace = True)
    new_data['difference_from_original5'] =  cdatamat5['All']-cdatamat['All']  

    ### plot the difference from original flows
    fig, ax = plt.subplots(figsize=(10,10))
    x = new_data['original_flows'][:-1]
    y2 = new_data['difference_from_original2'][:-1] 
    y3 = new_data['difference_from_original3'][:-1]  
    y4 = new_data['difference_from_original4'][:-1]
    y5 = new_data['difference_from_original5'][:-1]
    #plt.scatter(x, y2, marker="d",color='g',label='Scenarios A1',alpha=0.5)
    plt.scatter(x, y3, marker="+",color='r',label='Scenarios A')
    plt.scatter(x, y4, marker="o",color='b',label='Scenarios B1',alpha=0.6)
    plt.scatter(x, y5, marker="v",color='y',label='Scenarios B2',alpha=0.8)
    yy = copy.deepcopy(x)
    yy = [0 for i in yy]
    plt.plot(x, yy, 'k-', color = 'k')
    plt.xlabel('original out flows from station_origin', fontsize = 15)
    plt.ylabel("difference from original out flows", fontsize= 15)
    plt.title("difference", fontsize = 20)
    plt.legend()
    ax.set_xlim(0,70000)
    ax.set_ylim(-30,20)
    plt.grid(True)
    fig.savefig(path_out+'differece_from_original_flows.png',dpi=1000)
    
    ### plot the hisgram of difference
    fig, ax = plt.subplots(figsize=(10,10))   
    #plt.hist(y2, histtype="step" , bins = 50,color='g',label='Scenarios A1')
    plt.hist(y3, histtype="step" , bins = 50,color='r',label='Scenarios A')
    plt.hist(y4, histtype="step" , bins = 50,color='b',label='Scenarios B1')
    plt.hist(y5, histtype="step" , bins = 50,color='y',label='Scenarios B2')
    plt.xlabel("difference from original out flows from station_origin", fontsize = 15)
    plt.ylabel("Count", fontsize= 15)
    plt.title("difference histogram", fontsize = 20)
    plt.legend()
    #plt.grid(True)
    fig.savefig(path_out+'differece_from_original_flows_hisgram.png',dpi=1000)