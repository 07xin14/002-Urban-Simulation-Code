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
    
    new_in_data = pd.DataFrame(cdatamat.loc['All'])
    new_in_data.rename(columns={"All":"original_in_flows"}, inplace = True)
    new_in_data = pd.concat([new_in_data,cdatamat2.loc['All']],axis=1)
    new_in_data.rename(columns={"All":"prodsimest2_2_in_flows"}, inplace = True)
    new_in_data = pd.concat([new_in_data,cdatamat3.loc['All']],axis=1)
    new_in_data.rename(columns={"All":"prosimfitted3_in_flows"}, inplace = True)
    new_in_data = pd.concat([new_in_data,cdatamat4.loc['All']],axis=1)
    new_in_data.rename(columns={"All":"prodsimest4_4_in_flows"}, inplace = True)
    new_in_data = pd.concat([new_in_data,cdatamat5.loc['All']],axis=1)
    new_in_data.rename(columns={"All":"prodsimest5_5_in_flows"}, inplace = True)
    new_in_data['station_destination']=new_in_data.index
    new_in_data.index = range(0,len(new_in_data.index))


    ### plot the difference from original flows
    fig, ax = plt.subplots(figsize=(10,10))
    x = new_in_data['original_in_flows'][:-1]
    y2 = new_in_data['prodsimest2_2_in_flows'][:-1] 
    y3 = new_in_data['prosimfitted3_in_flows'][:-1]  
    y4 = new_in_data['prodsimest4_4_in_flows'][:-1]
    y5 = new_in_data['prodsimest5_5_in_flows'][:-1]
    #plt.scatter(x, y2, marker="d",color='g',label='Scenarios A1',alpha=0.5)
    plt.scatter(x, y3-x, marker="+",color='r',label='Scenarios A')
    plt.scatter(x, y4-x, marker="o",color='b',label='Scenarios B1',alpha=0.6)
    plt.scatter(x, y5-x, marker="v",color='y',label='Scenarios B2',alpha=0.8)
    yy = copy.deepcopy(x)
    yy = [0 for i in yy]
    plt.plot(x, yy, 'k-', color = 'k')
    plt.xlabel('original in flows to station_destination', fontsize = 15)
    plt.ylabel("difference from original in flows", fontsize= 15)
    plt.title("difference from original in flows to station_destination", fontsize = 20)
    plt.legend()
    #ax.set_xlim(0,70000)
    #ax.set_ylim(-30,20)
    plt.grid(True)
    fig.savefig(path_out+'in_flows_differece_from_original_flows.png',dpi=1000)
    
    ### plot the hisgram of difference
    fig, ax = plt.subplots(figsize=(10,10))   
    #plt.hist(y2, histtype="step" , bins = 50,color='g',label='Scenarios A1')
    plt.hist(y3-x, histtype="step" , bins = 50,color='r',label='Scenarios A')
    plt.hist(y4-x, histtype="step" , bins = 50,color='b',label='Scenarios B1')
    plt.hist(y5-x, histtype="step" , bins = 50,color='y',label='Scenarios B2')
    plt.xlabel("difference from original in flows to station_destination", fontsize = 15)
    plt.ylabel("Count", fontsize= 15)
    plt.title("difference histogram", fontsize = 20)
    plt.legend()
    #plt.grid(True)
    fig.savefig(path_out+'in_flows_differece_from_original_flows_hisgram.png',dpi=1000)