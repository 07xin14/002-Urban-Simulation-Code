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
    cdatamat = pd.read_csv("../result/III.2/production_constrained/cdatamat.csv",index_col=0)
    cdatamat2 = pd.read_csv("../result/IV.1/cdatamat2_2.csv",index_col=0)
    cdatamat3 = pd.read_csv("../result/IV.1/cdatamat3.csv",index_col=0)
    cdatamat4 = pd.read_csv("../result/IV.2-new/cdatamat4_4.csv",index_col=0)
    cdatamat5 = pd.read_csv("../result/IV.2-new/cdatamat5_5.csv",index_col=0)  
    station_origins = cdatamat.index[:-1]
    station_destinations = cdatamat.columns[:-1]
    
    ### plot the scatter for out flows ###
    fig, ax = plt.subplots(figsize=(10,10))
    xtick_labels = station_origins
    x  = list(range(0,len(station_origins)))
    for i in station_destinations:
        plt.scatter(x, cdatamat[i][:-1], marker="d",color='gray',alpha=0.5)
        plt.scatter(x, cdatamat3[i][:-1], marker="+",color='r')
        plt.scatter(x, cdatamat4[i][:-1], marker="o",color='b',alpha=0.6)
        plt.scatter(x, cdatamat5[i][:-1], marker="v",color='y',alpha=0.8)
        if i == station_destinations[-1]:
            plt.scatter(x, cdatamat[i][:-1], marker="d",color='gray',label='Original',alpha=0.5)
            plt.scatter(x, cdatamat3[i][:-1], marker="+",color='r',label='Scenarios A')
            plt.scatter(x, cdatamat4[i][:-1], marker="o",color='b',label='Scenarios B1',alpha=0.6)
            plt.scatter(x, cdatamat5[i][:-1], marker="v",color='y',label='Scenarios B2',alpha=0.8)

    # plt.scatter(x, y3, marker="+",color='r',label='Scenarios A')
    # plt.scatter(x, y4, marker="o",color='b',label='Scenarios B1',alpha=0.6)
    # plt.scatter(x, y5, marker="v",color='y',label='Scenarios B2',alpha=0.8)
    plt.xlabel('station_origin', fontsize = 15)
    plt.xticks(x[::100],xtick_labels[::100])#,rotation=20)
    plt.ylabel("ut flows", fontsize= 15)
    plt.title("ALL out flows from station_origin", fontsize = 20)
    plt.legend()
    ax.set_xlim(-10,410)
    sta = 'Canary Wharf'
    ii = list(xtick_labels).index(sta)
    plt.text(ii,max(cdatamat.loc[sta][:-1]),sta)
    plt.grid(True)
    fig.savefig(path_out+'ALL_out_flows_from_station_origin.png',dpi=500)

    ### plot the scatter for in flows ###
    fig, ax = plt.subplots(figsize=(10,10))
    xtick_labels = station_destinations
    x  = list(range(0,len(station_destinations)))
    for i in station_origins:
        plt.scatter(x, cdatamat.loc[i][:-1], marker="d",color='gray',alpha=0.5)
        plt.scatter(x, cdatamat3.loc[i][:-1], marker="+",color='r')
        plt.scatter(x, cdatamat4.loc[i][:-1], marker="o",color='b',alpha=0.6)
        plt.scatter(x, cdatamat5.loc[i][:-1], marker="v",color='y',alpha=0.8)
        if i == station_origins[-1]:
            plt.scatter(x, cdatamat.loc[i][:-1], marker="d",color='gray',label='Original',alpha=0.5)
            plt.scatter(x, cdatamat3.loc[i][:-1], marker="+",color='r',label='Scenarios A')
            plt.scatter(x, cdatamat4.loc[i][:-1], marker="o",color='b',label='Scenarios B1',alpha=0.6)
            plt.scatter(x, cdatamat5.loc[i][:-1], marker="v",color='y',label='Scenarios B2',alpha=0.8)

    plt.xlabel('station_destination', fontsize = 15)
    plt.xticks([x[0],x[99],x[199],x[299]],[xtick_labels[0],xtick_labels[99],xtick_labels[199],xtick_labels[299]])
    #plt.xticks(x[::100],xtick_labels[::100])#,rotation=20)
    plt.ylabel("in flows", fontsize= 15)
    plt.title("ALL in flows to station_destination", fontsize = 20)
    plt.legend()
    ax.set_xlim(-10,410)
    sta = 'Canary Wharf'
    ii = list(xtick_labels).index(sta)
    plt.text(ii,max(cdatamat[sta][:-1]),sta)
    plt.grid(True)
    fig.savefig(path_out+'ALL_in_flows_to_station_origin.png',dpi=500)
    
    cdata1 =  pd.read_csv("../result/IV.1/cdata_new2.csv",index_col=0)
    cdata = pd.read_csv("../result/IV.2-new/cdata_new5.csv",index_col=0)
    ### plot the hisgram of out flows ###
    fig, ax = plt.subplots(figsize=(10,10))   
    plt.hist(cdata['flows'], histtype="step" , bins = 50,color='gray',label='Original')
    plt.hist(cdata1['prosimfitted3'], histtype="step" , bins = 50,color='r',label='Scenarios A')
    plt.hist(cdata['prodsimest4_4'], histtype="step" , bins = 50,color='b',label='Scenarios B1')
    plt.hist(cdata['prodsimest5_5'], histtype="step" , bins = 50,color='y',label='Scenarios B2')
    plt.xlabel("flows", fontsize = 15)
    
    plt.ylabel("Count", fontsize= 15)
    plt.title("histogram of ALL flows", fontsize = 20)
    plt.legend()
    #plt.grid(True)
    fig.savefig(path_out+'ALL_histogram_of_total_flows.png',dpi=500)