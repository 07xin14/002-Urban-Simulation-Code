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
    
    new_out_data = pd.DataFrame(cdatamat['All'])
    new_out_data = new_out_data.merge(cdatamat2['All'], left_on="station_origin", right_on="station_origin", how = "left")
    new_out_data.rename(columns={"All_x":"original_out_flows"}, inplace = True)
    new_out_data.rename(columns={"All_y":"prodsimest2_2_out_flows"}, inplace = True)

    new_out_data = new_out_data.merge(cdatamat3['All'], left_on="station_origin", right_on="station_origin", how = "left")
    new_out_data.rename(columns={"All":"prosimfitted3_out_flows"}, inplace = True)
    new_out_data = new_out_data.merge(cdatamat4['All'], left_on="station_origin", right_on="station_origin", how = "left")
    new_out_data.rename(columns={"All":"prodsimest4_4_out_flows"}, inplace = True) 
    new_out_data = new_out_data.merge(cdatamat5['All'], left_on="station_origin", right_on="station_origin", how = "left")
    new_out_data.rename(columns={"All":"prodsimest5_5_out_flows"}, inplace = True)
    new_out_data['station_origin']=new_out_data.index
    new_out_data.index = range(0,len(new_out_data.index))
    
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
    
    ### plot the scatter for out flows ###
    fig, ax = plt.subplots(figsize=(10,10))
    xtick_labels = new_out_data['station_origin'][:-1]
    x  = new_out_data.index[:-1]
    y1 = new_out_data['original_out_flows'][:-1]
    y2 = new_out_data['prodsimest2_2_out_flows'][:-1] 
    y3 = new_out_data['prosimfitted3_out_flows'][:-1]  
    y4 = new_out_data['prodsimest4_4_out_flows'][:-1]
    y5 = new_out_data['prodsimest5_5_out_flows'][:-1]
    plt.scatter(x, y1, marker="d",color='gray',label='Original',alpha=0.5)
    plt.scatter(x, y3, marker="+",color='r',label='Scenarios A')
    plt.scatter(x, y4, marker="o",color='b',label='Scenarios B1',alpha=0.6)
    plt.scatter(x, y5, marker="v",color='y',label='Scenarios B2',alpha=0.8)
    plt.xlabel('station_origin', fontsize = 15)
    plt.xticks(x[::100],xtick_labels[::100])#,rotation=20)
    plt.ylabel("total out flows", fontsize= 15)
    plt.title("total out flows from station_origin", fontsize = 20)
    plt.legend()
    sta = 'Canary Wharf'
    ii = xtick_labels[xtick_labels==sta].index.values.mean()
    plt.text(ii+2,y1[ii],sta)
    ax.set_xlim(0,400)
    #ax.set_ylim(-30,20)
    plt.grid(True)
    fig.savefig(path_out+'total_out_flows_from_station_origin.png',dpi=500)
    
    ### plot the hisgram of out flows ###
    fig, ax = plt.subplots(figsize=(10,10))   
    plt.hist(y1, histtype="step" , bins = 50,color='gray',label='Original')
    plt.hist(y3, histtype="step" , bins = 50,color='r',label='Scenarios A')
    plt.hist(y4, histtype="step" , bins = 50,color='b',label='Scenarios B1')
    plt.hist(y5, histtype="step" , bins = 50,color='y',label='Scenarios B2')
    plt.xlabel("total out flows", fontsize = 15)
    
    plt.ylabel("Count", fontsize= 15)
    plt.title("histogram of total out flows from station_origin", fontsize = 20)
    plt.legend()
    #plt.grid(True)
    fig.savefig(path_out+'histogram_of_total_out_flows_from_station_origin.png',dpi=500)
    
    ### plot the scatter for in flows ###
    fig, ax = plt.subplots(figsize=(10,10))
    xtick_labels = new_in_data['station_destination'][:-1]
    x  = new_in_data.index[:-1]
    y1 = new_in_data['original_in_flows'][:-1]
    y2 = new_in_data['prodsimest2_2_in_flows'][:-1] 
    y3 = new_in_data['prosimfitted3_in_flows'][:-1]  
    y4 = new_in_data['prodsimest4_4_in_flows'][:-1]
    y5 = new_in_data['prodsimest5_5_in_flows'][:-1]
    plt.scatter(x, y1, marker="d",color='gray',label='Original',alpha=0.5)
    plt.scatter(x, y3, marker="+",color='r',label='Scenarios A')
    plt.scatter(x, y4, marker="o",color='b',label='Scenarios B1',alpha=0.6)
    plt.scatter(x, y5, marker="v",color='y',label='Scenarios B2',alpha=0.8)
    plt.xlabel('station_destination', fontsize = 15)
    plt.xticks(x[::100],xtick_labels[::100])#,rotation=20)
    plt.ylabel("total in flows", fontsize= 15)
    plt.title("total in flows to station_destination", fontsize = 20)
    plt.legend()
    ax.set_xlim(0,400)
    #ax.set_ylim(-30,20)
    plt.grid(True)
    #plot the largest 5 in flows's station names#
    df = new_in_data.sort_values('original_in_flows',ascending=False,inplace=False)
    for i in df.index[1:6]:
        plt.text(i+2,df['original_in_flows'][i],df['station_destination'][i])
    fig.savefig(path_out+'total_in_flows_to_station_destination.png',dpi=500)
    
    ### plot the hisgram of out flows ###
    fig, ax = plt.subplots(figsize=(10,10))   
    plt.hist(y1, histtype="step" , bins = 50,color='gray',label='Original')
    plt.hist(y3, histtype="step" , bins = 50,color='r',label='Scenarios A')
    plt.hist(y4, histtype="step" , bins = 50,color='b',label='Scenarios B1')
    plt.hist(y5, histtype="step" , bins = 50,color='y',label='Scenarios B2')
    plt.xlabel("total in flows", fontsize = 15)
    
    plt.ylabel("Count", fontsize= 15)
    plt.title("histogram of total in flows to station_destination", fontsize = 20)
    plt.legend()
    #plt.grid(True)
    fig.savefig(path_out+'histogram_of_total_in_flows_to_station_destination.png',dpi=500)