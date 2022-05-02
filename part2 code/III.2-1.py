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
import seaborn as sns
from matplotlib.colors import ListedColormap

def CalcRSqaured(observed, estimated):
    """Calculate the r^2 from a series of observed and estimated target values
    inputs:
    Observed: Series of actual observed values
    estimated: Series of predicted values"""
    
    r, p = scipy.stats.pearsonr(observed, estimated)
    R2 = r **2
    
    return R2

def CalcRMSE(observed, estimated):
    """Calculate Root Mean Square Error between a series of observed and estimated values
    inputs:
    Observed: Series of actual observed values
    estimated: Series of predicted values"""
    
    res = (observed -estimated)**2
    RMSE = round(sqrt(res.mean()), 3)
    
    return RMSE

def GetRangeId(dd,range_list):
    """give the number dd, get the id which block it belongs to
    """
    
    for i,block in enumerate(range_list[:-1]):
        if (dd>=range_list[i] and dd<range_list[i+1]):
            dd_id = i
    if dd>=range_list[-1]:
        dd_id = len(range_list)-1
    return dd_id
            

if __name__ == '__main__':
    path_out = "../result/III.2/original/"
    # ## 1. Constructing the networks
    
    # We are going to use the tube network file called "london_tubenetwork.graphml". This file has everything we need to construct the graph. A __graphml__ is a format that describes the structural properties of a graph. 
    
    #OK, let us start with the graphml file for London's underground
    #since coords tuples are stored as string, need to convert them back to tuples using eval()
    G = nx.read_graphml('../data/london.graph.xml')
    for node in G.nodes():
        G.nodes[node]['coords'] = eval(G.nodes[node]['coords'])
    for edge in G.edges():
        G.edges[edge]['flows'] = eval(G.edges[edge]['flows'])
    
    #print(nx.info(G))
    
    #read in your London Commuting Data
    cdata = pd.read_csv("../data/london_flows.csv")
    g_dad = nx.from_pandas_edgelist(cdata, 'station_origin', 'station_destination','flows')
    
    ### plot the scatter figure for flows with distance, population or jobs ###
    for ii in ['distance','population','jobs']:
        y = cdata[ii]
        x = cdata["flows"]
        
        #create the subplot
        fig, ax = plt.subplots(figsize = (10,10))
        #plot the results along with the line of best fit
        plt.scatter(x, y, marker="+")
        ax.set_xlim(0,20000)
        ax.set_xlabel("flows", fontsize = 20)
        ax.set_ylabel(ii, fontsize = 20)
        fig.savefig(path_out+f'london_underground_flows-{ii}.png',dpi=1000)

    ### plot the disgram of flows
    fig, ax = plt.subplots(figsize=(10,10))
    cdata_flows = cdata["flows"]    
    plt.hist(cdata_flows, histtype="stepfilled" , bins = 50)
    plt.xlabel("Flows", fontsize = 15)
    plt.ylabel("Count", fontsize= 15)
    plt.title("London underground flows  histogram", fontsize = 20)
    plt.grid(True)
    fig.savefig(path_out+'london_underground_flows_hisgram.png',dpi=1000)
    
    # cc = sorted(g_dad0.edges(data=True),reverse = False,key=lambda edge:edge[2]['flows'])
    # g_dad = nx.MultiGraph()
    # for i in cc:
    #     #g_dad.add_weight_edges_from(i[0],i[1],i[2]['flows'])
    #     g_dad.add_edges_from([i])
    # #print(nx.info(g_dad))
    
    # remove the nodes in G which are not in g_dad
    G_copy = copy.deepcopy(G)
    for i  in G.nodes():
        if i not in g_dad.nodes():
            #print(i)
            G_copy.remove_node(i)
    #print(nx.info(G_copy))
    
    ### Let's plot the tube network! ###
    
    # We can plot the tube network with the names of the stations as labels
    fig, ax = plt.subplots(figsize=(25,20))
    
    node_labels = dict([(i,i) for i in G_copy.nodes()])
    
    pos = nx.get_node_attributes(G_copy, 'coords')
    
    nx.draw_networkx_nodes(G_copy,pos,node_size=50,node_color='b')
    nx.draw_networkx_edges(G_copy,pos,arrows=False,width=0.2)
    nx.draw_networkx_labels(G_copy,pos, node_labels, font_size=10, font_color='black')
    
    plt.title("London tube network",fontsize=40)
    plt.axis("off")
    plt.show()
    fig.savefig(path_out+'network_with_names.png',dpi=1000)
    
    fig, ax = plt.subplots(figsize=(25,20))
    nx.draw_networkx_nodes(G_copy,pos,node_size=50,node_color='b')
    nx.draw_networkx_edges(G_copy,pos,arrows=False,width=0.2)
    
    plt.title("London tube network",fontsize=40)
    plt.axis("off")
    plt.show()
    fig.savefig(path_out+'network_without_names.png',dpi=1000)
    
    # ####plot the flow lines####
    # betweenness_e_values=[(i[2]['flows']) for i in g_dad.edges(data=True)]
    # #bet_e_color=[(i[2]['flows']-min(betweenness_e_values))/(max(betweenness_e_values)-min(betweenness_e_values)) for i in g_dad.edges(data=True)]
    # #bet_e_width=[(i[2]['flows']-min(betweenness_e_values))/(max(betweenness_e_values)-min(betweenness_e_values)) for i in g_dad.edges(data=True)]
    # bet_e_color=[(i[2]['flows']/max(betweenness_e_values)) for i in g_dad.edges(data=True)]
    # bet_e_width=[(i[2]['flows']/max(betweenness_e_values)*10) for i in g_dad.edges(data=True)]
    
    # # Plot graph
    # fig, ax = plt.subplots(figsize=(12,12))
    
    # #pos=nx.spring_layout(X)
    # edg=nx.draw_networkx_edges(g_dad, pos,edge_color=bet_e_color,width=bet_e_width
    #                            ,edge_cmap=plt.get_cmap('rainbow'))
    
    # nx.draw_networkx_nodes(g_dad,
    #         pos = pos,
    #         node_color= 'black',
    #         node_size= 1)
     
    # cb = plt.colorbar(edg,label="Flows",orientation="horizontal", shrink=0.5)
    # plt.axis("off")
    # plt.title("London underground Flows",fontsize=30)
    # cb.ax.tick_params(labelsize=20)
    # font = {'color' :'black','size':25}
    # cb.set_label("Flows",fontdict=font)
    # plt.show()
    # fig.savefig(path_out+'london_underground_flows.png',dpi=1000)
    
    ####plot the flow lines type 2####
    betweenness_e_values=[(i[2]['flows']) for i in g_dad.edges(data=True)]
    #bet_e_color=[(i[2]['flows']-min(betweenness_e_values))/(max(betweenness_e_values)-min(betweenness_e_values)) for i in g_dad.edges(data=True)]
    #bet_e_width=[(i[2]['flows']-min(betweenness_e_values))/(max(betweenness_e_values)-min(betweenness_e_values)) for i in g_dad.edges(data=True)]
    range_list = [0,50,100,150,200,500,1000,2000,3000,5000,8000,10000,20000]
    colors = ['skyblue','royalblue','lightgreen','seagreen','orange','darkgoldenrod',
              'yellow','lightpink','red','darkred','darkviolet','purple']
    cmap = ListedColormap(colors)
    bet_e_color=[GetRangeId(i[2]['flows'],range_list) for i in g_dad.edges(data=True)]
    bet_e_width=[(i[2]['flows']/max(betweenness_e_values)*10) for i in g_dad.edges(data=True)]
    
    # Plot graph
    fig, ax = plt.subplots(figsize=(12,12))
    
    #pos=nx.spring_layout(X)
    edg=nx.draw_networkx_edges(g_dad, pos,edge_color=bet_e_color,width=bet_e_width
                               ,edge_cmap=cmap,edge_vmin=0,edge_vmax=12)#plt.get_cmap('Paired'))
    
    nx.draw_networkx_nodes(g_dad,
            pos = pos,
            node_color= 'black',
            node_size= 1)
     
    cb = plt.colorbar(edg,label="Flows",orientation="horizontal")#, shrink=0.5)
    plt.axis("off")
    plt.title("London underground Flows",fontsize=30)
    cb.ax.tick_params(labelsize=20,rotation = 30)
    font = {'color' :'black','size':25}
    cb.set_label("Flows",fontdict=font)
    cb.set_ticks(np.arange(0,13))
    cb.set_ticklabels([str(i) for i in range_list])
    #cb.set_ticks(np.array(range_list))
    #cb.ax.locator_params(nbins=len(range_list))
    plt.show()
    fig.savefig(path_out+'london_underground_flows_type2.png',dpi=1000)
    
    
    ### plot the disgram of flows for flows>0
    fig, ax = plt.subplots(figsize=(10,10))
    
    #remove all 0 values (logarithms can't deal with 0 values)
    cdata_flows = cdata["flows"]
    cdata_flows = cdata_flows[(cdata_flows!=0)]
    plt.hist(np.log(cdata_flows), histtype="stepfilled" , bins = 50)
    plt.xlabel("log(Flows)", fontsize = 15)
    plt.ylabel("Count", fontsize= 15)
    plt.title("London underground flows  histogram", fontsize = 20)
    plt.grid(True)
    fig.savefig(path_out+'london_underground_log_flows_hisgram.png',dpi=1000)

    
    ### plot the log(dist) & log(flows) ###
    #extract the x and y converting to log
    cdata_flows = cdata[["distance", "flows"]]
    #remove all 0 values (logarithms can't deal with 0 values)
    cdata_flows = cdata_flows[(cdata_flows!=0).all(1)]
    
    x = np.log(cdata_flows["flows"])
    y = np.log(cdata_flows["distance"])
    
    #create the subplot
    fig, ax = plt.subplots(figsize = (10,10))
    #plot the results along with the line of best fit
    sns.regplot(x=x, y=y, marker="+", ax=ax)
    ax.set_xlabel("log(flows)", fontsize = 20)
    ax.set_ylabel("log(distance)", fontsize = 20)
    fig.savefig(path_out+'london_underground_log(flows-distance).png',dpi=1000)