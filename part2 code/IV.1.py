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
import seaborn as sns

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

def store_difference(cdatamat,cdatamat_new,keyword):
    ### store the original flows, prosimfitted flows and their difference ###
    new_data = pd.DataFrame(cdatamat['All'])
    new_data = new_data.merge(cdatamat_new['All'], left_on="station_origin", right_on="station_origin", how = "left")
    new_data['difference_from_original'] =  cdatamat_new['All']-cdatamat['All']  
    new_data.rename(columns={"All_x":"original_flows"}, inplace = True)
    new_data.rename(columns={"All_y":f"{keyword}_flows"}, inplace = True)
    #temp = pd.read_csv("../result/III.2/production_constrained/difference.csv")
    new_data.to_csv(path_out+f'{keyword}_difference.csv',index=True)
    
def plots(path_out,cdatafile,keyword):
    G = nx.read_graphml('../data/london.graph.xml')
    for node in G.nodes():
        G.nodes[node]['coords'] = eval(G.nodes[node]['coords'])
    for edge in G.edges():
        G.edges[edge]['flows'] = eval(G.edges[edge]['flows'])
    
    #print(nx.info(G))
    
    #read in your London Commuting Data
    cdata = pd.read_csv(cdatafile)
    g_dad = nx.from_pandas_edgelist(cdata, 'station_origin', 'station_destination',keyword)
    
    ### plot the scatter figure for flows with distance, population or jobs ###
    for ii in ['distance','population','jobs2']:
        y = cdata[ii]
        x = cdata[keyword]
        
        #create the subplot
        fig, ax = plt.subplots(figsize = (10,10))
        #plot the results along with the line of best fit
        plt.scatter(x, y, marker="+")
        ax.set_xlim(0,20000)
        ax.set_xlabel(keyword+ " flows", fontsize = 20)
        ax.set_ylabel(ii, fontsize = 20)
        fig.savefig(path_out+f'london_underground_{keyword}_flows-{ii}.png',dpi=1000)

    ### plot the disgram of flows
    fig, ax = plt.subplots(figsize=(10,10))
    cdata_flows = cdata[keyword]    
    plt.hist(cdata_flows, histtype="stepfilled" , bins = 50)
    plt.xlabel(keyword+" Flows", fontsize = 15)
    plt.ylabel("Count", fontsize= 15)
    plt.title(f"London underground {keyword} flows histogram", fontsize = 20)
    plt.grid(True)
    fig.savefig(path_out+f'london_underground_{keyword}_flows_hisgram.png',dpi=1000)
    
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
    pos = nx.get_node_attributes(G_copy, 'coords')
    
    ####plot the flow lines type 2####
    betweenness_e_values=[(i[2][keyword]) for i in g_dad.edges(data=True)]
    #bet_e_color=[(i[2]['flows']-min(betweenness_e_values))/(max(betweenness_e_values)-min(betweenness_e_values)) for i in g_dad.edges(data=True)]
    #bet_e_width=[(i[2]['flows']-min(betweenness_e_values))/(max(betweenness_e_values)-min(betweenness_e_values)) for i in g_dad.edges(data=True)]
    range_list = [0,50,100,150,200,500,1000,2000,3000,5000,8000,10000,20000]
    colors = ['skyblue','royalblue','lightgreen','seagreen','orange','darkgoldenrod',
              'yellow','lightpink','red','darkred','darkviolet','purple']
    cmap = ListedColormap(colors)
    bet_e_color=[GetRangeId(i[2][keyword],range_list) for i in g_dad.edges(data=True)]
    bet_e_width=[(i[2][keyword]/max(betweenness_e_values)*10) for i in g_dad.edges(data=True)]
    
    # Plot graph
    fig, ax = plt.subplots(figsize=(12,12))
    
    #pos=nx.spring_layout(X)
    edg=nx.draw_networkx_edges(g_dad, pos,edge_color=bet_e_color,width=bet_e_width
                               ,edge_cmap=cmap,edge_vmin=0,edge_vmax=12)#plt.get_cmap('Paired'))
    
    nx.draw_networkx_nodes(g_dad,
            pos = pos,
            node_color= 'black',
            node_size= 1)
     
    cb = plt.colorbar(edg,label=keyword+" Flows",orientation="horizontal")#, shrink=0.5)
    plt.axis("off")
    plt.title(f"London underground {keyword} Flows",fontsize=30)
    cb.ax.tick_params(labelsize=20,rotation = 30)
    font = {'color' :'black','size':25}
    cb.set_label(keyword+" Flows",fontdict=font)
    cb.set_ticks(np.arange(0,13))
    cb.set_ticklabels([str(i) for i in range_list])
    #cb.set_ticks(np.array(range_list))
    #cb.ax.locator_params(nbins=len(range_list))
    plt.show()
    fig.savefig(path_out+f'london_underground_{keyword}_flows_type2.png',dpi=1000)
    
    
    ### plot the disgram of flows for flows>0
    fig, ax = plt.subplots(figsize=(10,10))
    
    #remove all 0 values (logarithms can't deal with 0 values)
    cdata_flows = cdata[keyword]
    cdata_flows = cdata_flows[(cdata_flows!=0)]
    plt.hist(np.log(cdata_flows), histtype="stepfilled" , bins = 50)
    plt.xlabel(f"log({keyword} Flows)", fontsize = 15)
    plt.ylabel("Count", fontsize= 15)
    plt.title(f"London underground {keyword} flows histogram", fontsize = 20)
    plt.grid(True)
    fig.savefig(path_out+f'london_underground_log_{keyword}_flows_hisgram.png',dpi=1000)

    
    ### plot the log(dist) & log(flows) ###
    #extract the x and y converting to log
    cdata_flows = cdata[["distance", keyword]]
    #remove all 0 values (logarithms can't deal with 0 values)
    cdata_flows = cdata_flows[(cdata_flows!=0).all(1)]
    
    x = np.log(cdata_flows[keyword])
    y = np.log(cdata_flows["distance"])
    
    #create the subplot
    fig, ax = plt.subplots(figsize = (10,10))
    #plot the results along with the line of best fit
    sns.regplot(x=x, y=y, marker="+", ax=ax)
    ax.set_xlabel(f"log({keyword} flows)", fontsize = 20)
    ax.set_ylabel("log(distance)", fontsize = 20)
    fig.savefig(path_out+f'london_underground_log({keyword}_flows-distance).png',dpi=1000)

def plot_difference(path_out,datafile,keyword):
   
    cdata = pd.read_csv(datafile)
   
    ### plot the scatter figure for flows with prosimfitted flows ###
    xlabel = 'original_flows'
    ylabel = f'{keyword}_flows'
    x = cdata[xlabel]
    y = cdata[ylabel]
    
    #create the subplot
    fig, ax = plt.subplots(figsize = (10,10))
    #plot the results along with the line of best fit
    #plt.scatter(x, y, marker="+")
    sns.regplot(x=x, y=y, marker="+", ax=ax)
    ax.set_xlim(0,70000)
    ax.set_ylim(0,70000)
    ax.set_xlabel(xlabel, fontsize = 20)
    ax.set_ylabel(ylabel, fontsize = 20)
    fig.savefig(path_out+f'{keyword}{xlabel}_{ylabel}.png',dpi=1000)

    ### plot the difference from original flows
    fig, ax = plt.subplots(figsize=(10,10))
    x = cdata[xlabel][:-1]
    y = cdata['difference_from_original'][:-1]   
    plt.scatter(x, y, marker="+")
    yy = copy.deepcopy(x)
    yy = [0 for i in yy]
    plt.plot(x, yy, 'k-', color = 'r')
    plt.xlabel('original flows', fontsize = 15)
    plt.ylabel("difference from original flows", fontsize= 15)
    plt.title(f"{keyword} difference", fontsize = 20)
    ax.set_xlim(0,70000)
    plt.grid(True)
    fig.savefig(path_out+f'{keyword}differece_from_original_flows.png',dpi=1000)

if __name__ == '__main__':
    path_out = "../result/IV.1/"
    cdata = pd.read_csv("../result/III.2/production_constrained/cdata_new.csv")  
    params = pd.read_csv("../result/III.2/production_constrained/prodsim_params.csv")
    cdatamat = pd.read_csv("../result/III.2/production_constrained/cdatamat.csv",index_col=0)
    
    
    
    length = len(params['alpha_i'])
    alpha_i = params['alpha_i'][:length-3]
    gamma = params['alpha_i'][length-2]
    beta = 0.0-params['alpha_i'][length-1]
    
    ### Scenarios A: change the jobs of Canary Wharf from 100% to 50% ###
    station_sel = 'Canary Wharf'
    job = int(cdata.loc[cdata["station_destination"] == station_sel,'jobs'].mean())
    cdata['jobs2'] = cdata['jobs']
    cdata.loc[cdata["station_destination"] == station_sel,'jobs2'] = job/2
    
    cdata["prodsimest2"] = np.exp(cdata["alpha_i"]+gamma*np.log(cdata["jobs2"]) - beta*cdata["log_distance"])

    cdata["prodsimest2"] = round(cdata["prodsimest2"],0)
    #now we can convert the pivot table into a matrix
    cdatamat2 = cdata.pivot_table(values ="prodsimest2", index="station_origin", columns = "station_destination",
                            aggfunc=np.sum, margins=True)
    cdatamat2.to_csv(path_out+'cdatamat2.csv',index=True)
    
    #cdatamat2.insert(0,"station_origin",cdatamat2.index)
    #.index = cdatamat.index
    store_difference(cdatamat,cdatamat2,'prodsimest2')
    with open(path_out+'prodsim2.txt','w') as f:
        f.write(' r^2 : '+str(CalcRSqaured(cdata["flows"],cdata["prodsimest2"])))
        f.write('\n RMSE : '+str(CalcRMSE(cdata["flows"],cdata["prodsimest2"])))
  
    cdata['log_jobs2'] = np.log(cdata['jobs2'])
    
    #calculate some new wj^alpha and d_ij^beta values
    Dj2_gamma = cdata["jobs"]**gamma
    dist_beta = cdata["distance"]**(-1.0*beta)
    #calcualte the first stage of the Ai values
    cdata["Ai1"] = Dj2_gamma * dist_beta
    #now do the sum over all js bit
    A_i = pd.DataFrame(cdata.groupby(["station_origin"])["Ai1"].agg(np.sum))
    #now divide into 1
    A_i["Ai1"] = 1/A_i["Ai1"]
    A_i.rename(columns={"Ai1":"A_i"}, inplace=True)
    #and write the A_i values back into the dataframe
    cdata = cdata.merge(A_i, left_on="station_origin", right_index=True, how="left")
    
    #to check everything works, recreate the original estimates
    cdata["prodsimest2_test"] = cdata["A_i"]*cdata["O_i"]*Dj2_gamma*dist_beta
    #round
    cdata["prodsimest2_test"] = round(cdata["prodsimest2_test"])
    #check

    #calculate some new wj^alpha and d_ij^beta values
    Dj3_gamma = cdata["jobs2"]**gamma
    dist_beta = cdata["distance"]**(-1.0*beta)
    #calcualte the first stage of the Ai values
    cdata["Ai1"] = Dj3_gamma * dist_beta
    #now do the sum over all js bit
    A_i = pd.DataFrame(cdata.groupby(["station_origin"])["Ai1"].agg(np.sum))
    #now divide into 1
    A_i["Ai1"] = 1/A_i["Ai1"]
    A_i.rename(columns={"Ai1":"A_i2"}, inplace=True)
    #and write the A_i values back into the dataframe
    cdata = cdata.merge(A_i, left_on="station_origin", right_index=True, how="left")
    
    #to check everything works, recreate the original estimates
    cdata["prodsimest2_2"] = cdata["A_i2"]*cdata["O_i"]*Dj3_gamma*dist_beta
    #round
    cdata["prodsimest2_2"] = round(cdata["prodsimest2_2"])
    #check
    
    cdatamat2_2 = cdata.pivot_table(values ="prodsimest2_2", index="station_origin", columns = "station_destination",
                            aggfunc=np.sum, margins=True)
    cdatamat2_2.to_csv(path_out+'cdatamat2_2.csv',index=True)
    
    #cdatamat2.insert(0,"station_origin",cdatamat2.index)
    #.index = cdatamat.index
    with open(path_out+'prodsim2_2.txt','w') as f:
        f.write(' r^2 : '+str(CalcRSqaured(cdata["flows"],cdata["prodsimest2_2"])))
        f.write('\n RMSE : '+str(CalcRMSE(cdata["flows"],cdata["prodsimest2_2"])))
    store_difference(cdatamat,cdatamat2_2,'prodsimest2_2')
    
    
    ### Redo Produnction_Constrained Model
    
    #create the formula (the "-1" indicates no intercept in the regression model).
    formula = 'flows ~ station_origin + log_jobs2 + log_distance-1'
    #doubsim_form = "flows ~ population + jobs + log_distance-1"
    prodsim3 = smf.glm(formula=formula, data = cdata, family = sm.families.Poisson()).fit()

    #print(prodsim2.summary())
    cdata["prosimfitted3"] = np.round(prodsim3.mu,0)
    
    #here's the matrix
    cdatamat3 = cdata.pivot_table(values ="prosimfitted3", index="station_origin", columns = "station_destination",
                                        aggfunc=np.sum, margins=True)
    
    cdatamat3.to_csv(path_out+'cdatamat3.csv',index=True)
    #print(CalcRSqaured(cdata["flows"],cdata["prosimfitted3"]))
    #print(CalcRMSE(cdata["flows"],cdata["prosimfitted3"]))
    
    with open(path_out+'prodsim3.txt','w') as f:
        f.write(str(prodsim3.summary()))
        f.write('\n r^2 : '+str(CalcRSqaured(cdata["flows"],cdata["prosimfitted3"])))
        f.write('\n RMSE : '+str(CalcRMSE(cdata["flows"],cdata["prosimfitted3"])))
    
    #cdatamat3.insert(0,"station_origin",cdatamat3.index)
    #cdatamat3.index = cdatamat.index
    store_difference(cdatamat,cdatamat3,'prosimfitted3')
    
    cdata.to_csv(path_out+'cdata_new2.csv',index=False)
    ### plot the png we need ###
    plots(path_out,path_out+"cdata_new2.csv",'prosimfitted3')
    plots(path_out,path_out+"cdata_new2.csv",'prodsimest2_2')
    plot_difference(path_out,path_out+'prodsimest2_2_difference.csv','prodsimest2_2')
    plot_difference(path_out,path_out+'prosimfitted3_difference.csv','prosimfitted3')