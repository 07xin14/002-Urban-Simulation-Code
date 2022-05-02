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
    for ii in ['distance','population','jobs']:
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

def plot_difference(path_out,datafile):
   
    cdata = pd.read_csv(datafile)
   
    ### plot the scatter figure for flows with prosimfitted flows ###
    xlabel = 'original_flows'
    ylabel = 'prosimfitted_flows'
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
    fig.savefig(path_out+f'{xlabel}_{ylabel}.png',dpi=1000)

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
    plt.title("difference", fontsize = 20)
    ax.set_xlim(0,70000)
    plt.grid(True)
    fig.savefig(path_out+'differece_from_original_flows.png',dpi=1000)

if __name__ == '__main__':
    path_out = "../result/III.2/production_constrained/"
   
    #read in your London Commuting Data
    cdata = pd.read_csv("../data/london_flows.csv")
    g_dad = nx.from_pandas_edgelist(cdata, 'station_origin', 'station_destination','flows')
    
    #set the station code as Orig_code and Dest_code
    count_dict = { v:k for k,v in enumerate(list(g_dad.nodes()))}
    nx.set_node_attributes(g_dad, count_dict, name='node_code')
    
    Orig_code = [g_dad.nodes[i]['node_code'] for i in cdata['station_origin']]
    Dest_code = [g_dad.nodes[i]['node_code'] for i in cdata['station_origin']]
    cdata['Orig_code'] = Orig_code
    cdata['Dest_code'] = Dest_code
           
    # drop the data whose'distance' == 0 
    cdata.drop((cdata[cdata["distance"] ==0]).index, inplace=True)
    cdata.drop((cdata[cdata["jobs"] ==0]).index, inplace=True)
    cdata['log_distance'] = np.log(cdata['distance'])
    cdata['log_jobs'] = np.log(cdata['jobs'])
    
    
    ### Produnction_Constrained Model
    ### first params ###
    cdatamat = pd.pivot_table(cdata, values ="flows", index="station_origin", columns = "station_destination",
                            aggfunc=np.sum, margins=True)
    cdatamat.to_csv(path_out+'cdatamat.csv',index=True)
    
    #create the formula (the "-1" indicates no intercept in the regression model).
    formula = 'flows ~ station_origin + log_jobs + log_distance-1'
    #doubsim_form = "flows ~ population + jobs + log_distance-1"
    prodsim = smf.glm(formula=formula, data = cdata, family = sm.families.Poisson()).fit()

    print(prodsim.summary())
    cdata["prosimfitted"] = np.round(prodsim.mu,0)
    
    #here's the matrix
    cdatamat1 = cdata.pivot_table(values ="prosimfitted", index="station_origin", columns = "station_destination",
                                        aggfunc=np.sum, margins=True)
    cdatamat1.to_csv(path_out+'cdatamat1.csv',index=True)
    print(CalcRSqaured(cdata["flows"],cdata["prosimfitted"]))
    print(CalcRMSE(cdata["flows"],cdata["prosimfitted"]))
    
    with open(path_out+'prodsim.txt','w') as f:
        f.write(str(prodsim.summary()))
        f.write('\n r^2 : '+str(CalcRSqaured(cdata["flows"],cdata["prosimfitted"])))
        f.write('\n RMSE : '+str(CalcRMSE(cdata["flows"],cdata["prosimfitted"])))
    
    ### store the original flows, prosimfitted flows and their difference ###
    new_data = pd.DataFrame(cdatamat['All'])
    new_data = new_data.merge(cdatamat1['All'], left_on="station_origin", right_on="station_origin", how = "left")
    new_data['difference_from_original'] =  cdatamat1['All']-cdatamat['All']  
    new_data.rename(columns={"All_x":"original_flows"}, inplace = True)
    new_data.rename(columns={"All_y":"prosimfitted_flows"}, inplace = True)
    temp = nx.get_node_attributes(g_dad, name='node_code')
    temp = [temp[i] for i in new_data.index if i!='All']
    temp.append(1000)
    new_data['node_code'] = temp
    new_data.to_csv(path_out+'difference.csv',index=True)
        
    # # Run a doubly constrained SIM with a negative exponential cost function.
    # doubsim_form = "flows ~ station_origin + station_destination + distance-1"
    # doubsim1 = smf.glm(formula=doubsim_form, data = cdata, family = sm.families.Poisson()).fit()
    # print(doubsim1.summary())
    # cdata["doubsimfitted1"] = np.round(doubsim1.mu,0)
    
    #create some Oi and Dj columns in the dataframe and store row and column totals in them:
    #to create O_i, take cdatasub ...then... group by origcodenew ...then... summarise by calculating the sum of Total
    O_i = pd.DataFrame(cdata.groupby(["station_origin"])["flows"].agg(np.sum))
    O_i.rename(columns={"flows":"O_i"}, inplace = True)
    cdata = cdata.merge(O_i, on = "station_origin", how = "left" )
    
    D_j = pd.DataFrame(cdata.groupby(["station_destination"])["flows"].agg(np.sum))
    D_j.rename(columns={"flows":"D_j"}, inplace = True)
    cdata = cdata.merge(D_j, on = "station_destination", how = "left" )
    
    #We can do this by pulling out the parameter values
    coefs = pd.DataFrame(prodsim.params)
    coefs.reset_index(inplace=True)
    coefs.rename(columns = {0:"alpha_i", "index":"coef"}, inplace = True)
    to_repl = ["(station_origin)", "\[", "\]"]
    for x in to_repl:
        coefs["coef"] = coefs["coef"].str.replace(x, "")
    #then once you have done this you can join them back into the dataframes
    cdata = cdata.merge(coefs, left_on="station_origin", right_on="coef", how = "left")
    cdata.drop(columns = ["coef"], inplace = True)
    #check this has worked
    
    cdata.to_csv(path_out+'cdata_new.csv',index=False)
    coefs.to_csv(path_out+'prodsim_params.csv',index=False)
    
    ### plot the png we need ###
    plots(path_out,path_out+"cdata_new.csv",'prosimfitted')
    plot_difference(path_out,path_out+'difference.csv')