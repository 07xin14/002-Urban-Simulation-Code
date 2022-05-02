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
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity,performance
from cdlib import algorithms
import copy
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes

def centralities_first_node(G,choice=0):
    
    if choice==0:
        tmp =nx.degree_centrality(G)
    elif choice==1:
        tmp=nx.betweenness_centrality(G, normalized=True)
    elif choice==2:
        tmp=nx.closeness_centrality(G)
    else:
        print("Wrong Choice!")
        exit
    
    return max(tmp,key=tmp.get)
    

# Function for calculating the defined avg_shortest_path efficiency. 
# The forluma is : E = 2/N(N-1) * sum(1/dij)
# dij is the node shortest path length
def avg_shortest_path_efficiency(G,weight=None):
    avg=0.0
    if weight is None:
        for node in G:
            path_length=nx.single_source_shortest_path_length(G, node)
            avg += sum([1.0/i for i in path_length.values() if i != 0])
    else:
        for node in G:
            path_length=nx.single_source_dijkstra_path_length(G, node, weight=weight)
            avg += sum([1.0/i for i in path_length.values() if i != 0])
    n=len(G)
    return 2.0*avg/(n*(n-1)) 

# Function for calculating the average cluster coeffient. 
def avg_cluster_coefficient(G,weight=None):
    # let's measure the clustering coefficient
    c = nx.clustering(G)
    #print(type(c))
    
    # we can sort the dictionary by the clustering coefficient
    c = sorted(c.items(), key=lambda pair: pair[1], reverse=True)
    #print(c[:5])
    
    #what is the average clustering coefficient
    c_avg = sum([i[1] for i in c]) / len(c)
    #print(f'avg clustering coefficient: {c_avg}')
    return  c_avg


def greedy_m(G,pos,ii,df,fig,ax,cen=""):
    # 1.let's start with greedy modularity
    communities_fg = greedy_modularity_communities(G)
    print(f'number of greedy communities removal node {ii} for {cen}: {len(communities_fg)}')
    # we can sort this communities to plot only the largest communities 
    communities_fg = sorted(communities_fg)
     
    # plot the entire graph with gray nodes
    nx.draw_networkx_nodes(G, 
                           pos=pos,
                           node_color='grey',
                           node_size=20,
                           ax=ax, 
                          )
    
    nx.draw_networkx_edges(G,
                           pos=pos,
                           edge_color='gray',
                           width=1.2,
                           alpha=0.5,
                           ax=ax
                          )
    
    # set the colors we will be using for each community
    cls = ['r', 'c', 'm', 'y', 'b']
    i=0
    while i<5:
        nx.draw_networkx_nodes(G, 
                               pos=pos,
                               node_color=cls[i],
                               node_size=20,
                               ax=ax, 
                               nodelist= list(communities_fg[i])
                              )
        i+=1
        
    ax.axis('off')
    ax.set_title('Fast-Greedy Communnity Detection \n + Removal Node '+str(ii)+" + "+cen, fontsize=30)
    #plt.show()
    
    
    for i, community in enumerate(communities_fg[:5]):
        c = list(community)
        top_5 = [hero for hero in deg if hero[0] in c][:5]
        #print(f'community {i}:')
        #for (hero,dd) in top_5:
        #    print(f'\t {hero} : {dd}')
    
    #print(f'modularity of fast-greedy: {modularity(G, communities_fg)}')
    #print(f'performance of fast-greedy: {performance(G, communities_fg)}')
    df.loc['modularity of fast-greedy','remove node '+str(ii)] = modularity(G, communities_fg)
    df.loc['performance of fast-greedy','remove node '+str(ii)] = performance(G, communities_fg)

def louvain_m(G,pos,ii,df,fig,ax,cen=""):
    # 2.let's try a different algorithm using cdlib
    partitions = algorithms.louvain(G)
    communities_louvain = partitions.communities
    print(f'number of louvain communities removal node {ii} for {cen}: {len(communities_louvain)}')
    
    communities_louvain = sorted(communities_louvain, key=lambda x: len(x), reverse=True)
        
    nx.draw_networkx_nodes(G, 
                           pos=pos,
                           node_color='grey',
                           node_size=20,
                           ax=ax, 
                          )
    
    nx.draw_networkx_edges(G,
                           pos=pos,
                           edge_color='gray',
                           width=1.2,
                           alpha=0.5,
                           ax=ax
                          )
    
    cls = ['r', 'c', 'm', 'y', 'b']
    i=0
    while i<5:
        nx.draw_networkx_nodes(G, 
                               pos=pos,
                               node_color=cls[i],
                               node_size=20,
                               ax=ax, 
                               nodelist= list(communities_louvain[i])
                              )
        i+=1
        
    ax.axis('off')
    ax.set_title('Louvain Communities Detection \n + Removal Node '+str(ii)+" + "+cen, fontsize=30)
    #print(f'modularity of louvain: {modularity(G, communities_louvain)}')
    #print(f'performance of louvain: {performance(G, communities_louvain)}')
    df.loc['modularity of louvain','remove node '+str(ii)] = modularity(G, communities_louvain)
    df.loc['performance of louvain','remove node '+str(ii)] = performance(G, communities_louvain)

if __name__ == '__main__':
    path_out = "../result/I.3/"
    centralities = ['degree centrality',"closeness centrality",'betweenness centrality']
    
    # ## 1. Constructing the networks
    # We are going to use the tube network file called "london_tubenetwork.graphml". This file has everything we need to construct the graph. A __graphml__ is a format that describes the structural properties of a graph. 
    
    #OK, let us start with the graphml file for London's underground
    #since coords tuples are stored as string, need to convert them back to tuples using eval()
    G = nx.read_graphml('../data/london.graph.xml')
    for node in G.nodes():
        G.nodes[node]['coords'] = eval(G.nodes[node]['coords'])
    for edge in G.edges():
        G.edges[edge]['flows'] = eval(G.edges[edge]['flows'])
    
    deg = sorted(G.degree(), key=lambda pair: pair[1], reverse=True)
    print(nx.info(G))
    pos = nx.get_node_attributes(G, 'coords')
    
    
    c1 = ['number of nodes','modularity of fast-greedy','performance of fast-greedy','modularity of louvain',
          'performance of louvain','average shortest path efficiency','average shortest path','average cluster coefficient']
    df = pd.DataFrame(index=c1,columns=["remove node "+str(i) for i in range(0,11)])
    
    #get impact measures without removal
    fig, ax = plt.subplots(2,1,figsize=(12,24))
    greedy_m(G,pos,0,df,fig,ax[0],"all")
    louvain_m(G,pos,0,df,fig,ax[1],'all')
    df.loc['average shortest path efficiency','remove node 0'] = avg_shortest_path_efficiency(G)
    df.loc['average shortest path','remove node 0'] = nx.average_shortest_path_length(G)
    df.loc['average cluster coefficient','remove node 0'] = avg_cluster_coefficient(G)
    df.loc['number of nodes','remove node 0'] = len(G.nodes)
    
    df2 = copy.deepcopy(df) # for method 2
    fig.savefig(path_out+f'fast-greedy_louvain_remove_node_{0}.png',dpi=1000)
        
    G_copy = copy.deepcopy(G)
    
    for ic,cen in enumerate(centralities):
        rank = pd.read_csv("../result/I.1/"+cen+'.csv',header=0)
        G_copy = copy.deepcopy(G)
        for i in range(1,11):
            G_copy.remove_node(rank['nodes'][i-1]) #remove node
            Gcc = sorted(nx.connected_components(G_copy), key=len, reverse=True)
            Gsub = G_copy.subgraph(Gcc[0]) # the largest community
            pos = nx.get_node_attributes(Gsub, 'coords')
            
            #get impact measures 
            fig, ax = plt.subplots(2,1,figsize=(12,24))
            greedy_m(Gsub,pos,i,df,fig,ax[0],cen+' m1')
            louvain_m(Gsub,pos,i,df,fig,ax[1],cen+' m1')
            df.loc['average shortest path efficiency','remove node '+str(i)] = avg_shortest_path_efficiency(Gsub)
            df.loc['average shortest path','remove node '+str(i)] = nx.average_shortest_path_length(Gsub)
            df.loc['average cluster coefficient','remove node '+str(i)] = avg_cluster_coefficient(Gsub)
            df.loc['number of nodes','remove node '+str(i)] = len(Gsub.nodes)
            fig.savefig(path_out+cen+f'_m1_fast-greedy_louvain_remove_node_{i}.png',dpi=1000)
            
        
        df.to_csv(path_out+cen+'_m1_measures.csv',index=True)
    
        G_copy = copy.deepcopy(G)
        re_node = centralities_first_node(G_copy,ic)
        for i in range(1,11):
            G_copy.remove_node(re_node) #remove node
            Gcc = sorted(nx.connected_components(G_copy), key=len, reverse=True)
            Gsub = G_copy.subgraph(Gcc[0]) # the largest community
            pos = nx.get_node_attributes(Gsub, 'coords')
            
            #get impact measures 
            fig, ax = plt.subplots(2,1,figsize=(12,24))
            greedy_m(Gsub,pos,i,df2,fig,ax[0],cen+' m2')
            louvain_m(Gsub,pos,i,df2,fig,ax[1],cen+' m2')
            df2.loc['average shortest path efficiency','remove node '+str(i)] = avg_shortest_path_efficiency(Gsub)
            df2.loc['average shortest path','remove node '+str(i)] = nx.average_shortest_path_length(Gsub)
            df2.loc['average cluster coefficient','remove node '+str(i)] = avg_cluster_coefficient(Gsub)
            df2.loc['number of nodes','remove node '+str(i)] = len(Gsub.nodes)
            fig.savefig(path_out+cen+f'_m2_fast-greedy_louvain_remove_node_{i}.png',dpi=1000)
            
            re_node = centralities_first_node(G_copy,ic)
        
        df2.to_csv(path_out+cen+'_m2_measures.csv',index=True)

        
        plt.clf()
        plt.cla()
        plt.close()
        # draw the lines for average cluster coefficient and average shortest path efficiency
        fig2 = plt.figure()
        x_data = range(0,11)
        
        ax_cof = HostAxes(fig2, [0.1, 0.1, 0.5, 0.5]) #用[left, bottom, weight, height]的方式定义axes，0 <= l,b,w,h <= 1
     
        #parasite addtional axes, share x
        ax_temp = ParasiteAxes(ax_cof, sharex=ax_cof)
        ax_load = ParasiteAxes(ax_cof, sharex=ax_cof)
         
        #append axes
        ax_cof.parasites.append(ax_temp)
        ax_cof.parasites.append(ax_load)
         
        #invisible right axis of ax_cof
        ax_cof.axis['right'].set_visible(False)
        #ax_cof.axis['top'].set_visible(False)
        ax_temp.axis['right'].set_visible(True)
        ax_temp.axis['right'].major_ticklabels.set_visible(True)
        ax_temp.axis['right'].label.set_visible(True)
         
        #set label for axis
        ax_cof.set_ylabel('number of nodes')
        ax_cof.set_xlabel('number of removal nodes')
        ax_temp.set_ylabel('average shortest path efficiency')
        ax_load.set_ylabel('average cluster coefficient')
         
        load_axisline = ax_load.get_grid_helper().new_fixed_axis
         
        ax_load.axis['right2'] = load_axisline(loc='right', axes=ax_load, offset=(60,0))
         
        fig2.add_axes(ax_cof)
         
        ''' #set limit of x, y
        ax_cof.set_xlim(0,2)
        ax_cof.set_ylim(0,3)
        '''
         
        curve_cof = ax_cof.plot(x_data,df.loc['number of nodes'],'r*-',linewidth=2,label='number of nodes m1')
        curve_cof = ax_cof.plot(x_data,df2.loc['number of nodes'],'r*--',linewidth=3,alpha=0.5,label='number of nodes m2')
            
        curve_temp = ax_temp.plot(x_data,df.loc['average shortest path efficiency'],'bs-',linewidth=2,label='average shortest path efficiency m1')
        curve_temp = ax_temp.plot(x_data,df2.loc['average shortest path efficiency'],'bs--',linewidth=3,alpha=0.5,label='average shortest path efficiency m2')
        curve_load = ax_load.plot(x_data,df.loc['average cluster coefficient'],'go-',linewidth=2,label='average cluster coefficient m1')
        curve_load = ax_load.plot(x_data,df2.loc['average cluster coefficient'],'go--',linewidth=3,alpha=0.5,label='average cluster coefficient m2')
         
        ax_cof.set_ylim(150,420)
        ax_temp.set_ylim(0.15,0.25)
        ax_load.set_ylim(0.015,0.05)
         
        ax_cof.legend(fontsize=3,markerscale=0.2)
         
        #轴名称，刻度值的颜色
        ax_cof.axis['left'].label.set_color('r')
        ax_temp.axis['right'].label.set_color('b')
        ax_load.axis['right2'].label.set_color('g')
         
        ax_cof.axis['left'].major_ticks.set_color('r')
        ax_temp.axis['right'].major_ticks.set_color('b')
        ax_load.axis['right2'].major_ticks.set_color('g')
        
        ax_cof.axis['left'].major_ticklabels.set_color('r')
        ax_temp.axis['right'].major_ticklabels.set_color('b')
        ax_load.axis['right2'].major_ticklabels.set_color('g')
         
        ax_cof.axis['left'].line.set_color('r')
        ax_temp.axis['right'].line.set_color('b')
        ax_load.axis['right2'].line.set_color('g')
    
    
        #plt.legend()
        fig2.savefig(path_out+cen+'_result.png',dpi=1000)
        
        
    
        