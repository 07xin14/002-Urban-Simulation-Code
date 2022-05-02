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
    path_out = "../result/II.1/"
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
    
    # Inverse weights:
    inv_weights={(e1, e2):round(1./max([weight,0.00000001]),7) for e1, e2, weight in G.edges(data='flows')}
    weights={(e1, e2):weight for e1, e2, weight in G.edges(data='flows')}
    # Let us add the inversed weight as an attribute to the edges in the graph
    nx.set_edge_attributes(G, inv_weights, 'inv_weights')
    nx.set_edge_attributes(G, weights, 'weights')
   
    # #### Degree Centrality on nodes:

    #  We can calculate the degree centrality using networkx function: 
    deg_london =nx.degree_centrality(G)
    nx.set_node_attributes(G,dict(deg_london),'degree')
    
    # To dataframe using the nodes as the index
    #df = pd.DataFrame(index=G.nodes())
    #df['station_name'] = pd.Series(nx.get_node_attributes(G, 'station_name'))
    df = pd.DataFrame()
    df['nodes'] = G.nodes()
    df['degree'] = pd.Series(nx.get_node_attributes(G, 'degree'))
    
    df_sorted = df.sort_values(["degree"], ascending=False)
    # get the first 10 ranked nodes
    degree_10_out = path_out+"degree centrality.csv"
    df_sorted[0:10].to_csv(degree_10_out,index=False,encoding='utf_8')
    
    # Lets set colour and size of nodes according to betweenness values
    degree_values=[(i[1]['degree']) for i in G.nodes(data=True)]
    
    deg_color=[(i[1]['degree']/(max(degree_values))) for i in G.nodes(data=True)]
    deg_size=[(i[1]['degree']/(max(degree_values)))*50 for i in G.nodes(data=True)]
    
    # Plot graph
    
    pos=pos
    
    fig, ax = plt.subplots(figsize=(12,12))
    
    nx.draw_networkx_edges(G, pos,edge_color='gray', 
            width=0.4)
    nod=nx.draw_networkx_nodes(G,
            pos = pos,
            node_color= deg_color,
            node_size= deg_size)
    
    font = {'color' :'black','size':25}
    cb=plt.colorbar(nod,label = "Degree Centrality",
                 orientation="horizontal", shrink=0.5)
    plt.axis("off")
    plt.title("London tube degree centrality",fontsize=40)
    plt.show()
    cb.ax.tick_params(labelsize=20)
    cb.set_label("Degree Centrality",fontdict=font)
    fig.savefig(path_out+'weight degree_centrality.png',dpi=1000)


    # #### Betweenness Centrality on nodes:
    
    #Let us compute the betweenness centrality for the network, without using weights:
    bet_london_t=nx.betweenness_centrality(G, weight='inv_weights',normalized=True)
    # We can add these values to the nodes attributes:
    nx.set_node_attributes(G,bet_london_t,'betweenness_t')
    
    # To ataframe using the nodes as the index
    #df = pd.DataFrame(index=G.nodes())
    df = pd.DataFrame()
    df['nodes'] = G.nodes()
    #df['station_name'] = pd.Series(nx.get_node_attributes(G, 'station_name'))
    df['betweenness_t'] = pd.Series(nx.get_node_attributes(G, 'betweenness_t'))
    
    df_sorted = df.sort_values(["betweenness_t"], ascending=False)
    # get the first 10 ranked nodes
    degree_10_out = path_out+"weight betweenness centrality.csv"
    df_sorted[0:10].to_csv(degree_10_out,index=False,encoding='utf_8')
    
    # Lets set colour and size of nodes according to betweenness values
    betweenness_t_values=[(i[1]['betweenness_t']) for i in G.nodes(data=True)]
    
    bet_t_color=[(i[1]['betweenness_t']/max(betweenness_t_values)) for i in G.nodes(data=True)]
    bet_t_size=[(i[1]['betweenness_t']/max(betweenness_t_values))*100 for i in G.nodes(data=True)]
    
    # Plot graph
    fig, ax = plt.subplots(figsize=(12,12))
    
    nx.draw_networkx_edges(G, pos,edge_color='gray', width=0.4)
    
    nod=nx.draw_networkx_nodes(G, pos = pos, node_color= bet_t_color, node_size= bet_t_size)
    
    cb = plt.colorbar(nod,label="Betweenness Centrality",orientation="horizontal", shrink=0.5)
    plt.axis("off")
    plt.title("London tube topological weight betweenness centrality",fontsize=30)
    cb.ax.tick_params(labelsize=20)
    cb.set_label("Betweenness Centrality",fontdict=font)
    plt.show()
    fig.savefig(path_out+'weight_betweenness_centrality.png',dpi=1000)
    
    # #### Closeness Centrality:
    clos_t=nx.closeness_centrality(G,distance = 'weights')
    # We can add these values to the nodes attributes:
    nx.set_node_attributes(G,clos_t,'closeness_t')
    
    # To ataframe using the nodes as the index
    #df = pd.DataFrame(index=G.nodes())
    df = pd.DataFrame()
    df['nodes'] = G.nodes()
    #df['station_name'] = pd.Series(nx.get_node_attributes(G, 'station_name'))
    df['closeness_t'] = pd.Series(nx.get_node_attributes(G, 'closeness_t'))
    
    df_sorted = df.sort_values(["closeness_t"], ascending=False)
    # get the first 10 ranked nodes
    degree_10_out = path_out+"weight closeness centrality.csv"
    df_sorted[0:10].to_csv(degree_10_out,index=False,encoding='utf_8')
    
    # Lets set color and width of nodes according to the closeness values
    clos_t_val=[(i[1]['closeness_t']) for i in G.nodes(data=True)]
    
    closs_t_color=[(i[1]['closeness_t']-min(clos_t_val))/(max(clos_t_val)-min(clos_t_val)) for i in G.nodes(data=True)]
    closs_t_size=[((i[1]['closeness_t']-min(clos_t_val))/(max(clos_t_val)-min(clos_t_val))*50) for i in G.nodes(data=True)]
    
    # Plot graph
    fig, ax = plt.subplots(figsize=(12,12))
    
    nx.draw_networkx_edges(G, pos,edge_color='gray', 
            width=0.4)
    
    nod=nx.draw_networkx_nodes(G,
            pos = pos,
            node_color= closs_t_color,
            node_size= closs_t_size)
    
    cb = plt.colorbar(nod,label="Closeness Centrality",orientation="horizontal", shrink=0.5)
    plt.axis("off")
    plt.title("London tube topological closeness centrality",fontsize=30)
    cb.ax.tick_params(labelsize=20)
    cb.set_label("Closeness Centrality",fontdict=font)
    plt.show()
    fig.savefig(path_out+'weight_closeness_centrality.png',dpi=1000)
        
    #Let us compute the betweenness centrality for the network, but this time lets do it in the edges!
    bet_london_e=nx.edge_betweenness_centrality(G, normalized=True, weight='inv_weights')
    # We can add these values to the edges attributes:
    nx.set_edge_attributes(G,bet_london_e,'betweenness_e')
    # Lets set color and width of edges according to betweenness values
    betweenness_e_values=[(i[2]['betweenness_e']) for i in G.edges(data=True)]
    
    bet_e_color=[(i[2]['betweenness_e']/max(betweenness_e_values)) for i in G.edges(data=True)]
    bet_e_width=[(i[2]['betweenness_e']/max(betweenness_e_values)*5) for i in G.edges(data=True)]
    
    
    # Plot graph
    fig, ax = plt.subplots(figsize=(12,12))
    
    #pos=nx.spring_layout(X)
    edg=nx.draw_networkx_edges(G, pos,edge_color=bet_e_color, width=bet_e_width)
    
    nx.draw_networkx_nodes(G,
            pos = pos,
            node_color= 'black',
            node_size= 1)
     
    cb = plt.colorbar(edg,label="Betweenness Centrality",orientation="horizontal", shrink=0.5)
    plt.axis("off")
    plt.title("London tube topological betweenness centrality",fontsize=30)
    cb.ax.tick_params(labelsize=20)
    cb.set_label("Betweenness Centrality",fontdict=font)
    plt.show()
    fig.savefig(path_out+'weight_betweenness_edge_centrality.png',dpi=1000)
        