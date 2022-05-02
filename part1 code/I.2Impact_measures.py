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

path_out = "../result/I.2/"
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
# Let's plot the tube network! 

# # We can plot the tube network with the names of the stations as labels
# fig, ax = plt.subplots(figsize=(25,20))

# #node_labels = nx.get_node_attributes(G, 'station_name')
# #node_labels = dict([list(G.nodes),list(G.nodes)] for i in G.nodes())
# node_labels = dict([(i,i) for i in G.nodes()])

# pos = nx.get_node_attributes(G, 'coords')

# nx.draw_networkx_nodes(G,pos,node_size=50,node_color='b')
# nx.draw_networkx_edges(G,pos,arrows=False,width=0.2)
# nx.draw_networkx_labels(G,pos, node_labels, font_size=10, font_color='black')

# plt.title("London tube network",fontsize=40)
# plt.axis("off")
# plt.show()
# fig.savefig(path_out+'net.png')

# #We can print the dataframe from the shapefile to check the data
# df = nx.to_pandas_edgelist(G)

# 1.let's start with greedy modularity
communities_fg = greedy_modularity_communities(G)
print(f'number of communities: {len(communities_fg)}')
# we can sort this communities to plot only the largest communities 
communities_fg = sorted(communities_fg)

fig, ax = plt.subplots(figsize=(12,12))

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
    
plt.axis('off')
plt.title('Fast-Greedy Communnity Detection \n on London tube network', fontsize=30)
plt.show()
fig.savefig(path_out+'fast-greedy.png',dpi=1000)

for i, community in enumerate(communities_fg[:5]):
    c = list(community)
    top_5 = [hero for hero in deg if hero[0] in c][:5]
    print(f'community {i}:')
    for (hero,dd) in top_5:
        print(f'\t {hero} : {dd}')

print(f'modularity of fast-greedy: {modularity(G, communities_fg)}')
print(f'performance of fast-greedy: {performance(G, communities_fg)}')


# 2.let's try a different algorithm using cdlib
partitions = algorithms.louvain(G)
communities_louvain = partitions.communities
print(f'number of communities: {len(communities_louvain)}')

communities_louvain = sorted(communities_louvain, key=lambda x: len(x), reverse=True)

fig, ax = plt.subplots(figsize=(12,12))

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
    
plt.axis('off')
plt.title('Louvain Communities Detection \n on the London tube network', fontsize=30)
plt.show()
fig.savefig(path_out+'louvain.png',dpi=1000)
print(f'modularity of louvain: {modularity(G, communities_louvain)}')
print(f'performance of louvain: {performance(G, communities_louvain)}')
print(f'average shortest path efficiency: {avg_shortest_path_efficiency(G)}')
print(f'average shortest path: {nx.average_shortest_path_length(G)}')
print(f'average cluster coefficient: {avg_cluster_coefficient(G)}')

c1 = ['modularity of fast-greedy','performance of fast-greedy','modularity of louvain',
      'performance of louvain','average shortest path efficiency','average shortest path','average cluster coefficient']
c2 = [modularity(G, communities_fg),performance(G, communities_fg),modularity(G, communities_louvain),
      performance(G, communities_louvain),avg_shortest_path_efficiency(G),nx.average_shortest_path_length(G),avg_cluster_coefficient(G)]
c =[[c1[i],c2[i]] for i in range(0,len(c1))]
df = pd.DataFrame(c,columns = ['measures','values'])
df.to_csv(path_out+'measures.csv',index=False)