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

path_out = "../result/I.1/"
# ## 1. Constructing the networks

# We are going to use the tube network file called "london_tubenetwork.graphml". This file has everything we need to construct the graph. A __graphml__ is a format that describes the structural properties of a graph. 

#OK, let us start with the graphml file for London's underground
#since coords tuples are stored as string, need to convert them back to tuples using eval()
G = nx.read_graphml('../data/london.graph.xml')
for node in G.nodes():
    G.nodes[node]['coords'] = eval(G.nodes[node]['coords'])
for edge in G.edges():
    G.edges[edge]['flows'] = eval(G.edges[edge]['flows'])

print(nx.info(G))

# Let's plot the tube network! 

# We can plot the tube network with the names of the stations as labels
fig, ax = plt.subplots(figsize=(25,20))

#node_labels = nx.get_node_attributes(G, 'station_name')
#node_labels = dict([list(G.nodes),list(G.nodes)] for i in G.nodes())
node_labels = dict([(i,i) for i in G.nodes()])

pos = nx.get_node_attributes(G, 'coords')

nx.draw_networkx_nodes(G,pos,node_size=50,node_color='b')
nx.draw_networkx_edges(G,pos,arrows=False,width=0.2)
nx.draw_networkx_labels(G,pos, node_labels, font_size=10, font_color='black')

plt.title("London tube network",fontsize=40)
plt.axis("off")
plt.show()
fig.savefig(path_out+'net.png')

# #We can print the dataframe from the shapefile to check the data
# df = nx.to_pandas_edgelist(G)


# # diameter of the network considering the distance between stations (weighted diameter)

# nlen = {n:nx.single_source_dijkstra_path_length(G, n, weight='length') for n in G.nodes() }
# e = nx.eccentricity(G,sp=nlen)
# d = nx.diameter(G, e)
# d


# ## 3.  Centrality measures

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
fig.savefig(path_out+'degree_centrality.png',dpi=1000)

### Topological betweenness centrality:

#Let us compute the betweenness centrality for the network, without using weights:
bet_london_t=nx.betweenness_centrality(G, normalized=True)
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
degree_10_out = path_out+"betweenness centrality.csv"
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
plt.title("London tube topological betweenness centrality",fontsize=30)
cb.ax.tick_params(labelsize=20)
cb.set_label("Betweenness Centrality",fontdict=font)
plt.show()
fig.savefig(path_out+'betweenness_centrality.png',dpi=1000)

# #### Closeness Centrality:

#topological closeness centrality
clos_t=nx.closeness_centrality(G)
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
degree_10_out = path_out+"closeness centrality.csv"
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
fig.savefig(path_out+'closeness_centrality.png',dpi=1000)