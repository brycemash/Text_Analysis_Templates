from cv2 import minAreaRect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
from community import community_louvain
import matplotlib.cm as cm


#take filename return pandas df
def import_data(filepath: str, sheet_id: str):
    """
    reads data file
    drop uneccesary columns
    """
    xls = pd.ExcelFile(filepath)
    # print (xls.sheet_names)
    df = pd.read_excel(xls, sheet_id)
    # takes only 1st 8 columns to include keywords
    df = df.iloc[:,:11]
    # rename them
    df.columns = ['id','entry_ number', 'quote','nest', 'reference_number', 'c1','c2','c3', 'board', 'date', 'keywords']
    # replacing all missing value with -1
    df.fillna('-1', inplace=True)

    return df

# file path for data
#filepath = 'FullDataSheet.xlsx'
filepath = 'data_9.16.22.xlsx'
sheet_id = 'All_Data'
# run sentiment analysis on keywords for selected months (corresponds to excel sheet name)
df = import_data(filepath, sheet_id)


### CHANGE THEMES TO STRING ###
dic = {1: 'blah', 
       2: 'beep',
       3: 'boop',
       }
  
G = nx.Graph()

### MAKE A BIPARTITE GRAPH ###
### make list of unique keywords ###
list_of_keywords = []
list_of_themes = []
for index, row in df.iterrows():
    
        current_key = row['keywords']
        t1 = row['c1']
        t2 = row['c2']
        t3 = row['c3']
        
        if int(t1) != -1:
            t1 = dic[int(t1)]
        if int(t2) != -1:
            t2 = dic[int(t2)]
        if int(t3) != -1:
            t3 = dic[int(t3)]
        #search for non-empty keywords
        if current_key != '-1':
            #print(current_key)
            #iterate over keywords

            # Split the keys and clear empty spaces
            keys = str(current_key).lower().replace(' ', '').split(',')

            for k in keys:
                #print(k)   
                if k not in list_of_keywords:
                    list_of_keywords.append(k)
                    G.add_node(k, bipartite = 0)
            
                if t1 != '-1':
                    if t1 not in list_of_themes:
                        list_of_themes.append(t1)
                        G.add_node(k, bipartite = 1)
                    if G.has_edge(k,t1) == True:
                        w = G[k][t1]['weight']
                        w += 1
                        #print(w)
                        G.add_edge(k,t1,weight = w)
                    else:
                        G.add_edge(k, t1,weight = 1)
                if t2 != '-1':
                    if t1 not in list_of_themes:
                        list_of_themes.append(t2)
                        G.add_node(k, bipartite = 1)
                    if G.has_edge(k,t2) == True:
                        w = G[k][t2]['weight']
                        w += 1
                        G.add_edge(k,t2,weight = w)
                    else:
                        G.add_edge(k, t2,weight = 1)
                if t3 != '-1':
                    if t1 not in list_of_themes:
                        list_of_themes.append(t3)
                        G.add_node(k, bipartite = 1)
                    if G.has_edge(k,t3) == True:
                        w = G[k][t3]['weight']
                        w += 1
                        G.add_edge(k,t3,weight=w)
                    else:
                        G.add_edge(k, t3,weight = 1)

#print(G.nodes)
#print(G.edges)
#print(bipartite.is_bipartite(G))


### NETWORK METRICS ###
clustering_coefficient = bipartite.average_clustering(G)
print(f'The clustering coefficient of the network is {clustering_coefficient:0.3f}')
print(f'The diameter of the network is {nx.diameter(G)} ', f' and the average shortest path length is {nx.average_shortest_path_length(G):0.3f}')
x = bipartite.cluster.clustering(G)
fltX = {}
for node in x:
    if G.degree(node) > 1: 
       fltX[node]= x[node]
       
print(dict(sorted(fltX.items(), key=lambda item: item[1])))
#probability that two of neighboring nodes are also connected, shows power of local clustering






color_map = []
for h in G.nodes:
    if h in list_of_keywords:
        color_map.append('blue')
    else: 
        color_map.append('green')

#setting weight limit
min = 1
# filter out all edges above threshold and grab id's
long_edges = list(filter(lambda e: e[2] < min, (e for e in G.edges.data('weight'))))
le_ids = list(e[:2] for e in long_edges)


# remove filtered edges from graph G
G.remove_edges_from(le_ids)


"""
for n in range(len(G.nodes)):
    for i in range(len(G[n])):
        print(G[n][i])
"""       
        
        
#G.remove_nodes_from()
#print(len(G.edges)) 


#implement louvain clustering
partition = community_louvain.best_partition(G)
#filter only one group
partition_zero = {key:value for (key, value) in partition.items() if value == 0}


# draw the graph
pos = nx.spring_layout(G)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G, pos, partition.keys(), node_size=20, 
                           cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G, pos, alpha=0.1)
nx.draw_networkx_labels(G, pos)
plt.show()
#some have '' others don't. Makes it hard to process


                
    


    
