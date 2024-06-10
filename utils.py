import numpy as np
from pathlib import Path
import networkx as nx
from sklearn.neighbors import kneighbors_graph

def write_results(f1s, file_name, line_parameters, path):
    if not Path(path + file_name).is_file():
        file_ = open(path + file_name, 'w')
        string = 'Parameters'
        for i in range(len(f1s)):
            string += ';Iteration ' + str(i) 
        string += ';Average\n'
        file_.write(string)
        file_.close()
    file_ = open(path + file_name, 'a')
    string = line_parameters
    for f1 in f1s:
        string += ';' + str(f1)    
    string += ';' + str(np.mean(f1s)) + '\n'
    file_.write(string)
    file_.close()
    
    
def generate_graph(df, len_train, k, interest_class):
    y = df['class']
    df = df.drop(columns=['class'])

    A = kneighbors_graph(df.values, k, mode='connectivity', include_self=False)
    g = nx.Graph(A)
    
    for node in g.nodes():
        g.nodes[node]['features'] = df.loc[node].values
        g.nodes[node]['label'] = 1 if y.loc[node] == interest_class else 0
        if node < len_train:
            g.nodes[node]['train'] = 1
            g.nodes[node]['test'] = 0
        else:
            g.nodes[node]['train'] = 0
            g.nodes[node]['test'] = 1
    return g

def defining_training_instances(df_old, iteration, interest_class):
    if iteration == 0:
        df_t_i = df_old[df_old['class'] == interest_class]
        return df_t_i
    else:
        df_t_i = df_old[df_old['class'] == interest_class]
        l = len(df_t_i)
        if l <= 2500:
            return df_t_i
        else:
            return df_t_i[l-2500:]
