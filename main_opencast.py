import torch
from opencast import train_OPENCAST
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import numpy as np
import pandas as pd
import random
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
import argparse
from scipy.io import arff
from utils import write_results, generate_graph, defining_training_instances

def datastream_OPENCAST(df, k, lr, radius, mt, n_epochs, alpha, eps, num_layers):
    new_df = df.iloc[:2500]
    iteration = 0
    f1s = []
    for i in range(2500,len(df),2500):
        mantaining = defining_training_instances(new_df, iteration, interest_class)
        if i+5000 > len(df):
            end = len(df)
        else:
            end = i + 2500
        new_df = pd.concat([mantaining, df.iloc[i:end]], ignore_index=True)
        g = generate_graph(new_df, len(mantaining), k, interest_class)
        iteration+=1

        seed = 8
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        f1 = train_OPENCAST(g, 48, 2, lr, radius, mt, n_epochs, alpha, eps, num_layers)
        f1s.append(f1)
        if end == len(df):
            return f1s


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='OPENCAST')
    parser.add_argument("--k", type=int, default=1, help="k from graph modeling")
    parser.add_argument("--dataset", type=str, default="electricity", help="dataset")
    parser.add_argument("--numlayers", type=int, default=2, help="number of GNN layers")
    args = parser.parse_args()
    
    if args.dataset == 'electricity':
        data = arff.loadarff('./datasets/electricity.arff')
        df = pd.DataFrame(data[0])
        df = df.astype({'class': str})
        interest_class = "UP"
    elif args.dataset == 'bank':
        df = pd.read_pickle('./datasets/df_bank.pkl')
        interest_class = "yes"
    elif args.dataset == 'twitter':
        data, meta = arff.loadarff('./datasets/tweet500.arff')
        df = pd.DataFrame(data).astype(int)
        interest_class = 1
    elif args.dataset == 'argrawal':
        df = pd.read_pickle('./datasets/df_agrawal.pkl')
        interest_class = 0

    file_name = 'OPENCAST_' + args.dataset + '.csv'
    path = './resultados/'

    r = 0.3
    lr = 0.001

    for epoch in [2000, 3000]:
        for alpha in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
            for mt in [300, 500, 700]:
                for eps in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                    line_parameters = 'k:' + str(args.k) + ' | numlayers:' + str(args.numlayers) + ' | r:' + str(r) + ' | lr:' + str(lr) + ' | epoch:' + str(epoch) + ' | alpha:' + str(alpha) + ' | mt:' + str(mt) + ' | eps:' + str(eps)
                    f1s = datastream_OPENCAST(df, args.k, lr, r, mt, epoch, alpha, eps, args.numlayers)
                    write_results(f1s, file_name, line_parameters, path)
