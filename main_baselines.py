from scipy.io import arff
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report
import numpy as np
from sklearn.ensemble import IsolationForest
from CDD_github import CDD 
import argparse
from sklearn.neighbors import LocalOutlierFactor
from pathlib import Path

def train_one_class(X_train, X_unsup, y_true, one_class_classifier, clf, ht, alpha):
    
    if one_class_classifier == 'OCSVM':
        clf.fit(X_train)
        y_pred = clf.predict(X_unsup)
    elif one_class_classifier == 'IsoForest':
        clf.fit(X_train)
        y_pred = clf.predict(X_unsup)
    elif one_class_classifier == 'CDD':
        clf.fit(X_train)
        y_pred = clf.predict(X_unsup)
    elif one_class_classifier == 'LOF':
        y_pred = clf.fit_predict(np.concatenate([X_train,X_unsup]))[len(X_train):]
    elif one_class_classifier == 'CMA':
        y_pred = train_CMA(X_train, X_unsup, clf)
    elif one_class_classifier == 'FMA':
        y_pred = train_FMA(X_train, X_unsup, clf, ht, alpha)

    return classification_report(y_true, y_pred, output_dict=True)['macro avg']['f1-score']

def train_CMA(X_train, X_unsup, clf):
    y_pred = []
    for instance in X_unsup:
        F = clf.fit(X_train)
        if F.predict([instance]) == -1:
            y_pred.append(-1)
        else:
            y_pred.append(1)
        X_train = np.concatenate([X_train[1:,:], [instance]])
    
    return y_pred

def train_FMA(X_train, X_unsup, clf, ht, alpha):
    F = clf.fit(X_train)
    y_pred = []
    ct = 0
    nt = 0
    for t in range(1, len(X_unsup)+1):
        nt_ant = nt
        ct_ant = ct
        ht_ant = ht
        instance = X_unsup[t-1]
        if F.predict([instance]) == -1:
            y_pred.append(-1)
            ct = ct_ant + 1
        else:
            y_pred.append(1)
        X_train = np.concatenate([X_train[1:,:], [instance]])
        if(t % ht == 0):
            F_linha = clf.fit(X_train)
            if F.predict([instance]) == 1 and F_linha.predict([instance]) == 1:
                nt = nt_ant + 1
            F = F_linha
        
        if t == 1:
            delta_nt = (nt/t)
            delta_ct = (ct/t)
        else:
            delta_nt = (nt/t) - (nt_ant/(t-1))
            delta_ct = (ct/t) - (ct_ant/(t-1))
        if delta_nt >= 0 and delta_ct <= 0:
            try:
                ht = int(ht + (alpha * ht_ant))
            except:
                ht = 1000000
        else:
            ht = int(ht - (alpha * ht_ant))
            if ht == 0:
                ht=1
    return y_pred

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

def define_gammas():
  gammas = ['scale', 'auto']
  return gammas

def define_nus():
  nus = []
  for n in range(5,90,5):
    nus.append(n/100)
  for n in range(5,90,5):
    nus.append(n/1000)

  return nus

def define_kernels():
  return ['rbf', 'sigmoid','linear', 'poly']

def data_stream_pipeline(df, one_class_classifier_name, algorithm, ht=None, alpha=None):
    new_df = df.iloc[:2500]
    iteration = 0
    f1s = []
    for i in range(2500,len(df),2500):
        df_int = defining_training_instances(new_df, iteration, interest_class)
        if i+5000 > len(df):
            end = len(df)
        else:
            end = i + 2500
        df_test = df.iloc[i:end]
        new_df = pd.concat([df_int, df_test], ignore_index=True)
        X_train = df_int.drop(columns=['class']).values
        y_true = [1 if y==interest_class else -1 for y in df_test['class'].values]
        X_unsup = df_test.drop(columns=['class']).values        
        f1 = train_one_class(X_train, X_unsup, y_true, one_class_classifier_name, algorithm, ht, alpha)
        f1s.append(f1)
        iteration+=1
        if end == len(df):
            return f1s

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baselines')
    parser.add_argument("--dataset", type=str, help="dataset")
    parser.add_argument("--ocl", type=str, default="OCSVM", help="one-class algorithm name")
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
    file_name = args.ocl + '_' + args.dataset + '.csv'
    path = './resultados/'
    if args.ocl == 'OCSVM' or args.ocl == 'CMA':
         for kernel in define_kernels():
            for gamma in define_gammas():
                for nu in define_nus():
                    ocsvm = OneClassSVM(kernel=kernel,nu=nu,gamma=gamma)
                    f1s = data_stream_pipeline(df, args.ocl, ocsvm)
                    line_parameters =  'kernel:' + kernel + ' | gamma:' + gamma + ' | nu:' + str(nu)
                    write_results(f1s, file_name, line_parameters, path)
    
    elif args.ocl == 'IsoForest':
        for n_estimators in [1,2,5,10,50,100,200,500]:
            for max_samples in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                for max_features in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                    iso = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, max_features=max_features, random_state=81)
                    f1s = data_stream_pipeline(df, args.ocl, iso)
                    line_parameters =  'n_estimators:' + str(n_estimators) + '| max_samples:' + str(max_samples) + ' | max_features:' + str(max_features)
                    write_results(f1s, file_name, line_parameters, path)
    
    elif args.ocl == 'CDD':
        for max_k in [100]:
            for method in ['gmm']:
                cdd = CDD.CDD(max_k=max_k, method=method, random=81)
                f1s = data_stream_pipeline(df, args.ocl, cdd)
                line_parameters =  'max_k:' + str(max_k) + '| method:' + str(method)
                write_results(f1s, file_name, line_parameters, path)

    elif args.ocl == 'LOF':
        for n_neighbors in [1,2,5,10,50,100]:
            if args.dataset == 'twitter':
                metric='cosine'
            else:
                metric='euclidean'
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric=metric)
            f1s = data_stream_pipeline(df, args.ocl, lof)
            line_parameters =  'n_neighbors:' + str(n_neighbors)
            write_results(f1s, file_name, line_parameters, path)

    elif args.ocl == 'FMA':
        for kernel in define_kernels():
            for gamma in define_gammas():
                for nu in define_nus():
                    for ht in [50, 100, 1000, 2000]:
                        for alpha in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
                            ocsvm = OneClassSVM(kernel=kernel,nu=nu,gamma=gamma, max_iter=1000)
                            f1s = data_stream_pipeline(df, args.ocl, ocsvm, ht, alpha)
                            line_parameters =  'kernel:' + kernel + ' | gamma:' + gamma + ' | nu:' + str(nu) + ' | ht:' + str(ht) + ' | alpha:' + str(alpha)
                            write_results(f1s, file_name, line_parameters, path)
