import numpy as np
from pathlib import Path
import pandas as pd

path_results = './resultados/'

basepath = Path(path_results)
files_ = basepath.iterdir()

dict_ = {}

for file_ in files_:
  if file_.is_file():

    file_name = file_.name

    parts = file_name.split('_')

    method = parts[0]
    dataset = parts[1]
    
    df = pd.read_csv(path_results + file_name, sep=';')

    dataset = dataset.replace('.csv', '')

    best_f1 = max(df['Average'])

    cols = list(df.columns)
    cols.remove('Parameters')
  
    df_plot = df[df['Average'] == best_f1].iloc[0][cols]

    if dataset not in dict_:
      dict_[dataset] = {}

    dict_[dataset][method] = df_plot


for key in dict_:
    pd.DataFrame(dict_[key]).T.to_csv('./compiled_results/' + key + '_plot.csv')

