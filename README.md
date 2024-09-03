# One-Class Learning for Data Stream through Graph Neural Networks

- Marcos Gôlo (ICMC/USP) | marcosgolo@usp.br
- João Gama (Univeridade do Porto) | jgama@fep.up.pt
- Ricardo Marcacini (ICMC/USP) | ricardo.marcacini@icmc.usp.br

# Citing:

If you use any part of this code in your research, please cite it using the following BibTex entry
```latex
@article{ref:Golo2024,
  title={One-Class Learning for Data Stream through Graph Neural Networks},
  author={Gôlo, Marcos Paulo Silva and Gama, João and Marcacini, Ricardo Marcondes},
  booktitle={Brazilian Conference on Intelligent Systems},
  pages={X--X},
  year={2024},
  organization={Springer}
}
```

# Abstract 
In many data stream applications, there is a normal concept, and the objective is to identify normal and abnormal concepts by training only with normal concept instances. This scenario is known in the literature as one-class learning (OCL) for data stream. In this OCL scenario for data stream, we highlight two main gaps: (i) lack of methods based on graph neural networks (GNNs) and (ii) lack of interpretable methods. We introduce OPENCAST (**O**ne-class gra**P**h auto**ENC**oder for d**A**ta **ST**ream), a new method for data stream based on OCL and GNNs. Our method learns representations while encapsulating the instances of interest through a hypersphere. OPENCAST learns low-dimensional representations to generate interpretability in the representation learning process. OPENCAST achieved state-of-the-art results for data streams in the OCL scenario, outperforming seven other methods. Furthermore, OPENCAST learns low-dimensional representations, generating interpretability in the representation learning process and results.

# How to use || Replication of our results
```
python3 main_opencast.py --k 1 --dataset electricity --numlayers 1

python3 main_baselines.py --dataset electricity --ocl IsoForest
```
# How to apply OPENCAST to your datastream dataset

## Adding your dataset to our main file

You must add the name of your dataset in the conditional structures (if-else) to choose the dataset (lines 48 to 62 of the main file). Thus, you can use the name of your dataset as an argument when executing the main file.

## Dataset Format 

Our method expects a dataframe in which the instances are represented by the rows. Each column is an attribute of the instances, except the column with the class (**the name of this column always has to be class**). Our method expects numeric attributes. Example:

| - | attribute 1 | attribute 2 | attribute 3 | attribute 4 | attribute 5 | class |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| instance 1 | 0.1 | 0.2 | 1 | 10 | 0.001 | class_one |
| instance 2 | 0.2 | 0.1 | 2 | 6 | 0.01 | class_one |
| instance 3 | 0.4 | 0.1 | 7 | 40 | 0.1 | class_two |
| instance 4 | 0.9 | 0.2 | 90 | 100 | 0.9 | class_two |

## Code example to add to the main file
```
elif args.dataset == 'your_dataset':
  df = pd.read_pickle('./datasets/file_name_dataset.pkl') # You can read a CSV, ARFF, and other files, but you must transform the dataset into a dataframe
  interest_class = "class_one" # here you can use the class_one or class_two
```

# Requirements
 - python == 3.11.8
 - networkx == 3.2.1
 - pandas == 2.2.1
 - numpy == 1.26.4
 - scikit-learn == 1.4.1
 - torch == 2.2.2
 - torch-geometric == 2.5.2
