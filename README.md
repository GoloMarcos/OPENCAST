# One-Class Learning for Data Stream through Graph Neural Networks

# Abstract 
In many data stream applications, there is a normal concept, and the objective is to identify normal and abnormal concepts by training only with normal concept instances. This scenario is known in the literature as one-class learning (OCL) for data stream. In this OCL scenario for data stream, we highlight two main gaps: (i) lack of methods based on graph neural networks (GNNs) and (ii) lack of interpretable methods. We introduce OPENCAST (**O**ne-class gra**P**h auto**ENC**oder for d**A**ta **ST**ream), a new method for data stream based on OCL and GNNs. Our method learns representations while encapsulating the instances of interest through a hypersphere. OPENCAST learns low-dimensional representations to generate interpretability in the representation learning process. OPENCAST achieved state-of-the-art results for data streams in the OCL scenario, outperforming seven other methods. Furthermore, OPENCAST learns low-dimensional representations, generating interpretability in the representation learning process and results.

# Requiriments
 - python == 3.11.8
 - networkx == 3.2.1
 - pandas == 2.2.1
 - numpy == 1.26.4
 - scikit-learn == 1.4.1
 - torch == 2.2.2
 - torch-geometric == 2.5.2

# How to use
```
python3 main_opencast.py --k 1 --dataset electricity --numlayers 1

python3 main_baselines.py --dataset electricity --ocl IsoForest
```
