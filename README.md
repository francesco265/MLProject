# Machine Learning project 22/23
## Francesco Bottalico (787587)
## Importing

Code for the link prediction task on graphs using autoencoders.
`link_prediction` is the python package that can be easily imported by doing

``` python
import link_prediction
```

## Notebook
The code in this repository is the same present in the Colab Notebook, which was used to run the experiments (The notebook is in the _notebook_ folder or [here](https://colab.research.google.com/drive/1q-iMFNmc-2rKBgVWhGUVEeHg-jwnyoqq?usp=sharing)).

## Dependencies
The following dependencies are needed in order to use the package:
- scikit-learn
- dgl
- torch
- numpy
- scipy

## Features
The package exposes the model's classes used during the experiments:
- Graph Convolution AutoEncoder (GCAE)
- Graph Normalized Convolution AutoEncoder (GNCAE)
- Graph ATtention AutoEncoder (GATAE)
- Graph Transformer AutoEncoder (GTAE)

And the main functions to train them using k-fold:
- `train_kfold` for simple training
- `train_contrastive_kfold` for contrastive training
