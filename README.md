# Generalizable interactions
This repository contains the Python implementation of the paper "Defining and extracting generalizable interaction primitives from DNNs", ICLR 2024. The paper develops a new method to extract generalizable interactions shared by different DNNs (see [papers](https://arxiv.org/abs/2401.16318) for details and citations).

## Install
Generalizable interactions can be installed in the Python 3 environment:

```
pip3 install git+https://github.com/csluchen/generalizable-interaction
```


## How to use 
### Given two DNNs trained for the same task
- Sentiment Classification on the SST-2 dataset
```
python train.py
```

or directly access the pre-trained BERT-base and BERT-large models, which were fine-tined on SST-2 dataset in [Google Drive](https://drive.google.com/file/d/18NWWTXVvs6izdjj3fbEPNgsZqQw61WuA/view?usp=sharing). You can download `pretrained_model.zip` and unzip it into path like ```./pretrained_model/{DATASET}/{model}.pth```.

### Compute generalizable interactions shared by DNNs






## Sample notebooks

For image dataset, we provide a Jupyter notebook for the `CIFAR-10` and `MNIST` dataset for calculating Shapley values via HarsanyiNet under ```notebooks/CIFAR-10.ipynb``` and ```notebooks/MNIST.ipynb```, respectively.

For tabular dataset, we provide a Jupyter notebook for the `Census` dataset for calculating Shapley values via HarsanyiNet under ```notebooks/Census.ipynb```


## Citations
```
@InProceedings{chen2024defining,
  title = {Defining and extracting generalizable interaction primitives from DNNs},
  author = {Lu, Chen and Siyu, Lou and Benhao, Huang and Quanshi, Zhang},
  booktitle = {The Twelfth International Conference on Learning Representations},
  year = {2024}
}
```
