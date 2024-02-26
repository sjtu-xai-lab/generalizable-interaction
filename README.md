# Generalizable interactions
This repository contains the Python implementation of the paper "Defining and extracting generalizable interaction primitives from DNNs", ICLR 2024. The paper develops a new method to extract generalizable interactions shared by different DNNs (see [papers](https://arxiv.org/abs/2401.16318) for details and citations).

## Install
Generalizable interactions shared by different DNNs can be installed in the Python 3 environment:

```
pip3 install git+https://github.com/sjtu-xai-lab/generalizable-interaction
```


## How to use 
### Given two DNNs trained for the same task
- Sentiment classification with language models
```
python train.py
```
or directly access the pre-trained BERT-base and BERT-large models, which were fine-tined on the SST-2 dataset in [Google Drive](https://drive.google.com/file/d/18NWWTXVvs6izdjj3fbEPNgsZqQw61WuA/view?usp=sharing). You can download `pretrained_model.zip` and unzip it into path like ```./pretrained_model/{DATASET}/{model}.pth```.

- Dialogue task with large language models

Directly access the pre-trained LLaMA and OPT-1.3B models from ???. You can download to the path like ```./pretrained_model/{DATASET}/{model}.pth```.


- Image classification task with vision models

Directly access the pre-trained ResNet-20 and VGG-16 models, which were trained on the MNIST dataset in [Google Drive](https://drive.google.com/file/d/18NWWTXVvs6izdjj3fbEPNgsZqQw61WuA/view?usp=sharing). You can download `pretrained_model.zip` and unzip it into path like ```./pretrained_model/{DATASET}/{model}.pth```.




### Computing generalizable interactions shared by DNNs
To compute generalizable interactions shared by two DNNs, you can use codes like the following:

```
python eval_interaction_sst2.py  --interaction='generalizable'  --model_path_1='BERT-base.pth' --model_path_2='BERT-large.pth'
```

then you can get all the AND-OR interactions in path `????`.




## More details

In contrast, to compute traditional interactions extracted by a DNN ([github](https://github.com/sjtu-xai-lab/interaction-concept/tree/main)), you can use codes like the following:

```
python eval_interaction_sst2.py  --interaction='traditional'  --model_path='BERT-base.pth' 
```
then you can get all the AND-OR interactions in path `????`.



## Sample notebooks



## Citations
```
@InProceedings{chen2024defining,
  title = {Defining and extracting generalizable interaction primitives from DNNs},
  author = {Lu, Chen and Siyu, Lou and Benhao, Huang and Quanshi, Zhang},
  booktitle = {The Twelfth International Conference on Learning Representations},
  year = {2024}
}
```
