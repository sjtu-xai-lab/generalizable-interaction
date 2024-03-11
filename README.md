# Generalizable interactions
This repository contains the Python implementation of the paper "Defining and extracting generalizable interaction primitives from DNNs", ICLR 2024. The paper develops a new method to extract generalizable interactions shared by different DNNs (see [papers](https://arxiv.org/abs/2401.16318) for details and citations).

## Install
Generalizable interactions shared by different DNNs can be installed in the Python 3 environment:

```
pip3 install git+https://github.com/sjtu-xai-lab/generalizable-interaction
```


## How to use 
### Given two DNNs trained for the same task
- Task 1: Sentiment classification with language models

Directly access the pre-trained BERT-base and BERT-large models, which were fine-tined on the SST-2 dataset in [Google Drive](https://drive.google.com/drive/folders/1-HV-To2EiCATUaaWsnLEIeLSilFtYxbA?usp=sharing). You can download `pretrained_model.zip` and unzip it into path like ```./pretrained_model/task1/...```.



### Computing generalizable interactions shared by DNNs
To compute generalizable interactions shared by two DNNs, you can use codes like the following:

```
python eval_interaction_text.py  --interaction='generalizable'  --model_path_1='BERT-base.pth.tar' --model_path_2='BERT-large.pth.tar'
```

then you can get AND-OR interactions for the first DNN at paths `./output/task1/generalizable/Iand_1.npy` and `./output/task1/generalizable/Ior_1.npy`, and AND-OR interactions for the second DNN at paths  `./output/task1/generalizable/Iand_2.npy` and `./output/task1/generalizable/Ior_2.npy`.



## More details

In contrast, to compute traditional interactions extracted by a DNN ([github](https://github.com/sjtu-xai-lab/interaction-concept/tree/main)), you can use codes like the following:

```
python eval_interaction_text.py  --interaction='traditional'  --model_path='BERT-base.pth.tar' --sparsify-lr=1e-6
```
then you can get AND-OR interactions for the DNN at paths `./output/task1/traditional/{model}/Iand.npy` and `./output/task1/traditional/{model}/Ior.npy`



## Sample notebooks
We provide Jupyter notebooks for all tasks to compare generalizable interactions with traditional interactions in the folder `notebooks`.



## Citations
```
@InProceedings{chen2024defining,
  title = {Defining and extracting generalizable interaction primitives from DNNs},
  author = {Lu, Chen and Siyu, Lou and Benhao, Huang and Quanshi, Zhang},
  booktitle = {The Twelfth International Conference on Learning Representations},
  year = {2024}
}
```
