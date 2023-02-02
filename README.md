# Equivariant and Invariant Cross-Data Augmentation for Generalizable Graph Classification

We provide a detailed code for EI-CDA.

## Installations

Main packages: PyTorch, Pytorch Geometric, OGB.

```
pytorch==1.10.1
torch-cluster==1.5.9
torch-geometric==2.0.3
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-spline-conv==1.2.1
ogb==1.3.5
typed-argument-parser==1.7.2
gdown==4.6.0
tensorboard==2.10.1
ruamel-yaml==0.17.21
cilog==1.2.3
munch==2.5.0
rdkit==2020.09.1.0
```



## Preparations

Please download the graph OOD datasets and OGB datasets as described in the original paper. 
Create a folder ```dataset```, and then put the datasets into ```dataset```. Then modify the path by specifying ```--data_dir your/path/dataset```.



## Commands

 We use the NVIDIA GeForce RTX 3090 (24GB GPU) to conduct all our experiments.

To run the code on Motif, please use the following command:

```
CUDA_VISIBLE_DEVICES=$GPU python -u main_syn.py \
--data_dir $DATA_DIR \
--trails 10 \
--lr 0.001 \
--domain basis \
--dataset motif \
--epochs 100 \
--batch_size 512 \
--emb_dim 300 \
--single_linear False \
--cau_gamma 0.5 \
--inv 0.5 \
--equ 0.5 \
--reg 0.5
```

 To run the code on CMNIST, please use the following command:

 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main_syn.py \
--data_dir $DATA_DIR \
--trails 10 \
--lr 0.001 \
--eta_min 1e-6 \
--dataset cmnist \
--epochs 100 \
--batch_size 256 \
--emb_dim 300 \
--single_linear False \
--cau_gamma 0.5 \
--inv 0.5 \
--equ 0.5 \
--reg 0.1
 ```

 To run the code on Molhiv, please use the following command:

 ```
CUDA_VISIBLE_DEVICES=$GPU python -u main_mol.py \
--data_dir $DATA_DIR \
--domain size --dataset hiv \
--trails 10 \
--batch_size 512 --epochs 100  \
--emb_dim 300 \
--lr 0.001 \
--cau_gamma 0.2 \
--inv 0.5 \
--equ 0.5 \
--reg 0.5

 ```



