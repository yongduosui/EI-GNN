import argparse
import numpy as np
import torch
import random
from torch_geometric.loader import DataLoader
from ogb.graphproppred import Evaluator
from torch_geometric.transforms import BaseTransform


def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()


def get_mix_dataset(train_set, rand_perm):

    num_data = train_set.__len__()
    mix_train_set = []
    idx_list = range(num_data)
    for i, data in enumerate(train_set):
        mix_data = []
        mix_data.append(data)
        label = data.y
        while True:

            idx = random.sample(idx_list, 1)[0]
            if train_set[idx].y == label:
                continue
            else:

                break
                
        data2 = train_set[idx]
        if rand_perm:
            data2 = rand_permute(data2)

        mix_data.append(data2)
        mix_train_set.append(mix_data)

    return mix_train_set

def rand_permute(data):

    N = data.x.shape[0]
    perm = torch.randperm(N)
    inv_perm = perm.new_empty(N)
    inv_perm[perm] = torch.arange(N)
    data.x = data.x[inv_perm]
    data.edge_index = perm[data.edge_index]
    return data

def arg_parse():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--data_dir',  type=str, default='./dataset', help='dataset dir')
    parser.add_argument('--initw_name', type=str, default='kaiming', choices=['default','orthogonal','normal','xavier','kaiming'], help='method name to initialize neural weights')
    parser.add_argument('--seed',      type=int,   default=666)
    parser.add_argument('--emb_dim',    type=int,   default=64)
    parser.add_argument('--test_epoch', type=int, default=10)
    parser.add_argument('--cau_gamma', type=float, default=0.5)
    parser.add_argument('--inv',    type=float,     default=0.5)
    parser.add_argument('--equ',    type=float,     default=0.5)
    parser.add_argument('--reg',    type=float,     default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--eta_min', type=float, default=1e-6)
    parser.add_argument('--lr_decay', type=int, default=150)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--lr_scheduler',  type=str, default='cos')
    parser.add_argument('--milestones', nargs='+', type=int, default=[40,60,80])
    parser.add_argument('--lr',     type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--l2reg',           type=float, default=5e-6, help='L2 norm (default: 5e-6)')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--domain', type=str, default='size')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default="ogbg-molbbbp")
    parser.add_argument('--addlamb', type=float, default=0.5)
    parser.add_argument('--trails', type=int, default=10, help='number of runs (default: 10)')   
    parser.add_argument('--single_linear', type=str2bool, default=False)
    parser.add_argument('--equ_rep', type=str2bool, default=False)
    parser.add_argument('--one_dim', type=str2bool, default=False)
    parser.add_argument('--size', type=str, default='ls')
    # parser.add_argument('--random_add', type=str, default="everyadd")
    args = parser.parse_args()
    return args

def get_info_dataset(args, dataset, split_idx):

    total = []
    for mode in ['train', 'valid', 'test']:
        mode_max_node = 0
        mode_min_node = 9999
        mode_avg_node = 0
        mode_tot_node = 0.0

        dataset_name = dataset[split_idx[mode]]
        mode_num_graphs = len(dataset_name)
        for data in dataset_name:
            num_node = data.num_nodes
            mode_tot_node += num_node
            if num_node > mode_max_node:
                mode_max_node = num_node
            if num_node < mode_min_node:
                mode_min_node = num_node
        print("{} {:<5} | Graphs num:{:<5} | Node num max:{:<4}, min:{:<4}, avg:{:.2f}"
            .format(args.dataset, mode, mode_num_graphs,
                                        mode_max_node,
                                        mode_min_node, 
                                        mode_tot_node / mode_num_graphs))
        total.append(mode_num_graphs)
    all_graph_num = sum(total)
    print("train:{:.2f}%, val:{:.2f}%, test:{:.2f}%"
        .format(float(total[0]) * 100 / all_graph_num, 
                float(total[1]) * 100 / all_graph_num, 
                float(total[2]) * 100 / all_graph_num))

def size_split_idx(dataset, mode):

    num_graphs = len(dataset)
    num_val   = int(0.1 * num_graphs)
    num_test  = int(0.1 * num_graphs)
    num_train = num_graphs - num_test - num_val

    num_node_list = []
    train_idx = []
    valtest_list = []

    for data in dataset:
        num_node_list.append(data.num_nodes)

    sort_list = np.argsort(num_node_list)

    if mode == 'ls':
        train_idx = sort_list[2 * num_val:]
        valid_test_idx = sort_list[:2 * num_val]
    else:
        train_idx = sort_list[:-2 * num_val]
        valid_test_idx = sort_list[-2 * num_val:]
    random.shuffle(valid_test_idx)
    valid_idx = valid_test_idx[:num_val]
    test_idx = valid_test_idx[num_val:]

    split_idx = {'train': torch.tensor(train_idx, dtype = torch.long), 
                 'valid': torch.tensor(valid_idx, dtype = torch.long), 
                 'test': torch.tensor(test_idx, dtype = torch.long)}
    return split_idx
    
 

class ToEnvs(BaseTransform):
    
    def __init__(self, envs=10):
        self.envs = envs

    def __call__(self, data):

        data.env_id = torch.randint(0, self.envs, (1,))
        return data