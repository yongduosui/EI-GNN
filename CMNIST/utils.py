import argparse
import numpy as np
import torch
import random
from torch_geometric.loader import DataLoader
from ogb.graphproppred import Evaluator
from GOOD.data.good_datasets.good_cmnist import GOODCMNIST
from GOOD.data.good_datasets.good_motif import GOODMotif
from GOOD.data.good_datasets.good_hiv import GOODHIV
from GOOD.data.good_datasets.good_pcba import GOODPCBA
def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        print(arg + '.' * (str_num - len(arg) - len(str(val))) + str(val))
    print()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.CEX = False

def load_data(args):
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.BCEWithLogitsLoss()
    if args.dataset == "cmnist":
        dataset, meta_info = GOODCMNIST.load(args.data_dir, domain='color', shift='covariate', generate=False)
        num_class = 10
        num_layer = 5
        in_dim = 3
        eval_metric = "rocauc"
        cri = criterion1
        eval_name = None
        test_batch_size = args.batch_size

    elif args.dataset == "motif":
        dataset, meta_info = GOODMotif.load(args.data_dir, domain=args.domain, shift='concept', generate=False)
        num_class = 3
        num_layer = 3
        in_dim = 1
        eval_metric = "rocauc"
        cri = criterion1
        eval_name = None
        test_batch_size = args.batch_size

    elif args.dataset == "hiv":
        dataset, meta_info = GOODHIV.load(args.data_dir, domain=args.domain, shift='concept', generate=False)
        num_class = 1
        num_layer = 3
        in_dim = 9
        eval_metric = "rocauc"
        cri = criterion2
        eval_name = "ogbg-molhiv"
        test_batch_size = 256

    elif args.dataset == "pcba":

        dataset, meta_info = GOODPCBA.load(args.data_dir, domain=args.domain, shift='concept', generate=False)
        in_dim = 9
        num_class = 128
        num_layer = 5
        eval_metric = "ap"
        cri = criterion2
        eval_name = "ogbg-molpcba"
        test_batch_size = 4096
    else:
        assert False
    print(meta_info)
    return dataset, meta_info, in_dim, num_class, num_layer, eval_metric, cri, eval_name, test_batch_size




def arg_parse():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--data_dir',  type=str, default='../dataset', help='dataset dir')
    parser.add_argument('--seed',      type=int,   default=666)
    parser.add_argument('--emb_dim',    type=int,   default=300)
    parser.add_argument('--test_epoch', type=int, default=10)
    parser.add_argument('--layer', type=int, default=-1)
    parser.add_argument('--cau_gamma', type=float, default=0.5)
    parser.add_argument('--inv',    type=float,     default=0.5)
    parser.add_argument('--equ',    type=float,     default=0.5)
    parser.add_argument('--reg',    type=float,     default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr_decay', type=int, default=30)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--eta_min', type=float, default=1e-4)
    parser.add_argument('--lr_scheduler',  type=str, default='cos')
    parser.add_argument('--milestones', nargs='+', type=int, default=[40,60,80])
    parser.add_argument('--lr',     type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--l2reg',           type=float, default=5e-6, help='L2 norm (default: 5e-6)')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--domain', type=str, default='color', help='color, background')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default="motif")
    parser.add_argument('--addlamb', type=float, default=0.5)
    parser.add_argument('--trails', type=int, default=10, help='number of runs (default: 10)')   
    parser.add_argument('--single_linear', type=str2bool, default=False)
    parser.add_argument('--equ_rep', type=str2bool, default=False)
    parser.add_argument('--one_dim', type=str2bool, default=False)
    # parser.add_argument('--random_add', type=str, default="everyadd")
    args = parser.parse_args()
    return args