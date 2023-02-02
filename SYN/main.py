import torch
import os
import time
import warnings
warnings.filterwarnings('ignore')
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from model import EIGNN
import utils
from utils import load_data, arg_parse, print_args, set_seed
from train import train_eignn, eval
import pdb


def config_and_run(args):
    
    print_args(args)
    set_seed(args.seed)
    final_test_acc_cau = []
    for trail in range(1, args.trails+1):
        args.seed += 10
        set_seed(args.seed)
        test_auc_cau = main(args, trail)
        final_test_acc_cau.append(test_auc_cau)
    print("wsy: final: Test ACC CAU: [{:.2f}±{:.2f}]".format(np.mean(final_test_acc_cau), np.std(final_test_acc_cau)))



def main(args, trail):


    device = torch.device("cuda:" + str(args.device))
    dataset, meta_info, in_dim, num_class, num_layer, eval_metric, criterion, eval_name, test_batch_size = load_data(args) 

    if args.layer != -1:
        num_layer = args.layer
    model_func = utils.get_model(args)
    model = model_func(num_class, in_dim, args.emb_dim, dropout_rate=args.dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2reg)
    if args.lr_scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.lr_decay, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'muti':
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)
    else:
        pass
    
    # if args.model in ["GIN","GCN", "GAT"]:
    #     train_baseline(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    if args.model in ["EIGCN", "EIGIN", "EIGAT"]:
        return train_eignn(model=model, optimizer=optimizer, scheduler=scheduler, args=args, trail=trail)
    else:
        # assert False
        return train_eignn(model=model, optimizer=optimizer, scheduler=scheduler, args=args, trail=trail)
    # model = EIGNN_Syn(args=args, 
    #                     num_class=num_class, 
    #                     in_dim=in_dim, 
    #                     emb_dim=emb_dim, 
    #                     cau_gamma=args.cau_gamma, 
    #                     single_linear=args.single_linear,
    #                     equ_rep=args.equ_rep).to(device)








    # if args.model in ["GIN","GCN", "GAT"]:
    #     model_func = opts.get_model(args)

    #     train_baseline_syn(train_set, val_set, test_set, model_func=model_func, args=args)

    # elif args.model in ["CausalGCN", "CausalGIN", "CausalGAT"]:
    #     model_func = opts.get_model(args)

    #     train_causal_syn(train_set, val_set, test_set, model_func=model_func, args=args)

    # else:
    #     assert False

if __name__ == "__main__":

    args = utils.arg_parse()
    config_and_run(args)
    
