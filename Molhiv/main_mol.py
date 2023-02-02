import torch
from torch_geometric.loader import DataLoader
from ogb.graphproppred import Evaluator
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import numpy as np
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from utils import load_data, arg_parse, print_args, set_seed
from model import EIGNN_Mol
import pdb


def eval(model, evaluator, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(loader):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model.forward_causal(batch)
            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    input_dict = {"y_true": y_true, "y_pred": y_pred}
    output = evaluator.eval(input_dict)
    return output

def main(args):

    device = torch.device("cuda:" + str(args.device))
    
    dataset, meta_info, in_dim, num_class, num_layer, eval_metric, criterion, eval_name, test_batch_size = load_data(args)
    emb_dim = args.emb_dim
    # if args.layer != -1:
    #     num_layer = args.layer

    evaluator = Evaluator(eval_name)
    train_loader     = DataLoader(dataset["train_mix"],  batch_size=args.batch_size, shuffle=True)
    valid_loader_ood = DataLoader(dataset["val"],        batch_size=test_batch_size, shuffle=False)
    test_loader_ood  = DataLoader(dataset["test"],       batch_size=test_batch_size, shuffle=False)

    
    model = EIGNN_Mol(args=args, 
                        num_class=num_class, 
                        in_dim=in_dim, 
                        emb_dim=emb_dim, 
                        cau_gamma=args.cau_gamma, 
                        single_linear=args.single_linear,
                        equ_rep=args.equ_rep).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2reg)

    if args.lr_scheduler == 'step':
        scheduler = StepLR(optimizer, step_size=args.lr_decay, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'muti':
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_gamma)
    elif args.lr_scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        pass

    results = {'highest_valid': 0, 
               'update_test': 0,  
               'update_epoch': 0, }
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):

        start_time_local = time.time()
        total_loss_cau = 0
        total_loss_inv = 0
        total_loss_equ = 0
        total_loss_reg = 0
        total_loss = 0
        show = int(float(len(train_loader)) / 4.0)
        node_cau_list, edge_cau_list  = [], []
        for step, batch in enumerate(train_loader):
            
            batch1, batch2 = batch
            batch1, batch2 = batch1.to(device), batch2.to(device)
            
            model.train()
            optimizer.zero_grad()
            output_dict = model(batch1, batch2)

            env_num = (batch1.batch[-1] + 1) * 2
            one_hot_target = torch.cat([batch1.y, batch2.y])
            # one_hot_target_rep = one_hot_target.repeat_interleave(env_num, dim=0)

            loss_cau = criterion(output_dict["pred_cau"], one_hot_target)
            loss_inv = criterion(output_dict["pred_inv"], batch1.y)
            # if args.random_add =='shuffle':
            #     loss_inv = criterion(output_dict["pred_add"], one_hot_target)
            # elif args.random_add =='everyadd':
            #     loss_inv = criterion(output_dict["pred_add"], one_hot_target_rep)
            # else:
            #     assert False

            loss_reg = output_dict["loss_reg"]
            loss_equ = output_dict["loss_equ"]
            node_cau_list.append(output_dict["node_cau"])
            edge_cau_list.append(output_dict["edge_cau"])
 
            loss = loss_cau + loss_inv * args.inv + loss_equ * args.equ + loss_reg * args.reg
            loss.backward()

            optimizer.step()
            total_loss_cau += loss_cau.item()
            total_loss_inv += loss_inv.item()
            total_loss_equ += loss_equ.item()
            total_loss_reg += loss_reg.item()
            total_loss += loss.item()
            if step % show == 0:
                print("Train Iter:[{:<3}/{}], Total (CIER):[{:.4f} = {:.4f}+{:.4f}+{:.4f}+{:.4f}], Cau n/e:[{:.2f}/{:.2f}]"
                        .format(step, len(train_loader), 
                                total_loss / (step + 1),
                                total_loss_cau / (step + 1),
                                total_loss_inv / (step + 1),
                                total_loss_equ / (step + 1),
                                total_loss_reg / (step + 1),
                                np.mean(node_cau_list), 
                                np.mean(edge_cau_list)))

        epoch_loss_cau = total_loss_cau / len(train_loader)
        epoch_loss_inv = total_loss_inv / len(train_loader)
        epoch_loss_equ = total_loss_equ / len(train_loader)
        epoch_loss_reg = total_loss_reg / len(train_loader)
        epoch_loss = total_loss / len(train_loader)
        #################################### End training #####################################
        valid_acc = eval(model, evaluator, valid_loader_ood, device)[eval_metric] 
        test_acc  = eval(model, evaluator, test_loader_ood,  device)[eval_metric] 

        if valid_acc > results['highest_valid'] and epoch > args.test_epoch:
            results['highest_valid'] = valid_acc
            results['update_test'] = test_acc
            results['update_epoch'] = epoch

        if args.lr_scheduler in ["cos", "step", "muti"]:
            scheduler.step()

        print("-" * 150)
        epoch_time = (time.time()-start_time_local) / 60
        print("Epoch:[{}/{}], Total (cier):[{:.4f} = {:.4f}+{:.4f}+{:.4f}+{:.4f}], Val:[{:.2f}], Test:[{:.2f}] | Best Val:[{:.2f}] Update Test:[{:.2f}] at epoch:{} | time:{:.2f} min"
                        .format(epoch, args.epochs, 
                                epoch_loss,
                                epoch_loss_cau, 
                                epoch_loss_inv, 
                                epoch_loss_equ, 
                                epoch_loss_reg,
                                valid_acc* 100,
                                test_acc* 100, 
                                results['highest_valid']* 100,
                                results['update_test']* 100,
                                results['update_epoch'],
                                epoch_time))
        print("-" * 150)

    total_time = time.time() - start_time
    print("Update test:[{:.2f}] at epoch:[{}] | Total time:{}"
            .format(results['update_test']* 100,
                    results['update_epoch'],
                    time.strftime('%H:%M:%S', time.gmtime(total_time))))
    return results['update_test']


def config_and_run(args):
    
    print_args(args)
    set_seed(args.seed)
    final_test_acc_cau = []
    for _ in range(args.trails):
        args.seed += 10
        set_seed(args.seed)
        test_auc_cau = main(args)
        final_test_acc_cau.append(test_auc_cau)
    print("wsy: finall: Test ACC CAU: [{:.2f}±{:.2f}]".format(np.mean(final_test_acc_cau)* 100, np.std(final_test_acc_cau)* 100))

if __name__ == "__main__":
    args = arg_parse()
    config_and_run(args)
