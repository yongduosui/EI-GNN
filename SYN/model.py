import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool, GINConv, GATConv
import torch.nn.functional as F
# from conv_syn import GINConv, GNNSynEncoder, GraphSynMasker
from conv import GCNEncoder, GINEncoder, GATEncoder, GraphSynMasker, GraphMasker
import pdb
import numpy as np
from functools import partial
from gcn_conv import GCNConv
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
import random


class EI(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, args,
                       num_class, 
                       in_dim,
                       emb_dim=300,
                       fro_layer=2,
                       bac_layer=2,
                       cau_layer=2,
                       dropout_rate=0.5,
                       cau_gamma=0.4,
                       single_linear=False):

        super(EI, self).__init__()
        self.args = args
        self.cau_gamma = args.cau_gamma
        self.dropout_rate = args.dropout_rate
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.single_linear = args.single_linear
        self.beta1 = 1
        self.beta2 = 1
        if args.model =="EIGCN2": 
            self.graph_front = GCNEncoder(fro_layer, in_dim,  emb_dim, dropout_rate)
            self.graph_backs = GCNEncoder(bac_layer, emb_dim, emb_dim, dropout_rate)
        elif args.model =="EIGIN2":
            self.graph_front = GINEncoder(fro_layer, in_dim,  emb_dim, dropout_rate)
            self.graph_backs = GINEncoder(bac_layer, emb_dim, emb_dim, dropout_rate)
        elif args.model =="EIGAT2":
            self.graph_front = GATEncoder(num_layer=fro_layer, in_dim=in_dim,  emb_dim=emb_dim, head=4, dropout_rate=dropout_rate)
            self.graph_backs = GATEncoder(num_layer=bac_layer, in_dim=emb_dim,  emb_dim=emb_dim, head=4, dropout_rate=dropout_rate)
        else:
            assert False
        self.causaler = GraphMasker(args.model, cau_layer, in_dim, emb_dim, dropout_rate)

        # self.pool = global_mean_pool
        self.pool = global_add_pool
        self.addlamb = args.addlamb
        if self.single_linear:
            self.predictor = nn.Linear(emb_dim, num_class)
        else:
            self.predictor = nn.Sequential(nn.Linear(emb_dim, 2*emb_dim), 
                                                 nn.BatchNorm1d(2*emb_dim), 
                                                 nn.ReLU(), 
                                                 nn.Dropout(), 
                                                 nn.Linear(2*emb_dim, num_class))

    def forward(self, batch1, batch2, epoch=0):

        x1 = batch1.x if batch1.x is not None else batch1.feat
        x2 = batch2.x if batch2.x is not None else batch2.feat
        edge_index1, batch_idx1, label1 = batch1.edge_index, batch1.batch, batch1.y
        edge_index2, batch_idx2, label2 = batch2.edge_index, batch2.batch, batch2.y

        x_encode1 = self.graph_front(x1, edge_index1)
        x_encode2 = self.graph_front(x2, edge_index2)

        # causaler_output1 = self.causaler(batch1)
        # causaler_output2 = self.causaler(batch2)
        causaler_output1 = self.causaler(batch1, x_encode1)
        causaler_output2 = self.causaler(batch2, x_encode2)

        node_cau1, edge_cau1 = causaler_output1["node_key"], causaler_output1["edge_key"]
        node_env1, edge_env1 = 1 - node_cau1, 1 - edge_cau1
        node_cau_num1, node_env_num1 = causaler_output1["node_key_num"], causaler_output1["node_env_num"]
        edge_cau_num1, edge_env_num1 = causaler_output1["edge_key_num"], causaler_output1["edge_env_num"]
        non_zero_node_ratio1, non_zero_edge_ratio1 = causaler_output1["non_zero_node_ratio"], causaler_output1["non_zero_edge_ratio"]

        node_cau2, edge_cau2 = causaler_output2["node_key"], causaler_output2["edge_key"]
        node_env2, edge_env2 = 1 - node_cau2, 1 - edge_cau2
        node_cau_num2, node_env_num2 = causaler_output2["node_key_num"], causaler_output2["node_env_num"]
        edge_cau_num2, edge_env_num2 = causaler_output2["edge_key_num"], causaler_output2["edge_env_num"]
        non_zero_node_ratio2, non_zero_edge_ratio2 = causaler_output2["non_zero_node_ratio"], causaler_output2["non_zero_edge_ratio"]

        x_encode_equ, x_encode_inv, edge_weight_equ, edge_weight_inv, edge_index_mix, batch_index, equ_lamb = self.graph_mixup(batch1, x_encode1, node_cau1, edge_cau1, node_env1, edge_env1, batch2, x_encode2, node_cau2, edge_cau2, node_env2, edge_env2)

        h_node_cau1 = self.graph_backs(x_encode1, edge_index1, node_cau1, edge_cau1)
        h_node_cau2 = self.graph_backs(x_encode2, edge_index2, node_cau2, edge_cau2)
        h_node_equ = self.graph_backs(x_encode_equ, edge_index_mix, m_edge=edge_weight_equ.unsqueeze(0).t())
        h_node_inv = self.graph_backs(x_encode_inv, edge_index_mix, m_edge=edge_weight_inv.unsqueeze(0).t())

        h_graph_cau1 = self.pool(h_node_cau1, batch_idx1)
        h_graph_cau2 = self.pool(h_node_cau2, batch_idx2)
        h_graph_equ = self.pool(h_node_equ, batch_index)
        h_graph_inv = self.pool(h_node_inv, batch_index)

        node_cau = torch.cat([node_cau1, node_cau2])
        edge_cau = torch.cat([edge_cau1, edge_cau2])
        h_graph_cauu = torch.cat([h_graph_cau1, h_graph_cau2])
        node_cau_num = torch.cat([node_cau_num1, node_cau_num2])
        node_env_num = torch.cat([node_env_num1, node_env_num2])
        edge_cau_num = torch.cat([edge_cau_num1, edge_cau_num2])
        edge_env_num = torch.cat([edge_env_num1, edge_env_num2])

        non_zero_node_ratio = torch.cat([non_zero_node_ratio1, non_zero_node_ratio2])
        non_zero_edge_ratio = torch.cat([non_zero_edge_ratio1, non_zero_edge_ratio2])

        pred_cau = self.predictor(h_graph_cauu)
        pred_equ = self.predictor(h_graph_equ)
        pred_inv = self.predictor(h_graph_inv)

        if epoch >= self.args.pretrain and epoch % self.args.frequency == 0:
            loss_equ = self.equiv_loss(equ_lamb, h_graph_equ, pred_equ, h_graph_cau1, h_graph_cau2, label1, label2)
        else:
            loss_equ = 0

        cau_node_reg = self.reg_mask_loss(node_cau_num, node_env_num, self.cau_gamma, non_zero_node_ratio)
        cau_edge_reg = self.reg_mask_loss(edge_cau_num, edge_env_num, self.cau_gamma, non_zero_edge_ratio)
        loss_reg = cau_node_reg + cau_edge_reg

        """
        Equivance loss is accurate if pretrain some epoch, we use the pretrain method to improve the precision of causal parts
        """
        
        output = {'pred_cau': pred_cau,
                  'pred_inv': pred_inv,
                  'loss_reg': loss_reg,
                  "loss_equ": loss_equ,
                  "node_cau": node_cau.mean().item(),
                  "edge_cau": edge_cau.mean().item()}

        return output

    def forward_causal(self, data):
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        x_encode = self.graph_front(x, edge_index)
        # causaler_output = self.causaler(data)
        causaler_output = self.causaler(data, x_encode)
        node_cau, edge_cau = causaler_output["node_key"], causaler_output["edge_key"]
        h_node_cau = self.graph_backs(x_encode, edge_index, node_cau, edge_cau)
        h_graph_cau = self.pool(h_node_cau, batch)
        pred_cau = self.predictor(h_graph_cau)
        return pred_cau


    def equiv_loss(self, lamb, h_graph_mix, pred_equ, h_graph_cau1, h_graph_cau2, label1, label2):

        device = label1.device
        batch_size = label1.shape[0]
        class_num = self.num_class
        one_hot_label1 = torch.zeros(batch_size, class_num).to(device).scatter_(1, label1.unsqueeze(1), 1)
        one_hot_label2 = torch.zeros(batch_size, class_num).to(device).scatter_(1, label2.unsqueeze(1), 1)
        mix_label = lamb.unsqueeze(1) * one_hot_label1 + (1 - lamb).unsqueeze(1) * one_hot_label2
        mix_logis = F.log_softmax(pred_equ, dim=-1)
        equ_loss = F.kl_div(mix_logis, mix_label.float(), reduction='batchmean')
        return equ_loss



    def graph_mixup(self, batch1, x_encode1, node_cau1, edge_cau1, node_env1, edge_env1, batch2, x_encode2, node_cau2, edge_cau2, node_env2, edge_env2):
        device = x_encode1.device
        DN = x_encode1.shape[1]
        data_list1 = batch1.to_data_list()
        data_list2 = batch2.to_data_list()
        num_graphs = len(data_list1)
        inv_lamb = self.addlamb
        equ_lamb = torch.from_numpy(np.random.beta(self.beta1, self.beta2, size=(num_graphs))).to(device)

        ptr = [0]
        batch_index = []
        for i, (data1, data2) in enumerate(zip(data_list1, data_list2)):
            if self.args.dataset == 'SYN':
                N = max(data1.feat.shape[0], data2.feat.shape[0])
            else:
                N = max(data1.x.shape[0], data2.x.shape[0])
            ptr.append(ptr[-1] + N)
            batch_index.append(torch.full((N,), i, dtype=torch.long))

        batch_index = torch.cat(batch_index, dim=0).to(device)
        edge_index_list1, edge_index_list2 = [], []
        x_pos1, x_pos2 = [], []
        x_f1, x_f2 = [], []
        x_ls1, x_ls2 = [], []
        e_ls1, e_ls2 = [], []

        for l, s, data1, data2 in zip(equ_lamb, ptr, data_list1, data_list2):
            if self.args.dataset == 'SYN':
                N1, N2 = data1.feat.shape[0], data2.feat.shape[0]
                x_f1.append(data1.feat)
                x_f2.append(data2.feat)
            else:
                N1, N2 = data1.x.shape[0], data2.x.shape[0]
                x_f1.append(data1.x)
                x_f2.append(data2.x)
            x_pos1.append(torch.arange(s, s+N1))
            x_pos2.append(torch.arange(s, s+N2))

            edge_index_list1.append(data1.edge_index + s)
            edge_index_list2.append(data2.edge_index + s)
            # lamb list for equ
            x_ls1.append(torch.full((N1,), l))
            x_ls2.append(torch.full((N2,), 1-l))
            if len(data1.edge_index) > 0:
                e_ls1.append(torch.full((edge_index_list1[-1].shape[1],), l))
            if len(data2.edge_index) > 0:
                e_ls2.append(torch.full((edge_index_list2[-1].shape[1],), 1-l))

        x_pos1 = torch.cat(x_pos1, dim=0)
        x_pos2 = torch.cat(x_pos2, dim=0)
        x_f1 = torch.cat(x_f1, dim=0)
        x_f2 = torch.cat(x_f2, dim=0)
        edge_index1 = torch.cat(edge_index_list1, dim=1)
        edge_index2 = torch.cat(edge_index_list2, dim=1)

        x_ls1 = torch.cat(x_ls1, dim=0).to(device)
        x_ls2 = torch.cat(x_ls2, dim=0).to(device)
        e_ls1 = torch.cat(e_ls1, dim=0).to(device)
        e_ls2 = torch.cat(e_ls2, dim=0).to(device)

        inv_x = torch.zeros((ptr[-1], DN)).to(device)
        equ_x = torch.zeros((ptr[-1], DN)).to(device)

        x1 = x_encode1 * node_cau1 * x_ls1.unsqueeze(0).t()
        x2 = x_encode2 * node_cau2 * x_ls2.unsqueeze(0).t()
        equ_x[x_pos1] += x1
        equ_x[x_pos2] += x2

        x3 = x_encode1 * node_cau1* inv_lamb
        x4 = x_encode2 * node_env2* (1 - inv_lamb)
        inv_x[x_pos1] += x3
        inv_x[x_pos2] += x4

        adj1 = torch.sparse_coo_tensor(edge_index1, e_ls1 * edge_cau1.view(-1), (ptr[-1], ptr[-1]))
        adj2 = torch.sparse_coo_tensor(edge_index2, e_ls2 * edge_cau2.view(-1), (ptr[-1], ptr[-1]))
        equ_adj = (adj1 + adj2).coalesce()
        edge_weight_equ = equ_adj.values()
        edge_index = equ_adj._indices()

        adj3 = torch.sparse_coo_tensor(edge_index1, inv_lamb * edge_cau1.view(-1), (ptr[-1], ptr[-1]))
        adj4 = torch.sparse_coo_tensor(edge_index2, (1 - inv_lamb) * edge_env2.view(-1), (ptr[-1], ptr[-1]))
        inv_adj = (adj3 + adj4).coalesce()
        edge_weight_inv = inv_adj.values()

        return equ_x, inv_x, edge_weight_equ, edge_weight_inv, edge_index, batch_index, equ_lamb

    def reg_mask_loss(self, key_mask, env_mask, gamma, non_zero_ratio):

        loss_reg =  torch.abs(key_mask / (key_mask + env_mask) - gamma * torch.ones_like(key_mask)).mean()
        loss_reg += (non_zero_ratio - gamma  * torch.ones_like(key_mask)).mean()
        return loss_reg


class EIGNN(torch.nn.Module):
    def __init__(self, args,
                       num_class, 
                       in_dim,
                       emb_dim=300,
                       fro_layer=2,
                       bac_layer=2,
                       cau_layer=2,
                       dropout_rate=0.5,
                       cau_gamma=0.4,
                       single_linear=False):

        super(EIGNN, self).__init__()

        self.args = args
        self.cau_gamma = args.cau_gamma
        self.dropout_rate = args.dropout_rate
        self.emb_dim = emb_dim
        self.num_class = num_class
        self.single_linear = args.single_linear
        self.beta1 = 1
        self.beta2 = 1
        if args.model =="EIGCN": 
            self.graph_front = GCNEncoder(fro_layer, in_dim,  emb_dim, args.dropout_rate)
            self.graph_backs = GCNEncoder(bac_layer, emb_dim, emb_dim, args.dropout_rate)
        elif args.model =="EIGIN":
            self.graph_front = GINEncoder(fro_layer, in_dim,  emb_dim, dropout_rate)
            self.graph_backs = GINEncoder(bac_layer, emb_dim, emb_dim, dropout_rate)
        elif args.model =="EIGAT":
            self.graph_front = GATEncoder(num_layer=fro_layer, in_dim=in_dim,  emb_dim=emb_dim, head=4, dropout_rate=dropout_rate)
            self.graph_backs = GATEncoder(num_layer=fro_layer, in_dim=emb_dim,  emb_dim=emb_dim, head=4, dropout_rate=dropout_rate)
        else:
            assert False
        self.causaler = GraphSynMasker(args.model, cau_layer, in_dim, emb_dim, dropout_rate)

        # self.pool = global_mean_pool
        self.pool = global_add_pool
        self.addlamb = args.addlamb
        if self.single_linear:
            self.predictor = nn.Linear(emb_dim, num_class)
        else:
            self.predictor = nn.Sequential(nn.Linear(emb_dim, 2*emb_dim), 
                                                 nn.BatchNorm1d(2*emb_dim), 
                                                 nn.ReLU(), 
                                                 nn.Dropout(), 
                                                 nn.Linear(2*emb_dim, num_class))

    def forward(self, batch1, batch2, epoch=0):

        x1 = batch1.x if batch1.x is not None else batch1.feat
        x2 = batch2.x if batch2.x is not None else batch2.feat
        edge_index1, batch_idx1, label1 = batch1.edge_index, batch1.batch, batch1.y
        edge_index2, batch_idx2, label2 = batch2.edge_index, batch2.batch, batch2.y

        x_encode1 = self.graph_front(x1, edge_index1)
        x_encode2 = self.graph_front(x2, edge_index2)

        causaler_output1 = self.causaler(batch1)
        causaler_output2 = self.causaler(batch2)

        node_cau1, edge_cau1 = causaler_output1["node_key"], causaler_output1["edge_key"]
        node_env1, edge_env1 = 1 - node_cau1, 1 - edge_cau1
        node_cau_num1, node_env_num1 = causaler_output1["node_key_num"], causaler_output1["node_env_num"]
        edge_cau_num1, edge_env_num1 = causaler_output1["edge_key_num"], causaler_output1["edge_env_num"]
        non_zero_node_ratio1, non_zero_edge_ratio1 = causaler_output1["non_zero_node_ratio"], causaler_output1["non_zero_edge_ratio"]

        node_cau2, edge_cau2 = causaler_output2["node_key"], causaler_output2["edge_key"]
        node_env2, edge_env2 = 1 - node_cau2, 1 - edge_cau2
        node_cau_num2, node_env_num2 = causaler_output2["node_key_num"], causaler_output2["node_env_num"]
        edge_cau_num2, edge_env_num2 = causaler_output2["edge_key_num"], causaler_output2["edge_env_num"]
        non_zero_node_ratio2, non_zero_edge_ratio2 = causaler_output2["non_zero_node_ratio"], causaler_output2["non_zero_edge_ratio"]

        x_encode_equ, x_encode_inv, edge_weight_equ, edge_weight_inv, edge_index_mix, batch_index, equ_lamb = self.graph_mixup(batch1, x_encode1, node_cau1, edge_cau1, node_env1, edge_env1, batch2, x_encode2, node_cau2, edge_cau2, node_env2, edge_env2)

        h_node_cau1 = self.graph_backs(x_encode1, edge_index1, node_cau1, edge_cau1)
        h_node_cau2 = self.graph_backs(x_encode2, edge_index2, node_cau2, edge_cau2)
        h_node_equ = self.graph_backs(x_encode_equ, edge_index_mix, m_edge=edge_weight_equ.unsqueeze(0).t())
        h_node_inv = self.graph_backs(x_encode_inv, edge_index_mix, m_edge=edge_weight_inv.unsqueeze(0).t())

        h_graph_cau1 = self.pool(h_node_cau1, batch_idx1)
        h_graph_cau2 = self.pool(h_node_cau2, batch_idx2)
        h_graph_equ = self.pool(h_node_equ, batch_index)
        h_graph_inv = self.pool(h_node_inv, batch_index)

        node_cau = torch.cat([node_cau1, node_cau2])
        edge_cau = torch.cat([edge_cau1, edge_cau2])
        h_graph_cauu = torch.cat([h_graph_cau1, h_graph_cau2])
        node_cau_num = torch.cat([node_cau_num1, node_cau_num2])
        node_env_num = torch.cat([node_env_num1, node_env_num2])
        edge_cau_num = torch.cat([edge_cau_num1, edge_cau_num2])
        edge_env_num = torch.cat([edge_env_num1, edge_env_num2])

        non_zero_node_ratio = torch.cat([non_zero_node_ratio1, non_zero_node_ratio2])
        non_zero_edge_ratio = torch.cat([non_zero_edge_ratio1, non_zero_edge_ratio2])

        pred_cau = self.predictor(h_graph_cauu)
        pred_equ = self.predictor(h_graph_equ)
        pred_inv = self.predictor(h_graph_inv)

        if epoch >= self.args.pretrain and epoch % self.args.frequency == 0:
            loss_equ = self.equiv_loss(equ_lamb, h_graph_equ, pred_equ, h_graph_cau1, h_graph_cau2, label1, label2)
        else:
            loss_equ = 0

        cau_node_reg = self.reg_mask_loss(node_cau_num, node_env_num, self.cau_gamma, non_zero_node_ratio)
        cau_edge_reg = self.reg_mask_loss(edge_cau_num, edge_env_num, self.cau_gamma, non_zero_edge_ratio)
        loss_reg = cau_node_reg + cau_edge_reg

        """
        Equivance loss is accurate if pretrain some epoch, we use the pretrain method to improve the precision of causal parts
        """
        
        output = {'pred_cau': pred_cau,
                  'pred_inv': pred_inv,
                  'loss_reg': loss_reg,
                  "loss_equ": loss_equ,
                  "node_cau": node_cau.mean().item(),
                  "edge_cau": edge_cau.mean().item()}

        return output

    def forward_causal(self, data):
        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        x_encode = self.graph_front(x, edge_index)
        causaler_output = self.causaler(data)
        node_cau, edge_cau = causaler_output["node_key"], causaler_output["edge_key"]
        h_node_cau = self.graph_backs(x_encode, edge_index, node_cau, edge_cau)
        h_graph_cau = self.pool(h_node_cau, batch)
        pred_cau = self.predictor(h_graph_cau)
        return pred_cau


    def equiv_loss(self, lamb, h_graph_mix, pred_equ, h_graph_cau1, h_graph_cau2, label1, label2):

        device = label1.device
        batch_size = label1.shape[0]
        class_num = self.num_class
        one_hot_label1 = torch.zeros(batch_size, class_num).to(device).scatter_(1, label1.unsqueeze(1), 1)
        one_hot_label2 = torch.zeros(batch_size, class_num).to(device).scatter_(1, label2.unsqueeze(1), 1)
        mix_label = lamb.unsqueeze(1) * one_hot_label1 + (1 - lamb).unsqueeze(1) * one_hot_label2
        mix_logis = F.log_softmax(pred_equ, dim=-1)
        equ_loss = F.kl_div(mix_logis, mix_label.float(), reduction='batchmean')
        return equ_loss



    def graph_mixup(self, batch1, x_encode1, node_cau1, edge_cau1, node_env1, edge_env1, batch2, x_encode2, node_cau2, edge_cau2, node_env2, edge_env2):
        device = x_encode1.device
        DN = x_encode1.shape[1]
        data_list1 = batch1.to_data_list()
        data_list2 = batch2.to_data_list()
        num_graphs = len(data_list1)
        inv_lamb = self.addlamb
        equ_lamb = torch.from_numpy(np.random.beta(self.beta1, self.beta2, size=(num_graphs))).to(device)

        ptr = [0]
        batch_index = []
        for i, (data1, data2) in enumerate(zip(data_list1, data_list2)):
            if self.args.dataset == 'SYN':
                N = max(data1.feat.shape[0], data2.feat.shape[0])
            else:
                N = max(data1.x.shape[0], data2.x.shape[0])
            ptr.append(ptr[-1] + N)
            batch_index.append(torch.full((N,), i, dtype=torch.long))

        batch_index = torch.cat(batch_index, dim=0).to(device)
        edge_index_list1, edge_index_list2 = [], []
        x_pos1, x_pos2 = [], []
        x_f1, x_f2 = [], []
        x_ls1, x_ls2 = [], []
        e_ls1, e_ls2 = [], []

        for l, s, data1, data2 in zip(equ_lamb, ptr, data_list1, data_list2):
            if self.args.dataset == 'SYN':
                N1, N2 = data1.feat.shape[0], data2.feat.shape[0]
                x_f1.append(data1.feat)
                x_f2.append(data2.feat)
            else:
                N1, N2 = data1.x.shape[0], data2.x.shape[0]
                x_f1.append(data1.x)
                x_f2.append(data2.x)
            x_pos1.append(torch.arange(s, s+N1))
            x_pos2.append(torch.arange(s, s+N2))

            edge_index_list1.append(data1.edge_index + s)
            edge_index_list2.append(data2.edge_index + s)
            # lamb list for equ
            x_ls1.append(torch.full((N1,), l))
            x_ls2.append(torch.full((N2,), 1-l))
            if len(data1.edge_index) > 0:
                e_ls1.append(torch.full((edge_index_list1[-1].shape[1],), l))
            if len(data2.edge_index) > 0:
                e_ls2.append(torch.full((edge_index_list2[-1].shape[1],), 1-l))

        x_pos1 = torch.cat(x_pos1, dim=0)
        x_pos2 = torch.cat(x_pos2, dim=0)
        x_f1 = torch.cat(x_f1, dim=0)
        x_f2 = torch.cat(x_f2, dim=0)
        edge_index1 = torch.cat(edge_index_list1, dim=1)
        edge_index2 = torch.cat(edge_index_list2, dim=1)

        x_ls1 = torch.cat(x_ls1, dim=0).to(device)
        x_ls2 = torch.cat(x_ls2, dim=0).to(device)
        e_ls1 = torch.cat(e_ls1, dim=0).to(device)
        e_ls2 = torch.cat(e_ls2, dim=0).to(device)

        inv_x = torch.zeros((ptr[-1], DN)).to(device)
        equ_x = torch.zeros((ptr[-1], DN)).to(device)

        x1 = x_encode1 * node_cau1 * x_ls1.unsqueeze(0).t()
        x2 = x_encode2 * node_cau2 * x_ls2.unsqueeze(0).t()
        equ_x[x_pos1] += x1
        equ_x[x_pos2] += x2

        x3 = x_encode1 * node_cau1* inv_lamb
        x4 = x_encode2 * node_env2* (1 - inv_lamb)
        inv_x[x_pos1] += x3
        inv_x[x_pos2] += x4

        adj1 = torch.sparse_coo_tensor(edge_index1, e_ls1 * edge_cau1.view(-1), (ptr[-1], ptr[-1]))
        adj2 = torch.sparse_coo_tensor(edge_index2, e_ls2 * edge_cau2.view(-1), (ptr[-1], ptr[-1]))
        equ_adj = (adj1 + adj2).coalesce()
        edge_weight_equ = equ_adj.values()
        edge_index = equ_adj._indices()

        adj3 = torch.sparse_coo_tensor(edge_index1, inv_lamb * edge_cau1.view(-1), (ptr[-1], ptr[-1]))
        adj4 = torch.sparse_coo_tensor(edge_index2, (1 - inv_lamb) * edge_env2.view(-1), (ptr[-1], ptr[-1]))
        inv_adj = (adj3 + adj4).coalesce()
        edge_weight_inv = inv_adj.values()

        return equ_x, inv_x, edge_weight_equ, edge_weight_inv, edge_index, batch_index, equ_lamb

    def reg_mask_loss(self, key_mask, env_mask, gamma, non_zero_ratio):

        loss_reg =  torch.abs(key_mask / (key_mask + env_mask) - gamma * torch.ones_like(key_mask)).mean()
        loss_reg += (non_zero_ratio - gamma  * torch.ones_like(key_mask)).mean()
        return loss_reg




