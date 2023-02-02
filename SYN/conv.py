import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GATConv
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree
from torch_scatter import scatter_add
from torch_geometric.nn.inits import reset
from gcn_conv import GCNConv, GINConv
import math
import pdb


class GCNEncoder(torch.nn.Module):
    def __init__(self, num_layer, in_dim, emb_dim, dropout_rate=0.5):
        super(GCNEncoder, self).__init__()
        self.num_layer = num_layer
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate

        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(num_layer - 1)])
        self.batch_norm1 = nn.BatchNorm1d(in_dim)
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer - 1)])
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropouts = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(num_layer - 1)])
        self.conv1 = GCNConv(in_dim, emb_dim, gfn=True)
        self.convs = nn.ModuleList([GCNConv(emb_dim, emb_dim) for _ in range(num_layer - 1)])

    def forward(self, x, edge_index, m_node=None, m_edge=None):
        if m_node is not None:
            x = x * m_node
        post_conv = self.conv1(self.batch_norm1(x), edge_index, m_edge)
        if self.num_layer > 1:
            post_conv = self.relu1(post_conv)
            post_conv = self.dropout1(post_conv)
        for i, (conv, batch_norm, relu, dropout) in enumerate(zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = conv(batch_norm(post_conv), edge_index, m_edge)
            if i != len(self.convs) - 1:            # not for final layer
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
        return post_conv


class GINEncoder(torch.nn.Module):
    def __init__(self, num_layer, in_dim, emb_dim, dropout_rate):
        super(GINEncoder, self).__init__()
        self.num_layer = num_layer
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate

        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(num_layer - 1)])
        self.batch_norm1 = nn.BatchNorm1d(emb_dim)
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer - 1)])
        self.dropout1 = nn.Dropout(self.dropout_rate) # 
        self.dropouts = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(num_layer - 1)]) #
        self.conv1 = GINConv(in_dim, emb_dim)  
        self.convs = nn.ModuleList([ (emb_dim, emb_dim) for _ in range(num_layer - 1)])

    def forward(self, x, edge_index, m_node=None, m_edge=None):
        if m_node is not None:
            x = x * m_node
        post_conv = self.batch_norm1(self.conv1(x, edge_index, m_edge))
        if self.num_layer > 1:
            post_conv = self.relu1(post_conv)
            post_conv = self.dropout1(post_conv)
        for i, (conv, batch_norm, relu, dropout) in enumerate(zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = conv(batch_norm(post_conv), edge_index, m_edge)
            if i != len(self.convs) - 1:            # not for final layer
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
        return post_conv


class GATEncoder(torch.nn.Module):
    def __init__(self, num_layer, in_dim, emb_dim, head=4, dropout_rate=0.5):
       
        super(GATEncoder, self).__init__()
        self.num_layer = num_layer
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.dropout_rate = dropout_rate

        self.relu1 = nn.ReLU()
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(num_layer - 1)])
        self.batch_norm1 = nn.BatchNorm1d(in_dim)
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer - 1)])
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.dropouts = nn.ModuleList([nn.Dropout(self.dropout_rate) for _ in range(num_layer - 1)])
        self.conv1 = GCNConv(in_dim, emb_dim, gfn=True)
        # self.convs = nn.ModuleList([GCNConv(emb_dim, emb_dim) for _ in range(num_layer - 1)])
        # self.convs = torch.nn.ModuleList()
        # self.convs = nn.ModuleList([GATConv(emb_dim, heads=head) for _ in range(num_layer - 1)])
        self.convs = nn.ModuleList()
        for _ in range(num_layer-1):
            self.convs.append(GATConv(emb_dim, int(emb_dim / head), heads=head, dropout=dropout_rate))
        # for i in range(self.num_layer):
        #     self.bns_conv.append(BatchNorm1d(emb_dim))
        #     self.convs.append(GATConv(emb_dim, int(emb_dim / head), heads=head, dropout=dropout_rate))

    def forward(self, x, edge_index, m_node=None, m_edge=None):
        if m_node is not None:
            x = x * m_node
        post_conv = self.conv1(self.batch_norm1(x), edge_index, m_edge)
        if self.num_layer > 1:
            post_conv = self.relu1(post_conv)
            post_conv = self.dropout1(post_conv)
        
        for i, (conv, batch_norm, relu, dropout) in enumerate(zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # post_conv = conv(batch_norm(post_conv), edge_index, m_edge)
            post_conv = conv(batch_norm(post_conv), edge_index)
            if i != len(self.convs) - 1:            # not for final layer
                post_conv = relu(post_conv)
            post_conv = dropout(post_conv)
        return post_conv


class GraphMasker(torch.nn.Module):

    def __init__(self, model_name, num_layer, in_dim, emb_dim, dropout_rate=0.5):
        super(GraphMasker, self).__init__()

        if model_name =="EIGCN2":
            self.gnn_encoder = GCNEncoder(num_layer, in_dim, emb_dim, dropout_rate)
        elif model_name =="EIGIN2":
            self.gnn_encoder = GINEncoder(num_layer, in_dim, emb_dim, dropout_rate)
        elif model_name =="EIGAT2":
            self.gnn_encoder = GATEncoder(num_layer=num_layer, in_dim=in_dim,  emb_dim=emb_dim, head=4, dropout_rate=dropout_rate)
        else:
            assert False
        self.edge_att_mlp = nn.Linear(emb_dim * 2, 2)
        self.node_att_mlp = nn.Linear(emb_dim, 2)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gnn_encoder)
        reset(self.edge_att_mlp)
        reset(self.node_att_mlp)

    def forward(self, data, node_rep):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        # node_rep = self.gnn_encoder(x, edge_index)

        size = batch[-1].item() + 1
        row, col = edge_index
        edge_rep = torch.cat([node_rep[row], node_rep[col]], dim=-1)
        # node_key = torch.sigmoid(self.node_att_mlp(node_rep))
        # edge_key = torch.sigmoid(self.edge_att_mlp(edge_rep))
        node_key = F.softmax(self.node_att_mlp(node_rep))[:,0:1]
        edge_key = F.softmax(self.edge_att_mlp(edge_rep))[:,0:1]
        node_key_num, node_env_num, non_zero_node_ratio = self.reg_mask(node_key, batch, size)
        edge_key_num, edge_env_num, non_zero_edge_ratio = self.reg_mask(edge_key, batch[edge_index[0]], size)

        self.non_zero_node_ratio = non_zero_node_ratio
        self.non_zero_edge_ratio = non_zero_edge_ratio

        output = {"node_key": node_key, "edge_key": edge_key,
                  "node_key_num": node_key_num, 
                  "node_env_num": node_env_num,
                  "edge_key_num": edge_key_num, 
                  "edge_env_num": edge_env_num,
                  "non_zero_node_ratio": non_zero_node_ratio,
                  "non_zero_edge_ratio": non_zero_edge_ratio}
        return output
    
    def reg_mask(self, mask, batch, size):

        # 对属于同一graph的node mask值相加
        key_num = scatter_add(mask, batch, dim=0, dim_size=size)            
        env_num = scatter_add((1 - mask), batch, dim=0, dim_size=size)
        non_zero_mask = scatter_add((mask > 0).to(torch.float32), batch, dim=0, dim_size=size) # 判断非零数目
        all_mask = scatter_add(torch.ones_like(mask).to(torch.float32), batch, dim=0, dim_size=size)
        non_zero_ratio = non_zero_mask / all_mask       # 判断非零比例

        return key_num + 1e-8, env_num + 1e-8, non_zero_ratio


class GraphSynMasker(torch.nn.Module):

    def __init__(self, model_name, num_layer, in_dim, emb_dim, dropout_rate=0.5):
        super(GraphSynMasker, self).__init__()

        if model_name =="EIGCN":
            self.gnn_encoder = GCNEncoder(num_layer, in_dim, emb_dim, dropout_rate)
        elif model_name =="EIGIN":
            self.gnn_encoder = GINEncoder(num_layer, in_dim, emb_dim, dropout_rate)
        elif model_name =="EIGAT":
            self.gnn_encoder = GATEncoder(num_layer, in_dim, emb_dim, dropout_rate)
        else:
            assert False
        self.edge_att_mlp = nn.Linear(emb_dim * 2, 1)
        self.node_att_mlp = nn.Linear(emb_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.gnn_encoder)
        reset(self.edge_att_mlp)
        reset(self.node_att_mlp)

    def forward(self, data):

        x = data.x if data.x is not None else data.feat
        edge_index, batch = data.edge_index, data.batch
        
        node_rep = self.gnn_encoder(x, edge_index)

        size = batch[-1].item() + 1
        row, col = edge_index
        edge_rep = torch.cat([node_rep[row], node_rep[col]], dim=-1)
        node_key = torch.sigmoid(self.node_att_mlp(node_rep))
        edge_key = torch.sigmoid(self.edge_att_mlp(edge_rep))

        node_key_num, node_env_num, non_zero_node_ratio = self.reg_mask(node_key, batch, size)
        edge_key_num, edge_env_num, non_zero_edge_ratio = self.reg_mask(edge_key, batch[edge_index[0]], size)

        self.non_zero_node_ratio = non_zero_node_ratio
        self.non_zero_edge_ratio = non_zero_edge_ratio

        output = {"node_key": node_key, "edge_key": edge_key,
                  "node_key_num": node_key_num, 
                  "node_env_num": node_env_num,
                  "edge_key_num": edge_key_num, 
                  "edge_env_num": edge_env_num,
                  "non_zero_node_ratio": non_zero_node_ratio,
                  "non_zero_edge_ratio": non_zero_edge_ratio}
        return output
    
    def reg_mask(self, mask, batch, size):

        # 对属于同一graph的node mask值相加
        key_num = scatter_add(mask, batch, dim=0, dim_size=size)            
        env_num = scatter_add((1 - mask), batch, dim=0, dim_size=size)
        non_zero_mask = scatter_add((mask > 0).to(torch.float32), batch, dim=0, dim_size=size) # 判断非零数目
        all_mask = scatter_add(torch.ones_like(mask).to(torch.float32), batch, dim=0, dim_size=size)
        non_zero_ratio = non_zero_mask / all_mask       # 判断非零比例

        return key_num + 1e-8, env_num + 1e-8, non_zero_ratio

