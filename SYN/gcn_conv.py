
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros
import pdb

class GCNConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 edge_norm=True,
                 gfn=False):
        super(GCNConv, self).__init__('add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.edge_norm = edge_norm
        self.gfn = gfn
        self.message_mask = None
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)
        
        edge_weight = edge_weight.view(-1)
        
        
        assert edge_weight.size(0) == edge_index.size(1)
        
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        # Add edge_weight for loop edges.
        loop_weight = torch.full((num_nodes, ),
                                 1 if not improved else 2,
                                 dtype=edge_weight.dtype,
                                 device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        # pdb.set_trace()
        x = torch.matmul(x, self.weight)
        if self.gfn:
            return x
    
        if not self.cached or self.cached_result is None:
            if self.edge_norm:
                edge_index, norm = GCNConv.norm(
                    edge_index, 
                    x.size(0), 
                    edge_weight, 
                    self.improved, 
                    x.dtype)
            else:
                norm = None
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):

        if self.edge_norm:
            return norm.view(-1, 1) * x_j
        else:
            return x_j
        
    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class GINConv(MessagePassing):
    def __init__(self, in_dim, emb_dim):
        
        super(GINConv, self).__init__(aggr = "add")
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, 2*emb_dim),
                                       torch.nn.BatchNorm1d(2*emb_dim), 
                                       torch.nn.ReLU(), 
                                       torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_weight=None):
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_weight=edge_weight))
        return out
    def message(self, x_j, edge_weight=None):
        if edge_weight is not None:
            mess = F.relu(x_j * edge_weight)
        else:
            mess = F.relu(x_j)
        return mess

    def update(self, aggr_out):
        return aggr_out

class GATConv(MessagePassing):
	def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add", input_layer=False):
		super(GATConv, self).__init__()
		self.aggr = aggr

		self.emb_dim = emb_dim
		self.heads = heads
		self.negative_slope = negative_slope
    
		self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
		self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

		self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

		### Mapping 0/1 edge features to embedding
		self.edge_encoder = torch.nn.Linear(9, heads * emb_dim)

		### Mapping uniform input features to embedding.
		self.input_layer = input_layer
		if self.input_layer:
			self.input_node_embeddings = torch.nn.Embedding(2, emb_dim)
			torch.nn.init.xavier_uniform_(self.input_node_embeddings.weight.data)

		self.reset_parameters()

	def reset_parameters(self):
		glorot(self.att)
		zeros(self.bias)

	def forward(self, x, edge_index, edge_weight=None):
		# add self loops in the edge space
		edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

		# add features corresponding to self-loop edges.
		# self_loop_attr = torch.zeros(x.size(0), 9)
		# self_loop_attr[:, 7] = 1  # attribute for self-loop edge
		# self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
		# edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

		if self.input_layer:
			x = self.input_node_embeddings(x.to(torch.int64).view(-1, ))

		x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
		return self.propagate(self.aggr, edge_index, x=x, edge_weight=edge_weight)

	def message(self, edge_index, x_i, x_j, edge_weight=None):

		alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

		if edge_weight is not None:
			alpha = F.leaky_relu(alpha * edge_weight, self.negative_slope)
		else:
			alpha = F.leaky_relu(alpha, self.negative_slope)

		
		alpha = softmax(alpha, edge_index[0])

		return x_j * alpha.view(-1, self.heads, 1)

	def update(self, aggr_out):
		aggr_out = aggr_out.mean(dim=1)
		aggr_out = aggr_out + self.bias

		return aggr_out