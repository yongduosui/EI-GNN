"""
The GOOD-Motif dataset motivated by `Spurious-Motif
<https://arxiv.org/abs/2201.12872>`_.
"""
import math
import os
import os.path as osp
import random

import gdown
import torch
from munch import Munch
from torch_geometric.data import InMemoryDataset, extract_zip
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from GOOD import register
from GOOD.utils.synthetic_data.BA3_loc import *
from GOOD.utils.synthetic_data import synthetic_structsim
import pdb
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from torch_geometric.data.separate import separate
from torch_geometric.data.dataset import Dataset, IndexType
import copy

@register.dataset_register
class GOODMotif_mix(InMemoryDataset):
    r"""
    The GOOD-Motif dataset motivated by `Spurious-Motif
    <https://arxiv.org/abs/2201.12872>`_.

    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'basis' and 'size'.
        shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
        subset (str): The split set. Allowed: 'train', 'id_val', 'id_test', 'val', and 'test'. When shift='no_shift',
            'id_val' and 'id_test' are not applicable.
        generate (bool): The flag for regenerating dataset. True: regenerate. False: download.
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False, rand_permute=False):

        self.name = self.__class__.__name__
        self.domain = domain
        self.metric = 'Accuracy'
        self.task = 'Multi-label classification'
        self.url = 'https://drive.google.com/file/d/15YRuZG6wI4HF7QgrLI52POKjuObsOyvb/view?usp=sharing'

        self.generate = generate
        self.rand_permute = rand_permute

        self.all_basis = ["wheel", "tree", "ladder", "star", "path"]
        self.basis_role_end = {'wheel': 0, 'tree': 0, 'ladder': 0, 'star': 1, 'path': 1}
        self.all_motifs = [[["house"]], [["dircycle"]], [["crane"]]]
        self.num_data = 30000
        self.train_spurious_ratio = [0.99, 0.97, 0.95]

        super().__init__(root, transform, pre_transform)
        if shift == 'covariate':
            subset_pt = 3
        elif shift == 'concept':
            subset_pt = 8
        elif shift == 'no_shift':
            subset_pt = 0
        else:
            raise ValueError(f'Unknown shift: {shift}.')
        if subset == 'train':
            subset_pt += 0
        elif subset == 'val':
            subset_pt += 1
        elif subset == 'test':
            subset_pt += 2
        elif subset == 'id_val':
            subset_pt += 3
        else:
            subset_pt += 4
        
        data_path = self.processed_paths[subset_pt]
        
        new_file_name = ""
        file_name = data_path.split("/")
        for key in file_name:
            if "mix" in key:
                key = "GOODMotif"
            new_file_name += key
            if ".pt" not in key:
                new_file_name += "/"
        self.data, self.slices = torch.load(new_file_name)
        
    @property
    def raw_dir(self):
        return osp.join(self.root)


    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, self.domain, 'processed')

    @property
    def processed_file_names(self):
        return ['no_shift_train.pt', 'no_shift_val.pt', 'no_shift_test.pt',
                'covariate_train.pt', 'covariate_val.pt', 'covariate_test.pt', 'covariate_id_val.pt',
                'covariate_id_test.pt',
                'concept_train.pt', 'concept_val.pt', 'concept_test.pt', 'concept_id_val.pt', 'concept_id_test.pt']

    def gen_data(self, basis_id, width_basis, motif_id):
        basis_type = self.all_basis[basis_id]
        if basis_type == 'tree':
            width_basis = int(math.log2(width_basis)) - 1
            if width_basis <= 0:
                width_basis = 1
        list_shapes = self.all_motifs[motif_id]
        G, role_id, _ = synthetic_structsim.build_graph(
            width_basis, basis_type, list_shapes, start=0, rdm_basis_plugins=True
        )
        G = perturb([G], 0.05, id=role_id)[0]
        # from GOOD.causal_engine.graph_visualize import plot_graph
        # print(G.edges())
        # plot_graph(G, colors=[1 for _ in G.nodes()])

        # --- Convert networkx graph into pyg data ---
        data = from_networkx(G)
        data.x = torch.ones((data.num_nodes, 1))
        role_id = torch.tensor(role_id, dtype=torch.long)
        role_id[role_id <= self.basis_role_end[basis_type]] = 0
        role_id[role_id != 0] = 1

        edge_gt = torch.stack([role_id[data.edge_index[0]], role_id[data.edge_index[1]]]).sum(0) > 1.5

        data.node_gt = role_id
        data.edge_gt = edge_gt
        data.basis_id = basis_id
        data.motif_id = motif_id

        # --- noisy labels ---
        if random.random() < 0.1:
            data.y = random.randint(0, 2)
        else:
            data.y = motif_id

        return data

    def get(self, idx: int) -> Data:

        # if self.len() == 1:
        #     return copy.copy(self.data)

        # if not hasattr(self, '_data_list') or self._data_list is None:
        #     self._data_list = self.len() * [None]
        # elif self._data_list[idx] is not None:
        #     return copy.copy(self._data_list[idx])

        num_data = self.len()
        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False)
        label = data.y
        all_label = self.data.y
        sample_idx_list = (all_label!=label).nonzero().squeeze().t().tolist()
        idx2 = random.sample(sample_idx_list, 1)[0]
        
        data2 = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx2,
            slice_dict=self.slices,
            decrement=False) 
        pdb.set_trace()
        if self.rand_permute:
            pdb.set_trace()
            data2 = rand_permute(data2)
        return data, data2



def rand_permute(data):
    pdb.set_trace()
    N = data.x.shape[0]
    perm = torch.randperm(N)
    inv_perm = perm.new_empty(N)
    inv_perm[perm] = torch.arange(N)
    data.x = data.x[inv_perm]
    data.edge_index = perm[data.edge_index]
    return data


