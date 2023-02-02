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
class GOODHIV_mix(InMemoryDataset):
    r"""
    The GOOD-HIV dataset. Adapted from `MoleculeNet
    <https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a>`_.

    Args:
        root (str): The dataset saving root.
        domain (str): The domain selection. Allowed: 'scaffold' and 'size'.
        shift (str): The distributional shift we pick. Allowed: 'no_shift', 'covariate', and 'concept'.
        subset (str): The split set. Allowed: 'train', 'id_val', 'id_test', 'val', and 'test'. When shift='no_shift',
            'id_val' and 'id_test' are not applicable.
        generate (bool): The flag for regenerating dataset. True: regenerate. False: download.
    """

    def __init__(self, root: str, domain: str, shift: str = 'no_shift', subset: str = 'train', transform=None,
                 pre_transform=None, generate: bool = False, rand_permute=False):

        self.name = self.__class__.__name__
        self.mol_name = 'HIV'
        self.domain = domain
        self.metric = 'ROC-AUC'
        self.task = 'Binary classification'
        self.url = 'https://drive.google.com/file/d/1GNc0HUee5YQH4Vtlk8ZbDjyJBYTEyabo/view?usp=sharing'

        self.generate = generate
        self.rand_permute = rand_permute
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

        # self.data, self.slices = torch.load(self.processed_paths[subset_pt])
        data_path = self.processed_paths[subset_pt]
        
        new_file_name = ""
        file_name = data_path.split("/")
        for key in file_name:
            if "mix" in key:
                key = "GOODHIV"
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

    def get(self, idx: int) -> Data:

        num_data = self.len()
        data = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx,
            slice_dict=self.slices,
            decrement=False)
        
        label = data.y.squeeze()
        all_label = self.data.y.squeeze()
        sample_idx_list = (all_label!=label).nonzero().squeeze().t().tolist()
        idx2 = random.sample(sample_idx_list, 1)[0]
        data2 = separate(
            cls=self.data.__class__,
            batch=self.data,
            idx=idx2,
            slice_dict=self.slices,
            decrement=False) 
        
        if self.rand_permute:
            data2 = rand_permute(data2)
        
        return data, data2



def rand_permute(data):

    N = data.x.shape[0]
    perm = torch.randperm(N)
    inv_perm = perm.new_empty(N)
    inv_perm[perm] = torch.arange(N)
    data.x = data.x[inv_perm]
    data.edge_index = perm[data.edge_index]
    return data