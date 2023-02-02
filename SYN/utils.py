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
from model import EIGNN, EI
import data_utils.featgen as featgen
import data_utils.gengraph as gengraph
from torch_geometric.utils import from_networkx
from tqdm import tqdm
from torch_geometric.utils.convert import to_networkx
from sklearn.model_selection import StratifiedKFold
import pdb

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
        dataset, meta_info = GOODCMNIST.load(args.data_dir, domain='color', shift='concept', generate=False)
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

    elif args.dataset == "SYN":
        try:
            dataset_load = torch.load(args.data_dir + "/syn_dataset.pt")

        except:
            dataset_load = graph_dataset_generate(args, args.data_dir)
        rand_perm = True
        train_set, val_set, test_set, the = dataset_bias_split(dataset_load, args, bias=args.bias, split=[7, 1, 2], total=args.data_num * 4)
        train_mix_set = get_mix_dataset(train_set, rand_perm)
        group_counts = print_dataset_info(train_set, val_set, test_set, the)

        dataset = {"train": train_set, "val": val_set, "test": test_set, "train_mix": train_mix_set}
        meta_info = {'dataset_type': 'syn', 'model_level': 'graph', 'dim_node': 10, 'dim_edge': 0, 'num_envs': 2, 'num_classes': 4, 'group_counts': group_counts}
        in_dim = 10
        num_class = 4
        num_layer = 3
        eval_metric = None
        cri = criterion1
        eval_name = None
        test_batch_size = args.batch_size
    else:
        assert False
    print(meta_info)
    return dataset, meta_info, in_dim, num_class, num_layer, eval_metric, cri, eval_name, test_batch_size


def get_model(args):

    def model_func1(num_classes, num_features, hidden):
        return GCNNet(num_classes, num_features, hidden)  

    def model_func2(num_classes, num_features, hidden):
        return GINNet(num_classes, num_features, hidden) 
    
    def model_func3(num_classes, num_features, hidden):
        return GATNet(num_classes, num_features, hidden) 

    def model_func4(num_classes, num_features, hidden, dropout_rate):
        return EIGNN(args, num_classes, num_features, hidden, dropout_rate=dropout_rate) 

    def model_func5(num_classes, num_features, hidden, dropout_rate):
        return EIGNN(args, num_classes, num_features, hidden, dropout_rate=dropout_rate) 

    def model_func6(num_classes, num_features, hidden, dropout_rate):
        return EIGNN(args, num_classes, num_features, hidden, dropout_rate=dropout_rate) 
    
    def model_func7(num_classes, num_features, hidden, dropout_rate):
        return EI(args=args, num_class=num_classes, in_dim=num_features, emb_dim=hidden, fro_layer=args.fro_layer, bac_layer=args.bac_layer, cau_layer=args.cau_layer, dropout_rate=dropout_rate)

    if args.model == "GCN":
        model_func = model_func1
    elif args.model == "GIN":
        model_func = model_func2
    elif args.model == "GAT":
        model_func = model_func3
    elif args.model == "EIGCN":
        model_func = model_func4
    elif args.model == "EIGIN":
        model_func = model_func5
    elif args.model == "EIGAT":
        model_func = model_func6
    elif args.model in ['EIGCN2', 'EIGIN2', 'EIGAT2']:
        model_func = model_func7
    else:
        assert False
    return model_func

def arg_parse():
    str2bool = lambda x: x.lower() == "true"
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--data_dir',  type=str, default='../dataset', help='dataset dir')
    parser.add_argument('--seed',      type=int,   default=666)
    parser.add_argument('--emb_dim',    type=int,   default=300)
    parser.add_argument('--test_epoch', type=int, default=10)
    parser.add_argument('--layer', type=int, default=-1)
    parser.add_argument('--fro_layer', type=int, default=-1)
    parser.add_argument('--bac_layer', type=int, default=-1)
    parser.add_argument('--cau_layer', type=int, default=-1)
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

    parser.add_argument('--bias', type=float, default=0.5)
    parser.add_argument('--data_num', type=int, default=2000)

    parser.add_argument('--model', type=str, default="EIGCN")
    parser.add_argument('--node_num', type=int, default=15)
    parser.add_argument('--feature_dim', type=int, default=-1)
    parser.add_argument('--shape_num', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--max_degree', type=int, default=10)
    
    parser.add_argument('--dropout_rate', type=float, default=0)
    parser.add_argument('--pretrain', type=int, default=60)
    parser.add_argument('--frequency', type=int, default=3)
    args = parser.parse_args()
    return args

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

    N = data.feat.shape[0]
    perm = torch.randperm(N)
    inv_perm = perm.new_empty(N)
    inv_perm[perm] = torch.arange(N)
    data.feat = data.feat[inv_perm]
    data.edge_index = perm[data.edge_index]
    return data

def print_graph_info(G, c, o):
    print('-' * 100)
    print("| graph: {}-{} | nodes num:{} | edges num:{} |".format(c, o, G.num_nodes, G.num_edges))
    print('-' * 100)
    return G.num_nodes, G.num_edges

def print_dataset_info(train_set, val_set, test_set, the):

    class_list = ["house", "cycle", "grid", "diamond"]
    dataset_group_dict = {}
    dataset_group_dict["Train"] = dataset_context_object_info(train_set, "Train", class_list, the)
    dataset_group_dict["Val"] = dataset_context_object_info(val_set, "Val   ", class_list, the)
    dataset_group_dict["Test"] = dataset_context_object_info(test_set, "Test  ", class_list, the)
    return dataset_group_dict

def graph_dataset_generate(args, save_path):

    class_list = ["house", "cycle", "grid", "diamond"]
    settings_dict = {"ba": {"width_basis": args.node_num ** 2, "m": 2},
                     "tree": {"width_basis":2, "m": args.node_num}}

    feature_dim = args.feature_dim
    shape_num = args.shape_num
    class_num = class_list.__len__()
    dataset = {}
    dataset['tree'] = {}
    dataset['ba'] = {}

    for label, shape in enumerate(class_list):
        tr_list = []
        ba_list = []
        print("create shape:{}".format(shape))
        for i in tqdm(range(args.data_num)):
            tr_g, label1 = creat_one_pyg_graph(context="tree", shape=shape, label=label, feature_dim=feature_dim, 
                                               shape_num=shape_num, settings_dict=settings_dict, args=args)
            ba_g, label2 = creat_one_pyg_graph(context="ba", shape=shape, label=label, feature_dim=feature_dim, 
                                               shape_num=shape_num, settings_dict=settings_dict, args=args)
            tr_list.append(tr_g)
            ba_list.append(ba_g)
        dataset['tree'][shape] = tr_list
        dataset['ba'][shape] = ba_list

    save_path += "/syn_dataset.pt"
    torch.save(dataset, save_path)
    print("save at:{}".format(save_path))
    return dataset

def test_dataset_generate(args, save_path):

    class_list = ["house", "cycle", "grid", "diamond"]
    settings_dict = {"ba": {"width_basis": (args.node_num) ** 2, "m": 2},
                     "tree": {"width_basis":2, "m": args.node_num}}

    feature_dim = args.feature_dim
    shape_num = args.shape_num
    class_num = class_list.__len__()
    dataset = {}
    dataset['tree'] = {}
    dataset['ba'] = {}
    data_num = int(0.2 * args.data_num)
    for label, shape in enumerate(class_list):
        tr_list = []
        ba_list = []
        print("test set create shape:{}".format(shape))
        for i in tqdm(range(data_num)):
            tr_g, label1 = creat_one_pyg_graph(context="tree", shape=shape, label=label, feature_dim=feature_dim, 
                                               shape_num=shape_num, settings_dict=settings_dict, args=args)
            ba_g, label2 = creat_one_pyg_graph(context="ba", shape=shape, label=label, feature_dim=feature_dim, 
                                               shape_num=shape_num, settings_dict=settings_dict, args=args)
            tr_list.append(tr_g)
            ba_list.append(ba_g)
        dataset['tree'][shape] = tr_list
        dataset['ba'][shape] = ba_list

    save_path += "/syn_dataset_test.pt"
    torch.save(dataset, save_path)
    print("save at:{}".format(save_path))
    return dataset

def creat_one_pyg_graph(context, shape, label, feature_dim, shape_num, settings_dict, args=None):
    if args is None:
        noise = 0
    else:
        noise = args.noise
    if feature_dim == -1:
        # use degree as feature
        feature = featgen.ConstFeatureGen(None, max_degree=args.max_degree)
    else:
        feature = featgen.ConstFeatureGen(np.random.uniform(0, 1, feature_dim))
    G, node_label = gengraph.generate_graph(basis_type=context,
                                            shape=shape,
                                            nb_shapes=shape_num,
                                            width_basis=settings_dict[context]["width_basis"],
                                            feature_generator=feature,
                                            m=settings_dict[context]["m"],
                                            random_edges=noise) 
    pyg_G = from_networkx(G)
    pyg_G.y = torch.tensor([label])
    return pyg_G, node_label

def dataset_bias_split(dataset, args, bias=None, split=None, total=20000):
    
    class_list = ["house", "cycle", "grid", "diamond"]
    bias_dict = {"house": bias, "cycle": 1 - bias, "grid": 1 - bias, "diamond": 1 - bias}
    
    ba_dataset = dataset['ba']
    tr_dataset = dataset['tree']
    
    train_split, val_split, test_split = float(split[0]) / 10, float(split[1]) / 10, float(split[2]) / 10
    assert train_split + val_split + test_split == 1
    train_num, val_num, test_num = total * train_split, total * val_split, total * test_split
    # blance class
    class_num = 4
    train_class_num, val_class_num, test_class_num = train_num / class_num, val_num / class_num, test_num / class_num
    train_list, val_list, test_list  = [], [], []
    edges_num = 0
    
    for shape in class_list:
        bias = bias_dict[shape]
        train_tr_num = int(train_class_num * bias)
        train_ba_num = int(train_class_num * (1 - bias))
        val_tr_num = int(val_class_num * bias)
        val_ba_num = int(val_class_num * (1 - bias))
        test_tr_num = int(test_class_num * 0.5)
        test_ba_num = int(test_class_num * 0.5)
        train_list += tr_dataset[shape][:train_tr_num] + ba_dataset[shape][:train_ba_num]
        val_list += tr_dataset[shape][train_tr_num:train_tr_num + val_tr_num] + ba_dataset[shape][train_ba_num:train_ba_num + val_ba_num]
        test_list += tr_dataset[shape][train_tr_num + val_tr_num:train_tr_num + val_tr_num + test_tr_num] + ba_dataset[shape][train_ba_num + val_ba_num:train_ba_num + val_ba_num + test_ba_num]
        _, e1 = print_graph_info(tr_dataset[shape][0], "Tree", shape)
        _, e2 = print_graph_info(ba_dataset[shape][0], "BA", shape)
        
        edges_num += e1 + e2
    random.shuffle(train_list)
    random.shuffle(val_list)
    random.shuffle(test_list)
    the = float(edges_num) / (class_num * 2)
    return train_list, val_list, test_list, the


def dataset_context_object_info(dataset, title, class_list, the):

    class_num = len(class_list)
    tr_list = [0] * class_num
    ba_list = [0] * class_num
    for g in dataset:
        if g.num_edges > the: # ba
            ba_list[g.y.item()] += 1
        else: # tree
            tr_list[g.y.item()] += 1
    total = sum(tr_list) + sum(ba_list)
    info = "{} Total:{}\n| Tree: House:{:<5d}, Cycle:{:<5d}, Grids:{:<5d}, Diams:{:<5d} \n" +\
                        "| BA  : House:{:<5d}, Cycle:{:<5d}, Grids:{:<5d}, Diams:{:<5d} \n" +\
                        "| All : House:{:<5d}, Cycle:{:<5d}, Grids:{:<5d}, Diams:{:<5d} \n" +\
                        "| BIAS: House:{:.1f}%, Cycle:{:.1f}%, Grids:{:.1f}%, Diams:{:.1f}%"
    print("-" * 150)
    print(info.format(title, total, tr_list[0], tr_list[1], tr_list[2], tr_list[3],
                                    ba_list[0], ba_list[1], ba_list[2], ba_list[3],
                                    tr_list[0] +  ba_list[0],    
                                    tr_list[1] +  ba_list[1], 
                                    tr_list[2] +  ba_list[2], 
                                    tr_list[3] +  ba_list[3],
                                    100 *float(tr_list[0]) / (tr_list[0] +  ba_list[0]),
                                    100 *float(tr_list[1]) / (tr_list[1] +  ba_list[1]),
                                    100 *float(tr_list[2]) / (tr_list[2] +  ba_list[2]),
                                    100 *float(tr_list[3]) / (tr_list[3] +  ba_list[3]),
                     ))
    print("-" * 150)
    total_list = ba_list + tr_list
    group_counts = torch.tensor(total_list).float()
    return group_counts