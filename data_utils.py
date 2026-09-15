import os
import random
import statistics
from datetime import datetime

import torch
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_add

from nets.ifegatv2 import IFEGAT
from nets.ifegcn import IFEGCN
from nets.ifesage import IFESAGE

def load_dataset(name, path, split_type='full'):
    import torch_geometric.transforms as T

    if name == "Cora" or name == "CiteSeer" or name == "PubMed":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures(), split=split_type)
        data = dataset[0]
    elif name == 'Amazon-Computers':
        from torch_geometric.datasets import Amazon
        dataset = Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
        data = dataset[0]
    elif name == 'Amazon-Photo':
        from torch_geometric.datasets import Amazon
        dataset = Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
        data = dataset[0]
    elif name == 'Coauthor-CS':
        from torch_geometric.datasets import Coauthor
        dataset = Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
        data = dataset[0]
    else:
        raise NotImplementedError("Not Implemented Dataset!")

    data.y = data.y.long()
    n_cls = int(data.y.max().item()) + 1
    num_features = int(data.x.size(-1))

    return data, n_cls, num_features


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def construct_dataset_imbalance(dataset, data, n_cls, device, imb_ratio):
    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        data_train_mask, data_val_mask, data_test_mask = data.train_mask.clone(), data.val_mask.clone(), data.test_mask.clone()

        stats = data.y[data_train_mask]
        n_data = []
        for i in range(n_cls):
            data_num = (stats == i).sum()
            n_data.append(int(data_num.item()))
        idx_info = get_idx_info(data.y, n_cls, data_train_mask)
        class_num_list = n_data

        class_num_list, data_train_mask, idx_info, train_node_mask, train_edge_mask = make_longtailed_data_remove(
            data.edge_index, \
            data.y, n_data, n_cls, imb_ratio, data_train_mask.clone())

    elif dataset in ['Coauthor-CS', 'Amazon-Computers', 'Amazon-Photo']:
        train_idx, valid_idx, test_idx, train_node = get_step_split(imb_ratio=imb_ratio, \
                                                                    valid_each=int(data.x.shape[0] * 0.1 / n_cls), \
                                                                    labeling_ratio=0.1, \
                                                                    all_idx=[i for i in range(data.x.shape[0])], \
                                                                    all_label=data.y.cpu().detach().numpy(), \
                                                                    nclass=n_cls)

        data_train_mask = torch.zeros(data.x.shape[0]).bool().to(device)
        data_val_mask = torch.zeros(data.x.shape[0]).bool().to(device)
        data_test_mask = torch.zeros(data.x.shape[0]).bool().to(device)
        data_train_mask[train_idx] = True
        data_val_mask[valid_idx] = True
        data_test_mask[test_idx] = True
        train_idx = data_train_mask.nonzero().squeeze()
        train_edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool, device=device)

        train_node_mask = torch.zeros(data.x.shape[0]).bool().to(device)
        train_node_mask[sum(train_node, [])] = True

        class_num_list = [len(item) for item in train_node]
        idx_info = [torch.tensor(item) for item in train_node]

    return data_train_mask, data_val_mask, data_test_mask, train_edge_mask, train_node_mask, class_num_list, idx_info

def get_step_split(imb_ratio, valid_each, labeling_ratio, all_idx, all_label, nclass):
    base_valid_each = valid_each

    head_list = [i for i in range(nclass//2)]

    all_class_list = [i for i in range(nclass)]
    tail_list = list(set(all_class_list) - set(head_list))

    h_num = len(head_list)
    t_num = len(tail_list)

    base_train_each = int( len(all_idx) * labeling_ratio / (t_num + h_num * imb_ratio) )

    idx2train,idx2valid = {},{}

    total_train_size = 0
    total_valid_size = 0

    for i_h in head_list:
        idx2train[i_h] = int(base_train_each * imb_ratio)
        idx2valid[i_h] = int(base_valid_each * 1)

        total_train_size += idx2train[i_h]
        total_valid_size += idx2valid[i_h]

    for i_t in tail_list:
        idx2train[i_t] = int(base_train_each * 1)
        idx2valid[i_t] = int(base_valid_each * 1)

        total_train_size += idx2train[i_t]
        total_valid_size += idx2valid[i_t]

    train_list = [0 for _ in range(nclass)]
    train_node = [[] for _ in range(nclass)]
    train_idx  = []

    for iter1 in all_idx:
        iter_label = all_label[iter1]
        if train_list[iter_label] < idx2train[iter_label]:
            train_list[iter_label]+=1
            train_node[iter_label].append(iter1)
            train_idx.append(iter1)

        if sum(train_list)==total_train_size:break

    assert sum(train_list)==total_train_size

    after_train_idx = list(set(all_idx)-set(train_idx))

    valid_list = [0 for _ in range(nclass)]
    valid_idx  = []
    for iter2 in after_train_idx:
        iter_label = all_label[iter2]
        if valid_list[iter_label] < idx2valid[iter_label]:
            valid_list[iter_label]+=1
            valid_idx.append(iter2)
        if sum(valid_list)==total_valid_size:break

    test_idx = list(set(after_train_idx)-set(valid_idx))

    return train_idx, valid_idx, test_idx, train_node


def get_idx_info(label, n_cls, train_mask):
    index_list = torch.arange(len(label), device=label.device)
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
    return idx_info


def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(len(n_data))).sum().abs() < 1e-12

    mu = np.power(1/ratio, 1/(n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        class_num_list.append(int(min(sorted_n_data[0].item() * np.power(mu, i), sorted_n_data[i])))

        if i < 1:
            n_round.append(1)
        else:
            n_round.append(10)
    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]

    remove_class_num_list = [n_data[i].item()-class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask), device=train_mask.device)
    original_mask = train_mask.clone()
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) & original_mask])

    for i in indices.numpy():
        for r in range(1,n_round[i]+1):
            node_mask = label.new_ones(label.size(), dtype=torch.bool)
            node_mask[sum(remove_idx_list,[])] = False

            row, col = edge_index[0], edge_index[1]
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = row_mask & col_mask

            degree = scatter_add(torch.ones_like(col[edge_mask]), col[edge_mask], dim_size=label.size(0)).to(row.device)
            degree = degree[cls_idx_list[i]]

            _, remove_idx = torch.topk(degree, (r*remove_class_num_list[i])//n_round[i], largest=False)
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.cpu().numpy())

    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list,[])] = False

    row, col = edge_index[0], edge_index[1]
    row_mask = node_mask[row]
    col_mask = node_mask[col]
    edge_mask = row_mask & col_mask

    train_mask = node_mask & train_mask
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[(label == i) & train_mask]
        idx_info.append(cls_indices)

    return list(class_num_list), train_mask, idx_info, node_mask, edge_mask


def get_model(net, num_features, feat_dim, n_cls, y, n_head=8, dropout=0.5):
    model = None
    if net == 'GCN':
        model = IFEGCN(num_features, feat_dim, n_cls, y, normalize=True, is_add_self_loops=True, dropout_rate=dropout)
    elif net == 'GATV2':
        model = IFEGAT(num_features, feat_dim, n_cls, y, n_head, is_add_self_loops=True, dropout_rate=dropout)
    elif net == "SAGE":
        model = IFESAGE(num_features, feat_dim, n_cls, y, dropout_rate = dropout)
    else:
        raise NotImplementedError("Not Implemented Architecture!")

    return model


def get_optimizer(model, args):
    opt_if = torch.optim.Adam([
        {'params': list(model.x_encoder.parameters())
                   + list(model.global_encoder.parameters())
                   + list(model.augmentor.parameters())
                   + list(model.invariant_feature_extractor.parameters()),
         'weight_decay': args.weight_decay},

        {'params': list(model.classifier.parameters()), 'weight_decay': 0.0}
    ],
        lr=args.if_lr)

    opt_env = torch.optim.Adam(list(model.environment_feature_extractor.parameters()), lr=args.env_lr,
                               weight_decay=args.weight_decay)

    return opt_if, opt_env


def get_scheduler(opt_if, opt_env, args):
    sch_if = torch.optim.lr_scheduler.ReduceLROnPlateau(opt_if, mode='min',
                                                           factor=0.5,
                                                           patience=100,
                                                           verbose=False)
    sch_env = CosineAnnealingLR(opt_env, T_max=args.epochs, eta_min=args.lr_min)

    return sch_if, sch_env


def calculate_statistics(result_metrics, setting_log, args_list, args):
    n_runs = len(result_metrics['avg_test_acc'])
    if n_runs > 1:
        acc_CI = (statistics.stdev(result_metrics['avg_test_acc']) / (n_runs ** (1 / 2)))
        bacc_CI = (statistics.stdev(result_metrics['avg_test_bacc']) / (n_runs ** (1 / 2)))
        f1_CI = (statistics.stdev(result_metrics['avg_test_f1']) / (n_runs ** (1 / 2)))
    else:
        acc_CI = bacc_CI = f1_CI = 0.0
    avg_acc = statistics.mean(result_metrics['avg_test_acc'])
    avg_val_acc = statistics.mean(result_metrics['avg_val_acc'])
    avg_val_f1 = statistics.mean(result_metrics['avg_val_f1'])
    avg_bacc = statistics.mean(result_metrics['avg_test_bacc'])
    avg_f1 = statistics.mean(result_metrics['avg_test_f1'])

    avg_log = 'Test Acc: {:.4f} +- {:.4f}, BAcc: {:.4f} +- {:.4f}, F1: {:.4f} +- {:.4f}, Val Acc: {:.4f}, Val F1: {:.4f}'
    avg_log = avg_log.format(avg_acc, acc_CI, avg_bacc, bacc_CI, avg_f1, f1_CI, avg_val_acc, avg_val_f1)

    per_repetition_acc_log = (f'Acc in repetition:{result_metrics["avg_test_acc"]}\n'
                              f'Bacc in repetition: {result_metrics["avg_test_bacc"]}\n'
                              f'F1 in repetition: {result_metrics["avg_test_f1"]}\n')

    log = "{}\n{}".format(setting_log, per_repetition_acc_log)
    log = "{}\n{}".format(log, avg_log)

    if args.write:
        log = f"{args_list}\n\n{log}"
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{args.dataset}_{args.net}_{args.repetitions}_acc_{avg_acc:.4f}_{current_time}.txt"
        result_dir = './result'
        os.makedirs(result_dir, exist_ok=True)
        file_path = os.path.join(result_dir, file_name)
        with open(file_path, 'w') as f:
            f.write(log)

    print(log)


def compute_label_homophily_per_class(data, data_train_mask):
    edge_index = to_undirected(data.edge_index, num_nodes=data.y.size(0))
    y = data.y
    train_mask = data_train_mask
    num_nodes = y.size(0)

    src, dst = edge_index
    edge_mask = train_mask[src] & train_mask[dst]
    src = src[edge_mask]
    dst = dst[edge_mask]

    same_label = (y[src] == y[dst]).float()
    same_label_counts = scatter_add(same_label, src, dim=0, dim_size=num_nodes)
    neighbor_counts = scatter_add(torch.ones_like(same_label), src, dim=0, dim_size=num_nodes)
    label_ratio = same_label_counts / (neighbor_counts + 1e-10)

    num_classes = int(torch.max(y).item()) + 1
    train_labels = y[train_mask]
    label_ratio_train = label_ratio[train_mask]
    class_sums = scatter_add(label_ratio_train, train_labels, dim=0, dim_size=num_classes)
    class_counts = scatter_add(torch.ones_like(label_ratio_train), train_labels, dim=0, dim_size=num_classes)
    class_avg_ratios = class_sums / class_counts.clamp_min(1)
    class_avg_ratios[class_counts == 0] = float('nan')

    return class_avg_ratios, label_ratio, train_mask

def compute_node_weights(label_ratio, class_avg_ratios, train_mask, data):
    num_classes = len(class_avg_ratios)
    node_weights = torch.zeros_like(label_ratio)

    y = data.y

    for cls in range(num_classes):
        class_mask = (y == cls) & train_mask
        class_avg = class_avg_ratios[cls]

        node_class_ratio = label_ratio[class_mask]

        if class_avg > 0:
            min_val = node_class_ratio.min()
            max_val = node_class_ratio.max()
            normalized_ratio = (node_class_ratio - min_val) / (max_val - min_val + 1e-10)

            node_weights[class_mask] = normalized_ratio

    node_weights[~train_mask] = 0

    node_weights[train_mask] += 1
    return node_weights
