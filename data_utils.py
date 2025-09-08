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


# depr
def get_dataset(name, path, split_type='full'):
    import torch_geometric.transforms as T

    if name == "Cora" or name == "CiteSeer" or name == "PubMed":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures(), split=split_type)
    else:
        raise NotImplementedError("Not Implemented Dataset!")

    return dataset


def load_dataset(name, path, split_type='full'):
    import torch_geometric.transforms as T

    if name == "Cora" or name == "CiteSeer" or name == "PubMed":
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(path, name, transform=T.NormalizeFeatures(), split=split_type)
    elif name == 'Amazon-Computers':
        from torch_geometric.datasets import Amazon
        # return Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
        dataset = Amazon(root=path, name='computers', transform=T.NormalizeFeatures())
    elif name == 'Amazon-Photo':
        from torch_geometric.datasets import Amazon
        # return Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
        dataset = Amazon(root=path, name='photo', transform=T.NormalizeFeatures())
    elif name == 'Coauthor-CS':
        from torch_geometric.datasets import Coauthor
        # return Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
        dataset = Coauthor(root=path, name='cs', transform=T.NormalizeFeatures())
    else:
        raise NotImplementedError("Not Implemented Dataset!")

    data = dataset[0]
    n_cls = data.y.max().item() + 1

    return data, n_cls, dataset.num_features


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
        ## Data statistic ##
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
        train_edge_mask = torch.ones(data.edge_index.shape[1], dtype=torch.bool)

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




def split_semi_dataset(total_node, n_data, n_cls, class_num_list, idx_info, device):
    new_idx_info = []
    _train_mask = idx_info[0].new_zeros(total_node, dtype=torch.bool, device=device)
    for i in range(n_cls):
        if n_data[i] > class_num_list[i]:
            # 生成了下标的随机排序，并满足条件地取样
            cls_idx = torch.randperm(len(idx_info[i]))
            cls_idx = idx_info[i][cls_idx]
            cls_idx = cls_idx[:class_num_list[i]]
            new_idx_info.append(cls_idx)
        else:
            new_idx_info.append(idx_info[i])
        _train_mask[new_idx_info[i]] = True

    # 判断新的训练集数量是否满足取样要求
    assert _train_mask.sum().long() == sum(class_num_list)
    # 判断新的 训练集中类别为i的元素的数量是否满足取样要求
    assert sum([len(idx) for idx in new_idx_info]) == sum(class_num_list)

    return _train_mask, new_idx_info


def get_idx_info(label, n_cls, train_mask):
    # index_list相当于一个坐标集
    index_list = torch.arange(len(label))
    idx_info = []
    for i in range(n_cls):
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
    return idx_info


## Construct LT ##
def make_longtailed_data_remove(edge_index, label, n_data, n_cls, ratio, train_mask):
    # Sort from major to minor
    n_data = torch.tensor(n_data)
    sorted_n_data, indices = torch.sort(n_data, descending=True)
    inv_indices = np.zeros(n_cls, dtype=np.int64)
    for i in range(n_cls):
        inv_indices[indices[i].item()] = i
    assert (torch.arange(len(n_data))[indices][torch.tensor(inv_indices)] - torch.arange(len(n_data))).sum().abs() < 1e-12

    # Compute the number of nodes for each class following LT rules
    mu = np.power(1/ratio, 1/(n_cls - 1))
    n_round = []
    class_num_list = []
    for i in range(n_cls):
        assert int(sorted_n_data[0].item() * np.power(mu, i)) >= 1
        class_num_list.append(int(min(sorted_n_data[0].item() * np.power(mu, i), sorted_n_data[i])))
        """
        Note that we remove low degree nodes sequentially (10 steps)
        since degrees of remaining nodes are changed when some nodes are removed
        """
        if i < 1: # We does not remove any nodes of the most frequent class
            n_round.append(1)
        else:
            n_round.append(10)
    class_num_list = np.array(class_num_list)
    class_num_list = class_num_list[inv_indices]
    n_round = np.array(n_round)[inv_indices]

    # Compute the number of nodes which would be removed for each class
    remove_class_num_list = [n_data[i].item()-class_num_list[i] for i in range(n_cls)]
    remove_idx_list = [[] for _ in range(n_cls)]
    cls_idx_list = []
    index_list = torch.arange(len(train_mask))
    original_mask = train_mask.clone()
    for i in range(n_cls):
        cls_idx_list.append(index_list[(label == i) & original_mask])

    for i in indices.numpy():
        for r in range(1,n_round[i]+1):
            # Find removed nodes
            node_mask = label.new_ones(label.size(), dtype=torch.bool)
            node_mask[sum(remove_idx_list,[])] = False

            # Remove connection with removed nodes
            row, col = edge_index[0], edge_index[1]
            row_mask = node_mask[row]
            col_mask = node_mask[col]
            edge_mask = row_mask & col_mask

            # Compute degree
            degree = scatter_add(torch.ones_like(col[edge_mask]), col[edge_mask], dim_size=label.size(0)).to(row.device)
            degree = degree[cls_idx_list[i]]

            # Remove nodes with low degree first (number increases as round increases)
            # Accumulation does not be problem since
            _, remove_idx = torch.topk(degree, (r*remove_class_num_list[i])//n_round[i], largest=False)
            remove_idx = cls_idx_list[i][remove_idx]
            remove_idx_list[i] = list(remove_idx.numpy())

    # Find removed nodes
    node_mask = label.new_ones(label.size(), dtype=torch.bool)
    node_mask[sum(remove_idx_list,[])] = False

    # Remove connection with removed nodes
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
        pass
    elif net == "SAGE":
        model = IFESAGE(num_features, feat_dim, n_cls, y, dropout_rate = dropout)

        pass
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
    acc_CI = (statistics.stdev(result_metrics['avg_test_acc']) / (args.repetitions ** (1 / 2)))
    bacc_CI = (statistics.stdev(result_metrics['avg_test_bacc']) / (args.repetitions ** (1 / 2)))
    f1_CI = (statistics.stdev(result_metrics['avg_test_f1']) / (args.repetitions ** (1 / 2)))
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
        file_path = os.path.join('./result', file_name)
        with open(file_path, 'w') as f:
            f.write(log)

    print(log)


def write_to_file(string, file_path):
    # Open the file in write mode ('w') or append mode ('a')
    with open(file_path, 'w') as file:
        file.write(string)


def compute_label_homophily_per_class(data):
    edge_index = to_undirected(data.edge_index)  # 确保边是无向的
    y = data.y
    # train_mask = data.train_mask
    num_nodes = y.size(0)

    same_label_counts = torch.zeros(num_nodes)
    neighbor_counts = torch.zeros(num_nodes)

    src, dst = edge_index
    for s, d in zip(src, dst):
        if y[s] == y[d]:
            same_label_counts[s] += 1
        neighbor_counts[s] += 1

    label_ratio = same_label_counts / (neighbor_counts + 1e-10)
    # label_ratio = label_ratio[train_mask]
    # train_labels = y[train_mask]

    num_classes = int(torch.max(y).item()) + 1
    class_avg_ratios = []
    for cls in range(num_classes):
        class_mask = y == cls
        if class_mask.sum() > 0:
            avg_ratio = label_ratio[class_mask].mean().item()
        else:
            avg_ratio = float('nan') 
        class_avg_ratios.append(avg_ratio)

    return torch.tensor(class_avg_ratios), torch.tensor(label_ratio)  


def compute_node_weights(label_ratio, class_avg_ratios, data):

    num_classes = len(class_avg_ratios)
    node_weights = torch.zeros_like(label_ratio)
    # train_mask = data.train_mask
    y = data.y

    for cls in range(num_classes):
        class_mask = (y == cls)
        class_avg = class_avg_ratios[cls]

        node_class_ratio = label_ratio[class_mask]

        if class_avg > 0:
            min_val = node_class_ratio.min()
            max_val = node_class_ratio.max()
            normalized_ratio = (node_class_ratio - min_val) / (max_val - min_val + 1e-10)
            node_weights[class_mask] = normalized_ratio

    return node_weights