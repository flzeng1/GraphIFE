import os
from collections import Counter
from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from args import parse_args
from data_utils import load_dataset, fix_seed, construct_dataset_imbalance, get_model, get_optimizer, get_scheduler, \
    calculate_statistics, get_idx_info
from losses import CrossEntropy
from models import get_ins_neighbor_dist
from models.MultiObjectiveOptimizer import DWA
from models.Trianer import NodeClassificationTrainer
from visualize_utils import parameter_visualization



def run(args):
    args_list = parameter_visualization(args)

    result_log = "Dataset: {}, ratio: {}, net: {}, feat_dim: {}, ife: {}, early_stop: {}".format(
        args.dataset, str(args.imb_ratio), args.net, str(args.feat_dim), str(args.ife),
        str(args.early_stop))

    data, n_cls, num_features = load_dataset(args.dataset, args.data_path, split_type='full')
    data = data.to(args.device)

    result_metrics = {
        'avg_test_acc': [],
        'avg_val_acc': [],
        'avg_val_f1': [],
        'avg_test_bacc': [],
        'avg_test_f1': []
    }

    seed = args.seed
    for _ in range(args.repetitions):
        if args.write:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = f"{args.dataset}_{args.net}_{args.repetitions}_{current_time}"
            file_path = os.path.join(args.verbose_path, args.dataset, file_name)

        torch.cuda.empty_cache()

        seed = seed + 1
        fix_seed(seed)

        data_train_mask, data_val_mask, data_test_mask, train_edge_mask, train_node_mask, class_num_list, idx_info = construct_dataset_imbalance(
            args.dataset, data, n_cls, args.device, args.imb_ratio)


        if args.ife:
            neighbor_dist_list = get_ins_neighbor_dist(data.y.size(0), data.edge_index[:, train_edge_mask],
                                                       data_train_mask,
                                                       args.device)
        else:
            neighbor_dist_list = None

        model = get_model(args.net,  num_features, args.feat_dim, n_cls, data.y, args.n_head, args.dropout)
        opt_if, opt_env = get_optimizer(model, args)
        sch_if, sch_env = get_scheduler(opt_if, opt_env, args)

        data_statistic = {
            'class_num_list': class_num_list,
            'idx_info': idx_info,
            'neighbor_dist_list': neighbor_dist_list
        }

        construct_data_mask = {
            'data_train_mask': data_train_mask,
            'data_val_mask': data_val_mask,
            'data_test_mask': data_test_mask,
            'train_edge_mask': train_edge_mask,
            'train_node_mask': train_node_mask
        }

        trainer = NodeClassificationTrainer(
            model=model.to(args.device),
            data=data,
            data_statistic=data_statistic,
            construct_data_mask=construct_data_mask,
            criterion=CrossEntropy().to(args.device) if args.loss_type == 'ce' else None,
            invariant_optimizer=opt_if,
            environment_optimizer=opt_env,
            invariant_schedule=sch_if,
            environment_schedule=sch_env,
            multi_objective_optimizer=DWA(T=args.DMA_T),
            writer=SummaryWriter(file_path) if args.verbose else None,
            n_cls=n_cls,
            training_args=args)

        best_val_acc, val_f1, test_acc, test_bacc, test_f1 = trainer.train()

        result_metrics['avg_val_acc'].append(best_val_acc)
        result_metrics['avg_val_f1'].append(val_f1)
        result_metrics['avg_test_acc'].append(test_acc)
        result_metrics['avg_test_bacc'].append(test_bacc)
        result_metrics['avg_test_f1'].append(test_f1)

    calculate_statistics(result_metrics, result_log, args_list, args)


if __name__ == '__main__':
    args = parse_args()
    run(args)
