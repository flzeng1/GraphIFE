import argparse

import yaml


def get_args_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='./config.yaml', help='Path of YAML config file')

    # Dataset
    parser.add_argument('--dataset', type=str,
                        choices=['Cora', 'CiteSeer', 'PubMed', 'Amazon-Photo', 'Amazon-Computers', 'Coauthor-CS'],
                        default='Coauthor-CS', help='dataset name')

    parser.add_argument('--data_path', type=str, default='./data/',
                        help='data path')

    # Basic setting #
    parser.add_argument('--device', type=str, default='cuda',
                        help='device')

    parser.add_argument('--imb_ratio', type=float, default=100,
                        help='Imbalance Ratio')
    parser.add_argument('--net', type=str, default='GCN',
                        help='Architecture name')

    parser.add_argument('--feat_dim', type=int, default=256,
                        help='Feature dimension')

    parser.add_argument('--early_stop', type=bool, default=False,
                        help='Whether to use early stopping')
    # GAT
    parser.add_argument('--n_head', type=int, default=8,
                        help='the number of heads in GAT')
    # Imbalance Loss
    parser.add_argument('--loss_type', type=str, default='ce',
                        help='Loss type')

    # Method
    parser.add_argument('--ife', type=bool, default=True,
                        help='Mixing node')



    ##  Training Parameters        ##
    parser.add_argument('--seed', type=int, default=100, help='random seed')

    parser.add_argument('--dis', type=float, default=0.5,
                        help='the weight of Wasserstein distance')
    parser.add_argument('--if_lr', type=float, default=9e-2,
                        help='learning rate of invariant feature optimizer')
    parser.add_argument('--env_lr', type=float, default=0.01,
                        help='learning rate of environment optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight_decay in optimizer')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='total epochs')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='the minimum learning rate of environment schedule')

    parser.add_argument('--DMA_T', type=float, default=1,
                        help='the temperature of DMA')

    parser.add_argument('--repetitions', type=int, default=5,
                        help='total experiment times')

    parser.add_argument('--alpha', type=float, default=0.5,
                        help='the regularization term of gated mechanism')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')

    parser.add_argument('--pred_temp', type=float, default=2,
                        help='Prediction temperature')
    parser.add_argument('--warmup', type=int, default=70,
                        help='warmup times')

    # visualize
    parser.add_argument('--verbose', type=bool, default=False,
                        help='Whether to use tensorboard record loss')

    parser.add_argument('--verbose_path', type=str, default='./runs',
                        help='the path for tensorboard logs')

    parser.add_argument('--write', type=bool, default=False,
                        help='Whether to write the essential log in ./result')





    yaml_config = get_args_from_yaml("config.yaml")

    parser.set_defaults(**yaml_config)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

