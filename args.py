import argparse
import math
import sys

import yaml


def get_args_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError('The YAML configuration must contain a mapping of parameter names to values.')
    return config


def str2bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower()
        if value in ('true', '1', 'yes', 'y', 't'):
            return True
        if value in ('false', '0', 'no', 'n', 'f'):
            return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    config_parser.add_argument('--config', type=str, default='./config.yaml', help='Path of YAML config file')
    config_args, _ = config_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(parents=[config_parser], allow_abbrev=False)

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['Cora', 'CiteSeer', 'PubMed', 'Amazon-Photo', 'Amazon-Computers', 'Coauthor-CS'],
        default='Coauthor-CS', help='dataset name')

    parser.add_argument('--data_path', type=str, default='./data/',
                        help='data path')

    parser.add_argument('--device', type=str, default='cuda',
                        help='device')

    parser.add_argument('--imb_ratio', type=float, default=100,
                        help='Imbalance Ratio')

    parser.add_argument('--net', type=str, default='GCN',
                        help='Architecture name')

    parser.add_argument('--feat_dim', type=int, default=256,
                        help='Feature dimension')

    parser.add_argument('--early_stop', type=str2bool, default=False,
                        help='Whether to use early stopping')

    parser.add_argument('--n_head', type=int, default=8,
                        help='the number of heads in GAT')

    parser.add_argument('--loss_type', type=str, default='ce',
                        help='Loss type')

    parser.add_argument('--ife', type=str2bool, default=True,
                        help='Mixing node')

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

    parser.add_argument('--verbose', type=str2bool, default=False,
                        help='Whether to use tensorboard record loss')

    parser.add_argument('--verbose_path', type=str, default='./runs',
                        help='the path for tensorboard logs')

    parser.add_argument('--write', type=str2bool, default=False,
                        help='Whether to write the essential log in ./result')

    if '-h' in argv or '--help' in argv:
        return parser.parse_args(argv)

    try:
        yaml_config = get_args_from_yaml(config_args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.error(f'Cannot load configuration {config_args.config!r}: {exc}')

    actions = {action.dest: action for action in parser._actions if action.dest != 'help'}
    unknown = set(yaml_config) - set(actions)
    if unknown:
        parser.error('Unknown YAML parameter(s): ' + ', '.join(sorted(map(str, unknown))))

    for name, value in yaml_config.items():
        action = actions[name]
        try:
            if action.type is str:
                if not isinstance(value, str):
                    raise ValueError('expected a string')
            elif action.type is int:
                if isinstance(value, bool) or not isinstance(value, (int, str)):
                    raise ValueError('expected an integer')
            elif action.type is float:
                if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                    raise ValueError('expected a number')
            value = action.type(value)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError('expected a finite number')
            if action.choices is not None and value not in action.choices:
                raise ValueError(f'expected one of {action.choices}')
        except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
            parser.error(f'Invalid YAML value for {name!r}: {exc}')
        yaml_config[name] = value

    parser.set_defaults(**yaml_config)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
