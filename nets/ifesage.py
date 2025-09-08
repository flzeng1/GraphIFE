"""
Pytorch Geometric
Ref: https://github.com/pyg-team/pytorch_geometric/blob/97d55577f1d0bf33c1bfbe0ef864923ad5cb844d/torch_geometric/nn/conv/sage_conv.py
"""
from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor, PairTensor

import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_scatter import scatter_add

from models import GatedAugmentor


class SAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.temp_weight = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.temp_weight.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.temp_weight(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class SAGE(nn.Module):
    def __init__(self, n_layer, input_dim, feat_dim, n_cls):
        super(SAGE, self).__init__()
        self.n_layer = n_layer
        self.n_cls = n_cls
        self.conv1 = nn.ModuleList()
        for i in range(n_layer):
            in_channels = input_dim if i == 0 else feat_dim
            self.conv1.append(SAGEConv(in_channels, feat_dim))

        self.classifier = torch.nn.Linear(feat_dim, n_cls)

        self.reg_params = list(self.conv1.parameters())
        self.non_reg_params = list(self.classifier.parameters())


    def forward(self, x, edge_index, edge_weight):
        ori_n_edge = edge_index.size(1)
        for i, conv in enumerate(self.conv1):
            x = conv(x, edge_index, edge_weight)
            x = x.relu()

        x = F.dropout(x, 0.5, training=self.training)
        x = self.classifier(x)

        return x


class SAGEConvwithRelu(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 root_weight: bool = True,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConvwithRelu, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        if self.root_weight:
            self.temp_weight = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.temp_weight.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_weight=None,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out += self.temp_weight(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        out = F.relu(out)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
class SAGEExtractor(torch.nn.Module):
    def __init__(self, input_dim, feat_dim, normalize=True, is_add_self_loops=True, dropout_rate=0.5):
        super(SAGEExtractor, self).__init__()

        self.n_layer = 1
        self.dropout_rate = dropout_rate
        self.conv1 = [SAGEConvwithRelu(input_dim, feat_dim)]
        self.conv1 = torch.nn.ModuleList(self.conv1)

        self.reg_params = list(self.conv1.parameters())
        self.is_add_self_loops = is_add_self_loops


    def forward(self, x, edge_index, edge_weight):
        h = x

        for i in range(self.n_layer):
            x = self.conv1[i](x, edge_index, edge_weight)

        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        x = x + h

        return x





class IFESAGE(torch.nn.Module):
    def __init__(self, input_dim, feat_dim, n_cls,  y,\
                 normalize=True, is_add_self_loops=True, dropout_rate=0.5):

        super(IFESAGE, self).__init__()

        self.n_cls = n_cls
        self.y = y
        self.dropout_rate=dropout_rate

        self.wasserstein_distance = nn.MSELoss()
        self.x_encoder = SAGEConvwithRelu(input_dim, feat_dim)
        self.global_encoder = SAGEConvwithRelu(feat_dim, feat_dim)

        self.invariant_feature_extractor = SAGEExtractor(feat_dim, feat_dim, dropout_rate)
        self.environment_feature_extractor = SAGEExtractor(feat_dim, feat_dim, dropout_rate)

        self.augmentor = GatedAugmentor(feat_dim)

        self.classifier = torch.nn.Linear(feat_dim, n_cls)


    def forward_invariant(self, x, edge_index, only_vanilla_if=False):
        local_repr = self.x_encoder(x, edge_index)
        global_repr = self.global_encoder(local_repr, edge_index)

        vanilla_if = self.invariant_feature_extractor(global_repr, edge_index, None)

        if not only_vanilla_if:
            vanilla_env = self.environment_feature_extractor(global_repr, edge_index, None)
            vanilla_aug, gate = self.augmentor(vanilla_if, vanilla_env)

        vanilla_pred = self.classifier(vanilla_if)
        if only_vanilla_if:
            return vanilla_pred

        aug_pred = self.classifier(vanilla_aug)

        if_out = {
            'vanilla_pred': vanilla_pred,
            'augmentation_pred': aug_pred,
            'gate_reg_loss': gate.mean()
        }

        return if_out

    def forward_environment(self, x, edge_index):
        local_repr = self.x_encoder(x, edge_index)
        global_repr = self.global_encoder(local_repr, edge_index)

        env_f = self.environment_feature_extractor(global_repr, edge_index, None)
        loss_dis = (self.wasserstein_distance(env_f, global_repr)
                    + self.wasserstein_distance(env_f, local_repr))

        env_f_pred = self.classifier(env_f)

        env_f_out = {
            'loss_dis': loss_dis,
            'env_f_pred': env_f_pred,
        }

        return env_f_out
