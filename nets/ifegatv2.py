from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size,
                                    OptTensor, PairTensor)

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros

from models import GatedAugmentor


class GATv2ConvWithElu(MessagePassing):
    _alpha: OptTensor

    def __init__(
            self,
            in_channels: Union[int, Tuple[int, int]],
            out_channels: int,
            heads: int = 1,
            concat: bool = True,
            negative_slope: float = 0.2,
            dropout: float = 0.0,
            add_self_loops: bool = True,
            edge_dim: Optional[int] = None,
            fill_value: Union[float, Tensor, str] = 'mean',
            bias: bool = True,
            share_weights: bool = False,
            skip_connections=False,
            **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights
        self.skip_connections = skip_connections

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=bias,
                                weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels, heads * out_channels,
                                    bias=bias, weight_initializer='glorot')
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels,
                                bias=bias, weight_initializer='glorot')
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(in_channels[1], heads * out_channels,
                                    bias=bias, weight_initializer='glorot')

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False,
                                   weight_initializer='glorot')
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor = None,
                return_attention_weights: bool = None):
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, OptTensor, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, OptTensor, bool) -> Tuple[Tensor, SparseTensor]  # noqa

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(
                    edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index, edge_attr, fill_value=self.fill_value,
                    num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form")

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr,
                             size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if self.skip_connections:
            out = F.elu(out) + out
        else:
            out = F.elu(out)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out, edge_index

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x += edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class GATExtractor(torch.nn.Module):
    def __init__(self, input_dim, feat_dim, n_head=8, is_add_self_loops=True, dropout_rate=0.5):
        super(GATExtractor, self).__init__()

        self.n_layer = 1
        self.dropout_rate = dropout_rate
        self.conv1 = [GATv2ConvWithElu(input_dim, feat_dim // n_head, heads=n_head, add_self_loops=is_add_self_loops)]
        self.conv1 = torch.nn.ModuleList(self.conv1)

        self.reg_params = list(self.conv1.parameters())
        self.is_add_self_loops = is_add_self_loops

    def forward(self, x, edge_index, edge_weight):
        h = x

        for i in range(self.n_layer):
            x, edge_index = self.conv1[i](x, edge_index, edge_weight)

        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        x = x + h

        return x
class IFEGAT(torch.nn.Module):
    def __init__(self, input_dim, feat_dim, n_cls,  y, n_head=8, is_add_self_loops=True, dropout_rate=0.5):
        super(IFEGAT, self).__init__()

        self.n_cls = n_cls
        self.y = y

        self.dropout_rate = dropout_rate

        self.wasserstein_distance = nn.MSELoss()
        self.x_encoder = GATv2ConvWithElu(input_dim, feat_dim // n_head, heads=n_head, add_self_loops=is_add_self_loops, skip_connections=False)
        self.global_encoder = GATv2ConvWithElu(feat_dim, feat_dim // n_head, heads=n_head, add_self_loops=is_add_self_loops, skip_connections=True)

        self.invariant_feature_extractor = GATExtractor(feat_dim, feat_dim, n_head, is_add_self_loops=is_add_self_loops, dropout_rate=dropout_rate)
        self.environment_feature_extractor = GATExtractor(feat_dim, feat_dim, n_head, is_add_self_loops=is_add_self_loops, dropout_rate=dropout_rate)

        self.augmentor = GatedAugmentor(feat_dim)

        self.classifier = torch.nn.Linear(feat_dim, n_cls)

    def forward_invariant(self, x, edge_index, only_vanilla_if=False):
        local_repr, edge_index = self.x_encoder(x, edge_index)
        global_repr, edge_index = self.global_encoder(local_repr, edge_index)

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
        local_repr, edge_index = self.x_encoder(x, edge_index)
        global_repr, edge_index = self.global_encoder(local_repr, edge_index)

        env_f = self.environment_feature_extractor(global_repr, edge_index, None)
        loss_dis = (self.wasserstein_distance(env_f, global_repr)
                    + self.wasserstein_distance(env_f, local_repr))

        env_f_pred = self.classifier(env_f)

        env_f_out = {
            'loss_dis': loss_dis,
            'env_f_pred': env_f_pred,
        }

        return env_f_out
