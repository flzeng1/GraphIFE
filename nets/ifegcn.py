from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.nn.inits import glorot, zeros

from models import GatedAugmentor


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> PairTensor  # noqa
    pass


@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, Optional[int]) -> SparseTensor  # noqa
    pass


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class GCNConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 normalize: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.temp_weight = torch.nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.temp_weight.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, is_add_self_loops: bool = True) -> Tensor:
        original_size = edge_index.shape[1]

        x = self.temp_weight(x)

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, is_add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, is_add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out, edge_index

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class OneLayerGCN(nn.Module):
    def __init__(self, input_dim, feat_dim,
                 normalize=True, is_add_self_loops=True):
        super(OneLayerGCN, self).__init__()

        self.conv1 = [GCNConv(input_dim, feat_dim, cached=False, normalize=normalize)]

        self.conv1 = torch.nn.ModuleList(self.conv1)

        self.reg_params = list(self.conv1.parameters())

        self.is_add_self_loops = is_add_self_loops

    def forward(self, x, edge_index, edge_weight=None, is_add_self_loops=True):
        x, edge_index = self.conv1[0](x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)
        x = F.relu(x)

        return x, edge_index


class GCNExtractor(torch.nn.Module):
    def __init__(self, input_dim, feat_dim, normalize=True, is_add_self_loops=True, dropout_rate=0.5):
        super(GCNExtractor, self).__init__()

        self.n_layer = 1
        self.dropout_rate = dropout_rate
        self.conv1 = [OneLayerGCN(input_dim, feat_dim, normalize=normalize)]
        self.conv1 = torch.nn.ModuleList(self.conv1)

        self.reg_params = list(self.conv1.parameters())
        self.is_add_self_loops = is_add_self_loops

    def forward(self, x, edge_index, edge_weight):
        h = x

        for i in range(self.n_layer):
            x, edge_index = self.conv1[i](x, edge_index, edge_weight, is_add_self_loops=self.is_add_self_loops)

        x = F.dropout(x, training=self.training, p=self.dropout_rate)
        x = x + h

        return x


class IFEGCN(torch.nn.Module):
    def __init__(self, input_dim, feat_dim, n_cls,  y,\
                 normalize=True, is_add_self_loops=True, dropout_rate=0.5):
        super(IFEGCN, self).__init__()

        self.n_cls = n_cls
        self.y = y
        self.dropout_rate = dropout_rate

        self.wasserstein_distance = nn.MSELoss()
        self.x_encoder = OneLayerGCN(input_dim, feat_dim, normalize=normalize)
        self.global_encoder = OneLayerGCN(feat_dim, feat_dim, normalize=normalize)

        self.invariant_feature_extractor = GCNExtractor(feat_dim, feat_dim, normalize=normalize, is_add_self_loops=is_add_self_loops, dropout_rate=dropout_rate)
        self.environment_feature_extractor = GCNExtractor(feat_dim, feat_dim, normalize=normalize, is_add_self_loops=is_add_self_loops, dropout_rate=dropout_rate)

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
