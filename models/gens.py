import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


@torch.no_grad()
def build_csr_neighbor_index(num_nodes, edge_index, device):
    edge_index = edge_index.clone().to(device)
    src, dst = edge_index[0], edge_index[1]
    order = torch.argsort(dst)
    sorted_dst = dst[order]
    neighbors = src[order]
    degree = torch.bincount(sorted_dst, minlength=num_nodes).long()
    rowptr = torch.empty(num_nodes + 1, dtype=torch.long, device=device)
    rowptr[0] = 0
    rowptr[1:] = torch.cumsum(degree, dim=0)
    return {
        'format': 'csr',
        'rowptr': rowptr,
        'neighbors': neighbors,
        'degree': degree,
    }


@torch.no_grad()
def get_ins_neighbor_dist(num_nodes, edge_index, train_mask, device, dense_numel_limit=200_000_000):
    if num_nodes * num_nodes > dense_numel_limit:
        return build_csr_neighbor_index(num_nodes, edge_index, device)

    train_mask = train_mask.clone().to(device)
    edge_index = edge_index.clone().to(device)
    row, col = edge_index[0], edge_index[1]

    neighbor_dist_list = []
    for j in range(num_nodes):
        neighbor_dist = torch.zeros(num_nodes, dtype=torch.float32).to(device)
        idx = row[(col == j)]
        neighbor_dist[idx] = neighbor_dist[idx] + 1
        neighbor_dist_list.append(neighbor_dist)

    neighbor_dist_list = torch.stack(neighbor_dist_list, dim=0)
    neighbor_dist_list = F.normalize(neighbor_dist_list, dim=1, p=1)

    return neighbor_dist_list


@torch.no_grad()
def sampling_idx_individual_dst(class_num_list, idx_info, device):
    max_num = max(class_num_list)
    sampling_list = max_num * torch.ones(len(class_num_list)) - torch.tensor(class_num_list)
    new_class_num_list = torch.Tensor(class_num_list).to(device)

    src_chunks = []
    for cls_idx, samp_num in zip(idx_info, sampling_list):
        sample_num = int(samp_num.item())
        if sample_num <= 0:
            continue
        cls_idx = cls_idx.to(device)
        if cls_idx.numel() == 0:
            continue
        sampled_pos = torch.randint(len(cls_idx), (sample_num,), device=device)
        src_chunks.append(cls_idx[sampled_pos])

    if not src_chunks:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty

    sampling_src_idx = torch.cat(src_chunks)
    prob = torch.log(new_class_num_list.float()) / new_class_num_list.float()
    prob = prob.repeat_interleave(new_class_num_list.long())
    temp_idx_info = torch.cat(idx_info).to(device)
    dst_idx = torch.multinomial(prob, sampling_src_idx.shape[0], True)
    sampling_dst_idx = temp_idx_info[dst_idx]

    sampling_src_idx, sorted_idx = torch.sort(sampling_src_idx)
    sampling_dst_idx = sampling_dst_idx[sorted_idx]
    return sampling_src_idx, sampling_dst_idx


@torch.no_grad()
def duplicate_neighbor(total_node, edge_index, sampling_src_idx):
    device = edge_index.device
    if sampling_src_idx.numel() == 0:
        return edge_index

    row, col = edge_index[0], edge_index[1]
    row, sort_idx = torch.sort(row)
    col = col[sort_idx]
    degree = scatter_add(torch.ones_like(col), col, dim=0, dim_size=total_node)
    new_row =(torch.arange(len(sampling_src_idx)).to(device)+ total_node).repeat_interleave(degree[sampling_src_idx])
    temp = scatter_add(torch.ones_like(sampling_src_idx), sampling_src_idx).to(device)

    node_mask = torch.zeros(total_node, dtype=torch.bool, device=device)
    unique_src = torch.unique(sampling_src_idx)
    node_mask[unique_src] = True
    row_mask = node_mask[row]
    edge_mask = col[row_mask]
    b_idx = torch.arange(len(unique_src)).to(device).repeat_interleave(degree[unique_src])
    edge_dense, _ = to_dense_batch(edge_mask, b_idx, fill_value=-1)
    if len(temp[temp!=0]) != edge_dense.shape[0]:
        cut_num =len(temp[temp!=0]) - edge_dense.shape[0]
        cut_temp = temp[temp!=0][:-cut_num]
    else:
        cut_temp = temp[temp!=0]
    edge_dense  = edge_dense.repeat_interleave(cut_temp, dim=0)
    new_col = edge_dense[edge_dense!= -1]
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index


def get_dist_kl(prev_out, sampling_src_idx, sampling_dst_idx):
    device = prev_out.device
    dist_kl = F.kl_div(torch.log(prev_out[sampling_dst_idx.to(device)]), prev_out[sampling_src_idx.to(device)], \
                    reduction='none').sum(dim=1,keepdim=True)
    dist_kl[dist_kl<0] = 0
    return dist_kl


def is_csr_neighbor_index(neighbor_dist_list):
    return isinstance(neighbor_dist_list, dict) and neighbor_dist_list.get('format') == 'csr'


@torch.no_grad()
def sparse_neighbor_sampling(total_node, edge_index, sampling_src_idx, sampling_dst_idx,
                             neighbor_index, prev_out, train_node_mask=None):
    device = edge_index.device
    sampling_src_idx = sampling_src_idx.clone().to(device)
    sampling_dst_idx = sampling_dst_idx.clone().to(device)
    rowptr = neighbor_index['rowptr'].to(device)
    neighbors = neighbor_index['neighbors'].to(device)
    degree = neighbor_index['degree'].to(device)

    if sampling_src_idx.numel() == 0:
        return edge_index, None

    if prev_out is not None:
        dist_kl = get_dist_kl(prev_out, sampling_src_idx, sampling_dst_idx)
        ratio = F.softmax(torch.cat([dist_kl.new_zeros(dist_kl.size(0), 1), -dist_kl], dim=1), dim=1)
        src_ratio = ratio[:, 0]
    else:
        dist_kl = None
        src_ratio = torch.ones(sampling_src_idx.numel(), device=device)

    if train_node_mask is None:
        train_node_mask = torch.ones_like(degree, dtype=torch.bool)

    observed_degree = degree[train_node_mask]
    if observed_degree.numel() == 0:
        return edge_index, dist_kl
    degree_dist = torch.bincount(observed_degree, minlength=int(degree.max().item()) + 1).float()
    if degree_dist.sum() <= 0:
        return edge_index, dist_kl

    aug_degree = torch.multinomial(degree_dist, sampling_src_idx.numel(), replacement=True).to(device)
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx]).long()
    if aug_degree.sum() <= 0:
        return edge_index, dist_kl

    syn_idx = torch.arange(sampling_src_idx.numel(), device=device).repeat_interleave(aug_degree)
    src_nodes = sampling_src_idx[syn_idx]
    dst_nodes = sampling_dst_idx[syn_idx]

    dst_degree = degree[dst_nodes]
    choose_src = torch.rand(syn_idx.numel(), device=device) < src_ratio[syn_idx]
    base_nodes = torch.where((~choose_src) & (dst_degree > 0), dst_nodes, src_nodes)
    base_degree = degree[base_nodes].clamp_min(1)
    starts = rowptr[base_nodes]

    offsets = (torch.rand(syn_idx.numel(), device=device) * base_degree.float()).long()
    new_col = neighbors[starts + offsets]
    new_row = total_node + syn_idx

    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)
    return new_edge_index, dist_kl


@torch.no_grad()
def neighbor_sampling(total_node, edge_index, sampling_src_idx, sampling_dst_idx,
        neighbor_dist_list, prev_out, train_node_mask=None):
    if is_csr_neighbor_index(neighbor_dist_list):
        return sparse_neighbor_sampling(total_node, edge_index, sampling_src_idx, sampling_dst_idx,
                                        neighbor_dist_list, prev_out, train_node_mask)

    device = edge_index.device
    n_candidate = 1
    sampling_src_idx = sampling_src_idx.clone().to(device)

    if prev_out is not None:
        sampling_dst_idx = sampling_dst_idx.clone().to(device)
        dist_kl = get_dist_kl(prev_out, sampling_src_idx, sampling_dst_idx)

        ratio = F.softmax(torch.cat([dist_kl.new_zeros(dist_kl.size(0),1), -dist_kl], dim=1), dim=1)

        mixed_neighbor_dist = ratio[:,:1] * neighbor_dist_list[sampling_src_idx]
        for i in range(n_candidate):
            mixed_neighbor_dist += ratio[:,i+1:i+2] * neighbor_dist_list[sampling_dst_idx.unsqueeze(dim=1)[:,i]]
    else:
        mixed_neighbor_dist = neighbor_dist_list[sampling_src_idx]

    col = edge_index[1]
    degree = scatter_add(torch.ones_like(col), col, dim=0, dim_size=total_node)
    if len(degree) < total_node:
        degree = torch.cat([degree, degree.new_zeros(total_node-len(degree))],dim=0)
    if train_node_mask is None:
        train_node_mask = torch.ones_like(degree,dtype=torch.bool)

    degree_dist = scatter_add(torch.ones_like(degree[train_node_mask]), degree[train_node_mask]).to(device).type(torch.float32)

    aug_degree = torch.multinomial(degree_dist.float(), len(sampling_src_idx), replacement=True).to(device)
    max_degree = degree.max().item() + 1
    aug_degree = torch.min(aug_degree, degree[sampling_src_idx])

    new_tgt = torch.multinomial(mixed_neighbor_dist + 1e-12, max_degree)
    tgt_index = torch.arange(max_degree).unsqueeze(dim=0).to(device)

    new_col = new_tgt[(tgt_index - aug_degree.unsqueeze(dim=1) < 0)]
    new_row = (torch.arange(len(sampling_src_idx)).to(device)+ total_node)
    new_row = new_row.repeat_interleave(aug_degree)
    inv_edge_index = torch.stack([new_col, new_row], dim=0)
    new_edge_index = torch.cat([edge_index, inv_edge_index], dim=1)

    return new_edge_index, dist_kl


class MeanAggregation(MessagePassing):
    def __init__(self):
        super(MeanAggregation, self).__init__(aggr='mean')

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, x=x)

class GatedAugmentor(nn.Module):
    def __init__(self, emb_dim):
        super(GatedAugmentor, self).__init__()
        self.gate_layer = nn.Linear(emb_dim, emb_dim)

    def forward(self, if_feature, env_feature):
        gate = torch.sigmoid(self.gate_layer(env_feature))
        aug_feature = if_feature + gate * env_feature
        return aug_feature, gate
