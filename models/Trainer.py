from collections import OrderedDict

import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm

from data_utils import compute_label_homophily_per_class, compute_node_weights
from models import sampling_idx_individual_dst, duplicate_neighbor, neighbor_sampling, MeanAggregation

import torch.nn.functional as F

import warnings

warnings.filterwarnings('ignore')


class NodeClassificationTrainer:
    def __init__(self, model, data, data_statistic, construct_data_mask, criterion, invariant_optimizer,
                 environment_optimizer, invariant_schedule, environment_schedule, multi_objective_optimizer, writer, n_cls, training_args):
        self.epochs = training_args.epochs
        self.warm_up = training_args.warmup
        self.device = training_args.device
        self.ife = training_args.ife
        self.early_stop = training_args.early_stop
        self.dis = training_args.dis
        self.pred_temp = training_args.pred_temp
        self.alpha = training_args.alpha
        self.verbose = training_args.verbose

        self.model = model
        self.data = data
        self.data_statistic = data_statistic
        self.construct_data_mask = construct_data_mask
        self.n_cls = n_cls
        self.criterion = criterion
        self.writer = writer

        self.invariant_optimizer = invariant_optimizer
        self.environment_optimizer = environment_optimizer
        self.invariant_schedule = invariant_schedule
        self.environment_schedule = environment_schedule

        self.multi_objective_optimizer = multi_objective_optimizer

        self.aggregator = MeanAggregation()

        self.causaler_branch = [model.x_encoder, model.global_encoder,
                                model.invariant_feature_extractor, model.classifier,
                                model.augmentor]

        self.attacker_branch = [model.environment_feature_extractor]

        self.temp_metrics = {
            'best_epoch': -1,
            'best_select_score': -1,
            'best_val_acc': 0,
            'best_val_bacc': 0,
            'best_val_f1': 0,
            'test_acc': 0,
            'test_bacc': 0,
            'test_f1': 0
        }
        self.prev_out = None

        self.nld_weight = self.calculate_nld_weight()

    def train(self):
        pbar = tqdm(range(self.epochs), desc="Training")

        patient = 0
        for epoch in pbar:
            self.train_one_epoch(epoch)

            accs, baccs, f1s, _ = self.evaluate()
            train_acc, val_acc, tmp_test_acc = accs
            val_bacc = baccs[1]
            tmp_test_bacc = baccs[2]
            tmp_val_f1 = f1s[1]
            tmp_test_f1 = f1s[2]
            select_score = val_acc

            if self.writer != None and self.verbose:
                self.writer.add_scalar('Acc/train', train_acc, epoch)
                self.writer.add_scalar('Acc/val', val_acc, epoch)
                self.writer.add_scalar('Acc/test', tmp_test_acc, epoch)
                self.writer.add_scalar('BAcc/val', val_bacc, epoch)
                self.writer.add_scalar('F1/val', tmp_val_f1, epoch)

            if select_score > self.temp_metrics['best_select_score']:
                self.temp_metrics['best_epoch'] = epoch
                self.temp_metrics['best_select_score'] = select_score
                self.temp_metrics['best_val_acc'] = val_acc
                self.temp_metrics['best_val_bacc'] = val_bacc
                self.temp_metrics['best_val_f1'] = tmp_val_f1
                self.temp_metrics['test_acc'] = tmp_test_acc
                self.temp_metrics['test_bacc'] = tmp_test_bacc
                self.temp_metrics['test_f1'] = tmp_test_f1
                pbar.set_postfix(OrderedDict([
                    ('best_epoch', f"{epoch:.4f}"),
                    ('best_val_acc', f"{val_acc:.4f}"),
                    ('best_test_acc', f"{tmp_test_acc:.4f}")
                ]))

                patient = 0
            elif self.early_stop and epoch > self.warm_up:
                patient += 1
                if patient == 200:
                    print('Early Stop..........................')
                    break

        if self.writer != None:
            self.writer.close()

        return (self.temp_metrics['best_val_acc'], self.temp_metrics['best_val_f1'],
                self.temp_metrics['test_acc'], self.temp_metrics['test_bacc'], self.temp_metrics['test_f1'])

    def train_one_epoch(self, current_epoch):
        self.set_network_train_eval(train_mode="causaler")

        self.invariant_optimizer.zero_grad()
        self.environment_optimizer.zero_grad()

        if self.ife:
            sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(
                self.data_statistic['class_num_list'], self.data_statistic['idx_info'], self.device)
            beta = torch.distributions.beta.Beta(2.0, 2.0)

            lam = beta.sample((len(sampling_src_idx),)).unsqueeze(1)

            if current_epoch > self.warm_up:
                with torch.no_grad():
                    self.prev_out = self.aggregator(self.prev_out, self.data.edge_index[:,
                                                                   self.construct_data_mask['train_edge_mask']])
                    self.prev_out = self.nld_weight * self.prev_out
                    self.prev_out = F.softmax(self.prev_out / self.pred_temp, dim=1).detach().clone()

                new_edge_index, _ = neighbor_sampling(self.data.x.size(0), self.data.edge_index[:,
                                                                           self.construct_data_mask['train_edge_mask']],
                                                      sampling_src_idx, sampling_dst_idx,
                                                      self.data_statistic['neighbor_dist_list'], self.prev_out,
                                                      self.construct_data_mask['train_node_mask'])

                new_x = self.node_mixup(self.data.x, sampling_src_idx, sampling_dst_idx, lam)

            else:
                new_edge_index = duplicate_neighbor(self.data.x.size(0), self.data.edge_index[:,
                                                                         self.construct_data_mask['train_edge_mask']],
                                                    sampling_src_idx)
                new_x = self.node_mixup(self.data.x, sampling_src_idx, sampling_dst_idx, lam)

            new_x.requires_grad = True

            add_num = new_x.shape[0] - self.construct_data_mask['data_train_mask'].shape[0]
            new_train_mask = torch.ones(add_num, dtype=torch.bool, device=self.data.x.device)
            new_train_mask = torch.cat((self.construct_data_mask['data_train_mask'], new_train_mask), dim=0)

            new_y = self.data.y[sampling_src_idx].clone()
            new_y = torch.cat((self.data.y[self.construct_data_mask['data_train_mask']], new_y), dim=0)

            if_out = self.model.forward_invariant(new_x, new_edge_index, False)
            self.prev_out = (if_out['vanilla_pred'][:self.data.x.size(0)]).detach().clone()

            node_w_if = self.compute_weighted_loss(if_out['vanilla_pred'][new_train_mask], new_y, self.n_cls)
            if_loss = self.criterion(if_out['vanilla_pred'][new_train_mask], new_y, reduction='none')
            if_loss = (if_loss * node_w_if).mean()

            augmentation_loss = self.criterion(if_out['augmentation_pred'][new_train_mask], new_y)

            current_loss = [if_loss.item(), augmentation_loss.item()]
            weights = self.multi_objective_optimizer.step(current_loss)

            reg_loss = if_out['gate_reg_loss']
            total_loss = weights[0] * if_loss + weights[1] * augmentation_loss + self.alpha * reg_loss

            total_loss.backward()
            self.invariant_optimizer.step()

            if self.writer != None and self.verbose:
                self.writer.add_scalar('Loss/train_if', if_loss, current_epoch)
                self.writer.add_scalar('Loss/augmentation_pred', augmentation_loss, current_epoch)
                self.writer.add_scalar('Loss/train_total', total_loss, current_epoch)
                self.writer.add_scalar('Loss/reg', reg_loss, current_epoch)
                self.writer.add_scalar('Weight/if', weights[0], current_epoch)
                self.writer.add_scalar('Weight/aug', weights[1], current_epoch)

            self.set_network_train_eval(train_mode="attacker")

            env_out = self.model.forward_environment(new_x, new_edge_index)

            loss_env_ext = self.criterion(env_out['env_f_pred'][new_train_mask], new_y)
            loss_dis = env_out['loss_dis']
            env_loss = loss_env_ext - self.dis * loss_dis
            (-env_loss).backward()
            self.environment_optimizer.step()

            if self.writer != None and self.verbose:
                self.writer.add_scalar('Loss/train_env', env_loss, current_epoch)
                self.writer.add_scalar('Loss/env_ext', loss_env_ext, current_epoch)
                self.writer.add_scalar('Loss/dis', loss_dis, current_epoch)

        with torch.no_grad():
            self.model.eval()
            output = self.model.forward_invariant(self.data.x,
                                                  self.data.edge_index[:, self.construct_data_mask['train_edge_mask']],
                                                  True)
            val_loss = F.cross_entropy(output[self.construct_data_mask['data_val_mask']],
                                       self.data.y[self.construct_data_mask['data_val_mask']])

        self.invariant_schedule.step(val_loss)
        self.environment_schedule.step()

        if self.writer != None and self.verbose:
            self.writer.add_scalar('Loss/val', val_loss, current_epoch)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        logits = self.model.forward_invariant(self.data.x,
                                              self.data.edge_index[:, self.construct_data_mask['train_edge_mask']],
                                              True)
        accs, baccs, f1s = [], [], []

        for i, mask in enumerate([self.construct_data_mask['data_train_mask'],
                                  self.construct_data_mask['data_val_mask'],
                                  self.construct_data_mask['data_test_mask']]):
            pred = logits[mask].max(1)[1]
            y_pred = pred.cpu().numpy()
            y_true = self.data.y[mask].cpu().numpy()
            acc = pred.eq(self.data.y[mask]).sum().item() / mask.sum().item()
            bacc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')

            accs.append(acc)
            baccs.append(bacc)
            f1s.append(f1)

        return accs, baccs, f1s, logits


    def set_network_train_eval(self, train_mode="causaler"):
        if train_mode == 'causaler':
            for net in self.causaler_branch:
                net.train()
                net.zero_grad()
            for net in self.attacker_branch:
                net.eval()
        else:
            for net in self.attacker_branch:
                net.train()
                net.zero_grad()
            for net in self.causaler_branch:
                net.eval()

    def node_mixup(self, x, sampling_src_idx, sampling_dst_idx, lam):
        new_src = x[sampling_src_idx.to(x.device), :].clone()
        new_dst = x[sampling_dst_idx.to(x.device), :].clone()
        lam = lam.to(x.device)

        mixed_node = lam * new_src + (1 - lam) * new_dst
        new_x = torch.cat([x, mixed_node], dim=0)
        return new_x

    def compute_weighted_loss(self, node_repr, labels, num_classes):
        device = node_repr.device
        N, D = node_repr.size()
        weights = torch.ones(N, device=device)

        for c in range(num_classes):
            class_mask = (labels == c)
            class_indices = class_mask.nonzero(as_tuple=False).squeeze()

            if class_indices.numel() <= 1:
                continue

            class_repr = node_repr[class_mask]
            mu_c = class_repr.mean(dim=0, keepdim=True)

            class_repr_norm = F.normalize(class_repr, p=2, dim=1)
            mu_c_norm = F.normalize(mu_c, p=2, dim=1)

            sim = (class_repr_norm @ mu_c_norm.T).squeeze()
            mean_sim = sim.mean()

            adjust_mask = sim < mean_sim
            adjustment = (mean_sim - sim[adjust_mask])
            local_weights = torch.ones_like(sim)
            local_weights[adjust_mask] += adjustment

            weights[class_indices] = local_weights

        return weights

    def calculate_nld_weight(self):
        class_avg_ratios, label_ratio, train_mask = compute_label_homophily_per_class(self.data, self.construct_data_mask['data_train_mask'])
        nld_weight = compute_node_weights(label_ratio, class_avg_ratios, train_mask, self.data)
        nld_weight = nld_weight.unsqueeze(1).to(self.device)

        return nld_weight
