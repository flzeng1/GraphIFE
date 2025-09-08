import torch


class DWA:
    def __init__(self, num_losses=-1, T=2.0, device='cuda'):
        self.num_losses = num_losses
        self.T = T
        self.device = device
        self.loss_history = []
        self.weights = torch.ones(num_losses, device=device) if num_losses != -1 else None

    def update(self):
        if len(self.loss_history) < 2:
            return self.weights
        else:
            prev_loss = self.loss_history[-1]
            prev2_loss = self.loss_history[-2]
            r = prev_loss / (prev2_loss + 1e-8)
            r = r / self.T
            r = torch.exp(r)
            self.weights = r / r.sum()
            return self.weights

    def step(self, losses):
        if self.weights is None:
            self.weights = torch.ones(len(losses), device=self.device)

        if not isinstance(losses, torch.Tensor):
            losses = torch.tensor(losses, dtype=torch.float32, device=self.device)
        self.loss_history.append(losses)
        return self.update()