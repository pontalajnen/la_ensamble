import torch


class SGLD(torch.optim.Optimizer):
    """Stochastic Gradient Langevin Dynamics (Welling & Teh, 2011).

    Update rule per step:
        θ ← θ − lr * (∇loss + weight_decay * θ) + N(0, 2 * lr * noise_factor)

    During burn-in call step(add_noise=False) to do standard SGD.
    After burn-in call step(add_noise=True) and periodically snapshot the model.
    """

    def __init__(self, params, lr: float = 1e-2, weight_decay: float = 0.0, noise_factor: float = 1.0):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = dict(lr=lr, weight_decay=weight_decay, noise_factor=noise_factor)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, add_noise: bool = True):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            nf = group["noise_factor"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if wd != 0.0:
                    grad = grad.add(p, alpha=wd)

                p.add_(grad, alpha=-lr)

                if add_noise:
                    noise_std = (2.0 * lr * nf) ** 0.5
                    p.add_(torch.randn_like(p), alpha=noise_std)

        return loss
