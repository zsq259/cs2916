from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # raise NotImplementedError()

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1 = group["betas"][0]
                beta2 = group["betas"][1]
                eps = group["eps"]                                        

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # First moment vector
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Second moment vector
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq'] #m1, m2
                state['step'] += 1

                # Update first and second moments of the gradients
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980

                corrected_exp_avg = exp_avg / (1 - beta1 ** state['step'])                
                corrected_exp_avg_sq = exp_avg_sq / (1 - beta2 ** state['step'])

                # Update parameters
                denom = corrected_exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(-alpha, corrected_exp_avg, denom)

                # Add weight decay after the main gradient-based updates                
                p.data.add_(-alpha * group['weight_decay'], p.data)
                state['exp_avg'] = exp_avg
                state['exp_avg_sq'] = exp_avg_sq


        return loss