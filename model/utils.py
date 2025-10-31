
class AdaptiveScheduler:
    def __init__(self, kl_threshold = 0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr
    

class LinearScheduler(AdaptiveScheduler):
    def __init__(self, start_lr, min_lr=1e-6, max_steps=1000000, apply_to_entropy=False, **kwargs):
        super().__init__()

        self.start_lr = start_lr
        self.min_lr = min_lr
        self.max_steps = max_steps
        self.apply_to_entropy = apply_to_entropy
        if apply_to_entropy:
            self.start_entropy_coef = kwargs.pop('start_entropy_coef', 0.01)
            self.min_entropy_coef = kwargs.pop('min_entropy_coef', 0.0001)

    def update(self, steps, entropy_coef=0.):
        mul = max(0, self.max_steps - steps)/self.max_steps 
        lr = self.min_lr + (self.start_lr - self.min_lr) * mul
        if self.apply_to_entropy:
            entropy_coef = self.min_entropy_coef + (self.start_entropy_coef - self.min_entropy_coef) * mul

        return lr, entropy_coef    


def linearAmplifier(start_v, max_v, cur_step, max_steps, curr_rate=1):
    mul = min(max_steps, curr_rate * cur_step) / max_steps
    next_v = start_v + (max_v - start_v) * mul
    return next_v