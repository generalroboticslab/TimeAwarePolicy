import os
import torch
import torch.nn as nn
from torch.distributions import Beta, Normal
from copy import deepcopy

from core.agents.networks import (
    FourierEncoding,
    MLP,
)
from core.agents.normalization import (
    NormalizeObservation,
)

MIN_STD = 0.05
INITIAL_STD = 1


class Agent(nn.Module):
    def __init__(self, envs, args, num_actions=None):
        super().__init__()
        self.args = args
        self.envs = envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cost = get_args_attr(args, "use_cost", False)
        self.tensor_dtype = torch.float32
        self.activation = nn.Tanh() if not args.use_relu else nn.ReLU()
        self.deterministic = args.deterministic
        self.hidden_size = args.hidden_size
        self.obs_dim = envs.num_observations
        self.state_dim = envs.num_states
        self.action_logits_num = envs.num_actions * 2 if num_actions is None else num_actions * 2

        # Layer Number
        self.num_hidden_layer = 3
        
        self.init_preprocess_net(envs, args)
        self.init_policy_net(envs, args)
        if self.use_cost:
            self.init_costs_net()
        self._initial_critic_states = {
            name: deepcopy(module.state_dict())
            for name, module in self.critic_modules().items()
        }
        self.to(self.tensor_dtype)


    def init_preprocess_net(self, envs, args):
        if get_args_attr(args, 'use_fourier', False):
            self.fourier_encoding = FourierEncoding(1, args.fourier_hidden_size)
            self.obs_dim += len(envs.fourier_idxs) * args.fourier_hidden_size
            self.state_dim += len(envs.fourier_idxs) * args.fourier_hidden_size

        # Input Normalization Layer
        if get_args_attr(args, 'norm_obs', False):
            self.obs_normalizer = NormalizeObservation(self.obs_dim, device=self.device)
            self.state_normalizer = NormalizeObservation(self.state_dim, device=self.device)
    
    
    def init_policy_net(self, envs, args):
        # Use MLP for the critic and actor
        self.critic = MLP(
            input_size=self.state_dim,
            hidden_size=self.hidden_size,
            output_size=1,
            num_hidden_layers=len(self.hidden_size) if type(self.hidden_size) in [list, tuple] else self.num_hidden_layer,
            use_relu=args.use_relu,
            init_std=1.0,
        ).to(self.device)
        self.actor = MLP( # The output is the mean only
            input_size=self.obs_dim,
            hidden_size=self.hidden_size,
            output_size=self.action_logits_num // 2,
            num_hidden_layers=len(self.hidden_size) if type(self.hidden_size) in [list, tuple] else self.num_hidden_layer,
            use_relu=args.use_relu,
            init_std=0.01,
        ).to(self.device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_logits_num // 2))


    def init_costs_net(self):
        self.critic_t = MLP(
            input_size=self.state_dim,
            hidden_size=self.hidden_size,
            output_size=1,
            num_hidden_layers=len(self.hidden_size) if type(self.hidden_size) in [list, tuple] else self.num_hidden_layer,
            use_relu=self.args.use_relu,
            init_std=1.0,
            output_softplus=True
        ).to(self.device)

        self.critic_inst = MLP(
            input_size=self.state_dim,
            hidden_size=self.hidden_size,
            output_size=1,
            num_hidden_layers=len(self.hidden_size) if type(self.hidden_size) in [list, tuple] else self.num_hidden_layer,
            use_relu=self.args.use_relu,
            init_std=1.0,
            output_softplus=True
        ).to(self.device)


    def get_value(self, raw_state):
        x = self.preprocess_state(raw_state)
        return self.critic(x), self.get_cost_value(x)
    

    def get_cost_value(self, x):
        ### ! We use x after the preprocess ###
        value_c = torch.cat([self.critic_t(x), self.critic_inst(x)], dim=-1) if self.use_cost else torch.zeros(x.shape[0], 2, device=self.device)
        return value_c


    def get_action_and_value(self, raw_obs, raw_state=None, action=None, action_only=False):
        x = self.preprocess_obs(raw_obs)
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        self.probs = probs # Record the current probs for logging
        if action is None:
            action = probs.mean if self.deterministic else probs.sample()
        if action_only:
            return action, probs
        
        logprob = probs.log_prob(action).sum(1)
        self.prob_entropy = probs.entropy() # Record the current probs for logging
        entropy = self.prob_entropy.sum(1)
        if torch.isnan(logprob).any() or torch.isinf(logprob).any():
            print("logprob has inf or nan")
            import ipdb; ipdb.set_trace()

        return action, action_mean, logprob, entropy, *self.get_value(raw_state)
    

    def preprocess_obs(self, obs):
        if get_args_attr(self.args, 'use_fourier', False):
            encoding_values = obs[..., self.envs.fourier_idxs].unsqueeze(-1)
            new_obs = self.fourier_encoding(encoding_values).flatten(start_dim=-2)
            obs = torch.cat([obs, new_obs], dim=-1)
        obs = self.normalize_obs(obs)
        return obs
    

    def preprocess_state(self, state):
        state = self.normalize_state(state)
        return state
    
    
    def normalize_obs(self, obs):
        if get_args_attr(self.args, 'norm_obs', False):
            return self.obs_normalizer(obs)
        return obs
    

    def normalize_state(self, state):
        if get_args_attr(self.args, 'norm_obs', False):
            return self.state_normalizer(state)
        return state


    def set_mode(self, mode='train'):
        if mode == 'train': 
            self.train()
        elif mode == 'eval': 
            self.eval()


    def critic_modules(self):
        """Return every value network owned by this policy."""
        modules = {"critic": self.critic}
        if self.use_cost:
            modules.update({
                "critic_t": self.critic_t,
                "critic_inst": self.critic_inst,
            })
        return modules


    def reset_critics(self):
        """Restore all reward and cost critics to their fresh initialization."""
        for name, module in self.critic_modules().items():
            module.load_state_dict(self._initial_critic_states[name])
    

    def save_checkpoint(self, folder_path, ckpt_name="eps", suffix="", ckpt_path=None, reward_normalizer=None, verbose=False):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if ckpt_path is None:
            ckpt_path = "{}/{}_{}".format(folder_path, ckpt_name, suffix)
        if verbose:
            print('Saving models to {}'.format(ckpt_path))
        torch.save(self.state_dict(), ckpt_path)
        
        if reward_normalizer is not None:
            save_checkpoint(reward_normalizer, folder_path, ckpt_name="rew_norm_eps", suffix=suffix, verbose=verbose)


    # Load model parameters
    def load_checkpoint(self, ckpt_path, evaluate=False, map_location='cuda:0', reset_critic=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path, map_location=map_location)
            self.load_state_dict(checkpoint, strict=False)
            if reset_critic:
                self.reset_critics()
                del self._initial_critic_states

            if evaluate: 
                self.set_mode('eval')
            else:
                self.set_mode('train')
                # Do we need to reset the actor_logstd here?
            
            if getattr(self.args, 'freeze', False):
                # freeze the model parameters apart from the last layer
                last_layer_num = list(self.state_dict().keys())[-1].split('.')[-2] # Example: "actor.mlp.7.weight"
                for name, param in self.named_parameters():
                    if last_layer_num not in name:
                        param.requires_grad = False
                        print(name, "is frozen", "shape", param.shape)



class BetaAgent(Agent):
    def __init__(self, envs, args, num_actions=None):
        super().__init__(envs, args, num_actions=num_actions)
        

    def init_policy_net(self, envs, args):
        # Use MLP for the critic and actor
        self.critic = MLP(
            input_size=self.state_dim,
            hidden_size=self.hidden_size,
            output_size=1,
            num_hidden_layers=len(self.hidden_size) if type(self.hidden_size) in [list, tuple] else self.num_hidden_layer,
            use_relu=args.use_relu,
            init_std=1.0,
        ).to(self.device)
        self.actor = MLP( # The output is the alpha and beta for the Beta distribution
            input_size=self.obs_dim,
            hidden_size=self.hidden_size,
            output_size=self.action_logits_num,
            num_hidden_layers=len(self.hidden_size) if type(self.hidden_size) in [list, tuple] else self.num_hidden_layer,
            use_relu=args.use_relu,
            init_std=0.01,
        ).to(self.device)
    

    def get_action_and_value(self, raw_obs, raw_state=None, action=None, action_only=False):
        obs = self.preprocess_obs(raw_obs)
        action_logalpha_logbeta = self.actor(obs)
        action_logalpha, action_logbeta = torch.chunk(action_logalpha_logbeta, 2, dim=-1)
        action_alpha = torch.exp(action_logalpha)
        action_beta = torch.exp(action_logbeta)
        probs = Beta(action_alpha, action_beta)
        self.probs = probs # Record the current probs for logging
        if action is None:
            action = probs.mean if self.deterministic else probs.sample()
        if action_only: # Only return the action and probs for evaluation
            return action, probs

        logprob = probs.log_prob(action).sum(-1) # log_prb means prob density not mass! This could be larger than 1
        self.prob_entropy = probs.entropy() # Record the current probs for logging to avoid repeated computation
        entropy = self.prob_entropy.sum(-1)
        
        return action, probs.mean, logprob, entropy, *self.get_value(raw_state)
    

    def logprob_saliency(self, raw_obs, raw_state=None):
        obs = self.preprocess_obs(raw_obs)
        with torch.enable_grad():
            obs.requires_grad_(True)
            action_logalpha_logbeta = self.actor(obs)
            action_logalpha, action_logbeta = torch.chunk(action_logalpha_logbeta, 2, dim=-1)
            action_alpha = torch.exp(action_logalpha)
            action_beta = torch.exp(action_logbeta)
            probs = Beta(action_alpha, action_beta)
            action = probs.mean if self.deterministic else probs.rsample()

            logprob = probs.log_prob(action).sum(-1) # action logprob; we do not compute per action grad but treat action as a whole

            self.actor.zero_grad(set_to_none=True)
            if obs.grad is not None:
                obs.grad.zero_()
            grad_out = torch.ones_like(logprob)
            logprob.backward(grad_out)
            grad_obs = obs.grad.detach()
        
        grad_obs = grad_obs.abs()
        grad_obs = grad_obs / (grad_obs.max(dim=-1, keepdim=True)[0] + 1e-10) # Normalize to [0, 1]

        return action, grad_obs


class SquashedNormalAgent(Agent):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5
    def __init__(self, envs, args, num_actions=None):
        super().__init__(envs, args, num_actions=num_actions)

        self.action_scale = 0.5
        self.action_bias = 0.5

    
    def init_policy_net(self, envs, args):
        # Use MLP for the critic and actor
        self.critic = MLP(
            input_size=self.state_dim,
            hidden_size=self.hidden_size,
            output_size=1,
            num_hidden_layers=len(self.hidden_size) if type(self.hidden_size) in [list, tuple] else self.num_hidden_layer,
            use_relu=args.use_relu,
            init_std=1.0,
        ).to(self.device)
        self.actor = MLP( # The output is the mean and logstd
            input_size=self.obs_dim,
            hidden_size=self.hidden_size,
            output_size=self.action_logits_num,
            num_hidden_layers=len(self.hidden_size) if type(self.hidden_size) in [list, tuple] else self.num_hidden_layer,
            use_relu=args.use_relu,
            init_std=0.01,
        ).to(self.device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_logits_num // 2))
        self.to(self.tensor_dtype)


    def get_action_and_value(self, raw_obs, raw_state=None, action=None, action_only=False):
        obs = self.preprocess_obs(raw_obs)
        action_mean_logstd = self.actor(obs)
        action_mean, action_logstd = torch.chunk(action_mean_logstd, 2, dim=-1)
        action_logstd = torch.tanh(action_logstd)
        action_logstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (action_logstd + 1)  # From SpinUp / Denis Yarats
        action_std = action_logstd.exp()
        probs = Normal(action_mean, action_std)
        self.probs = probs # Record the current probs for logging
        if action is None:
            raw_action = probs.mean if self.deterministic else probs.sample()
            action = self.squashed_action(raw_action)
        else:
            raw_action = self.unsquashed_action(action)
        
        if action_only:
            return action, probs

        logprob = self.squashed_logprob(probs, raw_action).sum(1) # Enforcing Action Bound
        self.prob_entropy = self.squashed_entropy(probs) # Record the current probs for logging
        entropy = self.prob_entropy.sum(1)
        
        return action, action_mean, logprob, entropy, *self.get_value(raw_state)
    

    def squashed_action(self, raw_action):
        return torch.tanh(raw_action) * self.action_scale + self.action_bias
    

    def unsquashed_action(self, action):
        # Clamp the action to avoid numerical issues
        tanh_raw_action = (action - self.action_bias) / self.action_scale
        clamped_tanh_raw_action = torch.clamp(tanh_raw_action, -0.999, 0.999)
        return torch.atanh(clamped_tanh_raw_action)


    def squashed_logprob(self, normal, raw_action):
        logprob = normal.log_prob(raw_action)
        if (logprob==torch.inf).any() or (logprob==-torch.inf).any() or torch.isnan(logprob).any():
            print("logprob has inf or nan")
            import ipdb; ipdb.set_trace()
        action = self.squashed_action(raw_action)
        logprob -= torch.log(self.action_scale * (1 - action.pow(2)) + 1e-6)
        logprob = torch.clamp(logprob, min=-15, max=15) # Clip the logprob to avoid NaN during the training)
        return logprob.sum(1, keepdim=True)


    def squashed_entropy(self, normal, num_samples=20000): # This seems not very good in on-policy rl training
        """Monte Carlo approximation of the entropy."""
        samples = normal.sample((num_samples,))
        log_prob = self.squashed_logprob(normal, samples)
        return -torch.mean(log_prob, dim=0)


    # MultivariateNormal does not support bfloat16
    # def get_action_and_value(self, x, action=None):
    #     action_mean = self.actor(x)
    #     action_logstd = self.actor_logstd.expand_as(action_mean)
    #     action_std = torch.exp(action_logstd)
    #     cov_mat = torch.diag_embed(action_std)
    #     probs = MultivariateNormal(action_mean, cov_mat)
    #     if action is None:
    #         action = action_mean if self.deterministic else probs.sample()
    #     return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class LSTMAgent(Agent):
    def __init__(self, envs, args, num_actions=None):
        super().__init__(envs, args, num_actions=num_actions)
        self.lstm_hidden_size = args.lstm_hidden_size
        assert isinstance(self.lstm_hidden_size, int), f"lstm_hidden_size must be an integer, but got {self.lstm_hidden_size}"
        
    
    def init_policy_net(self, envs, args):
        self.crt_lstm = nn.LSTM(
            input_size=self.state_dim,
            hidden_size=args.lstm_hidden_size,
            num_layers=1
        ).to(self.device)
        self.act_lstm = nn.LSTM(
            input_size=self.obs_dim,
            hidden_size=args.lstm_hidden_size,
            num_layers=1
        ).to(self.device)

        # Use MLP for the critic and actor
        self.critic = MLP(
            input_size=args.lstm_hidden_size,
            hidden_size=self.hidden_size,
            output_size=1,
            num_hidden_layers=len(self.hidden_size) if type(self.hidden_size) in [list, tuple] else self.num_hidden_layer,
            use_relu=args.use_relu,
            init_std=1.0,
        ).to(self.device)
        self.actor = MLP( # The output is the mean only
            input_size=args.lstm_hidden_size,
            hidden_size=self.hidden_size,
            output_size=self.action_logits_num,
            num_hidden_layers=len(self.hidden_size) if type(self.hidden_size) in [list, tuple] else self.num_hidden_layer,
            use_relu=args.use_relu,
            init_std=0.01,
        ).to(self.device)
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_logits_num // 2))


    def lstm_fw(self, lstm, x, lstm_state, done):
        """
        lstm_state: (hidden, cell)
        """
        obs_ft = x

        # LSTM forward
        batch_size = lstm_state[0].shape[1]
        # batch_first cannot process parallel computation (start from each sequence).
        # Sequence len is 1 when doing roll-out and will be 32 during training.
        obs_ft = obs_ft.reshape((-1, batch_size, lstm.input_size)) 
        done = done.reshape((-1, batch_size))
        new_ft = []
        for ft, d in zip(obs_ft, done):
            ft, lstm_state = lstm(
                ft.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_ft.append(ft)
        # Why flatten: same as the observation flatten in ppo_training.py. To align the observation
        new_ft = torch.flatten(torch.cat(new_ft), start_dim=0, end_dim=1)
        return new_ft, lstm_state

    
    def get_value(self, raw_state, lstm_state, done):
        x = self.preprocess_state(raw_state)
        crt_lstm_state = lstm_state[:2]
        crt_obs_ft, crt_lstm_state = self.lstm_fw(self.crt_lstm, x, crt_lstm_state, done)
        lstm_state = crt_lstm_state + lstm_state[2:]  # combine the lstm states to one tuple
        return self.critic(crt_obs_ft), lstm_state, self.get_cost_value(x)


    def get_action_and_value(self, raw_obs, lstm_state, done, raw_state=None, action=None, action_only=False):
        obs = self.preprocess_obs(raw_obs)
        crt_lstm_state, act_lstm_state = lstm_state[:2], lstm_state[2:]
        
        act_obs_ft, lstm_state = self.lstm_fw(self.act_lstm, obs, act_lstm_state, done)
        action_logalpha_logbeta = self.actor(act_obs_ft)
        action_logalpha, action_logbeta = torch.chunk(action_logalpha_logbeta, 2, dim=-1)
        action_alpha = torch.exp(action_logalpha)
        action_beta = torch.exp(action_logbeta)
        # If the model of action is complicated, creating the distribution will take more time.
        self.probs = probs = Beta(action_alpha, action_beta) # Record the current probs for logging
        lstm_state = crt_lstm_state + act_lstm_state # combine the lstm states to one tuple
        if action is None:
            action = probs.mean if self.deterministic else probs.sample()
        if action_only: # Only return the action and probs for evaluation
            return action, probs, lstm_state

        logprob = probs.log_prob(action).sum(1) # log_prb means prob density not mass! This could be larger than 1
        self.prob_entropy = probs.entropy() # Record the current probs for logging to avoid repeated computation
        entropy = self.prob_entropy.sum(1)
        
        return action, probs.mean, logprob, entropy, *self.get_value(raw_state, lstm_state, done)


def get_agent(envs, args, device='cuda'):
    agent = None
    if args.beta:
        if args.use_lstm:
            agent = LSTMAgent(envs, args).to(device)
        else:
            agent = BetaAgent(envs, args).to(device)
    elif args.squashed:
        agent = SquashedNormalAgent(envs, args).to(device)
    else:
        agent = Agent(envs, args).to(device)
        
    return agent
    

def get_meta_agent(envs, args, num_actions=1, device='cuda'):
    if args.beta:
        return BetaAgent(envs, args, num_actions=num_actions).to(device)
    else:
        raise NotImplementedError


def save_checkpoint(model, folder_path, ckpt_name="rew_eps", suffix="", ckpt_path=None, verbose=False):
    if model is None:
        return
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if ckpt_path is None:
        ckpt_path = "{}/{}_{}".format(folder_path, ckpt_name, suffix)
    if verbose:
        print('Saving models to {}'.format(ckpt_path))
    # Don't save the pc_extractor; we have weights offline
    # filtered_state_dict = {k: v for k, v in self.state_dict().items() if 'pc_extractor' not in k}
    filtered_state_dict = {k: v for k, v in model.state_dict().items()}
    torch.save(filtered_state_dict, ckpt_path)


# Load model parameters
def load_checkpoint(model, ckpt_path, evaluate=False, map_location='cuda:0'):
    print('Loading models from {}'.format(ckpt_path))
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        model.load_state_dict(checkpoint, strict=False)

        if evaluate: model.eval()
        else: model.train()
    return model


def control_len(lst, length=100): # control a list length to be a certain number
    if len(lst) <= length: return lst
    else: return lst[len(lst)-length:]


def update_tensor_buffer(buffer, new_v):
    len_v = len(new_v)
    if len_v == 0:
        return
    elif len_v > len(buffer):
        buffer[:] = new_v[len_v-len(buffer):]
    else:
        buffer[:-len_v] = buffer[len_v:].clone()
        buffer[-len_v:] = new_v


def get_args_attr(args, attr_name, default_v=None):
    if hasattr(args, attr_name): 
        return getattr(args, attr_name)
    return default_v
