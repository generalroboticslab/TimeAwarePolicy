import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical, MultivariateNormal, Beta

from copy import deepcopy
import math

MIN_STD = 0.05
INITIAL_STD = 1


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def lstm_init(lstm, std=1.):
    for name, param in lstm.named_parameters():
        if 'weight' in name:
            torch.nn.init.orthogonal_(param, std)
        elif 'bias' in name:
            torch.nn.init.constant_(param, 0)
    return lstm
    

class FourierEncoding(nn.Module):
    def __init__(self, in_features, out_features, scale=1.0):
        super(FourierEncoding, self).__init__()
        assert out_features % 2 == 0, f"out_features (now is {out_features}) must be an even number"
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        weights = scale * torch.randn(in_features, out_features // 2)
        self.weights = nn.Parameter(weights, requires_grad=False)
    
    def forward(self, x):
        # x = 2 * torch.pi * x # x ranges from 0 to 1?
        return torch.concatenate([torch.sin(x @ self.weights), torch.cos(x @ self.weights)], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, batch_first=True):
        super().__init__()
        # Pad d_model to be even if it's odd
        self.batch_first = batch_first
        self.pad = (d_model % 2 != 0)
        if self.pad:
            d_model += 1

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        if not batch_first: # (batch_size, seq_len, d_model) -> (seq_len, batch_size, d_model)
            pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.pad:
            # Pad the input with zeros on the last dimension to make it even
            zero_padding = torch.zeros((x.size(0), x.size(1), 1), device=x.device)
            x = torch.cat((x, zero_padding), dim=-1)

        if self.batch_first: x = x + self.pe[:, :x.size(1)]
        else: x = x + self.pe[:x.size(0), :]

        if self.pad:
            # Remove the padded 0 after adding the positional encoding
            x = x[..., :-1]
        return x


class Transfromer_Linear(nn.Module):
    def __init__(self, input_size, hidden_size, num_transf_layers, lin_input_size, num_linear_layers, output_size, nhead=1, batch_first=True, init_std=0.01) -> None:
        super().__init__()
        self.positional_encoding = PositionalEncoding(input_size, batch_first=batch_first)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size, 
            nhead=nhead, 
            dim_feedforward=hidden_size, 
            batch_first=batch_first
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, 
            num_layers=num_transf_layers, 
            norm=nn.LayerNorm(input_size)
        )
        self.linear = MLP(
            input_size=lin_input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_hidden_layers=num_linear_layers,
            init_std=init_std
        )
    
    def forward(self, x):
        x = self.positional_encoding(x)
        embeddings = self.transformer(x)
        # last_embedding = embeddings[:, -1, :] # only use the last layer output
        # flatten the whole embeddings
        last_embedding = torch.flatten(embeddings, start_dim=1)
        output = self.linear(last_embedding)
        return output
    

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=None, use_relu=True, output_layernorm=False, output_softplus=False, init_std=1., auto_flatten=False, flatten_start_dim=1):
        super().__init__()
        self.activation = nn.ReLU() if use_relu else nn.Tanh()
        self.mlp = nn.Sequential() \
                   if not auto_flatten \
                   else nn.Sequential(AutoFlatten(start_dim=flatten_start_dim))
        
        if type(hidden_size) not in [list, tuple]:
            assert num_hidden_layers is not None, f"num_hidden_layers must be specified if hidden_size is an int number {hidden_size}"
            hidden_size = [hidden_size] * num_hidden_layers
        else:
            num_hidden_layers = len(hidden_size)

        for i in range(num_hidden_layers):
            input_shape = input_size if i==0 else hidden_size[i-1]
            self.mlp.append(layer_init(nn.Linear(input_shape, hidden_size[i])))
            self.mlp.append(nn.LayerNorm(hidden_size[i]))
            self.mlp.append(self.activation)
        self.mlp.append(layer_init(nn.Linear(hidden_size[i], output_size), std=init_std))
        if output_layernorm:
            self.mlp.append(nn.LayerNorm(output_size))
        if output_softplus:
            self.mlp.append(nn.Softplus())
    
    def forward(self, x):
        return self.mlp(x)
    

class PC_Encoder(nn.Module):
    def __init__(self, channels=3, output_dim=256):
        super().__init__()
        # Simple pointNet model
        # We only use xyz (channels=3) in this work
        # while our encoder also works for xyzrgb (channels=6) in our experiments
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(channels, 64)), 
            nn.LayerNorm(64), 
            nn.ReLU(),
            layer_init(nn.Linear(64, 128)), 
            nn.LayerNorm(128), 
            nn.ReLU(),
            layer_init(nn.Linear(128, 256)), 
            nn.LayerNorm(256), 
            nn.ReLU()
        )
        self.projection = nn.Sequential(
            layer_init(nn.Linear(256, output_dim)), 
            nn.LayerNorm(output_dim)
        )
    def forward(self, x):
        # x: B, N, 3
        x = self.mlp(x) # B, N, 256
        x = torch.max(x, 1)[0] # B, 256
        x = self.projection(x) # B, Output_dim
        return x
    

class HistoryEncoder(nn.Module):
    def __init__(self, envs, input_size, hidden_size, num_hidden_layers, output_size, act_size=6, use_relu=True, init_std=1.):
        super().__init__()
        
        self.envs = envs
        self.traj_hist_encoder = MLP(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            num_hidden_layers=num_hidden_layers
        )
        self.traj_hist_decoder = MLP(
            input_size=output_size,
            hidden_size=hidden_size,
            output_size=act_size,
            num_hidden_layers=2
        )
    

    def encoding(self, obs_list):
        """
        Preprocess the sequence observation and extract the features
        input seq_obs: (Env, Seq, Dim1)
        output seq_obs_ft: (Env, Dim2)
        """
        with torch.no_grad():
            traj_hist_ft, _, act_hist = self.forward(obs_list)
            post_seq_obs = torch.cat([traj_hist_ft, act_hist], dim=-1)
        
        if self.args.use_seq_obs_encoder:
            post_seq_obs = self.seq_obs_encoder(post_seq_obs)
        else:
            # Flatten the sequence observation from (Env, Seq, Dim) to (Env, Seq*Dim)
            post_seq_obs = torch.flatten(post_seq_obs, start_dim=1)
        return post_seq_obs


    def forward(self, obs_list, pred_act=False):
        seq_obs, scene_ft_tensor, obj_ft_tensor = obs_list
        qr_region_hist = seq_obs[:, :, self.envs.qr_region_slice]
        converted_act_hist = seq_obs[:, :, self.envs.converted_act_slice]
        traj_history = seq_obs[:, :, self.envs.traj_history_slice]
        act_hist = seq_obs[:, :, self.envs.action_slice]

        # TODO: The traj_hist contains the pos information, which will let the model to cheat
        scene_ft_tensor = scene_ft_tensor.unsqueeze(1).expand(-1, seq_obs.shape[1], -1)
        obj_ft_tensor = obj_ft_tensor.unsqueeze(1).expand(-1, seq_obs.shape[1], -1)
        raw_obs = torch.cat([scene_ft_tensor, qr_region_hist, obj_ft_tensor, converted_act_hist, traj_history], dim=-1)

        traj_hist_ft = self.traj_hist_encoder(raw_obs)
        traj_hist_ft = traj_hist_ft.view(*seq_obs.shape[:2], self.envs.history_ft_dim)

        if pred_act:
            pred_act_hist = self.traj_hist_decoder(traj_hist_ft)
        else:
            pred_act_hist = None
        return traj_hist_ft, pred_act_hist, act_hist


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
        self.init_critic_params = deepcopy(self.critic.state_dict())
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
                self.critic.load_state_dict(self.init_critic_params)
                del self.init_critic_params
            
            # try:
            #     self.load_state_dict(checkpoint, strict=False)
            # except RuntimeError as e:
            #     for name, param in checkpoint.items():
            #         if len(param.shape) == 2:
            #             ckpt_dim = param.shape[1]
            #             model_dim = self.state_dict()[name].shape[1]
            #             if ckpt_dim != model_dim:
            #                 self.state_dict()[name][:, :ckpt_dim].copy_(param)
            #                 print(f"############ Waring: The checkpoint is not strictly loaded. The ckpt input_dim is {ckpt_dim} while the current model is {model_dim} ###########")
            #             else:
            #                 self.state_dict()[name].copy_(param)
            #         elif len(param.shape) == 1:
            #             ckpt_dim = param.shape[0]
            #             model_dim = self.state_dict()[name].shape[0]
            #             if ckpt_dim != model_dim:
            #                 self.state_dict()[name][:ckpt_dim].copy_(param)
            #                 print(f"############ Waring: The checkpoint is not strictly loaded. The ckpt input_dim is {ckpt_dim} while the current model is {model_dim} ###########")
            #             else:
            #                 self.state_dict()[name].copy_(param)
            #         else:
            #             self.state_dict()[name].copy_(param)
                    
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


class RunningMeanStd(nn.Module):
    '''
    updates statistic from a full data
    '''
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False, device='cuda'):
        super(RunningMeanStd, self).__init__()
        self.insize = insize
        self.epsilon = epsilon

        self.norm_only = norm_only
        self.per_channel = per_channel
        if per_channel:
            if len(self.insize) == 3:
                self.axis = [0,2,3]
            if len(self.insize) == 2:
                self.axis = [0,2]
            if len(self.insize) == 1:
                self.axis = [0]
            in_size = self.insize[0] 
        else:
            self.axis = [0]
            in_size = insize

        self.register_buffer("running_mean", torch.zeros(in_size, dtype = torch.float64, device=device))
        self.register_buffer("running_var", torch.ones(in_size, dtype = torch.float64, device=device))
        self.register_buffer("count", torch.ones((), dtype = torch.float64, device=device))

    
    def reset(self, reset_slice=None):
        if reset_slice is None:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.count.fill_(0)
        else:
            self.running_mean[reset_slice].zero_()
            self.running_var[reset_slice].fill_(1)
            self.count[reset_slice].fill_(0)
        
    
    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        with torch.no_grad():
            batch_mean = x.mean(self.axis)
            batch_var = x.var(self.axis)
            batch_count = x.size()[0]
            self.running_mean, self.running_var, self.count = \
                self.update_mean_var_count_from_moments(self.running_mean, self.running_var, self.count, batch_mean, batch_var, batch_count)


    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count


    def forward(self, input, denorm=False):
        with torch.no_grad():
            if self.training:
                self.update(input)

            # change shape
            if self.per_channel:
                if len(self.insize) == 3:
                    current_mean = self.running_mean.view([1, self.insize[0], 1, 1]).expand_as(input)
                    current_var = self.running_var.view([1, self.insize[0], 1, 1]).expand_as(input)
                if len(self.insize) == 2:
                    current_mean = self.running_mean.view([1, self.insize[0], 1]).expand_as(input)
                    current_var = self.running_var.view([1, self.insize[0], 1]).expand_as(input)
                if len(self.insize) == 1:
                    current_mean = self.running_mean.view([1, self.insize[0]]).expand_as(input)
                    current_var = self.running_var.view([1, self.insize[0]]).expand_as(input)        
            else:
                current_mean = self.running_mean
                current_var = self.running_var
            # get output


            if denorm:
                y = torch.clamp(input, min=-5.0, max=5.0)
                y = torch.sqrt(current_var.float() + self.epsilon)*y + current_mean.float()
            else:
                if self.norm_only:
                    y = input / torch.sqrt(current_var.float() + self.epsilon)
                else:
                    y = (input - current_mean.float()) / torch.sqrt(current_var.float() + self.epsilon)
                    y = torch.clamp(y, min=-5.0, max=5.0)
            return y


class NormalizeObservation(nn.Module):
    def __init__(self, insize, epsilon=1e-05, per_channel=False, norm_only=False, device='cuda'):
        super(NormalizeObservation, self).__init__()
        print("Creating Normalizer NormalizeObservation | Size: ", insize)
        self.running_mean_std = RunningMeanStd(insize, epsilon, per_channel, norm_only, device)
    
    def reset(self):
        self.running_mean_std.reset()

    def forward(self, input, denorm=False):
        return self.running_mean_std(input, denorm)
    

class NormalizeReward(nn.Module):
    def __init__(self, num_envs, insize: int = 1, gamma: float = 0.99, epsilon: float = 1e-8, device='cuda'):
        super(NormalizeReward, self).__init__()
        print("Creating Normalizer NormalizeReward | Size: ", insize)
        self.return_rms = RunningMeanStd(insize, epsilon, norm_only=True)
        self.returns = torch.zeros((num_envs, insize), dtype=torch.float32, device=device)
        self.gamma = gamma if insize==1 else gamma.repeat(num_envs, 1)
        self.epsilon = epsilon

    
    def reset(self):
        self.return_rms.reset()
        self.returns = torch.zeros_like(self.returns)


    def normalize(self, rews, dones):
        """Normalizes the step rewards with the running mean returns and their variance."""
        org_shape = rews.shape
        rews = rews.view(self.returns.shape)
        self.returns = self.returns * self.gamma * (1 - dones).view(-1, 1) + rews
        self.return_rms.update(self.returns)
        post_rews = rews / torch.sqrt(self.return_rms.running_var + self.epsilon)
        return post_rews.view(org_shape)  # Return to the original shape


# Misc Functions
class AutoFlatten(nn.Module):
    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim
    
    def forward(self, x):
        return torch.flatten(x, start_dim=self.start_dim)
    

def bound_loss(mu, soft_bound=1.0):
    # constrain the action within [-soft_bound, soft_bound]
    mu_loss_high = torch.maximum(mu - soft_bound, torch.tensor(0, device=mu.device))**2
    mu_loss_low = torch.minimum(mu + soft_bound, torch.tensor(0, device=mu.device))**2
    b_loss = (mu_loss_low + mu_loss_high).mean()
    return b_loss
    

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


if __name__ == "__main__":
    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor_std = nn.Parameter(torch.zeros(1, 2), requires_grad=False)
