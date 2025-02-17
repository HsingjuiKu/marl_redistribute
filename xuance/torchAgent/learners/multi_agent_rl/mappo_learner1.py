from xuance.torchAgent.learners import *
from xuance.torchAgent.utils.value_norm import ValueNorm
from xuance.torchAgent.utils.operations import update_linear_decay

import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from redistribute import EnhancedCausalModel

class MAPPO_Clip_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 ):
        self.gamma = gamma
        self.clip_range = config.clip_range
        self.use_linear_lr_decay = config.use_linear_lr_decay
        self.use_grad_norm, self.max_grad_norm = config.use_grad_norm, config.max_grad_norm
        self.use_value_clip, self.value_clip_range = config.use_value_clip, config.value_clip_range
        self.use_huber_loss, self.huber_delta = config.use_huber_loss, config.huber_delta
        self.use_value_norm = config.use_value_norm
        self.use_global_state = config.use_global_state
        self.vf_coef, self.ent_coef = config.vf_coef, config.ent_coef
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(reduction="none", delta=self.huber_delta)
        super(MAPPO_Clip_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)
        if self.use_value_norm:
            self.value_normalizer = ValueNorm(1).to(device)
        else:
            self.value_normalizer = None
        self.lr = config.learning_rate
        self.end_factor_lr_decay = config.end_factor_lr_decay

        self.causal_model = EnhancedCausalModel(config.n_agents, config.obs_shape[0], config.act_shape[0], device)
        self.n_iters = config.running_steps

    def lr_decay(self, i_step):
        if self.use_linear_lr_decay:
            update_linear_decay(self.optimizer, i_step, self.running_steps, self.lr, self.end_factor_lr_decay)

    def update(self, sample):
        info = {}
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        values = torch.Tensor(sample['values']).to(self.device)
        returns = torch.Tensor(sample['returns']).to(self.device)
        advantages = torch.Tensor(sample['advantages']).to(self.device)
        log_pi_old = torch.Tensor(sample['log_pi_old']).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
        batch_size = obs.shape[0]
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)

        # Calculate social contribution index and redistribute rewards
        social_contribution_index = self.causal_model.calculate_social_contribution_index(obs, actions)
        tax_rates = self.causal_model.calculate_tax_rates(social_contribution_index)
        alpha = 1.0 - (self.iterations / self.n_iters)  # Calculate alpha which decays from 1 to 0 over iterations
        new_returns = self.causal_model.redistribute_rewards(returns, social_contribution_index, tax_rates, beta=0.5, alpha=alpha)

        # actor loss
        _, pi_dist = self.policy(obs, IDs)
        log_pi = pi_dist.log_prob(actions)
        ratio = torch.exp(log_pi - log_pi_old).reshape(batch_size, self.n_agents, 1)
        advantages_mask = advantages.detach() * agent_mask
        surrogate1 = ratio * advantages_mask
        surrogate2 = torch.clip(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages_mask
        loss_a = -torch.sum(torch.min(surrogate1, surrogate2), dim=-2, keepdim=True).mean()

        # entropy loss
        entropy = pi_dist.entropy().reshape(agent_mask.shape) * agent_mask
        loss_e = entropy.mean()

        # critic loss
        critic_in = torch.Tensor(obs).reshape([batch_size, 1, -1]).to(self.device)
        critic_in = critic_in.expand(-1, self.n_agents, -1)
        _, value_pred = self.policy.get_values(critic_in, IDs)
        value_pred = value_pred.reshape(-1, 1)
        value_target = new_returns.reshape(-1, 1)
        values = values.reshape(-1, 1)
        agent_mask_flatten = agent_mask.reshape(-1, 1)
        if self.use_value_clip:
            value_clipped = values + (value_pred - values).clamp(-self.value_clip_range, self.value_clip_range)
            if self.use_value_norm:
                self.value_normalizer.update(value_target)
                value_target = self.value_normalizer.normalize(value_target)
            if self.use_huber_loss:
                loss_v = self.huber_loss(value_pred, value_target)
                loss_v_clipped = self.huber_loss(value_clipped, value_target)
            else:
                loss_v = (value_pred - value_target) ** 2
                loss_v_clipped = (value_clipped - value_target) ** 2
            loss_c = torch.max(loss_v, loss_v_clipped) * agent_mask_flatten
            loss_c = loss_c.sum() / agent_mask_flatten.sum()
        else:
            if self.use_value_norm:
                self.value_normalizer.update(value_target)
                value_target = self.value_normalizer.normalize(value_target)
            if self.use_huber_loss:
                loss_v = self.huber_loss(value_pred, value_target) * agent_mask_flatten
            else:
                loss_v = ((value_pred - value_target) ** 2) * agent_mask_flatten
            loss_c = loss_v.sum() / agent_mask_flatten.sum()

        loss = loss_a + self.vf_coef * loss_c - self.ent_coef * loss_e
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            info["gradient_norm"] = grad_norm.item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "actor_loss": loss_a.item(),
            "critic_loss": loss_c.item(),
            "entropy": loss_e.item(),
            "loss": loss.item(),
            "predict_value": value_pred.mean().item()
        })

        return info

    def update_recurrent(self, sample):
        info = {}
        self.iterations += 1
        state = torch.Tensor(sample['state']).to(self.device)
        if self.use_global_state:
            state = state.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        values = torch.Tensor(sample['values']).to(self.device)
        returns = torch.Tensor(sample['returns']).to(self.device)
        advantages = torch.Tensor(sample['advantages']).to(self.device)
        log_pi_old = torch.Tensor(sample['log_pi_old']).to(self.device)
        avail_actions = torch.Tensor(sample['avail_actions']).float().to(self.device)
        filled = torch.Tensor(sample['filled']).float().to(self.device)
        batch_size = obs.shape[0]
        episode_length = actions.shape[2]
        IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, episode_length + 1, -1).to(
            self.device)

        # Calculate social contribution index and redistribute rewards
        social_contribution_index = self.causal_model.calculate_social_contribution_index(obs, actions)
        tax_rates = self.causal_model.calculate_tax_rates(social_contribution_index)
        alpha = 1.0 - (self.iterations / self.n_iters)  # Calculate alpha which decays from 1 to 0 over iterations
        new_returns = self.causal_model.redistribute_rewards(returns, social_contribution_index, tax_rates, beta=0.5, alpha=alpha)

        # actor loss
        rnn_hidden_actor = self.policy.representation.init_hidden(batch_size * self.n_agents)
        _, pi_dist = self.policy(obs[:, :, :-1].reshape(-1, episode_length, self.dim_obs),
                                 IDs[:, :, :-1].reshape(-1, episode_length, self.n_agents),
                                 *rnn_hidden_actor,
                                 avail_actions=avail_actions[:, :, :-1].reshape(-1, episode_length, self.dim_act))
        log_pi = pi_dist.log_prob(actions.reshape(-1, episode_length)).reshape(batch_size, self.n_agents, episode_length)
        ratio = torch.exp(log_pi - log_pi_old).unsqueeze(-1)
        filled_n = filled.unsqueeze(1).expand(batch_size, self.n_agents, episode_length, 1)
        surrogate1 = ratio * advantages
        surrogate2 = torch.clip(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        loss_a = -(torch.min(surrogate1, surrogate2) * filled_n).sum() / filled_n.sum()

        # entropy loss
        entropy = pi_dist.entropy().reshape(batch_size, self.n_agents, episode_length, 1)
        entropy = entropy * filled_n
        loss_e = entropy.sum() / filled_n.sum()

        # critic loss
        rnn_hidden_critic = self.policy.representation_critic.init_hidden(batch_size * self.n_agents)
        if self.use_global_state:
            critic_in_obs = obs[:, :, :-1].transpose(1, 2).reshape(batch_size, episode_length, -1)
            critic_in_obs = critic_in_obs.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
            critic_in_state = state[:, :, :-1]
            critic_in = torch.concat([critic_in_obs, critic_in_state], dim=-1)
        else:
            critic_in = obs[:, :, :-1].transpose(1, 2).reshape(batch_size, episode_length, -1)
            critic_in = critic_in.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
        _, value_pred = self.policy.get_values(critic_in, IDs[:, :, :-1], *rnn_hidden_critic)
        value_target = new_returns.reshape(-1, 1)
        values = values.reshape(-1, 1)
        value_pred = value_pred.reshape(-1, 1)
        filled_all = filled_n.reshape(-1, 1)
        if self.use_value_clip:
            value_clipped = values + (value_pred - values).clamp(-self.value_clip_range, self.value_clip_range)
            if self.use_value_norm:
                self.value_normalizer.update(value_target)
                value_target = self.value_normalizer.normalize(value_target)
            if self.use_huber_loss:
                loss_v = self.huber_loss(value_pred, value_target)
                loss_v_clipped = self.huber_loss(value_clipped, value_target)
            else:
                loss_v = (value_pred - value_target) ** 2
                loss_v_clipped = (value_clipped - value_target) ** 2
            loss_c = torch.max(loss_v, loss_v_clipped) * filled_all
            loss_c = loss_c.sum() / filled_all.sum()
        else:
            if self.use_value_norm:
                self.value_normalizer.update(value_target)
                value_target = self.value_normalizer.normalize(value_target)
            if self.use_huber_loss:
                loss_v = self.huber_loss(value_pred, value_target)
            else:
                loss_v = (value_pred - value_target) ** 2
            loss_c = (loss_v * filled_all).sum() / filled_all.sum()

        loss = loss_a + self.vf_coef * loss_c - self.ent_coef * loss_e
        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            info["gradient_norm"] = grad_norm.item()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        # Logger
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info.update({
            "learning_rate": lr,
            "actor_loss": loss_a.item(),
            "critic_loss": loss_c.item(),
            "entropy": loss_e.item(),
            "loss": loss.item(),
            "predict_value": value_pred.mean().item()
        })

        return info
