"""
Multi-Agent Deep Deterministic Policy Gradient
Paper link:
https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf
Implementation: Pytorch
Trick: Parameter sharing for all agents, with agents' one-hot IDs as actor-critic's inputs.
"""
import torch
from torch import nn
from xuance.torch.learners import LearnerMAS
from typing import Optional, List
from argparse import Namespace
from operator import itemgetter
from ASIRR import EnhancedCausalModel

class MADDPG_Learner(LearnerMAS):
    def __init__(self, config: Namespace, model_keys: List[str], agent_keys: List[str], episode_length: int, policy: nn.Module, optimizer: Optional[dict], scheduler: Optional[dict] = None):
        self.gamma = config.gamma
        self.tau = config.tau
        self.mse_loss = nn.MSELoss()
        super(MADDPG_Learner, self).__init__(config, model_keys, agent_keys, episode_length, policy, optimizer, scheduler)
        self.optimizer = {key: {'actor': optimizer[key][0], 'critic': optimizer[key][1]} for key in self.model_keys}
        self.scheduler = {key: {'actor': scheduler[key][0], 'critic': scheduler[key][1]} for key in self.model_keys}
        self.causal_model = EnhancedCausalModel(len(agent_keys), config.obs_shape[0], config.act_shape[0], config.device)

    def update(self, sample):
        self.iterations += 1
        info = {}

        # prepare training data
        sample_Tensor = self.build_training_data(sample, use_parameter_sharing=self.use_parameter_sharing, use_actions_mask=False)
        batch_size = sample_Tensor['batch_size']
        obs = sample_Tensor['obs']
        actions = sample_Tensor['actions']
        obs_next = sample_Tensor['obs_next']
        rewards = sample_Tensor['rewards']
        terminals = sample_Tensor['terminals']
        agent_mask = sample_Tensor['agent_mask']
        IDs = sample_Tensor['agent_ids']

        alpha = 1.0 - (self.iterations / self.config.n_iters)  # Alpha decays from 1 to 0

        if self.use_parameter_sharing:
            key = self.model_keys[0]
            bs = batch_size * self.n_agents
            obs_joint = obs[key].reshape(batch_size, -1)
            next_obs_joint = obs_next[key].reshape(batch_size, -1)
            actions_joint = actions[key].reshape(batch_size, -1)
            rewards[key] = rewards[key].reshape(batch_size * self.n_agents)
            terminals[key] = terminals[key].reshape(batch_size * self.n_agents)
        else:
            bs = batch_size
            obs_joint = torch.cat(itemgetter(*self.agent_keys)(obs), dim=-1).reshape(batch_size, -1)
            next_obs_joint = torch.cat(itemgetter(*self.agent_keys)(obs_next), dim=-1).reshape(batch_size, -1)
            actions_joint = torch.cat(itemgetter(*self.agent_keys)(actions), dim=-1).reshape(batch_size, -1)

        # get actions
        _, actions_eval = self.policy(observation=obs, agent_ids=IDs)
        _, actions_next = self.policy.Atarget(next_observation=obs_next, agent_ids=IDs)
        # get values
        if self.use_parameter_sharing:
            key = self.model_keys[0]
            actions_next_joint = actions_next[key].reshape(batch_size, self.n_agents, -1).reshape(batch_size, -1)
        else:
            actions_next_joint = torch.cat(itemgetter(*self.model_keys)(actions_next), -1).reshape(batch_size, -1)
        _, q_eval = self.policy.Qpolicy(joint_observation=obs_joint, joint_actions=actions_joint, agent_ids=IDs)
        _, q_next = self.policy.Qtarget(joint_observation=next_obs_joint, joint_actions=actions_next_joint, agent_ids=IDs)

        for key in self.model_keys:
            mask_values = agent_mask[key]
            # update critic
            q_eval_a = q_eval[key].reshape(bs)
            q_next_i = q_next[key].reshape(bs)
            # Calculate new rewards
            social_contribution_index = self.causal_model.calculate_social_contribution_index(obs[key], actions[key])
            tax_rates = self.causal_model.calculate_tax_rates(social_contribution_index)
            new_rewards = self.causal_model.redistribute_rewards(rewards[key], social_contribution_index, tax_rates, beta=0.5, alpha=alpha)
            q_target = new_rewards + (1 - terminals[key]) * self.gamma * q_next_i

            td_error = (q_eval_a - q_target.detach()) * mask_values
            loss_c = (td_error ** 2).sum() / mask_values.sum()
            self.optimizer[key]['critic'].zero_grad()
            loss_c.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic[key], self.grad_clip_norm)
            self.optimizer[key]['critic'].step()
            if self.scheduler[key]['critic'] is not None:
                self.scheduler[key]['critic'].step()

            # update actor
            if self.use_parameter_sharing:
                act_eval = actions_eval[key].reshape(batch_size, self.n_agents, -1).reshape(batch_size, -1)
            else:
                a_joint = {k: actions_eval[k] if k == key else actions[k] for k in self.agent_keys}
                act_eval = torch.cat(itemgetter(*self.agent_keys)(a_joint), dim=-1).reshape(batch_size, -1)
            _, q_policy = self.policy.Qpolicy(joint_observation=obs_joint, joint_actions=act_eval, agent_ids=IDs, agent_key=key)
            q_policy_i = q_policy[key].reshape(bs)
            loss_a = -(q_policy_i * mask_values).sum() / mask_values.sum()
            self.optimizer[key]['actor'].zero_grad()
            loss_a.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor[key], self.grad_clip_norm)
            self.optimizer[key]['actor'].step()
            if self.scheduler[key]['actor'] is not None:
                self.scheduler[key]['actor'].step()

            lr_a = self.optimizer[key]['actor'].state_dict()['param_groups'][0]['lr']
            lr_c = self.optimizer[key]['critic'].state_dict()['param_groups'][0]['lr']

            info.update({
                f"{key}/learning_rate_actor": lr_a,
                f"{key}/learning_rate_critic": lr_c,
                f"{key}/loss_actor": loss_a.item(),
                f"{key}/loss_critic": loss_c.item(),
                f"{key}/predictQ": q_eval[key].mean().item()
            })

        self.policy.soft_update(self.tau)
        return info
