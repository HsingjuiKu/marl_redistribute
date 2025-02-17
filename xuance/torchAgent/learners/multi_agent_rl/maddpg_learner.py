"""
Multi-Agent Deep Deterministic Policy Gradient
Paper link:
https://proceedings.neurips.cc/paper/2017/file/68a9750337a418a86fe06c1991a1d64c-Paper.pdf
Implementation: Pytorch
Trick: Parameter sharing for all agents, with agents' one-hot IDs as actor-critic's inputs.
"""
from xuance.torchAgent.learners import *
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from redistribute import EnhancedCausalModel


class MADDPG_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: Sequence[torch.optim.Optimizer],
                 scheduler: Sequence[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100
                 ):
        self.gamma = gamma
        self.tau = config.tau
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(MADDPG_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)
        self.optimizer = {
            'actor': optimizer[0],
            'critic': optimizer[1]
        }
        self.scheduler = {
            'actor': scheduler[0],
            'critic': scheduler[1]
        }
        self.causal_model = EnhancedCausalModel(config.n_agents, config.obs_shape[0], config.act_shape[0],device)
        self.n_iters = config.running_steps

    def update(self, sample):
        self.iterations += 1
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        obs_next = torch.Tensor(sample['obs_next']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().reshape(-1, self.n_agents, 1).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

        # Calculate alpha which decays from 1 to 0 over iterations
        alpha = 1.0 - (self.iterations / self.n_iters)

        # Train actor
        _, actions_eval = self.policy(obs, IDs)
        loss_a = -(self.policy.Qpolicy(obs, actions_eval, IDs) * agent_mask).sum() / agent_mask.sum()
        self.optimizer['actor'].zero_grad()
        loss_a.backward()
        if self.args.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters_actor, self.args.grad_clip_norm)
        self.optimizer['actor'].step()
        if self.scheduler['actor'] is not None:
            self.scheduler['actor'].step()

        # Train critic
        actions_next = self.policy.Atarget(obs_next, IDs)
        q_eval = self.policy.Qpolicy(obs, actions, IDs)
        q_next = self.policy.Qtarget(obs_next, actions_next, IDs)

        # Calculate new rewards
        social_contribution_index = self.causal_model.calculate_social_contribution_index(obs, actions)
        tax_rates = self.causal_model.calculate_tax_rates(social_contribution_index)
        new_rewards = self.causal_model.redistribute_rewards(rewards, social_contribution_index, tax_rates, beta=0.5, alpha=alpha)

        q_target = new_rewards + (1 - terminals) * self.gamma * q_next
        td_error = (q_eval - q_target.detach()) * agent_mask
        loss_c = (td_error ** 2).sum() / agent_mask.sum()
        self.optimizer['critic'].zero_grad()
        loss_c.backward()
        if self.args.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters_critic, self.args.grad_clip_norm)
        self.optimizer['critic'].step()
        if self.scheduler['critic'] is not None:
            self.scheduler['critic'].step()

        self.policy.soft_update(self.tau)

        lr_a = self.optimizer['actor'].state_dict()['param_groups'][0]['lr']
        lr_c = self.optimizer['critic'].state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate_actor": lr_a,
            "learning_rate_critic": lr_c,
            "loss_actor": loss_a.item(),
            "loss_critic": loss_c.item(),
            "predictQ": q_eval.mean().item()
        }

        return info