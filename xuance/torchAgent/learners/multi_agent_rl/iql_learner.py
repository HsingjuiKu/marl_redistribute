"""
Independent Q-learning (IQL)
Implementation: Pytorch
"""
from xuance.torchAgent.learners import *


class IQL_Learner(LearnerMAS):
    def __init__(self,
                 config: Namespace,
                 policy: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 env,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: Optional[Union[int, str, torch.device]] = None,
                 model_dir: str = "./",
                 gamma: float = 0.99,
                 sync_frequency: int = 100):
        self.gamma = gamma
        self.sync_frequency = sync_frequency
        self.mse_loss = nn.MSELoss()
        super(IQL_Learner, self).__init__(config, policy, optimizer, scheduler, device, model_dir)

    def update(self, sample):
        self.iterations += 1
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        obs_next = torch.Tensor(sample['obs_next']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().reshape(-1, self.n_agents, 1).to(self.device)
        agent_mask = torch.Tensor(sample['agent_mask']).float().reshape(-1, self.n_agents, 1).to(self.device)
        IDs = torch.eye(self.n_agents).unsqueeze(0).expand(self.args.batch_size, -1, -1).to(self.device)

        _, _, q_eval = self.policy(obs, IDs)
        q_eval_a = q_eval.gather(-1, actions.long().reshape([self.args.batch_size, self.n_agents, 1]))
        _, q_next = self.policy.target_Q(obs_next, IDs)

        if self.args.double_q:
            _, action_next_greedy, q_next_eval = self.policy(obs_next, IDs)
            q_next_a = q_next.gather(-1, action_next_greedy.unsqueeze(-1).long())
        else:
            q_next_a = q_next.max(dim=-1, keepdim=True).values

        q_target = rewards + (1 - terminals) * self.args.gamma * q_next_a

        # calculate the loss function
        q_eval_a *= agent_mask
        q_target *= agent_mask
        loss = self.mse_loss(q_eval_a, q_target.detach())
        self.optimizer.zero_grad()  
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate": lr,
            "loss_Q": loss.item(),
            "predictQ": q_eval_a.mean().item()
        }

        return info

    def update_recurrent(self, sample):
        self.iterations += 1
        obs = torch.Tensor(sample['obs']).to(self.device)
        actions = torch.Tensor(sample['actions']).to(self.device)
        rewards = torch.Tensor(sample['rewards']).mean(dim=1, keepdims=True).to(self.device)
        terminals = torch.Tensor(sample['terminals']).float().to(self.device)
        avail_actions = torch.Tensor(sample['avail_actions']).float().to(self.device)
        filled = torch.Tensor(sample['filled']).float().to(self.device)
        batch_size = actions.shape[0]
        episode_length = actions.shape[2]
        IDs = torch.eye(self.n_agents).unsqueeze(1).unsqueeze(0).expand(batch_size, -1, episode_length + 1, -1).to(
            self.device)

        # Current Q
        rnn_hidden = self.policy.representation.init_hidden(batch_size * self.n_agents)
        _, actions_greedy, q_eval = self.policy(obs.reshape(-1, episode_length + 1, self.dim_obs),
                                                IDs.reshape(-1, episode_length + 1, self.n_agents),
                                                *rnn_hidden,
                                                avail_actions=avail_actions.reshape(-1, episode_length + 1, self.dim_act))
        q_eval = q_eval[:, :-1].reshape(batch_size, self.n_agents, episode_length, self.dim_act)
        actions_greedy = actions_greedy.reshape(batch_size, self.n_agents, episode_length + 1, 1)
        q_eval_a = q_eval.gather(-1, actions.long().reshape([self.args.batch_size, self.n_agents, episode_length, 1]))

        # Target Q
        target_rnn_hidden = self.policy.target_representation.init_hidden(batch_size * self.n_agents)
        _, q_next = self.policy.target_Q(obs.reshape(-1, episode_length + 1, self.dim_obs),
                                         IDs.reshape(-1, episode_length + 1, self.n_agents),
                                         *target_rnn_hidden)
        q_next = q_next[:, 1:].reshape(batch_size, self.n_agents, episode_length, self.dim_act)
        q_next[avail_actions[:, :, 1:] == 0] = -9999999

        # use double-q trick
        if self.args.double_q:
            action_next_greedy = actions_greedy[:, :, 1:]
            q_next_a = q_next.gather(-1, action_next_greedy.long().detach())
        else:
            q_next_a = q_next.max(dim=-1, keepdim=True).values

        filled_n = filled.unsqueeze(1).expand(-1, self.n_agents, -1, -1)
        rewards = rewards.expand(-1, self.n_agents, -1, -1)
        terminals = terminals.unsqueeze(1).expand(batch_size, self.n_agents, episode_length, 1)
        q_target = rewards + (1 - terminals) * self.args.gamma * q_next_a

        # calculate the loss function
        td_errors = q_eval_a - q_target.detach()
        td_errors *= filled_n
        loss = (td_errors ** 2).sum() / filled_n.sum()
        self.optimizer.zero_grad()
        loss.backward()
        if self.args.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.args.grad_clip_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()
        lr = self.optimizer.state_dict()['param_groups'][0]['lr']

        info = {
            "learning_rate": lr,
            "loss_Q": loss.item(),
            "predictQ": q_eval_a.mean().item()
        }

        return info

