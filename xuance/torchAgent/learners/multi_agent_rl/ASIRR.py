import torch.nn as nn
import torch.nn.functional as F
import torch
class SocialInfluenceModel(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, obs, action):
        return self.network(torch.cat([obs, action], dim=-1))


class EnhancedCausalModel(nn.Module):
    def __init__(self, num_agents, obs, action, device):
        super().__init__()
        self.num_agents = num_agents
        self.obs = obs
        self.action = action
        self.obs_dim = obs.shape[-1]
        self.action_dim = action.shape[-1]
        self.device = device

        self.sim = SocialInfluenceModel(num_agents, self.obs_dim, self.action_dim).to(device)

    def predict_others_actions(self, obs, action):
        return self.sim(obs, action)

    def calculate_social_influence(self, obs, actions):
        batch_size, time_steps, num_agents, _ = obs.shape
        influences = []

        for k in range(self.num_agents):
            obs_k = obs[:, :, k]
            action_k = actions[:, :, k]
            # Calculate conditional policy with agent k's action
            p_with_k = self.predict_others_actions(obs_k, action_k)
            # Calculate marginal policy by averaging over several counterfactual actions
            counterfactual_actions = torch.zeros_like(action_k).to(self.device)
            p_without_k = self.predict_others_actions(obs_k, counterfactual_actions)
            for _ in range(10):  # Sample 10 counterfactual actions
                counterfactual_actions = torch.rand_like(action_k).to(self.device)
                p_without_k += self.predict_others_actions(obs_k, counterfactual_actions)
            p_without_k /= 11  # Including the initial zero action

            # Calculate KL divergence between conditional and marginal policies
            influence = F.kl_div(
                p_with_k.log_softmax(dim=-1),
                p_without_k.softmax(dim=-1),
                reduction='batchmean'
            )
            influences.append(influence.unsqueeze(-1))
        influences = torch.stack(influences, dim=-1)  # Shape: [batch_size, time_steps, num_agents, 1]
        influences = F.softmax(influences, dim=-2)
        influences = influences.unsqueeze(2)  # Shape: [batch_size, time_steps, 1, num_agents]
        return influences

    def calculate_social_contribution_index(self, obs, actions):
        influences = self.calculate_social_influence(obs, actions)
        return influences

    def calculate_tax_rates(self, social_contribution_index):
        return torch.sigmoid(social_contribution_index)

    def redistribute_rewards(self, original_rewards, social_contribution_index, tax_rates, beta=0.5, alpha=1.0):
        central_pool = (tax_rates * original_rewards).sum(dim=(1, 2), keepdim=True)
        normalized_contributions = social_contribution_index / (
                social_contribution_index.sum(dim=2, keepdim=True) + 1e-8)
        redistributed_rewards = (1 - tax_rates) * original_rewards + beta * normalized_contributions * central_pool
        return alpha * redistributed_rewards + (1 - alpha) * original_rewards

    def get_new_rewards(self,  rewards,  beta=0.5, alpha=1.0):
        social_contribution_index = self.calculate_social_contribution_index(self.obs, self.action)
        tax_rates = self.calculate_tax_rates(social_contribution_index)
        new_rewards = self.redistribute_rewards(rewards, social_contribution_index, tax_rates, beta, alpha)
        return new_rewards