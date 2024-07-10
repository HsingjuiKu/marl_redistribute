import torch.nn as nn
import torch.nn.functional as F
import torch



class EnhancedCausalModel(nn.Module):
    def __init__(self, num_agents, obs_dim, action_dim, device):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(device)

    def predict_others_actions(self, obs, action):
        return self.network(torch.cat([obs, action], dim=-1))

    def calculate_social_influence(self, obs, actions):
        batch_size, num_agents, obs_dim = obs.shape
        influences = []

        for k in range(num_agents):  # Loop range increased by 10 times
            agent_idx = k % num_agents  # Ensure index stays within bounds
            obs_k = obs[:, agent_idx]
            action_k = actions[:, agent_idx]
            p_with_k = self.predict_others_actions(obs_k, action_k)
            # Calculate p_without_k by averaging over several counterfactual actions
            p_without_k = self.predict_others_actions(obs_k, torch.zeros_like(action_k).to(self.device))
            for _ in range(10):  # Sample 10 counterfactual actions
                counterfactual_actions = torch.rand_like(action_k).to(self.device)  # Generate random actions
                p_without_k += self.predict_others_actions(obs_k, counterfactual_actions)
            p_without_k /= 11  # Including the initial zero action

            influence = F.kl_div(
                p_with_k.log_softmax(dim=-1),
                p_without_k.softmax(dim=-1),
                reduction='batchmean'
            )
            influences.append(influence.unsqueeze(-1))
        influences = torch.stack(influences, dim=-1)  # Shape: [batch_size, 10*num_agents, 1]
        influences = F.softmax(influences, dim=-2)
        influences = influences.unsqueeze(1)  # Shape: [batch_size, 1, 10*num_agents]
        return influences

    def calculate_social_contribution_index(self, obs, actions):
        influences = self.calculate_social_influence(obs, actions)
        return influences

    def calculate_tax_rates(self, social_contribution_index):
        return torch.sigmoid(social_contribution_index)

    def redistribute_rewards(self, original_rewards, social_contribution_index, tax_rates, beta=0.5, alpha=1.0):
        central_pool = (tax_rates * original_rewards).sum(dim=1, keepdim=True)
        normalized_contributions = social_contribution_index / (social_contribution_index.sum(dim=1, keepdim=True) + 1e-8)
        redistributed_rewards = (1 - tax_rates) * original_rewards + beta * normalized_contributions * central_pool
        return alpha * redistributed_rewards + (1 - alpha) * original_rewards
        # return redistributed_rewards
