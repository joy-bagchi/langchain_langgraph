import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.softmax(self.fc3(x))



class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def collect_trajectories(env, policy_net, num_episodes=10):
    trajectories = []
    for _ in range(num_episodes):
        state = env.reset()[0]
        episode_data = []
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = policy_net(state_tensor)
            action = np.random.choice(len(action_probs.detach().numpy()), p=action_probs.detach().numpy())
            next_state, reward, done, _, _ = env.step(action)
            episode_data.append((state, action, reward, next_state, done))
            state = next_state
        trajectories.append(episode_data)
    return trajectories


def compute_grouped_advantages(trajectories, value_net, gamma=0.99, normalize=True):
    all_advantages = []

    # Compute advantage for each trajectory
    for trajectory in trajectories:
        rewards = np.array([step[2] for step in trajectory])
        values = np.array([value_net(torch.tensor(step[0], dtype=torch.float32)).item() for step in trajectory])
        deltas = rewards + gamma * np.append(values[1:], 0) - values  # TD residual
        advantages = []
        adv = 0
        for delta in reversed(deltas):
            adv = delta + gamma * adv
            advantages.insert(0, adv)

        all_advantages.append(advantages)

    # Normalize advantages within each group
    if normalize:
        import math
        group_mean = sum(all_advantages) / len(all_advantages)
        group_std = math.sqrt(sum((x - group_mean) ** 2 for x in all_advantages) / len(all_advantages)) + 1e-8
        all_advantages = [(adv - group_mean) / group_std for adv in all_advantages]

    return all_advantages


def grpo_loss(policy_net, value_net, optimizer, trajectories, gamma=0.99, clip_epsilon=0.2):
    grouped_advantages = compute_grouped_advantages(trajectories, value_net, gamma)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.01)

    for i, trajectory in enumerate(trajectories):
        for j, (state, action, reward, next_state, done) in enumerate(trajectory):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = policy_net(state_tensor)
            old_prob = action_probs[action].detach()

            # Compute surrogate loss with clipping
            ratio = action_probs[action] / (old_prob + 1e-10)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            policy_loss = -torch.min(ratio * grouped_advantages[i][j], clipped_ratio * grouped_advantages[i][j])

            # Value loss
            value_target = reward + (1 - done) * gamma * value_net(
                torch.tensor(next_state, dtype=torch.float32)).detach()
            value_loss = (value_net(state_tensor) - value_target) ** 2

            # Optimize networks
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()



def grpo_loss(policy_net, value_net, optimizer, trajectories, gamma=0.99, clip_epsilon=0.2):
    grouped_advantages = compute_grouped_advantages(trajectories, value_net, gamma)

    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.001)
    value_optimizer = optim.Adam(value_net.parameters(), lr=0.01)

    for i, trajectory in enumerate(trajectories):
        for j, (state, action, reward, next_state, done) in enumerate(trajectory):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_probs = policy_net(state_tensor)
            old_prob = action_probs[action].detach()

            # Compute surrogate loss with clipping
            ratio = action_probs[action] / (old_prob + 1e-10)
            clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
            policy_loss = -torch.min(ratio * grouped_advantages[i][j], clipped_ratio * grouped_advantages[i][j])

            # Value loss
            value_target = reward + (1 - done) * gamma * value_net(
                torch.tensor(next_state, dtype=torch.float32)).detach()
            value_loss = (value_net(state_tensor) - value_target) ** 2

            # Optimize networks
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()


env = gym.make("CartPole-v1")
policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
value_net = ValueNetwork(env.observation_space.shape[0])

for episode in range(500):
    trajectories = collect_trajectories(env, policy_net, num_episodes=10)
    grpo_loss(policy_net, value_net, optimizer=None, trajectories=trajectories)

    if episode % 50 == 0:
        print(f"Episode {episode} - Training in Progress")