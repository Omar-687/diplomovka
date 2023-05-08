import torch
import torch.nn as nn
import torch.optim as optim
# sample initial state s
# 2. for each time step:
# 3.     Sample action a_t from actor network using current state s_t
# 4.     Take action a_t and observe reward r_t and new state s_{t+1}.
# 5.
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) # assuming action space is bounded between -1 and 1
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, lr):
        super(ActorCritic, self).__init__()
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr)

    def forward(self, state):
        action = self.actor(state)
        value = self.critic(state, action)
        return action, value

    def update(self, state, action, target_value):
        # update critic
        predicted_value = self.critic(state, action)
        critic_loss = nn.functional.mse_loss(predicted_value, target_value)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # update actor
        advantage = target_value - predicted_value.detach()
        actor_loss = -(self.actor(state) * advantage).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
