import gym
import numpy as np
import matplotlib.pyplot as plt
import random


import torch
import torch.nn as nn
import torch.optim as optim
from DQN import DQN, ReplayMemory


# Hyper-parameters
NUM_EPISODES = 1600
NUM_TEST_EPISODES = 100

BATCH_SIZE = 64
MEMORY_SIZE = 10000
LEARNING_RATE = 0.001
GAMMA = 0.99

UPDATE_FREQ = 2
TAU = 0.001

EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 0.995

SEED = 6891


# Control parameters
SAVE_NET = True  # whether save the model to local
PATH = 'model.pt'

RENDER = False  # whether render the agent after training the agent
RENDER_NUM_EPISODE = 3


# Gym env
env = gym.make('LunarLander-v2')

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n


# seed
env.seed(SEED)
random.seed(SEED)


# neural network
policy_net = DQN(n_states, n_actions, seed=SEED)
target_net = DQN(n_states, n_actions, seed=SEED)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


# optimisation function
optimizer = optim.Adam(params=policy_net.parameters(), lr=LEARNING_RATE)


# loss function
loss_function = nn.MSELoss()


# Epsilon-greedy selection algorithm
def select_action(s, epsilon=0.01, seed=0):
    random.seed(seed)

    if random.random() < epsilon:
        return torch.tensor([[random.randrange(n_actions)]],
                            dtype=torch.long)
    else:
        policy_net.eval()
        with torch.no_grad():
            a = policy_net(s).max(1)[1].view(1, 1)
        policy_net.train()
        return a


# Compute walking mean of 100 consecutive runs
def walking_mean(scores):
    avg_score = [np.mean(scores[_:_+100]) for _ in range(NUM_EPISODES - 100)]
    return avg_score


if __name__ == '__main__':
    
    # Training
    memory = ReplayMemory(MEMORY_SIZE, BATCH_SIZE)

    total_scores = []
    eps = EPS_START

    for i in range(NUM_EPISODES):
        state = env.reset()
        state = torch.tensor(state).unsqueeze(0)

        score = 0
        t = 0
        done = False

        while not done:

            action = select_action(state, epsilon=eps)
            eps = max(eps * EPS_DECAY, EPS_END)

            next_state, reward, done, _ = env.step(action.item())

            score += reward
            t += 1

            action = torch.tensor([action]).unsqueeze(0)
            reward = torch.tensor([reward])
            if not done:
                next_state = torch.tensor(next_state).unsqueeze(0)
            else:
                next_state = None

            memory.push(state, action, reward, next_state, done)

            state = next_state

            if t % UPDATE_FREQ == 0:
                if len(memory) >= BATCH_SIZE:

                    batch = memory.sample_batch()

                    state_batch = torch.cat(batch.state).float()
                    action_batch = torch.cat(batch.action)
                    reward_batch = torch.cat(batch.reward)

                    Q_exp = policy_net(state_batch).gather(1, action_batch).float()

                    # mask for non-terminal states
                    non_terminal_state_index = torch.tensor(tuple(map(lambda s: s is not None,
                                                                      batch.next_state)),
                                                            dtype=torch.bool)
                    non_terminal_state = torch.cat([_ for _ in batch.next_state if _ is not None]).float()

                    # 0 for terminal states
                    Q_opt = torch.zeros(BATCH_SIZE)
                    Q_opt[non_terminal_state_index] = target_net(non_terminal_state).detach().max(1)[0]
                    Q_target = Q_opt * GAMMA + reward_batch

                    # gradient descent
                    loss = loss_function(Q_exp, Q_target.float().unsqueeze(1))
                    optimizer.zero_grad()
                    loss.backward()

                    optimizer.step()

                    # soft update target network
                    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
                        target_param.data.copy_(TAU * policy_param.data + (1-TAU) * target_param.data)

        total_scores.append(score)

    walking_avg_scores = walking_mean(total_scores)
    success_episode = np.argmax(np.array(walking_avg_scores) >= 200) + 100
    success_avg = walking_avg_scores[success_episode-100]

    plt.xlabel("episodes")
    plt.ylabel("rewards")
    plt.title("Reward for each training episode")
    plt.plot(total_scores)
    plt.plot(np.arange(100, 1600), walking_avg_scores, c="orange", linewidth=4)
    plt.plot(success_episode, success_avg, c='red', markersize=10, marker='o')
    plt.savefig("Reward_for_Training_Episodes.png")
    plt.show()

    # Testing
    test_scores = []
    for _ in range(NUM_TEST_EPISODES):
        state = env.reset()
        score = 0
        done = False

        while not done:
            state = torch.tensor(state).unsqueeze(0).float()
            action = select_action(state)
            state, reward, done, _ = env.step(action.item())
            score += reward
        test_scores.append(score)

    plt.xlabel("episodes")
    plt.ylabel("rewards")
    plt.title("Reward per episode using trained agent")
    plt.plot(test_scores)
    plt.hlines(np.mean(test_scores), 0, 100, color='orange', linewidth=4)
    plt.savefig('Reward_of_Testing_Episodes.png')
    plt.show()

    if SAVE_NET:
        torch.save({
            'NUM_EPISODES': NUM_EPISODES,
            'BATCH_SIZE': BATCH_SIZE,
            'MEMORY_SIZE': MEMORY_SIZE,
            'LEARNING_RATE': LEARNING_RATE,
            'GAMMA': GAMMA,
            'UPDATE_FREQ': UPDATE_FREQ,
            'TAU': TAU,
            'EPS_START': EPS_START,
            'EPS_END': EPS_END,
            'EPS_DECAY': EPS_DECAY,
            'SEED': SEED,
            'policy_model_state_dict': policy_net.state_dict(),
            'target_model_state_dict': target_net.state_dict(),
            'total_scores': total_scores
        }, PATH)

    if RENDER:
        for _ in range(RENDER_NUM_EPISODE):
            state = env.reset()
            env.render()
            done = False

            while not done:
                state = torch.tensor(state).unsqueeze(0).float()
                action = select_action(state)
                state, reward, done, _ = env.step(action.item())
                env.render()
        env.close()
