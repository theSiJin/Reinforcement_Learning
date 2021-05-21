import numpy as np
import cvxpy as cp


class Learner:

    def __init__(self, num_iter=500000, gamma=0.9,
                 alpha_start=0.1, alpha_decay=0.99999, alpha_end=0.001,
                 seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.n_position = 8
        self.n_action = 5

        self.num_iter = num_iter
        self.gamma = gamma
        self.alpha = alpha_start
        self.alpha_decay = alpha_decay
        self.alpha_end = alpha_end

        self.Qa = np.zeros((self.n_position, self.n_position, 2, self.n_action, self.n_action))
        self.Qb = np.zeros((self.n_position, self.n_position, 2, self.n_action, self.n_action))
        self.q, self.q_prev = 0, 0
        self.q_diff = np.zeros(self.num_iter)

        self.cnt = 0

    def learn(self, game):

        game.reset()
        done = False
        state = game.state()

        for i in range(self.num_iter):

            if done:
                game.reset()
                state = game.state()

            # if i % 1000 == 0:
            #     print(i)

            pos_a, pos_b, ball_owner = state
            self.q_prev = self.Qa[1, 5, 0, 1, 4]

            action_a = np.random.choice(self.n_action)
            action_b = np.random.choice(self.n_action)

            next_state, reward_a, reward_b, done = game.step(action_a, action_b)
            next_pos_a, next_pos_b, next_ball_owner = next_state

            Ra = self.Qa[next_pos_a, next_pos_b, next_ball_owner]
            Rb = self.Qb[next_pos_a, next_pos_b, next_ball_owner]

            self.pi = cp.Variable((self.n_action, self.n_action))
            total_reward = cp.sum(cp.multiply(self.pi, Ra)) + cp.sum(cp.multiply(self.pi, Rb))
            obj = cp.Maximize(total_reward)

            constraints = [self.pi >= 0, cp.sum(self.pi) == 1]
            for j in range(self.n_action):
                for k in range(self.n_action):
                    if j != k:
                        constraints.append(self.pi[j, :] @ Ra[j, :] >= self.pi[j, :] @ Ra[k, :])
                        constraints.append(self.pi[:, j] @ Rb[:, j] >= self.pi[:, j] @ Rb[:, k])

            self.prob = cp.Problem(obj, constraints)

            self.result = self.prob.solve(solver=cp.GLPK)

            if self.prob.status != 'optimal':
                self.Va, self.Vb = 0, 0
            else:
                self.Va = np.sum(self.pi.value * Ra)
                self.Vb = np.sum(self.pi.value * Rb)

            self.Qa[pos_a, pos_b, ball_owner, action_a, action_b] = \
                (1 - self.alpha) * self.Qa[pos_a, pos_b, ball_owner, action_a, action_b] + \
                self.alpha * ((1 - self.gamma) * reward_a + self.gamma * self.Va)

            self.Qb[pos_a, pos_b, ball_owner, action_a, action_b] = \
                (1 - self.alpha) * self.Qb[pos_a, pos_b, ball_owner, action_a, action_b] + \
                self.alpha * ((1 - self.gamma) * reward_b + self.gamma * self.Vb)

            # record diff in state s and action S as shown in Fig 4
            if (pos_a, pos_b, ball_owner, action_a, action_b) == (1, 5, 0, 1, 4):
                self.q = self.Qa[1, 5, 0, 1, 4]
                err = self.q - self.q_prev
                self.q_diff[i] = abs(err)
                self.cnt += 1

            state = next_state
            self.alpha = max(self.alpha_decay * self.alpha, self.alpha_end)
