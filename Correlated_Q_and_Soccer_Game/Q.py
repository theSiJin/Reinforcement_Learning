import numpy as np


class Learner:
    
    def __init__(self, num_iter=1000000, gamma=0.9,
                 alpha_start=0.1, alpha_decay=0.99999, alpha_end=0.001,
                 eps_start=1.0, eps_end=0.01,
                 seed=None):

        if seed is not None:
            np.random.seed(seed)

        self.n_position = 8
        self.n_action = 5
        self.seed = seed

        self.num_iter = num_iter
        self.gamma = gamma
        self.alpha = alpha_start
        self.alpha_decay = alpha_decay
        self.alpha_end = alpha_end
        self.eps = eps_start
        self.eps_decay = (eps_start - eps_end) / num_iter
        self.eps_end = eps_end

        self.qa, self.qb = None, None
        self.q_diff = np.zeros(self.num_iter)
        self.q, self.q_prev = 0, 0
        self.tch = 0

    def learn(self, game):
        self.qa = np.zeros((self.n_position, self.n_position, 2, self.n_action))
        self.qb = np.zeros((self.n_position, self.n_position, 2, self.n_action))

        game.reset()
        done = False
        state = game.state()

        for i in range(self.num_iter):
            # self.eps = max(self.eps_decay * self.eps, self.eps_end)
            self.eps -= self.eps_decay
            self.q_prev = self.qa[1, 5, 0, 1]

            if done:
                game.reset()
                state = game.state()

            pos_a, pos_b, ball_owner = state

            if self.eps > np.random.random():
                action_a = np.random.choice(self.n_action)
                # action_b = np.random.choice(self.n_action)
            else:
                action_a = np.argmax(self.qa[pos_a, pos_b, ball_owner, :])
                # action_b = np.argmax(self.qb[pos_b, pos_b, ball_owner, :])
            action_b = np.random.choice(self.n_action)

            next_state, reward_a, reward_b, done = game.step(action_a, action_b)

            next_pos_a, next_pos_b, next_ball_owner = next_state

            Va = np.max(self.qa[next_pos_a, next_pos_b, next_ball_owner])
            self.qa[pos_a, pos_b, ball_owner, action_a] = \
                (1 - self.alpha) * self.qa[pos_a, pos_b, ball_owner, action_a] + \
                self.alpha * ((1 - self.gamma) * reward_a + self.gamma * Va)

            # Vb = np.max(self.qb[next_pos_a, next_pos_b, next_ball_owner])
            # self.qb[pos_a, pos_b, ball_owner, action_b] = \
            #     (1 - self.alpha) * self.qb[pos_a, pos_b, ball_owner, action_b] + \
            #     self.alpha * ((1 - self.gamma) * reward_b + self.gamma * Vb)

            # record diff in state s and action S as shown in Fig 4
            if (pos_a, pos_b, ball_owner, action_a) == (1, 5, 0, 1):
                self.q = self.qa[1, 5, 0, 1]
                err = self.q - self.q_prev
                self.q_diff[i] = abs(err)
                self.tch += 1

            state = next_state
            self.alpha = max(self.alpha_decay * self.alpha, self.alpha_end)
            # self.alpha -= self.alpha_decay

