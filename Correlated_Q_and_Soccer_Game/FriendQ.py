import numpy as np


class Learner:

    def __init__(self, num_iter=1000000, gamma=0.9,
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

        self.Q = None
        self.Qb = None
        self.q, self.q_prev = 0, 0
        self.q_diff = np.zeros(self.num_iter)
        
        self.b_c = []

    def learn(self, game):
        self.Q = np.zeros((self.n_position, self.n_position, 2, self.n_action, self.n_action))
        self.Qb = np.zeros((self.n_position, self.n_position, 2, self.n_action, self.n_action))

        game.reset()
        done = False
        state = game.state()

         
        for i in range(self.num_iter):

            if done:
                game.reset()
                state = game.state()

            pos_a, pos_b, ball_owner = state
            self.q_prev = self.Q[1, 5, 0, 1, 4]

            action_a = np.random.choice(self.n_action)
            action_b = np.random.choice(self.n_action)

            next_state, reward_a, reward_b, done = game.step(action_a, action_b)

            next_pos_a, next_pos_b, next_ball_owner = next_state

            V = np.max(self.Q[next_pos_a, next_pos_b, next_ball_owner])
            Vb = np.max(self.Qb[next_pos_a, next_pos_b, next_ball_owner])

            self.Q[pos_a, pos_b, ball_owner, action_a, action_b] = \
                (1 - self.alpha) * self.Q[pos_a, pos_b, ball_owner, action_a, action_b] + \
                self.alpha * ((1 - self.gamma) * reward_a + self.gamma * V)
            
            self.Qb[pos_a, pos_b, ball_owner, action_a, action_b] = \
                (1 - self.alpha) * self.Qb[pos_a, pos_b, ball_owner, action_a, action_b] + \
                self.alpha * ((1 - self.gamma) * reward_b + self.gamma * Vb)
            

            # record diff in state s and action S as shown in Fig 4
            if (pos_a, pos_b, ball_owner, action_a, action_b) == (1, 5, 0, 1, 4):
                self.q = self.Q[1, 5, 0, 1, 4]
                err = self.q - self.q_prev
                self.q_diff[i] = abs(err)
                
                self.b_c.append(np.argmax(self.Qb[1,5,0]))

            state = next_state
            self.alpha = max(self.alpha_decay * self.alpha, self.alpha_end)
