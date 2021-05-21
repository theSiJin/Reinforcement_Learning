import random


class Soccer:
    """
    Grid size: 2 x 4
        -----------------
        | 0 | 1 | 2 | 3 |
        -----------------
        | 4 | 5 | 6 | 7 |
        -----------------
    Action space:
        0: North / Up
        1: South / Down
        2: East / Left
        3: West / Right
        4: stick
    Goal position:
        A: 0 and 4
        B: 3 and 7
    Reward:
        +100 if player moves the ball into the correct goal,
                or the opposite moves the ball into the wrong goal
        -100 if player moves the ball into the wrong goal,
                or the opposite moves the ball into the correct goal
    """

    def __init__(self, A, B, seed=None):

        if seed is not None:
            random.seed(seed)

        self.num_row = 2
        self.num_col = 4
        self.action_space = 5

        self.A = A
        self.B = B
        self.goals = {self.A.name: [0, 4], self.B.name: [3, 7]}
        self.ball_owner = None

        self.done = False
        self.reset()

    def reset(self):

        self.A.reset()
        self.B.reset()
        self.done = False

        self.A.position, self.B.position = random.sample([1, 2, 5, 6], k=2)

        self.ball_owner = random.choice([0, 1])
        if self.ball_owner == 0:
            self.A.with_ball = True
        else:
            self.B.with_ball = True

    def _move(self, player, action):

        player.previous_position = player.position

        if action == 0:
            player.position = player.position - 4 if player.position in [5, 6] else player.position
        elif action == 1:
            player.position = player.position + 4 if player.position in [1, 2] else player.position
        elif action == 2:
            player.position = player.position + 1 if player.position not in [3, 7] else player.position
        elif action == 3:
            player.position = player.position - 1 if player.position not in [0, 4] else player.position

    def _determine_end(self, p1, p2):
        """determine if game ends after p1 moves"""
        if p1.with_ball:
            if p1.position in self.goals[p1.name]:
                self.done = True
                p1.score += 100
                p2.score += -100
            elif p1.position in self.goals[p2.name]:
                self.done = True
                p1.score += -100
                p2.score += 100

    def _update_ball_owner(self):
        self.ball_owner = 0 if self.A.with_ball else 1

    def _determine_position(self, p1, p2, a1):
        """p1 is the player who moves"""

        # p1 moves
        self._move(p1, a1)
        if p1.position == p2.position:
            p1.position = p1.previous_position
            if p1.with_ball:
                p1.with_ball = False
                p2.with_ball = True

        # if game ends after the move
        self._determine_end(p1, p2)
        # update ball position
        self._update_ball_owner()

        # end the game if done
        if self.done:
            return

    def step(self, actionA, actionB):

        who_move = random.choice([0, 1])
        if who_move == 0:
            self._determine_position(self.A, self.B, actionA)
            self._determine_position(self.B, self.A, actionB)

        else:
            self._determine_position(self.B, self.A, actionB)
            self._determine_position(self.A, self.B, actionA)

        return self.state(), self.A.score, self.B.score, self.done

    def state(self):
        return self.A.position, self.B.position, self.ball_owner

class Player:

    def __init__(self, player_name):
        self.name = player_name
        self.position = None
        self.previous_position = 0
        self.score = None
        self.with_ball = False

        self.reset()

    def reset(self, initial_position=0):
        self.position = initial_position
        self.previous_position = initial_position
        self.score = 0
        self.with_ball = False
