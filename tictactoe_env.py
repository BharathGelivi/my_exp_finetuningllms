import random

class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [" "] * 9
        return self.render()

    def render(self):
        b = self.board
        return f"""
{b[0]} | {b[1]} | {b[2]}
---------
{b[3]} | {b[4]} | {b[5]}
---------
{b[6]} | {b[7]} | {b[8]}
"""

    def available_moves(self):
        return [i for i in range(9) if self.board[i] == " "]

    def step(self, action):
        # ❌ Invalid move → strong penalty
        if action not in self.available_moves():
            return self.render(), -1, True

        # ✅ Player move
        self.board[action] = "X"

        # ✅ Win
        if self.check_win("X"):
            return self.render(), +3, True

        # ✅ Draw
        if not self.available_moves():
            return self.render(), +1, True

        # 🤖 Opponent move (slightly smarter)
        opp = self.smart_opponent_move()
        self.board[opp] = "O"

        # ❌ Lose
        if self.check_win("O"):
            return self.render(), -2, True

        # ✅ Continue → small positive reward
        return self.render(), +0.2, False

    def smart_opponent_move(self):
        # 1. Try to win
        for move in self.available_moves():
            self.board[move] = "O"
            if self.check_win("O"):
                self.board[move] = " "
                return move
            self.board[move] = " "

        # 2. Block player
        for move in self.available_moves():
            self.board[move] = "X"
            if self.check_win("X"):
                self.board[move] = " "
                return move
            self.board[move] = " "

        # 3. Otherwise random
        return random.choice(self.available_moves())

    def check_win(self, p):
        lines = [
            [0,1,2],[3,4,5],[6,7,8],
            [0,3,6],[1,4,7],[2,5,8],
            [0,4,8],[2,4,6]
        ]
        return any(all(self.board[i] == p for i in line) for line in lines)