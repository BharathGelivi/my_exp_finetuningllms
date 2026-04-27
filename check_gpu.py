from tictactoe_env import TicTacToeEnv

env = TicTacToeEnv()

state = env.reset()
print(state)

state, reward, done = env.step(4)

print(state)
print("Reward:", reward, "Done:", done)