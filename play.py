import torch
from policy_model import PolicyNetwork, board_to_tensor
from tictactoe_env import TicTacToeEnv

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- Load trained model --------
model = PolicyNetwork().to(device)
model.load_state_dict(torch.load("policy_model.pth", map_location=device))
model.eval()

env = TicTacToeEnv()
env.reset()

print("\n=== TIC TAC TOE: YOU (X) vs AI (O) ===\n")
print("Positions:")
print("""
0 | 1 | 2
---------
3 | 4 | 5
---------
6 | 7 | 8
""")

# -------- Game Loop --------
done = False

while not done:

    # -------- USER TURN --------
    print("\nCurrent Board:")
    print(env.render())

    try:
        user_input = int(input("Your move (0-8): "))
    except:
        print("Invalid input. Enter a number 0-8.")
        continue

    if user_input not in env.available_moves():
        print("Invalid move! Try again.")
        continue

    # Apply user move
    env.board[user_input] = "X"

    if env.check_win("X"):
        print(env.render())
        print("🎉 You WIN!")
        break

    if not env.available_moves():
        print(env.render())
        print("🤝 It's a DRAW!")
        break

    # -------- AI TURN --------
    state_tensor = board_to_tensor(env.board).to(device)

    with torch.no_grad():
        logits, _ = model(state_tensor)

        # Mask invalid moves
        mask = torch.zeros_like(logits)
        mask[env.available_moves()] = 1

        masked_logits = logits.clone()
        masked_logits[mask == 0] = -1e9

        probs = torch.softmax(masked_logits, dim=-1)

        action = torch.argmax(probs).item()

    print(f"\nAI chooses: {action}")

    env.board[action] = "O"

    if env.check_win("O"):
        print(env.render())
        print("🤖 AI WINS!")
        break

    if not env.available_moves():
        print(env.render())
        print("🤝 It's a DRAW!")
        break
