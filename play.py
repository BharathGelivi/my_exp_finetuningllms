import torch
from policy_model import PolicyNetwork, board_to_tensor
from tictactoe_env import TicTacToeEnv

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PolicyNetwork().to(device)
model.load_state_dict(torch.load("policy_model.pth"))
model.eval()

env = TicTacToeEnv()
state = env.reset()

print("\n=== STARTING GAME ===\n")

for step in range(9):
    state_tensor = board_to_tensor(env.board).to(device)

    with torch.no_grad():
        logits, _ = model(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        mask = torch.zeros_like(logits)
        mask[env.available_moves()] = 1

        masked_logits = logits.clone()
        masked_logits[mask == 0] = -1e9

        probs = torch.softmax(masked_logits, dim=-1)

        action = torch.argmax(probs).item()

    print(f"\nStep {step+1}")
    print("Chosen action:", action)

    state, reward, done = env.step(action)

    print("\nBoard:")
    print(state)
    print("Reward:", reward)

    if done:
        print("\n=== GAME OVER ===")
        break