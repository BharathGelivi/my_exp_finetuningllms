import torch
import torch.optim as optim
from policy_model import PolicyNetwork, board_to_tensor
from tictactoe_env import TicTacToeEnv
import os
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PolicyNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

save_path = "policy_model.pth"

# Load if exists
if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path, map_location=device))
    print("Loaded saved model")

env = TicTacToeEnv()

gamma = 0.99

for episode in range(2000):

    env.reset()
    log_probs = []
    values = []
    rewards = []

    done = False

    while not done:
        state_tensor = board_to_tensor(env.board).to(device)

        logits, value = model(state_tensor)

        # -------- Action Masking --------
        mask = torch.zeros_like(logits)
        mask[env.available_moves()] = 1

        masked_logits = logits.clone()
        masked_logits[mask == 0] = -1e9

        probs = torch.softmax(masked_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        # -------- Exploration (FIXED) --------
        if random.random() < 0.1:
            action_value = random.choice(env.available_moves())
            action = torch.tensor(action_value, device=device)  # scalar tensor
        else:
            action = dist.sample()

        # IMPORTANT: always compute log_prob from SAME dist
        log_prob = dist.log_prob(action)

        # Step environment
        _, reward, done = env.step(action.item())

        log_probs.append(log_prob.squeeze())
        values.append(value.squeeze())
        rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))

    # -------- Compute Returns --------
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.stack(returns)
    values = torch.stack(values)

    # -------- Normalize returns (stability boost) --------
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # -------- Advantage --------
    advantages = returns - values.detach()

    # -------- Loss --------
    log_probs = torch.stack(log_probs)

    policy_loss = -(log_probs * advantages).mean()
    value_loss = torch.nn.functional.mse_loss(values, returns)

    # -------- Entropy Bonus (PREVENT COLLAPSE) --------
    entropy = -(probs * torch.log(probs + 1e-8)).sum()
    
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 50 == 0:
        total_reward = sum([r.item() for r in rewards])
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

    if episode % 200 == 0:
        torch.save(model.state_dict(), save_path)
        print("Model saved!")

torch.save(model.state_dict(), save_path)
print("Training complete.")