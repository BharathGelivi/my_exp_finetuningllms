from tictactoe_env import TicTacToeEnv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -------- Setup --------
model_name = "distilgpt2"
save_path = "./trained_tictactoe_model"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load existing trained model if exists
try:
    model = AutoModelForCausalLM.from_pretrained(save_path).to(device)
    print("Loaded saved model")
except:
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    print("Loaded base model")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

env = TicTacToeEnv()

# -------- Restrict Action Space (CRITICAL FIX) --------
valid_actions = [str(i) for i in range(9)]
action_ids = tokenizer(valid_actions, add_special_tokens=False)["input_ids"]
action_ids = [item[0] for item in action_ids]  # flatten

# -------- Training Loop --------
for episode in range(1000):
    state = env.reset()
    total_reward = 0

    for step in range(9):
        prompt = f"TicTacToe:\n{state}\nChoose move (0-8):"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        outputs = model(**inputs)
        logits = outputs.logits

        # Only allow valid actions (0–8)
        action_logits = logits[:, -1, action_ids]

        # Temperature for better learning
        probs = torch.softmax(action_logits / 1.0, dim=-1)

        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample()

        token_id = action_ids[action_idx]
        action = int(valid_actions[action_idx])

        next_state, reward, done = env.step(action)
        total_reward += reward

        # Policy gradient loss
        log_prob = dist.log_prob(action_idx)
        loss = -log_prob * (reward + 1e-6)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = next_state

        if done:
            break

    print(f"Episode {episode}, Reward: {total_reward}")

    # -------- Save model every 100 episodes --------
    if episode % 100 == 0:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("Model saved!")

# -------- Final save --------
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("Training complete, model saved.")