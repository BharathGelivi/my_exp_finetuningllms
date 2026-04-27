from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
import torch
import re
from tictactoe_env import TicTacToeEnv

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "distilgpt2"

# -------- Tokenizer --------
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# -------- PPO Model --------
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to(device)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to(device)

# -------- PPO Config --------
config = PPOConfig(
    learning_rate=1e-5,
    batch_size=8,
    mini_batch_size=4,
)

ppo_trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    config=config,
)

env = TicTacToeEnv()

# -------- Extract move --------
def extract_action(text):
    match = re.search(r"[0-8]", text)
    return int(match.group()) if match else -1

# -------- Training Loop --------
for episode in range(500):

    queries = []
    responses = []
    rewards = []

    for _ in range(8):  # batch
        state = env.reset()

        prompt = f"TicTacToe:\n{state}\nChoose move (0-8):"
        query_tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Generate response
        response_tensor = ppo_trainer.generate(
            query_tensor,
            max_new_tokens=1,
            do_sample=True,
            temperature=1.0
        )

        response_text = tokenizer.decode(response_tensor[0], skip_special_tokens=True)

        action = extract_action(response_text)

        _, reward, _ = env.step(action)

        queries.append(query_tensor[0])
        responses.append(response_tensor[0])
        rewards.append(torch.tensor(reward).to(device))

    # -------- PPO Update --------
    stats = ppo_trainer.step(queries, responses, rewards)

    print(f"Episode {episode}, Avg reward: {sum([r.item() for r in rewards]) / len(rewards)}")

# -------- Save Model --------
model.save_pretrained("./ppo_tictactoe")
tokenizer.save_pretrained("./ppo_tictactoe")

print("PPO training complete.")