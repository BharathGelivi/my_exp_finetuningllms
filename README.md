# 🧠 RL-based Fine-Tuning of Models for Tic-Tac-Toe Decision Making

## 📌 Overview

This project explores **reinforcement learning (RL) for decision-making tasks**, starting from Large Language Models (LLMs) and evolving into a more suitable **policy-based neural network approach**.

The goal was to train an agent to play **Tic-Tac-Toe optimally** using different RL strategies and understand the limitations of applying LLM-based RL to structured environments.

---

## 🎯 Objectives

- Apply RL techniques to train a model for a structured environment (Tic-Tac-Toe)
- Experiment with **LLM fine-tuning using RL**
- Compare different approaches:
  - LLM + RL (token-based action space)
  - Custom policy network (discrete action space)
- Identify limitations and optimize learning performance

---

## 🔬 Experiments Conducted

### 1️⃣ LLM + Reinforcement Learning (Initial Approach)

We started by using transformer models from:
- Hugging Face Transformers
- TRL (Transformers Reinforcement Learning)

#### Setup:
- Model: `distilgpt2`
- Environment: Tic-Tac-Toe
- Action: Generated text → parsed into moves (0–8)
- RL Method:
  - Custom policy gradient
  - Attempted PPO using TRL

#### Problems Observed:
- ❌ Extremely large action space (~50,000 tokens)
- ❌ Invalid move generation
- ❌ Weak reward signal propagation
- ❌ Training instability
- ❌ Dependency conflicts with TRL (PPOTrainer missing / API mismatch)

#### Conclusion:
LLMs are **not well-suited for small discrete action spaces** like board games.

---

### 2️⃣ Custom RL Loop with LLM (Intermediate Fix)

We improved:
- Action restriction (only tokens 0–8)
- Reward shaping
- Policy gradient correction

#### Outcome:
- Slight improvement
- Still unstable
- Slow convergence
- Policy collapse issues

---

## 🚀 Final Approach (Successful)

### ✅ Policy Network + Reinforcement Learning

We replaced the LLM with a **custom neural network policy model**.

#### Model Architecture:
- Input: Board state (9 cells)
- Hidden layers: Fully connected layers
- Output:
  - Policy (9 actions)
  - Value function (state evaluation)

#### RL Algorithm:
- Advantage Actor-Critic (A2C-style)
- Key improvements:
  - Action masking (only valid moves)
  - Reward shaping
  - Entropy regularization
  - Return normalization
  - Exploration (epsilon-greedy)

---

## 🎮 Environment

- Tic-Tac-Toe game engine
- Smart opponent:
  - Attempts to win
  - Blocks player moves
  - Random fallback

### Reward System:
| Event            | Reward |
|-----------------|--------|
| Win             | +3     |
| Draw            | +1     |
| Lose            | -2     |
| Invalid Move    | -1     |
| Valid Move Step | +0.2   |

---

## 📈 Results

### Before (LLM-based RL):
- ❌ Mostly invalid moves
- ❌ No meaningful learning
- ❌ Constant negative rewards

### After (Policy Network RL):
- ✅ Valid moves consistently
- ✅ Learns to avoid invalid actions
- ✅ Blocks opponent
- ✅ Achieves draws and wins
- ✅ Stable training behavior

Example gameplay:
Step 1 → Center move (optimal)
Step 2 → Strategic placement
Step 3 → Valid progression
Step 5 → Draw achieved

---

## 🧠 Key Learnings

### ❗ Why LLM + RL Failed:
- Mismatch between **token space** and **action space**
- Poor credit assignment
- Overly complex model for simple task

### ✅ Why Policy Network Worked:
- Direct mapping: state → action
- Small, structured action space
- Faster convergence
- Stable gradients

---

## 📌 Final Conclusion

> While LLMs are powerful for language tasks, they are **not ideal for structured decision-making problems with small discrete action spaces**.

A **simple policy network with RL**:
- Outperformed LLM-based approaches
- Trained faster
- Was easier to debug and stabilize

---

## 🔧 Tech Stack

- Python
- PyTorch
- Reinforcement Learning (A2C-style)
- Custom environment simulation

---

## 🚀 Future Work

- Implement **Minimax opponent for curriculum learning**
- Extend to more complex games
- Apply RL to:
  - Trading systems
  - Resource optimization problems
- Experiment with hybrid models (LLM + policy head)

---

## 📁 Project Structure
├── tictactoe_env.py
├── policy_model.py
├── train_rl.py
├── play.py
└── README.md


---

## 🙌 Acknowledgements

This project was built as an experimental exploration into:
- RL for structured tasks
- Limitations of LLM fine-tuning
- Practical RL system design

---

## 💡 Final Note

This project emphasizes an important engineering principle:

> **Choosing the right model for the problem is more important than using the most powerful model available.**
