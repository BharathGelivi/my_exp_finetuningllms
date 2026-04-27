import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, 9)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy_logits = self.policy_head(x)
        state_value = self.value_head(x)

        return policy_logits, state_value


# -------- Helper: Convert board to tensor --------
def board_to_tensor(board):
    mapping = {" ": 0, "X": 1, "O": -1}
    return torch.tensor([mapping[c] for c in board], dtype=torch.float32)