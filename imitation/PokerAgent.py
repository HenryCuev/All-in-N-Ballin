import pyspiel

import torch
import torch.nn as nn


import numpy as np


class PokerNN(nn.Module):
    def __init__(self, input_size=16, hidden_size=128, output_size=3):
        super(PokerNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


class PokerAgent:
    def __init__(self, model_path, input_size, device="cpu"):
        self.model = PokerNN(16, 128, 3)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.device = device

    def preprocess_observation(self, observation):
        
        hole_cards = observation["private_card"]
        board_cards = observation["community_card"]

        def encode_cards(cards):
            vec = np.zeros(52)
            for card in cards:
                vec[card] = 1
            return vec

        hole_vec = encode_cards(hole_cards)
        board_vec = encode_cards(board_cards)
        state_vec = np.concatenate([hole_vec, board_vec])

        return torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)

    def step(self, state):
        current_player = state.current_player()
    
        if state.is_terminal():
            return None
    
        obs = torch.tensor(state.observation_tensor(current_player), dtype=torch.float32).unsqueeze(0)

        action_probs = self.model(obs)
        action = torch.argmax(action_probs, dim=-1).item()
        return action

        obs = np.array(state.observation_tensor(current_player))

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_probs = self.model(obs_tensor)
        action = torch.argmax(action_probs, dim=1).item()
    
        return action

game = pyspiel.load_game("leduc_poker")
state = game.new_initial_state()

agent = PokerAgent("poker_model.pth", input_size=52)

while not state.is_terminal():
    if state.is_chance_node():
        outcomes, probs = zip(*state.chance_outcomes())
        state.apply_action(np.random.choice(outcomes, p=probs))
    else:
        current_player = state.current_player()
        action = agent.step(state) if current_player == 0 else np.random.choice([0, 1, 2])
        state.apply_action(action)

print("Final State:", state)

