import numpy as np
import pandas as pd

def encode_cards(cards):
    card_map = {f"{rank}{suit}": i for i, (rank, suit) in enumerate(
        [(r, s) for r in "23456789TJQKA" for s in "cdhs"])}  # 52 cards
    vec = np.zeros(52)
    for card in cards.split():
        if card in card_map:
            vec[card_map[card]] = 1
    return vec

df = pd.read_csv("poker_dataset.csv")


df["hole_cards_encoded"] = df["p1_hole_cards"].apply(encode_cards)
df["board_cards_encoded"] = df["flop"].apply(encode_cards)

action_map = {"f": 0, "c": 1, "r": 2}
df["action_label"] = df["actions"].apply(lambda x: action_map.get(x[0][0], 1))

df.to_csv("poker_training_data.csv", index=False)

