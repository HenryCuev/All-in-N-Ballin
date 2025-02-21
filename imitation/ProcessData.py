import os
import pandas as pd
import re

# directory with the logs - change as necessary
log_dir = "../Data/processed_logs_2pn_2017"

def parse_log_line(line):
    parts = line.strip().split(":")
    
    if len(parts) < 6:
        return None  

    hand_id = int(parts[1])
    actions = parts[2].split("/")
    hole_cards = parts[3].split("|")

    board_cards = parts[4].split("/") if parts[4] else []

    if all(c.isdigit() or c == '-' for c in parts[5] if c not in "|"):  
        ev_values = list(map(int, parts[5].split("|")))
        players = parts[6].split("|") if len(parts) > 6 else ["Unknown", "Unknown"]
    else:  
        ev_values = [0, 0]
        players = parts[5].split("|")

    return {
        "hand_id": hand_id,
        "p1_hole_cards": hole_cards[0],
        "p2_hole_cards": hole_cards[1],
        "flop": board_cards[0] if len(board_cards) > 0 else "",
        "turn": board_cards[1] if len(board_cards) > 1 else "",
        "river": board_cards[2] if len(board_cards) > 2 else "",
        "actions": actions,
        "ev_p1": ev_values[0],
        "ev_p2": ev_values[1],
        "player_1": players[0],
        "player_2": players[1],
    }

data = []

num_files = 0
for filename in os.listdir(log_dir):
    if num_files > 10:
        break
    if filename.endswith(".log"):
        with open(os.path.join(log_dir, filename), "r") as file:
            for line in file:
                if line.startswith("STATE"):
                    parsed = parse_log_line(line)
                    if parsed:
                        data.append(parsed)
            num_files = num_files + 1

df = pd.DataFrame(data)
df.to_csv("poker_dataset.csv", index=False)
print(f"Processed {len(df)} game states and saved to poker_dataset.csv")

