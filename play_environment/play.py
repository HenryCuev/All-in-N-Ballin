# Program Adapted From:
# https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/examples/kuhn_nfsp.py
# & https://texasholdem.readthedocs.io/en/stable/guis.html#abstract-gui
# # utilizing texasholden gui https://github.com/SirRender00/texasholdem

"""NFSP agents trained on Texas Hold 'Em Poker."""

import pyspiel # type: ignore

from absl import app # type: ignore
from absl import flags # type: ignore
from absl import logging # type: ignore
import tensorflow.compat.v1 as tf # type: ignore

from open_spiel.python import policy # type: ignore
from open_spiel.python import rl_environment # type: ignore
from nfsp import NFSP, FLAGS, CUSTOM_NO_LIMIT_HEADS_UP_TEXAS_HOLDEM_CONFIG

def main(unused_argv):

  game = "universal_poker"
  num_players = 2

  env = rl_environment.Environment(game, **CUSTOM_NO_LIMIT_HEADS_UP_TEXAS_HOLDEM_CONFIG)

  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]
  kwargs = {
      "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
      "epsilon_decay_duration": FLAGS.num_train_episodes,
      "epsilon_start": 0.06,
      "epsilon_end": 0.001,
  }

  with tf.Session() as sess:
    agent = NFSP(sess, 0, info_state_size, num_actions, hidden_layers_sizes,
                  FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param,
                  **kwargs)
    
    agent.restore("agent_1")


    # # play against the agent you train in nfsp.py!
    from texasholdem.game.game import TexasHoldEm # type: ignore
    from texasholdem.gui.text_gui import TextGUI # type: ignore


    game = TexasHoldEm(buyin=500, big_blind=5, small_blind=2, max_players=2)
    gui = TextGUI(game=game)

    while game.is_game_running():
        game.start_hand()

        print(game.hand_phase)

        while game.is_hand_running():
          if game.current_player % 2 == 0:
            game.take_action(*agent(game))
          else:
            gui.run_step()

if __name__ == "__main__":
  app.run(main)