# Program Adapted From:
# https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/examples/kuhn_nfsp.py


"""NFSP agents trained on Texas Hold 'Em Poker."""

import pyspiel # type: ignore

from absl import app # type: ignore
from absl import flags # type: ignore
from absl import logging # type: ignore
import tensorflow.compat.v1 as tf # type: ignore

from open_spiel.python import policy # type: ignore
from open_spiel.python import rl_environment # type: ignore
from open_spiel.python.algorithms import exploitability # type: ignore
from open_spiel.python.algorithms import nfsp # type: ignore

import numpy as np # type: ignore

universal_poker = pyspiel.universal_poker

CUSTOM_NO_LIMIT_HEADS_UP_TEXAS_HOLDEM_CONFIG = {
    "betting":"nolimit",
    "bettingAbstraction": "fcpa",
    "numPlayers" : 2,
    "numRounds" : 4,
    "raiseSize" : "20 20 20 20",
    "blind" : "10 20",
    "firstPlayer" : "1",
    "numSuits" : 4,
    "numRanks" : 13,
    "numHoleCards" : 2,
    "numBoardCards" : "0 3 1 1",
    "stack" : "2000 2000",
  }

# use a modification of NSFP to work with universal poker edge case
class NFSP(nfsp.NFSP):

  def _act(self, info_state, legal_actions):

    info_state = np.reshape(info_state, [1, -1])
    action_values, action_probs = self._session.run(
        [self._avg_policy, self._avg_policy_probs],
        feed_dict={self._info_state_ph: info_state})

    self._last_action_values = action_values[0]
    # Remove illegal actions, normalize probs
    probs = np.zeros(self._num_actions)
    # print()
    probs[legal_actions] = action_probs[0][legal_actions]

    if sum(probs) != 0:
      probs /= sum(probs)
    else:
      # handling for edge case where probs dont line up
      prob_per_action = 1/len(legal_actions)
      probs = [prob_per_action if idx in legal_actions else 0 for idx in range(self._num_actions)]

    action = np.random.choice(len(probs), p=probs)
    
    return action, probs

FLAGS = flags.FLAGS


flags.DEFINE_integer("num_train_episodes", int(3e6),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [
    128,
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e7),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e8),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")


class NFSPPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    player_ids = [0, 1]
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


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
    # pylint: disable=g-complex-comprehension
    agents = [
        NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                  FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param,
                  **kwargs) for idx in range(num_players)
    ]
    expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

    sess.run(tf.global_variables_initializer())
    for ep in range(FLAGS.num_train_episodes):
      if (ep + 1) % FLAGS.eval_every == 0:
        losses = [agent.loss for agent in agents]
        logging.info("Losses: %s", losses)
        # NOTE: exploitability measure class no longer works, action space too big
        # expl = exploitability.exploitability(env.game, expl_policies_avg)
        # logging.info("[%s] Exploitability AVG %s", ep + 1, expl)
        logging.info("_____________________________________________")

      time_step = env.reset()
      while not time_step.last():
        player_id = time_step.observations["current_player"]
        agent_output = agents[player_id].step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)

      # Episode is over, step all agents with final info state.
      for agent in agents:
        agent.step(time_step)

    # save agent policy to files
    # NOTE: saves only policy not training buffer
    for pos, agent in enumerate(agents):
      agent.save(f"agent_{pos+1}")

if __name__ == "__main__":
  app.run(main)
