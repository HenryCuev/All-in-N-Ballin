# Program Adapted From:
# https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/examples/universal_poker_cfr_cpp_load_from_acpc_gamedef_example.py


"""CFR algorithm on Texas Hold 'Em Poker."""

import pickle
import sys
from absl import app # type: ignore
from absl import flags # type: ignore

import pyspiel # type: ignore

universal_poker = pyspiel.universal_poker

FLAGS = flags.FLAGS

flags.DEFINE_enum("solver", "cfr", ["cfr", "cfrplus", "cfrbr"], "CFR solver")
_ITERATIONS = flags.DEFINE_integer("iterations", 100, "Number of iterations")


CUSTOM_NO_LIMIT_HEADS_UP_TEXAS_HOLDEM_GAMEDEF = """\
GAMEDEF
nolimit
numPlayers = 2
numRounds = 4
raiseSize = 20 20 20 20
blind = 10 20
firstPlayer = 1
numSuits = 4
numRanks = 13
numHoleCards = 2
numBoardCards = 0 3 1 1
stack = 2000
END GAMEDEF
"""


def main(_):
    game = universal_poker.load_universal_poker_from_acpc_gamedef(
        CUSTOM_NO_LIMIT_HEADS_UP_TEXAS_HOLDEM_GAMEDEF
    )

    solver = None
    if FLAGS.solver == "cfr":
        solver = pyspiel.CFRSolver(game)
    elif FLAGS.solver == "cfrplus":
        solver = pyspiel.CFRPlusSolver(game)
    elif FLAGS.solver == "cfrbr":
        solver = pyspiel.CFRBRSolver(game)
    else:
        print("Unknown solver")
        sys.exit(0)


    for i in range(int(_ITERATIONS.value / 2)):
        print("in evaluate and update policy")
        solver.evaluate_and_update_policy()
        print("Iteration {} exploitability: {:.6f}".format(
            i, pyspiel.exploitability(game, solver.average_policy())))

    filename = "/tmp/{}_solver.pickle".format(FLAGS.solver)
    print("Persisting the model...")
    with open(filename, "wb") as file:
        pickle.dump(solver, file, pickle.HIGHEST_PROTOCOL)

    print("Loading the model...")
    with open(filename, "rb") as file:
        loaded_solver = pickle.load(file)
    print("Exploitability of the loaded model: {:.6f}".format(
        pyspiel.exploitability(game, loaded_solver.average_policy())))

    for i in range(int(_ITERATIONS.value / 2)):
        loaded_solver.evaluate_and_update_policy()
        tabular_policy = loaded_solver.tabular_average_policy()
        print(f"Tabular policy length: {len(tabular_policy)}")
        print(
            "Iteration {} exploitability: {:.6f}".format(
                int(_ITERATIONS.value / 2) + i,
                pyspiel.exploitability(game, loaded_solver.average_policy()),
            )
        )

if __name__ == "__main__":
  app.run(main)
