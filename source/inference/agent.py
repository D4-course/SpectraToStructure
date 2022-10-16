# pylint: disable=consider-using-f-string,too-few-public-methods
"""
This module has serialized agent which doesn't use ray parallel processing.
"""
import time
import sys
from copy import deepcopy

import torch
# local imports
import numpy as np

from source.inference.model import ActionPredictionModel
from source.inference.mcts import MCTS
from source.inference.environment.environment import Env
from source.inference.environment.molecule_state import MolState
from source.inference.environment.RPNMR import predictor_new as P

MAX_NODES = 9
N_MCTS = 1
MAX_ACTIONS = MAX_NODES * MAX_NODES * 3

NUM_GPU = 1


def execute_episode(model, molform, spectra, episode_actor):
    """
    Executes an episode of search for a molform and spectra
    """

    if molform[0] < 1 or len(spectra) < 1:
        raise Exception("Molecule needs to have at least one Carbon")

    if molform[0] != len(spectra):
        raise Exception("Number of C13 signals and number of Carbons " +
                        "in molecular formula do not match")

    for each_c in spectra:
        if len(each_c) != 3:
            raise Exception("Input spectra does not have Shifts, Splits, and Index values")

    env = Env(molform, spectra, episode_actor)

    a_store = []

    start = time.time()
    state_0 = MolState(env.molForm, env.targetSpectra)
    mcts = MCTS(
        root_molstate=state_0,
        root=None,
        model=model,
        valuemodel=None,
        na=3 * MAX_NODES * MAX_NODES,
        gamma=1.0,
    )  # the object responsible for MCTS searches
    n_mcts = N_MCTS
    while True:
        # MCTS step
        mcts.search(n_mcts=n_mcts, c=1, Env=env)  # perform a forward search

        state, pi_tree, _value_tree, _fpib_tree = mcts.return_results()  # extract the root output
        if state.mol_state.rdmol.GetNumAtoms() > (sum(env.molForm) // 2):
            n_mcts = N_MCTS
        action_chosen = np.random.choice(len(pi_tree), p=pi_tree)
        a_store.append(action_chosen)
        state_next, reward_gotten, terminal = env.step(int(action_chosen))
        print(
            "{:18}".format(str(env.state)),
            ">>> Reward: {:.2f},".format(reward_gotten),
            "Model prior argmax:{:>5},".format(np.argmax(state.priors)),
            "Action taken:{:>5},".format(action_chosen),
            "Prior argmax:{:>5}".format(np.argmax(pi_tree)),
        )
        if terminal:
            break
        mcts.forward(action_chosen, state_next)
    del mcts
    # Finished episode
    sys.stdout.flush()

    print(
        "Finished episode, total return: {}, total time: {} sec".format(
            np.round(reward_gotten, 2), np.round((time.time() - start), 1)
        )
    )
    return str(env.state)


class EpisodeActor:
    """
    Serial analogue of episode actor which distributes forward calls to multiple GPUs
    """

    def __init__(self):
        self.nmr_pred_object = P.NMRPredictor(
            "trainedModels/RPNMRModels/best_model.meta",
            "trainedModels/RPNMRModels/best_model.00000000.state",
            False,
        )

    def predict(self, smile):
        """
        Predicts the NMR Spectra for a given SMILES
        """
        return self.nmr_pred_object.predict(smile)


### --------------models------------------ ###
model_instance = ActionPredictionModel(77, 6, 77, 64)
model_instance.load_state_dict(
    torch.load("trainedModels/default.state", map_location="cpu")
)
model_episode = ActionPredictionModel(77, 6, 77, 64)
model_episode.load_state_dict(deepcopy(model_instance.state_dict()))

episode_actor_instance = EpisodeActor()


def get_models():
    """
    Returns model instances
    """
    return model_instance, episode_actor_instance


def run():
    """
    Main Function
    """
    molform = [7, 0, 1, 0]
    spectra = [
        (25.2, "T", 1),
        (25.2, "T", 3),
        (26.1, "T", 0),
        (26.1, "T", 2),
        (26.1, "T", 4),
        (50.1, "D", 5),
        (204.7, "D", 6),
    ]

    print("Trying out molform : {}\nSpectra :{}".format(molform, spectra))
    print(
        "The molecule predicted by the agent is: {}".format(
            execute_episode(model_instance, molform, spectra, episode_actor_instance)
        )
    )

def predict(molform, spectra):
    print("Trying out molform : {}\nSpectra :{}".format(molform, spectra))
    prediction = execute_episode(model_instance, molform, spectra, episode_actor_instance)
    print(
        "The molecule predicted by the agent is: {}".format(
            prediction
        )
    )
    return prediction

if __name__ == "__main__":
    run()
