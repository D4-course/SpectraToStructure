MAX_NODES = 9
N_MCTS = 1
MAX_ACTIONS = MAX_NODES * MAX_NODES * 3

NUM_GPU = 1

from datetime import datetime

LOG_FILE = "./log_file_{}.txt".format(datetime.now().strftime("%d_%m"))
import torch

# local imports
from utils.helpers import  Database, store_safely
from model import ActionPredictionModel
from mcts import  MCTS
import numpy as np
import os
import time
import sys
from environment.environment import  Env
from environment.molecule_state import MolState
from copy import deepcopy
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from environment.RPNMR import predictor_new as P

def execute_episode(model, molform, spectra, episode_actor):
    env = Env(molform, spectra, episode_actor)

    a_store = []
    exps = []

    start = time.time()
    s = MolState(env.molForm, env.targetSpectra)
    mcts = MCTS(root_molstate=s, root=None, model=model, valuemodel=None,na = 3*MAX_NODES*MAX_NODES,gamma=1.0)  # the object responsible for MCTS searches
    n_mcts = N_MCTS
    t = 1
    while True:
        # MCTS step
        mcts.search(n_mcts=n_mcts, c=1, Env=env)  # perform a forward search

        state, pi, V,fpib = mcts.return_results(t)  # extract the root output
        if(state.mol_state.rdmol.GetNumAtoms() > (sum(env.molForm)//2)):
            n_mcts = N_MCTS
            # n_mcts = 50 # Fail test to see if the framework is configured properly
        exps.append(({'mol_graph':state.mol_graph,'action_mask':state.action_mask,'action_1_mask':state.action_1_mask,'index_mask':state.index_mask}, V, pi,fpib))
        # Make the true step
        a = np.random.choice(len(pi), p=pi)
        a_store.append(a)
        s1, r, terminal = env.step(int(a))
        print("{:18}".format(str(env.state)),
              ">>> Reward: {:.2f},".format(r),
              "Model prior argmax:{:>5},".format(np.argmax(state.priors)),
              "Action taken:{:>5},".format(a),
              "Prior argmax:{:>5}".format(np.argmax(pi)),)
        R = r
        if terminal:
            break
        else:
            mcts.forward(a, s1)
    del mcts
    # Finished episode
    sys.stdout.flush()

    print('Finished episode, total return: {}, total time: {} sec'.format(np.round(R, 2),np.round((time.time() - start), 1)))
    return str(env.state)

class EpisodeActor(object):
    def __init__(self):
        self.NMR_Pred = P.NMRPredictor("../../trainedModels/RPNMRModels/best_model.meta","../../trainedModels/RPNMRModels/best_model.00000000.state",False)

    def predict(self, smile):
        return self.NMR_Pred.predict(smile)


### --------------models------------------ ###
model =  ActionPredictionModel(77, 6, 77,64)
model.load_state_dict(torch.load("../../trainedModels/default.state",map_location='cpu'))
model_episode =  ActionPredictionModel(77, 6, 77,64)
model_episode.load_state_dict(deepcopy(model.state_dict()))

episode_actor = EpisodeActor()

def run():

    molform = [7,0,1,0]
    spectra = [(25.2, 'T', 1), (25.2, 'T', 3), (26.1, 'T', 0), (26.1, 'T', 2), (26.1, 'T', 4), (50.1, 'D', 5), (204.7, 'D', 6)]

    print("Trying out molform : {}\nSpectra :{}".format(molform, spectra))
    print("The molecule predicted by the agent is: {}".format(execute_episode(model, molform, spectra, episode_actor)))

if __name__ == "__main__":
    run()