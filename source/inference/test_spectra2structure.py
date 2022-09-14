import pytest
import numpy as np
from rdkit import Chem

from agent import get_models, execute_episode

from mcts import MCTS
from environment.environment import Env
from environment.molecule_state import MolState

MAX_NODES = 9
N_MCTS = 1
MAX_ACTIONS = MAX_NODES * MAX_NODES * 3
NUM_GPU = 1

model, episode_actor = get_models()

class TestRPNMRPredictor:
    """
    Class to test the working of the RPNMR predictor
    """
    def test_num_peaks(self):
        """
        When given a SMILES string to predict the NMR spectra, the RPNMR predictor has to return 
        """
        smiles_str = "CC"
        predicted_spectra = episode_actor.predict(smiles_str)
        mol_obj = Chem.MolFromSmiles(smiles_str)

        num_carbons = 0
        for atom in mol_obj.GetAtoms():
            if atom.GetSymbol() == "C":
                num_carbons += 1

        assert len(predicted_spectra) == num_carbons


class TestExecuteEpisode:
    """
    Class to test the working of executing an episode - where an input spectra is given and a SMILES is returned
    """

    def test_execute_episode(self):
        """
        Main test that simulates the execute_episode() function, and runs without any exceptions
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


    def test_no_carbon(self):
        """
        Test that checks if an Exception is properly raised when the
        input NMR spectra and molecular formula have no Carbon atoms
        """

        molform = [0, 0, 0, 0]
        spectra = []

        with pytest.raises(Exception, match="Molecule needs to have at least one Carbon"):
            execute_episode(model, molform, spectra, episode_actor)

    def test_improper_spectra(self):
        """
        Test that checks if an Exception is properly raised when the NMR spectra
        is incorrect with respect to the molecular formula
        """

        molform = [6, 0, 0, 0]
        spectra = [
            (128.5, 'D', 0),
            (128.5, 'D', 1),
            (128.5, 'D', 2),
            (128.5, 'D', 3),
            (128.5, 'D', 4),
        ]

        with pytest.raises(Exception, match="Number of C13 signals and number of Carbons in molecular formula do not match"):
            execute_episode(model, molform, spectra, episode_actor)

    def test_improper_mol_form(self):
        """
        Test that checks if an Exception is properly raised when the molecular formula
        is incorrect with respect to the NMR spectra
        """

        molform = [4, 0, 0, 0]
        spectra = [
            (128.5, 'D', 0),
            (128.5, 'D', 1),
            (128.5, 'D', 2),
            (128.5, 'D', 3),
            (128.5, 'D', 4),
            (128.5, 'D', 5)
        ]

        with pytest.raises(Exception, match="Number of C13 signals and number of Carbons in molecular formula do not match"):
            execute_episode(model, molform, spectra, episode_actor)

    def test_improper_spectra_format(self):
        """
        Test that checks if an Exception is properly raised when the NMR spectra
        is not in the right format
        """

        molform = [6, 0, 0, 0]
        spectra = [
            (128.5, 'D', 0),
            (128.5, 'D', 1),
            (128.5, 'D'),
            (128.5, 'D', 3),
            (128.5, 'D', 4),
            (128.5, 'D', 5)
        ]

        with pytest.raises(Exception, match="Input spectra does not have Shifts, Splits, and Index values"):
            execute_episode(model, molform, spectra, episode_actor)

    def test_execute_episode(self):
        """
        Test that a simple molecule like Benzene is predicted
        """

        molform = [6, 0, 0, 0]
        spectra = [
            (128.5, 'D', 0),
            (128.5, 'D', 1),
            (128.5, 'D', 2),
            (128.5, 'D', 3),
            (128.5, 'D', 4),
            (128.5, 'D', 5)
        ]

        assert execute_episode(model, molform, spectra, episode_actor) == "C1=CC=CC=C1"
