import unittest
import networkx as nx
import cirq

from openqaoa import QAOA
from openqaoa.problems import MinimumVertexCover
from openqaoa.backends.devices import DeviceCirq


class TestingLoggerClass(unittest.TestCase):
    def test_plot_probabilities(self):
        # Create the problem
        g = nx.circulant_graph(6, [1])
        vc = MinimumVertexCover(g, field=1.0, penalty=10).qubo

        # Create the Cirq device
        device = DeviceCirq(device_name="Simulator")

        # Create the QAOA instance
        q_shot = QAOA(device)

        q_shot.compile(vc)
        q_shot.optimize()

        q_shot.result.plot_probabilities()