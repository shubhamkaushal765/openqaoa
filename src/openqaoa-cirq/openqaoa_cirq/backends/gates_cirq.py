from typing import Callable
import numpy as np
import sympy as sp

import cirq
from cirq.circuits import Circuit
from openqaoa.qaoa_components.ansatz_constructor.rotationangle import RotationAngle
import openqaoa.qaoa_components.ansatz_constructor.gates as gates_core


class RZXGate(cirq.Gate):
    def __init__(self, exponent):
        super().__init__()
        self.exponent = exponent
        self.theta = self.exponent * np.pi

    def _unitary_(self):
        return np.array(
            [
                [np.cos(self.theta / 2), 0, 0, -1j * np.sin(self.theta / 2)],
                [0, np.cos(self.theta / 2), -1j * np.sin(self.theta / 2), 0],
                [0, -1j * np.sin(self.theta / 2), np.cos(self.theta / 2), 0],
                [-1j * np.sin(self.theta / 2), 0, 0, np.cos(self.theta / 2)],
            ]
        )

    def _num_qubits_(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return f"RZX({self.theta})", f"RZX({self.theta})"


class RZZGate(cirq.Gate):
    def __init__(self, theta):
        super().__init__()
        self.theta = theta

    def _unitary_(self):
        if isinstance(self.theta, sp.Symbol):
            return np.array(
                [
                    [sp.exp(-1j * self.theta / 2), 0, 0, 0],
                    [0, sp.exp(1j * self.theta / 2), 0, 0],
                    [0, 0, sp.exp(1j * self.theta / 2), 0],
                    [0, 0, 0, sp.exp(-1j * self.theta / 2)],
                ]
            )
        else:
            # If theta is a numeric value, compute the unitary matrix directly
            return np.array(
                [
                    [np.exp(-1j * self.theta / 2), 0, 0, 0],
                    [0, np.exp(1j * self.theta / 2), 0, 0],
                    [0, 0, np.exp(1j * self.theta / 2), 0],
                    [0, 0, 0, np.exp(-1j * self.theta / 2)],
                ]
            )

    def _num_qubits_(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return f"RZZ({self.theta})", f"RZZ({self.theta})"


class CirqGateApplicator(gates_core.GateApplicator):
    """
    All the 2q-rotation-gates take exponent as input. Under the hood,
    the implementation takes care of converting the provided rotation angle
    to the corresponding exponent value that Cirq's built-in gates expect.
    This conversion is typically done by dividing the rotation angle by π (pi)
    to obtain the exponent.

    The user can simply specify the desired rotation angle, and the gate will
    handle the necessary conversions internally.
    """

    CIRQ_OQ_GATE_MAPPER = {
        gates_core.X.__name__: cirq.X,
        gates_core.RZ.__name__: cirq.rz,
        gates_core.RX.__name__: cirq.rx,
        gates_core.RY.__name__: cirq.ry,
        gates_core.CX.__name__: cirq.CNOT,
        gates_core.CZ.__name__: cirq.CZ,
        gates_core.RXX.__name__: cirq.XXPowGate,
        # gates_core.RZX.__name__: RZXGate,
        gates_core.RZZ.__name__: cirq.ZZPowGate,  # (rotation angle is exponent / np.pi)
        gates_core.RYY.__name__: cirq.YYPowGate,
        gates_core.CPHASE.__name__: cirq.CZPowGate,
    }

    library = "cirq"

    # def create_quantum_circuit(self, n_qubits) -> cirq.Circuit:
    #     """
    #     Function which creates an empty circuit specific to the cirq backend.
    #     Needed for SPAM twirling but more general than this.
    #     """
    #     qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
    #     parametric_circuit = cirq.Circuit(qubits)
    #     return parametric_circuit

    def gate_selector(self, gate: gates_core.Gate) -> Callable:
        selected_cirq_gate = CirqGateApplicator.CIRQ_OQ_GATE_MAPPER[gate.__name__]
        return selected_cirq_gate

    @staticmethod
    def apply_1q_rotation_gate(
        cirq_gate,
        qubit_1: int,
        rotation_object: RotationAngle,
        circuit: Circuit,
    ) -> Circuit:
        q1 = cirq.LineQubit(qubit_1)
        circuit.append(cirq_gate(rotation_object.rotation_angle).on(q1))
        return circuit

    @staticmethod
    def apply_2q_rotation_gate(
        cirq_gate,
        qubit_1: int,
        qubit_2: int,
        rotation_object: RotationAngle,
        circuit: Circuit,
    ) -> Circuit:
        q1 = cirq.LineQubit(qubit_1)
        q2 = cirq.LineQubit(qubit_2)
        circuit.append(
            cirq_gate(exponent=rotation_object.rotation_angle / np.pi).on(q1, q2)
        )
        return circuit

    @staticmethod
    def apply_1q_fixed_gate(cirq_gate, qubit_1: int, circuit: Circuit) -> Circuit:
        q1 = cirq.LineQubit(qubit_1)
        circuit.append(cirq_gate.on(q1))
        return circuit

    @staticmethod
    def apply_2q_fixed_gate(
        cirq_gate,
        qubit_1: int,
        qubit_2: int,
        circuit: Circuit,
    ) -> Circuit:
        q1 = cirq.LineQubit(qubit_1)
        q2 = cirq.LineQubit(qubit_2)
        circuit.append(cirq_gate.on(q1, q2))
        return circuit

    def apply_gate(self, gate: gates_core.Gate, *args):
        selected_cirq_gate = self.gate_selector(gate)
        if gate.n_qubits == 1:
            if hasattr(gate, "rotation_object"):
                # *args must be of the following format -- (qubit_1,rotation_object,circuit)
                return self.apply_1q_rotation_gate(selected_cirq_gate, *args)
            else:
                # *args must be of the following format -- (qubit_1,circuit)
                return self.apply_1q_fixed_gate(selected_cirq_gate, *args)
        elif gate.n_qubits == 2:
            if hasattr(gate, "rotation_object"):
                # *args must be of the following format -- (qubit_1,qubit_2,rotation_object,circuit)
                return self.apply_2q_rotation_gate(selected_cirq_gate, *args)
            else:
                # *args must be of the following format -- (qubit_1,qubit_2,circuit)
                return self.apply_2q_fixed_gate(selected_cirq_gate, *args)
        else:
            raise ValueError("Only 1 and 2-qubit gates are supported.")
