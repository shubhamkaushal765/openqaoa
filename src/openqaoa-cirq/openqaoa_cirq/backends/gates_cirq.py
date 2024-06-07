from typing import Callable

import cirq
from openqaoa.qaoa_components.ansatz_constructor.rotationangle import RotationAngle
import openqaoa.qaoa_components.ansatz_constructor.gates as gates_core

class CirqGateApplicator(gates_core.GateApplicator):
    CIRQ_OQ_GATE_MAPPER = {
        gates_core.X.__name__: cirq.X,
        gates_core.RZ.__name__: cirq.rz,
        gates_core.RX.__name__: cirq.rx,
        gates_core.RY.__name__: cirq.ry,
        gates_core.CX.__name__: cirq.CNOT,
        gates_core.CZ.__name__: cirq.CZ,
        gates_core.RXX.__name__: cirq.XXPowGate,
        gates_core.RZX.__name__: cirq.ZXPowGate,
        gates_core.RZZ.__name__: cirq.ZZPowGate,
        gates_core.RYY.__name__: cirq.YYPowGate,
        gates_core.CPHASE.__name__: cirq.CZPowGate,
    }

    library = "cirq"

    def create_quantum_circuit(self, n_qubits) -> cirq.Circuit:
        """
        Function which creates an empty circuit specific to the cirq backend.
        Needed for SPAM twirling but more general than this.
        """
        qubits = [cirq.LineQubit(i) for i in range(n_qubits)]
        parametric_circuit = cirq.Circuit()
        return parametric_circuit, qubits

    def gate_selector(self, gate: gates_core.Gate) -> Callable:
        selected_cirq_gate = CirqGateApplicator.CIRQ_OQ_GATE_MAPPER[gate.__name__]
        return selected_cirq_gate

    @staticmethod
    def apply_1q_rotation_gate(
        cirq_gate,
        qubit_1: cirq.LineQubit,
        rotation_object: RotationAngle,
        circuit: cirq.Circuit,
    ) -> cirq.Circuit:
        circuit.append(cirq_gate(rotation_object.rotation_angle)(qubit_1))
        return circuit

    @staticmethod
    def apply_2q_rotation_gate(
        cirq_gate,
        qubit_1: cirq.LineQubit,
        qubit_2: cirq.LineQubit,
        rotation_object: RotationAngle,
        circuit: cirq.Circuit,
    ) -> cirq.Circuit:
        circuit.append(cirq_gate(rotation_object.rotation_angle)(qubit_1, qubit_2))
        return circuit

    @staticmethod
    def apply_1q_fixed_gate(
        cirq_gate, qubit_1: cirq.LineQubit, circuit: cirq.Circuit
    ) -> cirq.Circuit:
        circuit.append(cirq_gate(qubit_1))
        return circuit

    @staticmethod
    def apply_2q_fixed_gate(
        cirq_gate, qubit_1: cirq.LineQubit, qubit_2: cirq.LineQubit, circuit: cirq.Circuit
    ) -> cirq.Circuit:
        circuit.append(cirq_gate(qubit_1, qubit_2))
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
