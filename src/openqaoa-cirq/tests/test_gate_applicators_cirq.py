import unittest
import openqaoa
import openqaoa.qaoa_components.ansatz_constructor.gates as oq_gate_mod
from openqaoa_cirq.backends.gates_cirq import CirqGateApplicator
from openqaoa.qaoa_components.ansatz_constructor.rotationangle import RotationAngle
import cirq


class TestCirqGateApplicator(unittest.TestCase):
    def setUp(self):
        self.available_qiskit_gates = {
            gate.__name__.lower(): gate
            for gate in [
                *cirq.SingleQubitGate.__subclasses__(),
                *cirq.TwoQubitGate.__subclasses__(),
            ]
        }

        self.qiskit_excluded_gates = [
            oq_gate_mod.OneQubitGate,
            oq_gate_mod.OneQubitRotationGate,
            oq_gate_mod.TwoQubitGate,
            oq_gate_mod.TwoQubitRotationGate,
            oq_gate_mod.RXY,
            oq_gate_mod.RiSWAP,
            oq_gate_mod.RYZ,
        ]

    def test_gate_applicator_mapper(self):
        """
        The mapper to the gate applicator should only contain gates that
        are trivially supported by the library.
        """

        for gate in CirqGateApplicator.CIRQ_OQ_GATE_MAPPER.values():
            self.assertTrue(
                gate.__name__.lower() in self.available_qiskit_gates.keys(),
                f"{gate.__name__}, {self.available_qiskit_gates.keys()}",
            )

    def test_gate_selector(self):
        """
        This method should return the Cirq Gate object based on the input OQ
        Gate object.
        """

        gate_applicator = CirqGateApplicator()

        oq_gate_list = (
            list(oq_gate_mod.Gate.__subclasses__())
            + list(oq_gate_mod.OneQubitGate.__subclasses__())
        )

        for gate in oq_gate_list:
            if gate not in self.qiskit_excluded_gates:
                returned_gate = gate_applicator.gate_selector(gate())
                self.assertEqual(
                    self.available_qiskit_gates[returned_gate.__name__.lower()],
                    returned_gate,
                )

    def test_static_methods_1q(self):
        """
        Checks that the static method, apply_1q_fixed_gate, apply the correct
        gate to the circuit object.
        """

        gate_applicator = CirqGateApplicator()
        circuit = cirq.Circuit()
        output_circuit = gate_applicator.apply_1q_fixed_gate(cirq.X, 0, circuit)

        self.assertEqual(
            [op.qubits[0].x for op in output_circuit.all_operations()], [0]
        )
        self.assertEqual(
            [op.gate.__class__.__name__ for op in output_circuit.all_operations()],
            ["XPowGate"],
        )

    def test_static_methods_1qr(self):
        """
        Checks that the static method, apply_1q_rotation_gate, apply the correct
        gate to the circuit object.
        """

        gate_applicator = CirqGateApplicator()
        input_angle = 1
        rot_obj = RotationAngle(lambda x: x, None, input_angle)

        each_sub_gate = [
            gate
            for gate in oq_gate_mod.OneQubitRotationGate.__subclasses__()
            if gate not in self.qiskit_excluded_gates
        ]

        for gate in each_sub_gate:
            circuit = cirq.Circuit()
            output_circuit = gate_applicator.apply_gate(
                gate(gate_applicator, 0, rot_obj), 0, rot_obj, circuit
            )

            self.assertEqual(
                [op.gate.exponent for op in output_circuit.all_operations()],
                [input_angle],
            )
            self.assertEqual(
                [op.qubits[0].x for op in output_circuit.all_operations()], [0]
            )

    def test_static_methods_2q(self):
        """
        Checks that the static method, apply_2q_fixed_gate, apply the correct
        gate to the circuit object.
        """

        gate_applicator = CirqGateApplicator()

        each_sub_gate = [
            gate
            for gate in oq_gate_mod.TwoQubitGate.__subclasses__()
            if gate not in self.qiskit_excluded_gates
        ]

        for gate in each_sub_gate:
            circuit = cirq.Circuit()
            output_circuit = gate_applicator.apply_gate(gate(gate_applicator, 0, 1), 0, 1, circuit)

            self.assertEqual(
                [op.qubits[i].x for op in output_circuit.all_operations() for i in range(2)],
                [0, 0, 1, 1],
            )
            self.assertEqual(
                [op.gate.__class__.__name__ for op in output_circuit.all_operations()],
                [gate.__name__],
            )

    def test_static_methods_2qr(self):
        """
        Checks that the static method, apply_2q_rotation_gate, apply the correct
        gate to the circuit object.
        """

        gate_applicator = CirqGateApplicator()
        input_angle = 1
        rot_obj = RotationAngle(lambda x: x, None, input_angle)

        each_sub_gate = [
            gate
            for gate in oq_gate_mod.TwoQubitRotationGate.__subclasses__()
            if gate not in self.qiskit_excluded_gates
        ]

        for gate in each_sub_gate:
            circuit = cirq.Circuit()
            output_circuit = gate_applicator.apply_gate(
                gate(gate_applicator, 0, 1, rot_obj), 0, 1, rot_obj, circuit
            )

            self.assertEqual(
                [op.gate.exponent for op in output_circuit.all_operations()],
                [input_angle],
            )
            self.assertEqual(
                [op.qubits[i].x for op in output_circuit.all_operations() for i in range(2)],
                [0, 1, 0, 1],
            )


if __name__ == "__main__":
    unittest.main()
