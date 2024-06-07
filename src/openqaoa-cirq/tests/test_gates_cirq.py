import unittest
import numpy as np

import cirq

from openqaoa.qaoa_components.ansatz_constructor.gates import (
    RY,
    RX,
    RZ,
    CZ,
    CX,
    RXX,
    RYY,
    RZZ,
    RZX,
    CPHASE,
    RiSWAP,
)
from openqaoa_qiskit.backends.gates_qiskit import QiskitGateApplicator


class TestingGate(unittest.TestCase):
    def setUp(self):
        self.qiskit_gate_applicator = QiskitGateApplicator()

    def test_ibm_gates_1q(self):
        # Qiskit Gate Applicator
        gate_applicator = self.qiskit_gate_applicator

        # One Qubit Gate Tests
        rotation_angle_obj = cirq.ParamResolver({'x': np.pi})

        empty_circuit = cirq.Circuit()
        llgate = RY(gate_applicator, 0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = cirq.Circuit(cirq.ry(np.pi).on(cirq.LineQubit(0)))

        self.assertEqual(
            str(test_circuit),
            str(output_circuit),
        )

        empty_circuit = cirq.Circuit()
        llgate = RX(gate_applicator, 0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = cirq.Circuit(cirq.rx(np.pi).on(cirq.LineQubit(0)))

        self.assertEqual(
            str(test_circuit),
            str(output_circuit),
        )

        empty_circuit = cirq.Circuit()
        llgate = RZ(gate_applicator, 0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = cirq.Circuit(cirq.rz(np.pi).on(cirq.LineQubit(0)))

        self.assertEqual(
            str(test_circuit),
            str(output_circuit),
        )

    def test_ibm_gates_2q(self):
        # Qiskit Gate Applicator
        gate_applicator = self.qiskit_gate_applicator

        # Two Qubit Gate Tests
        empty_circuit = cirq.Circuit()
        llgate = CZ(gate_applicator, 0, 1)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = cirq.Circuit(cirq.CZ(cirq.LineQubit(0), cirq.LineQubit(1)))

        self.assertEqual(
            str(test_circuit),
            str(output_circuit),
        )

        empty_circuit = cirq.Circuit()
        llgate = CX(gate_applicator, 0, 1)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = cirq.Circuit(cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(1)))

        self.assertEqual(
            str(test_circuit),
            str(output_circuit),
        )

    #         empty_circuit = cirq.Circuit()
    #         llgate = CX(gate_applicator)
    #         output_circuit = llgate.apply_ibm_gate([0, 1], empty_circuit)

    #         test_circuit = cirq.Circuit()
    #         test_circuit.ry(np.pi / 2, 1)
    #         test_circuit.rx(np.pi, 1)
    #         test_circuit.cz(0, 1)
    #         test_circuit.ry(np.pi / 2, 1)
    #         test_circuit.rx(np.pi, 1)

    #         self.assertEqual(
    #             test_circuit.to_instruction().definition,
    #             output_circuit.to_instruction().definition,
    #         )

    def test_ibm_gates_2q_w_gates(self):
        # Qiskit Gate Applicator
        gate_applicator = self.qiskit_gate_applicator

        # Two Qubit Gate with Angles Tests
        rotation_angle_obj = cirq.ParamResolver({'x': np.pi})

        empty_circuit = cirq.Circuit()
        llgate = RXX(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = cirq.Circuit(cirq.rxx(np.pi).on(cirq.LineQubit(0), cirq.LineQubit(1)))

        self.assertEqual(
            str(test_circuit),
            str(output_circuit),
        )

        empty_circuit = cirq.Circuit()
        llgate = RYY(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = cirq.Circuit(cirq.ryy(np.pi).on(cirq.LineQubit(0), cirq.LineQubit(1)))

        self.assertEqual(
            str(test_circuit),
            str(output_circuit),
        )

        empty_circuit = cirq.Circuit()
        llgate = RZZ(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = cirq.Circuit(cirq.rzz(np.pi).on(cirq.LineQubit(0), cirq.LineQubit(1)))

        self.assertEqual(
            str(test_circuit),
            str(output_circuit),
        )

        empty_circuit = cirq.Circuit()
        llgate = RZX(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = cirq.Circuit(cirq.rzx(np.pi).on(cirq.LineQubit(0), cirq.LineQubit(1)))

        self.assertEqual(
            str(test_circuit),
            str(output_circuit),
        )

        empty_circuit = cirq.Circuit()
        llgate = CPHASE(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = cirq.Circuit(cirq.CZPowGate(exponent=np.pi).on(cirq.LineQubit(0), cirq.LineQubit(1)))

        self.assertEqual(
            str(test_circuit),
            str(output_circuit),
        )


if __name__ == "__main__":
    unittest.main()
