import unittest
import numpy as np
import itertools
import cirq

from openqaoa.qaoa_components import (
    PauliOp,
    Hamiltonian,
    QAOADescriptor,
    QAOAVariationalExtendedParams,
    QAOAVariationalStandardParams,
)
from openqaoa_cirq.backends import (
    QAOACirqBackendStatevecSimulator,
    QAOACirqBackendShotBasedSimulator,
)
from openqaoa.backends import QAOAvectorizedBackendSimulator
from openqaoa.utilities import X_mixer_hamiltonian, ring_of_disagrees


def resolve_to_desired_precision(circuit, desired_precision=1e-5):
    modified_circuit = cirq.Circuit()
    for moment in circuit:
        modified_moment = []
        for op in moment.operations:
            if isinstance(op.gate, cirq.ops.eigen_gate.EigenGate):
                if isinstance(
                    op.gate,
                    cirq.ops.common_gates.XPowGate,
                ):
                    rounded_rads = (
                        np.round(
                            op.gate.exponent * np.pi / 2,
                            int(-np.log10(desired_precision)),
                        )
                        * 2
                        / np.pi
                    )
                    rounded_gate = cirq.rx(rads=rounded_rads)

                elif isinstance(op.gate, cirq.ops.common_gates.ZPowGate):
                    rounded_rads = np.round(
                        op.gate.exponent * np.pi, int(-np.log10(desired_precision))
                    )
                    rounded_gate = cirq.rz(rads=rounded_rads)
                else:
                    rounded_exponent = np.round(
                        op.gate.exponent, int(-np.log10(desired_precision))
                    )
                    rounded_gate = op.gate.__class__(
                        exponent=rounded_exponent, global_shift=op.gate.global_shift
                    )
                modified_op = op.with_gate(rounded_gate)
                modified_moment.append(modified_op)
            else:
                modified_moment.append(op)
        modified_circuit.append(modified_moment)

    return modified_circuit


class TestingQAOACirqSimulatorBackend(unittest.TestCase):
    """This Object tests the QAOA Cirq Simulator Backend objects, which is
    tasked with the creation and execution of a QAOA circuit for the cirq
    library and its local backends.
    """

    def test_circuit_angle_assignment_statevec_backend(self):
        """
        A tests that checks if the circuit created by the Cirq Backend
        has the appropriate angles assigned before the circuit is executed.
        Checks the circuit created on Cirq Simulator Backends.
        """

        ntrials = 10

        nqubits = 3
        q0, q1, q2 = cirq.LineQubit.range(nqubits)

        p = 2
        weights = [
            [np.random.rand(), np.random.rand(), np.random.rand()]
            for i in range(ntrials)
        ]
        init_hadamards = [np.random.choice([True, False]) for i in range(ntrials)]
        constants = [np.random.rand() for i in range(ntrials)]

        for i in range(ntrials):
            gammas = [np.random.rand() * np.pi for i in range(p)]
            betas = [np.random.rand() * np.pi for i in range(p)]

            print(gammas)
            print(betas)

            cost_hamil = Hamiltonian(
                [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
                weights[i],
                constants[i],
            )
            mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
            variate_params = QAOAVariationalStandardParams(
                qaoa_descriptor, betas, gammas
            )

            cirq_statevec_backend = QAOACirqBackendStatevecSimulator(
                qaoa_descriptor, None, None, False
            )

            statevec_circuit = cirq_statevec_backend.qaoa_circuit(variate_params)

            # Trivial Decomposition
            main_circuit = cirq.Circuit()
            main_circuit.append(
                cirq.ZZPowGate(exponent=2 * weights[i][0] * gammas[0] / np.pi)(q0, q1)
            )
            main_circuit.append(
                cirq.ZZPowGate(exponent=2 * weights[i][1] * gammas[0] / np.pi)(q1, q2)
            )
            main_circuit.append(
                cirq.ZZPowGate(exponent=2 * weights[i][2] * gammas[0] / np.pi)(q0, q2)
            )
            main_circuit.append(cirq.rx(-2 * betas[0])(q0))
            main_circuit.append(cirq.rx(-2 * betas[0])(q1))
            main_circuit.append(cirq.rx(-2 * betas[0])(q2))
            main_circuit.append(
                cirq.ZZPowGate(exponent=2 * weights[i][0] * gammas[1] / np.pi)(q0, q1)
            )
            main_circuit.append(
                cirq.ZZPowGate(exponent=2 * weights[i][1] * gammas[1] / np.pi)(q1, q2)
            )
            main_circuit.append(
                cirq.ZZPowGate(exponent=2 * weights[i][2] * gammas[1] / np.pi)(q0, q2)
            )
            main_circuit.append(cirq.rx(-2 * betas[1])(q0))
            main_circuit.append(cirq.rx(-2 * betas[1])(q1))
            main_circuit.append(cirq.rx(-2 * betas[1])(q2))

            modified_main_circuit = resolve_to_desired_precision(main_circuit)
            modified_statevec_circuit = resolve_to_desired_precision(statevec_circuit)

            cirq.testing.assert_same_circuits(
                modified_main_circuit, modified_statevec_circuit
            )

    def test_circuit_angle_assignment_statevec_backend_w_hadamard(self):
        """
        Checks for consistent if init_hadamard is set to True.
        """

        ntrials = 10

        nqubits = 3
        q0, q1, q2 = cirq.LineQubit.range(nqubits)
        p = 2
        weights = [
            [np.random.rand(), np.random.rand(), np.random.rand()]
            for i in range(ntrials)
        ]
        init_hadamards = [np.random.choice([True, False]) for i in range(ntrials)]
        constants = [np.random.rand() for i in range(ntrials)]

        for i in range(ntrials):
            gammas = [np.random.rand() * np.pi for i in range(p)]
            betas = [np.random.rand() * np.pi for i in range(p)]

            cost_hamil = Hamiltonian(
                [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
                weights[i],
                constants[i],
            )
            mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
            variate_params = QAOAVariationalStandardParams(
                qaoa_descriptor, betas, gammas
            )

            cirq_statevec_backend = QAOACirqBackendStatevecSimulator(
                qaoa_descriptor, None, None, True
            )

            statevec_circuit = cirq_statevec_backend.qaoa_circuit(variate_params)

            # Trivial Decomposition
            main_circuit = cirq.Circuit()
            main_circuit.append(cirq.H(q0))
            main_circuit.append(cirq.H(q1))
            main_circuit.append(cirq.H(q2))
            main_circuit.append(
                cirq.ZZPowGate(exponent=2 * weights[i][0] * gammas[0] / np.pi)(q0, q1)
            )
            main_circuit.append(
                cirq.ZZPowGate(exponent=2 * weights[i][1] * gammas[0] / np.pi)(q1, q2)
            )
            main_circuit.append(
                cirq.ZZPowGate(exponent=2 * weights[i][2] * gammas[0] / np.pi)(q0, q2)
            )
            main_circuit.append(cirq.rx(-2 * betas[0])(q0))
            main_circuit.append(cirq.rx(-2 * betas[0])(q1))
            main_circuit.append(cirq.rx(-2 * betas[0])(q2))
            main_circuit.append(
                cirq.ZZPowGate(exponent=2 * weights[i][0] * gammas[1] / np.pi)(q0, q1)
            )
            main_circuit.append(
                cirq.ZZPowGate(exponent=2 * weights[i][1] * gammas[1] / np.pi)(q1, q2)
            )
            main_circuit.append(
                cirq.ZZPowGate(exponent=2 * weights[i][2] * gammas[1] / np.pi)(q0, q2)
            )
            main_circuit.append(cirq.rx(-2 * betas[1])(q0))
            main_circuit.append(cirq.rx(-2 * betas[1])(q1))
            main_circuit.append(cirq.rx(-2 * betas[1])(q2))

            modified_main_circuit = resolve_to_desired_precision(main_circuit)
            modified_statevec_circuit = resolve_to_desired_precision(statevec_circuit)

            cirq.testing.assert_same_circuits(
                modified_main_circuit, modified_statevec_circuit
            )

    def test_prepend_circuit(self):
        """
        Checks if prepended circuit has been prepended correctly.
        """

        nqubits = 3
        q0, q1, q2 = cirq.LineQubit.range(nqubits)
        p = 1
        weights = [1, 1, 1]
        gammas = [1 / 8 * np.pi]
        betas = [1 / 8 * np.pi]
        shots = 10000

        # Prepended Circuit
        prepend_circuit = cirq.Circuit()
        prepend_circuit.append(cirq.X(q0))
        prepend_circuit.append(cirq.X(q1))
        prepend_circuit.append(cirq.X(q2))

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        cirq_statevec_backend = QAOACirqBackendStatevecSimulator(
            qaoa_descriptor, prepend_circuit, None, True
        )
        cirq_shot_backend = QAOACirqBackendShotBasedSimulator(
            qaoa_descriptor, shots, prepend_circuit, None, True, 1.0
        )

        statevec_circuit = cirq_statevec_backend.qaoa_circuit(variate_params)
        shot_circuit = cirq_shot_backend.qaoa_circuit(variate_params)

        # Trivial Decomposition
        main_circuit = cirq.Circuit()
        main_circuit.append(cirq.X(q0))
        main_circuit.append(cirq.X(q1))
        main_circuit.append(cirq.X(q2))
        main_circuit.append(cirq.H(q0))
        main_circuit.append(cirq.H(q1))
        main_circuit.append(cirq.H(q2))
        main_circuit.append(cirq.ZZPowGate(exponent=2 * gammas[0] / np.pi)(q0, q1))
        main_circuit.append(cirq.ZZPowGate(exponent=2 * gammas[0] / np.pi)(q1, q2))
        main_circuit.append(cirq.ZZPowGate(exponent=2 * gammas[0] / np.pi)(q0, q2))
        main_circuit.append(cirq.rx(-2 * betas[0])(q0))
        main_circuit.append(cirq.rx(-2 * betas[0])(q1))
        main_circuit.append(cirq.rx(-2 * betas[0])(q2))

        modified_main_circuit = resolve_to_desired_precision(main_circuit)
        modified_statevec_circuit = resolve_to_desired_precision(statevec_circuit)
        modified_shot_circuit = resolve_to_desired_precision(shot_circuit)

        cirq.testing.assert_same_circuits(
            modified_main_circuit, modified_statevec_circuit
        )
        cirq.testing.assert_same_circuits(modified_main_circuit, modified_shot_circuit)

    def test_append_circuit(self):
        """
        Checks if appended circuit is appropriately appended to the back of the
        QAOA Circuit.
        """

        nqubits = 3
        q0, q1, q2 = cirq.LineQubit.range(3)
        p = 1
        weights = [1, 1, 1]
        gammas = [1 / 8 * np.pi]
        betas = [1 / 8 * np.pi]
        shots = 10000

        # Appended Circuit
        append_circuit = cirq.Circuit()
        append_circuit.append(cirq.X(q0))
        append_circuit.append(cirq.X(q1))
        append_circuit.append(cirq.X(q2))

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        cirq_statevec_backend = QAOACirqBackendStatevecSimulator(
            qaoa_descriptor, None, append_circuit, True
        )
        cirq_shot_backend = QAOACirqBackendShotBasedSimulator(
            qaoa_descriptor, shots, None, append_circuit, True, 1.0
        )

        statevec_circuit = cirq_statevec_backend.qaoa_circuit(variate_params)
        shot_circuit = cirq_shot_backend.qaoa_circuit(variate_params)

        # Standard Decomposition
        main_circuit = cirq.Circuit()
        main_circuit.append(cirq.H(q0))
        main_circuit.append(cirq.H(q1))
        main_circuit.append(cirq.H(q2))
        main_circuit.append(cirq.ZZPowGate(exponent=2 * gammas[0] / np.pi)(q0, q1))
        main_circuit.append(cirq.ZZPowGate(exponent=2 * gammas[0] / np.pi)(q1, q2))
        main_circuit.append(cirq.ZZPowGate(exponent=2 * gammas[0] / np.pi)(q0, q2))
        main_circuit.append(cirq.rx(-2 * betas[0])(q0))
        main_circuit.append(cirq.rx(-2 * betas[0])(q1))
        main_circuit.append(cirq.rx(-2 * betas[0])(q2))
        main_circuit.append(cirq.X(q0))
        main_circuit.append(cirq.X(q1))
        main_circuit.append(cirq.X(q2))

        modified_main_circuit = resolve_to_desired_precision(main_circuit)
        modified_statevec_circuit = resolve_to_desired_precision(statevec_circuit)
        modified_shot_circuit = resolve_to_desired_precision(shot_circuit)

        cirq.testing.assert_same_circuits(
            modified_main_circuit, modified_statevec_circuit
        )
        cirq.testing.assert_same_circuits(modified_main_circuit, modified_shot_circuit)

    def test_qaoa_circuit_wavefunction_expectation_equivalence_1(self):
        """
        The following tests with a similar naming scheme check for consistency
        between the outputs of the cirq statevector simulator and the
        vectorized backend.
        We compare both the wavefunctions returned by the Backends and the
        expectation values.
        """

        nqubits = 3
        p = 1
        weights = [1, 2, 3]
        gammas = [[3], [2]]
        betas = [[1], [1 / 8]]

        for i in range(2):
            cost_hamil = Hamiltonian(
                [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
                weights,
                1,
            )
            mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
            variate_params = QAOAVariationalStandardParams(
                qaoa_descriptor, betas[i], gammas[i]
            )

            cirq_backend = QAOACirqBackendStatevecSimulator(
                qaoa_descriptor, None, None, True
            )

            cirq_wavefunction = cirq_backend.wavefunction(variate_params)
            cirq_expectation = cirq_backend.expectation(variate_params)

            vector_backend = QAOAvectorizedBackendSimulator(
                qaoa_descriptor, None, None, True
            )
            vector_wavefunction = vector_backend.wavefunction(variate_params)
            vector_expectation = vector_backend.expectation(variate_params)

            self.assertAlmostEqual(cirq_expectation, vector_expectation)

            for j in range(2**nqubits):
                self.assertAlmostEqual(
                    cirq_wavefunction[j].real, vector_wavefunction[j].real
                )
                self.assertAlmostEqual(
                    cirq_wavefunction[j].imag, vector_wavefunction[j].imag
                )

    def test_qaoa_circuit_wavefunction_expectation_equivalence_2(self):
        """Due to the difference in the constructions of the statevector simulators,
        there is a global phase difference between the results obtained from
        cirq's statevector simulator and OpenQAOA's vectorised simulator. In order to
        show the equivalence between the wavefunctions produced, the expectation
        of a random operator is computed.
        """

        nqubits = 3
        p = 1
        weights = [1, 2, 3]
        gammas = [[1 / 8 * np.pi], [1 / 8 * np.pi]]
        betas = [[1], [1 / 8 * np.pi]]

        for i in range(2):
            cost_hamil = Hamiltonian(
                [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
                weights,
                1,
            )

            mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)

            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
            variate_params = QAOAVariationalStandardParams(
                qaoa_descriptor, betas[i], gammas[i]
            )

            cirq_backend = QAOACirqBackendStatevecSimulator(
                qaoa_descriptor, None, None, True
            )

            cirq_wavefunction = cirq_backend.wavefunction(variate_params)
            cirq_expectation = cirq_backend.expectation(variate_params)

            vector_backend = QAOAvectorizedBackendSimulator(
                qaoa_descriptor, None, None, True
            )
            vector_wavefunction = vector_backend.wavefunction(variate_params)
            vector_expectation = vector_backend.expectation(variate_params)

            random_operator = (
                np.random.rand(2**nqubits, 2**nqubits)
                + np.random.rand(2**nqubits, 2**nqubits) * 1j
            )
            random_herm = random_operator + random_operator.conj().T

            expect_cirq = np.matmul(
                np.array(cirq_wavefunction).T.conjugate(),
                np.matmul(random_herm, np.array(cirq_wavefunction)),
            )
            expect_vector = np.matmul(
                np.array(vector_wavefunction).T.conjugate(),
                np.matmul(random_herm, np.array(vector_wavefunction)),
            )

            self.assertAlmostEqual(expect_cirq.real, expect_vector.real)
            self.assertAlmostEqual(expect_cirq.imag, expect_vector.imag)
            self.assertAlmostEqual(cirq_expectation, vector_expectation)

    def test_qaoa_circuit_wavefunction_expectation_equivalence_3(self):
        """Due to the difference in the constructions of the statevector simulators,
        there is a global phase difference between the results obtained from
        cirq's statevector simulator and OpenQAOA's vectorised simulator. In order to
        show the equivalence between the wavefunctions produced, the expectation
        of a random operator is computed.

        Nonuniform mixer weights.
        """

        nqubits = 3
        p = 1
        weights = [1, 2, 3]
        gammas = [[1 / 8 * np.pi], [1 / 8 * np.pi]]
        betas = [[1], [1 / 8 * np.pi]]

        for i in range(2):
            cost_hamil = Hamiltonian(
                [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
                weights,
                1,
            )

            mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits, coeffs=[1, 2, 3])

            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
            variate_params = QAOAVariationalStandardParams(
                qaoa_descriptor, betas[i], gammas[i]
            )

            cirq_backend = QAOACirqBackendStatevecSimulator(
                qaoa_descriptor, None, None, True
            )

            cirq_wavefunction = cirq_backend.wavefunction(variate_params)
            cirq_expectation = cirq_backend.expectation(variate_params)

            vector_backend = QAOAvectorizedBackendSimulator(
                qaoa_descriptor, None, None, True
            )
            vector_wavefunction = vector_backend.wavefunction(variate_params)
            vector_expectation = vector_backend.expectation(variate_params)

            random_operator = (
                np.random.rand(2**nqubits, 2**nqubits)
                + np.random.rand(2**nqubits, 2**nqubits) * 1j
            )
            random_herm = random_operator + random_operator.conj().T

            expect_cirq = np.matmul(
                np.array(cirq_wavefunction).T.conjugate(),
                np.matmul(random_herm, np.array(cirq_wavefunction)),
            )
            expect_vector = np.matmul(
                np.array(vector_wavefunction).T.conjugate(),
                np.matmul(random_herm, np.array(vector_wavefunction)),
            )

            self.assertAlmostEqual(expect_cirq.real, expect_vector.real)
            self.assertAlmostEqual(expect_cirq.imag, expect_vector.imag)
            self.assertAlmostEqual(cirq_expectation, vector_expectation)

    def test_qaoa_circuit_wavefunction_expectation_equivalence_4(self):
        """Due to the difference in the constructions of the statevector simulators,
        there is a global phase difference between the results obtained from
        cirq's statevector simulator and OpenQAOA's vectorised simulator. In order to
        show the equivalence between the wavefunctions produced, the expectation
        of a random operator is computed.

        Y, YY and XX mixers with nonuniform weights.
        """

        nqubits = 3
        p = 1
        weights = [1, 2, 3]
        gammas = [[1 / 8 * np.pi], [1 / 8 * np.pi]]
        betas = [[1], [1 / 8 * np.pi]]

        for i in range(2):
            cost_hamil = Hamiltonian(
                [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
                weights,
                1,
            )
            mixer_hamil = Hamiltonian(
                [
                    PauliOp("Y", (0,)),
                    PauliOp("YY", (0, 1)),
                    PauliOp("XX", (1, 2)),
                    PauliOp("XZ", (1, 2)),
                ],
                [1, 2, 3, 4],
                1,
            )

            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
            variate_params = QAOAVariationalStandardParams(
                qaoa_descriptor, betas[i], gammas[i]
            )

            cirq_backend = QAOACirqBackendStatevecSimulator(
                qaoa_descriptor, None, None, True
            )

            cirq_wavefunction = cirq_backend.wavefunction(variate_params)
            cirq_expectation = cirq_backend.expectation(variate_params)

            vector_backend = QAOAvectorizedBackendSimulator(
                qaoa_descriptor, None, None, True
            )
            vector_wavefunction = vector_backend.wavefunction(variate_params)
            vector_expectation = vector_backend.expectation(variate_params)

            random_operator = (
                np.random.rand(2**nqubits, 2**nqubits)
                + np.random.rand(2**nqubits, 2**nqubits) * 1j
            )
            random_herm = random_operator + random_operator.conj().T

            expect_cirq = np.matmul(
                np.array(cirq_wavefunction).T.conjugate(),
                np.matmul(random_herm, np.array(cirq_wavefunction)),
            )
            expect_vector = np.matmul(
                np.array(vector_wavefunction).T.conjugate(),
                np.matmul(random_herm, np.array(vector_wavefunction)),
            )

            self.assertAlmostEqual(expect_cirq.real, expect_vector.real)
            self.assertAlmostEqual(expect_cirq.imag, expect_vector.imag)
            self.assertAlmostEqual(cirq_expectation, vector_expectation)

    def test_cost_call(self):
        """
        testing the __call__ method of the base class.
        Only for vectorized and Cirq Local Statevector Backends.
        """

        n_qubits = 8
        register = range(n_qubits)
        p = 1

        betas = [np.pi / 8]
        gammas = [np.pi / 4]
        cost_hamiltonian = ring_of_disagrees(register)
        mixer_hamiltonian = X_mixer_hamiltonian(n_qubits)
        qaoa_descriptor = QAOADescriptor(cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = QAOAVariationalStandardParams(
            qaoa_descriptor, betas, gammas
        )

        backend_cirq_statevec = QAOACirqBackendStatevecSimulator(
            qaoa_descriptor, prepend_state=None, append_state=None, init_hadamard=True
        )

        exp_cirq_statevec = backend_cirq_statevec.expectation((variational_params_std))

        assert np.isclose(exp_cirq_statevec, -6)

    def test_get_wavefunction(self):
        n_qubits = 3
        terms = [[0, 1], [0, 2], [0]]
        weights = [1, 1, -0.5]
        p = 1

        betas_singles = [np.pi, 0, 0]
        betas_pairs = []
        gammas_singles = [np.pi]
        gammas_pairs = [[1 / 2 * np.pi] * 2]

        cost_hamiltonian = Hamiltonian.classical_hamiltonian(
            terms=terms, coeffs=weights, constant=0
        )
        mixer_hamiltonian = X_mixer_hamiltonian(n_qubits)
        qaoa_descriptor = QAOADescriptor(cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = QAOAVariationalExtendedParams(
            qaoa_descriptor,
            betas_singles=betas_singles,
            betas_pairs=betas_pairs,
            gammas_singles=gammas_singles,
            gammas_pairs=gammas_pairs,
        )

        backend_cirq_statevec = QAOACirqBackendStatevecSimulator(
            qaoa_descriptor, prepend_state=None, append_state=None, init_hadamard=True
        )

        wf_cirq_statevec = backend_cirq_statevec.wavefunction((variational_params_std))
        expected_wf = 1j * np.array([-1, 1, 1, -1, 1, -1, -1, 1]) / (2 * np.sqrt(2))

        try:
            assert np.allclose(wf_cirq_statevec, expected_wf)
        except AssertionError:
            assert np.allclose(
                np.real(np.conjugate(wf_cirq_statevec) * wf_cirq_statevec),
                np.conjugate(expected_wf) * expected_wf,
            )

    def test_exact_solution(self):
        """
        NOTE: Since the implementation of exact solution is backend agnostic,
            checking it once should be okay.

        Nevertheless, for the sake of completeness, it will be tested for all backend
        instances.
        """

        n_qubits = 8
        register = range(n_qubits)
        p = 1

        correct_energy = -8
        correct_config = [0, 1, 0, 1, 0, 1, 0, 1]

        # The tests pass regardless of the value of betas and gammas. Is this correct?
        betas = [np.pi / 8]
        gammas = [np.pi / 4]

        cost_hamiltonian = ring_of_disagrees(register)
        mixer_hamiltonian = X_mixer_hamiltonian(n_qubits)
        qaoa_descriptor = QAOADescriptor(cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = QAOAVariationalStandardParams(
            qaoa_descriptor, betas, gammas
        )

        backend_cirq_statevec = QAOACirqBackendStatevecSimulator(
            qaoa_descriptor, prepend_state=None, append_state=None, init_hadamard=True
        )

        # Exact solution is defined as the property of the cost function
        energy_cirq, config_cirq = backend_cirq_statevec.exact_solution

        assert np.isclose(energy_cirq, correct_energy)

        config_cirq = [config.tolist() for config in config_cirq]

        assert correct_config in config_cirq

    def test_expectation_w_uncertainty(self):
        """
        Test the standard deviation equality. Expectation w uncertainty.
        """

        n_qubits = 8
        register = range(n_qubits)
        p = 1

        betas = [np.pi / 8]
        gammas = [np.pi / 4]
        cost_hamiltonian = ring_of_disagrees(register)
        mixer_hamiltonian = X_mixer_hamiltonian(n_qubits)
        qaoa_descriptor = QAOADescriptor(cost_hamiltonian, mixer_hamiltonian, p)
        variational_params_std = QAOAVariationalStandardParams(
            qaoa_descriptor, betas, gammas
        )

        backend_cirq_statevec = QAOACirqBackendStatevecSimulator(
            qaoa_descriptor, prepend_state=None, append_state=None, init_hadamard=True
        )

        (
            exp_cirq_statevec,
            exp_unc_cirq_statevec,
        ) = backend_cirq_statevec.expectation_w_uncertainty(variational_params_std)

        assert np.isclose(exp_cirq_statevec, -6)

        terms = [(register[i], register[(i + 1) % n_qubits]) for i in range(n_qubits)]
        weights = [0.5] * len(terms)

        # Check standard deviation
        # Get the matrix form of the Hamiltonian (note we just keep the diagonal
        # part) and square it
        ham_matrix = np.zeros((2 ** len(register)))
        for i, term in enumerate(terms):
            out = np.real(weights[i])
            for qubit in register:
                if qubit in term:
                    out = np.kron([1, -1], out)
                else:
                    out = np.kron([1, 1], out)
            ham_matrix += out

        ham_matrix_sq = np.square(ham_matrix)

        # Get the wavefunction
        wf = backend_cirq_statevec.wavefunction(variational_params_std)

        # Get the probabilities
        probs = np.real(np.conjugate(wf) * wf)

        # Standard deviation
        exp_2 = np.dot(probs, ham_matrix)
        std_dev2 = np.sqrt(np.dot(probs, ham_matrix_sq) - exp_2**2)

        assert np.isclose(exp_unc_cirq_statevec, std_dev2)

    def test_expectation_w_randomizing_variables(self):
        """
        Run ntrials sets of randomized input parameters and compares the
        expectation value output between the cirq statevector simulator and
        vectorized simulator.
        """

        ntrials = 100

        nqubits = 3
        p = [np.random.randint(1, 4) for i in range(ntrials)]
        weights = [
            [np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand()]
            for i in range(ntrials)
        ]
        init_hadamards = [np.random.choice([True, False]) for i in range(ntrials)]
        constants = [np.random.rand() for i in range(ntrials)]

        for i in range(ntrials):
            gammas = [np.random.rand() * np.pi for i in range(p[i])]
            betas = [np.random.rand() * np.pi for i in range(p[i])]

            cost_hamil = Hamiltonian(
                [
                    PauliOp("Z", (0,)),
                    PauliOp("ZZ", (0, 1)),
                    PauliOp("ZZ", (1, 2)),
                    PauliOp("ZZ", (0, 2)),
                ],
                weights[i],
                constants[i],
            )
            mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p[i])
            variate_params = QAOAVariationalStandardParams(
                qaoa_descriptor, betas, gammas
            )

            cirq_backend = QAOACirqBackendStatevecSimulator(
                qaoa_descriptor, None, None, init_hadamards[i]
            )

            cirq_expectation = cirq_backend.expectation(variate_params)

            vector_backend = QAOAvectorizedBackendSimulator(
                qaoa_descriptor, None, None, init_hadamards[i]
            )
            vector_expectation = vector_backend.expectation(variate_params)

            self.assertAlmostEqual(cirq_expectation, vector_expectation)

    def test_shot_based_simulator(self):
        """
        Test get_counts in shot-based cirq simulator.
        """

        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [1 / 8 * np.pi]
        betas = [1 / 8 * np.pi]
        shots = 10000

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        cirq_shot_backend = QAOACirqBackendShotBasedSimulator(
            qaoa_descriptor, shots, None, None, True, 1.0
        )

        shot_result = cirq_shot_backend.get_counts(variate_params)

        self.assertEqual(type(shot_result), dict)

    def test_cvar_alpha_expectation(self):
        """
        Test computing the expectation value by changing the alpha of the cvar.
        """

        nqubits = 3
        p = 1
        weights = [1, -1, 1]
        gammas = [1 / 8 * np.pi]
        betas = [1 / 8 * np.pi]
        shots = 10000

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        cirq_shot_backend = QAOACirqBackendShotBasedSimulator(
            qaoa_descriptor, shots, None, None, True, 1.0, seed_simulator=1234
        )

        expectation_value_100 = cirq_shot_backend.expectation(variate_params)
        self.assertEqual(type(float(expectation_value_100)), float)

        # cvar_alpha = 0.5, 0.75
        cirq_shot_backend.cvar_alpha = 0.5
        expectation_value_05 = cirq_shot_backend.expectation(variate_params)
        cirq_shot_backend.cvar_alpha = 0.75
        expectation_value_075 = cirq_shot_backend.expectation(variate_params)
        self.assertNotEqual(expectation_value_05, expectation_value_075)
        self.assertEqual(type(float(expectation_value_05)), float)
        self.assertEqual(type(float(expectation_value_075)), float)

    def test_standard_decomposition_branch_in_circuit_construction(self):
        """
        XY Pauli is not an implemented low level gate. Produces NotImplementedError
        as the standard decomposition for the XY PauliGate doesn't exist.
        """

        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [1 / 8 * np.pi]
        betas = [1 / 8 * np.pi]
        shots = 10000

        cost_hamil = Hamiltonian(
            [PauliOp("XY", (0, 1)), PauliOp("XY", (1, 2)), PauliOp("XY", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        self.assertRaises(
            NotImplementedError,
            QAOACirqBackendShotBasedSimulator,
            qaoa_descriptor,
            shots,
            None,
            None,
            True,
            1.0,
        )

        self.assertRaises(
            NotImplementedError,
            QAOACirqBackendStatevecSimulator,
            qaoa_descriptor,
            None,
            None,
            True,
        )


if __name__ == "__main__":
    unittest.main()
