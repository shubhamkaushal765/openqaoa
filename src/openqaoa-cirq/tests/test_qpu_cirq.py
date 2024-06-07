import unittest
import numpy as np
import pytest
import cirq
import time
from unittest.mock import Mock

from openqaoa.qaoa_components import (
    PauliOp,
    Hamiltonian,
    QAOADescriptor,
    QAOAVariationalStandardParams,
)
from openqaoa.utilities import X_mixer_hamiltonian
from openqaoa_cirq.backends import DeviceCirq, QAOACirqQPUBackend, QAOACirqBackendStatevecSimulator


class TestingQAOACirqQPUBackend(unittest.TestCase):
    """Tests the QAOA Cirq QPU Backend objects for circuit creation and execution."""

    @pytest.mark.api
    def test_circuit_angle_assignment_qpu_backend(self):
        """
        Test if the circuit created by the Cirq Backend has the appropriate angles assigned
        before the circuit is executed.
        """
        nqubits = 3
        p = 2
        weights = [1, 1, 1]
        gammas = [0, 1 / 8 * np.pi]
        betas = [1 / 2 * np.pi, 3 / 8 * np.pi]

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))], weights, 1
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        cirq_device = DeviceCirq(device_name="Simulator")
        cirq_backend = QAOACirqQPUBackend(
            qaoa_descriptor, cirq_device, shots=None, prepend_state=None, append_state=None, init_hadamard=False
        )
        qpu_circuit = cirq_backend.qaoa_circuit(variate_params)

        # Construct the expected circuit
        expected_circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(nqubits)
        expected_circuit.append(cirq.CZ(qubits[0], qubits[1]).with_parameter(2 * gammas[0]))
        expected_circuit.append(cirq.CZ(qubits[1], qubits[2]).with_parameter(2 * gammas[0]))
        expected_circuit.append(cirq.CZ(qubits[0], qubits[2]).with_parameter(2 * gammas[0]))
        expected_circuit.append(cirq.rx(-2 * betas[0]).on_each(*qubits))
        expected_circuit.append(cirq.CZ(qubits[0], qubits[1]).with_parameter(2 * gammas[1]))
        expected_circuit.append(cirq.CZ(qubits[1], qubits[2]).with_parameter(2 * gammas[1]))
        expected_circuit.append(cirq.CZ(qubits[0], qubits[2]).with_parameter(2 * gammas[1]))
        expected_circuit.append(cirq.rx(-2 * betas[1]).on_each(*qubits))

        # Compare the constructed circuits
        assert qpu_circuit == expected_circuit

    @pytest.mark.api
    def test_circuit_angle_assignment_qpu_backend_w_hadamard(self):
        """Test if the circuit is consistent when init_hadamard is set to True."""
        nqubits = 3
        p = 2
        weights = [1, 1, 1]
        gammas = [0, 1 / 8 * np.pi]
        betas = [1 / 2 * np.pi, 3 / 8 * np.pi]

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))], weights, 1
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        cirq_device = DeviceCirq(device_name="Simulator")
        cirq_backend = QAOACirqQPUBackend(
            qaoa_descriptor, cirq_device, shots=None, prepend_state=None, append_state=None, init_hadamard=True
        )
        qpu_circuit = cirq_backend.qaoa_circuit(variate_params)

        # Construct the expected circuit
        expected_circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(nqubits)
        expected_circuit.append(cirq.H.on_each(*qubits))
        expected_circuit.append(cirq.CZ(qubits[0], qubits[1]).with_parameter(2 * gammas[0]))
        expected_circuit.append(cirq.CZ(qubits[1], qubits[2]).with_parameter(2 * gammas[0]))
        expected_circuit.append(cirq.CZ(qubits[0], qubits[2]).with_parameter(2 * gammas[0]))
        expected_circuit.append(cirq.rx(-2 * betas[0]).on_each(*qubits))
        expected_circuit.append(cirq.CZ(qubits[0], qubits[1]).with_parameter(2 * gammas[1]))
        expected_circuit.append(cirq.CZ(qubits[1], qubits[2]).with_parameter(2 * gammas[1]))
        expected_circuit.append(cirq.CZ(qubits[0], qubits[2]).with_parameter(2 * gammas[1]))
        expected_circuit.append(cirq.rx(-2 * betas[1]).on_each(*qubits))

        # Compare the constructed circuits
        assert qpu_circuit == expected_circuit

    @pytest.mark.api
    def test_prepend_circuit(self):
        """
        Checks if prepended circuit has been prepended correctly.
        """
        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [1 / 8 * np.pi]
        betas = [1 / 8 * np.pi]

        # Prepended Circuit
        prepend_circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(nqubits)
        prepend_circuit.append([cirq.X(q) for q in qubits])

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        cirq_device = DeviceCirq(device_name="Simulator")
        cirq_backend = QAOACirqQPUBackend(
            qaoa_descriptor,
            cirq_device,
            shots=None,
            prepend_state=prepend_circuit,
            append_state=None,
            init_hadamard=True,
        )
        qpu_circuit = cirq_backend.qaoa_circuit(variate_params)

        # Construct the expected circuit
        expected_circuit = cirq.Circuit()
        expected_circuit.append([cirq.X(q) for q in qubits])
        expected_circuit.append([cirq.H(q) for q in qubits])
        expected_circuit.append(cirq.CZ(qubits[0], qubits[1]).with_parameter(2 * gammas[0]))
        expected_circuit.append(cirq.CZ(qubits[1], qubits[2]).with_parameter(2 * gammas[0]))
        expected_circuit.append(cirq.CZ(qubits[0], qubits[2]).with_parameter(2 * gammas[0]))
        expected_circuit.append(cirq.rx(-2 * betas[0]).on_each(*qubits))

        # Compare the constructed circuits
        assert qpu_circuit == expected_circuit

    @pytest.mark.api
    def test_append_circuit(self):
        """
        Checks if appended circuit is appropriately appended to the back of the
        QAOA Circuit.
        """
        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [1 / 8 * np.pi]
        betas = [1 / 8 * np.pi]

        # Appended Circuit
        append_circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(nqubits)
        append_circuit.append([cirq.X(q) for q in qubits])

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        cirq_device = DeviceCirq(device_name="Simulator")
        cirq_backend = QAOACirqQPUBackend(
            qaoa_descriptor,
            cirq_device,
            shots=None,
            prepend_state=None,
            append_state=append_circuit,
            init_hadamard=True,
        )
        qpu_circuit = cirq_backend.qaoa_circuit(variate_params)

        # Construct the expected circuit
        expected_circuit = cirq.Circuit()
        expected_circuit.append([cirq.H(q) for q in qubits])
        expected_circuit.append(cirq.CZ(qubits[0], qubits[1]).with_parameter(2 * gammas[0]))
        expected_circuit.append(cirq.CZ(qubits[1], qubits[2]).with_parameter(2 * gammas[0]))
        expected_circuit.append(cirq.CZ(qubits[0], qubits[2]).with_parameter(2 * gammas[0]))
        expected_circuit.append(cirq.rx(-2 * betas[0]).on_each(*qubits))
        expected_circuit.append([cirq.X(q) for q in qubits])

        # Compare the constructed circuits
        assert qpu_circuit == expected_circuit
        
    @pytest.mark.api
    def test_expectations_in_init(self):
        """
        Testing the Exceptions in the init function of the QAOACirqQPUBackend
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

        # Check the creation of the variational parameters
        _ = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)

        # We mock the potential Exception that could occur in the Device class
        cirq_device = DeviceCirq()
        cirq_device._check_provider_connection = Mock(return_value=False)

        with self.assertRaises(Exception) as context:
            QAOACirqQPUBackend(
                qaoa_descriptor,
                cirq_device,
                shots,
                None,
                None,
                True,
            )
        self.assertTrue("An Exception has occurred when trying to connect with the provider." in str(context.exception))

        cirq_device = DeviceCirq(device_name="")
        cirq_device._check_backend_connection = Mock(return_value=False)

        with self.assertRaises(Exception) as context:
            QAOACirqQPUBackend(
                qaoa_descriptor,
                cirq_device,
                shots,
                None,
                None,
                True,
            )
        self.assertTrue("Please choose from " in str(context.exception))

    @pytest.mark.sim
    def test_remote_integration_sim_run(self):
        """
        Checks if Remote Cirq Simulator is similar/close to Local Cirq
        Statevector Simulator.
        This test also serves as an integration test for the QAOACirqQPUBackend.

        This test takes a long time to complete.
        """
        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [[0], [1 / 8 * np.pi], [0], [1 / 8 * np.pi]]
        betas = [[0], [0], [1 / 8 * np.pi], [1 / 8 * np.pi]]

        for i in range(4):
            cost_hamil = Hamiltonian(
                [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
                weights,
                1,
            )
            mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
            qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
            variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas[i], gammas[i])

            cirq_device = DeviceCirq(device_name="Simulator")
            cirq_backend = QAOACirqQPUBackend(
                qaoa_descriptor,
                cirq_device,
                shots=10000,
                prepend_state=None,
                append_state=None,
                init_hadamard=False,
            )
            qpu_expectation = cirq_backend.expectation(variate_params)

            cirq_statevec_backend = QAOACirqBackendStatevecSimulator(
                qaoa_descriptor, None, None, False
            )
            cirq_statevec_expectation = cirq_statevec_backend.expectation(variate_params)

            acceptable_delta = 0.05 * cirq_statevec_expectation
            self.assertAlmostEqual(qpu_expectation, cirq_statevec_expectation, delta=acceptable_delta)
            
    @pytest.mark.api
    def test_remote_qubit_overflow(self):
        """
        If the user creates a circuit that is larger than the maximum circuit size
        that is supported by the QPU, an Exception should be raised with the
        appropriate error message alerting the user to the error.
        """
        shots = 100

        # Create a random QUBO problem with 8 qubits
        set_of_numbers = np.random.randint(1, 10, 8).tolist()
        weights = [1] * len(set_of_numbers)
        qubit_pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
        qubo = Hamiltonian([PauliOp("ZZ", pair) for pair in qubit_pairs], weights, 1)

        mixer_hamil = X_mixer_hamiltonian(n_qubits=8)
        qaoa_descriptor = QAOADescriptor(qubo, mixer_hamil, p=1)

        # Check the creation of the variational parameters
        _ = QAOAVariationalStandardParams(qaoa_descriptor, "rand", "rand")

        # Use a device with fewer qubits than required
        cirq_device = DeviceCirq(device_name="Bristlecone")

        with self.assertRaises(Exception) as context:
            QAOACirqQPUBackend(
                qaoa_descriptor,
                cirq_device,
                shots,
                None,
                None,
                True,
            )
        self.assertTrue("There are fewer qubits on the device than the number of qubits required for the circuit." in str(context.exception))

    @pytest.mark.qpu
    def test_integration_on_emulator(self):
        """
        Test Emulated QPU Workflow. Checks if the expectation value is returned
        after the circuit run.
        """

        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [[1 / 8 * np.pi]]
        betas = [[1 / 8 * np.pi]]
        shots = 10000

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)
        cirq_device = DeviceCirq(device_name="Simulator", as_emulator=True)

        cirq_backend = QAOACirqQPUBackend(
            qaoa_descriptor, cirq_device, shots, None, None, False
        )
        cirq_expectation = cirq_backend.expectation(variate_params)

        self.assertEqual(type(cirq_expectation), float)
    
    @pytest.mark.qpu
    def test_remote_integration_qpu_run(self):
        """
        Test Actual QPU Workflow. Checks if the expectation value is returned
        after the circuit run.
        """

        nqubits = 3
        p = 1
        weights = [1, 1, 1]
        gammas = [[1 / 8 * np.pi]]
        betas = [[1 / 8 * np.pi]]
        shots = 10000

        cost_hamil = Hamiltonian(
            [PauliOp("ZZ", (0, 1)), PauliOp("ZZ", (1, 2)), PauliOp("ZZ", (0, 2))],
            weights,
            1,
        )
        mixer_hamil = X_mixer_hamiltonian(n_qubits=nqubits)
        qaoa_descriptor = QAOADescriptor(cost_hamil, mixer_hamil, p=p)
        variate_params = QAOAVariationalStandardParams(qaoa_descriptor, betas, gammas)
        cirq_device = DeviceCirq(device_name="Bristlecone")

        cirq_backend = QAOACirqQPUBackend(
            qaoa_descriptor, cirq_device, shots, None, None, False
        )
        circuit = cirq_backend.qaoa_circuit(variate_params)
        job = cirq_backend.device.backend_device.run(circuit, repetitions=cirq_backend.n_shots)

        # Check if the circuit is validated by Cirq when submitted for execution
        # Check the status of the job and keep retrying until it's completed or cancelled
        job_state = False
        while not job_state:
            try:
                result = job.results()
                job_state = True
            except cirq.google.api.v2.quantum_exception as e:
                if "CANCELLED" in str(e):
                    job_state = True
                else:
                    time.sleep(1)

        assert job_state

if __name__ == "__main__":
    unittest.main()
