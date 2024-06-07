import numpy as np
import cirq
from typing import Union, List, Tuple, Optional

from openqaoa.backends.basebackend import (
    QAOABaseBackendParametric,
    QAOABaseBackendShotBased,
    QAOABaseBackendStatevector,
)
from openqaoa.qaoa_components import QAOADescriptor
from openqaoa.qaoa_components.variational_parameters.variational_baseparams import (
    QAOAVariationalBaseParams,
)
from openqaoa.utilities import (
    flip_counts,
    generate_uuid,
    round_value,
)
from openqaoa.backends.cost_function import cost_function
from openqaoa.qaoa_components.ansatz_constructor import (
    RXGateMap,
    RYGateMap,
    RZGateMap,
    RXXGateMap,
    RYYGateMap,
    RZZGateMap,
    RZXGateMap,
)

class CirqGateApplicator:
    """
        Apply a gate to the given circuit with optional parameterization.

        Args:
            circuit (cirq.Circuit): The Cirq circuit to which the gate is applied.
            gate (cirq.Gate): The gate to be applied.
            qubits (list[cirq.Qid]): The qubits on which the gate acts.
            param (Optional[float]): The parameter for the gate, if applicable.
    """
    def apply_gate(self, circuit, gate, qubits, param=None):
        if param is not None:
            circuit.append(gate(*qubits).on(*qubits).with_parameter(param))
        else:
            circuit.append(gate(*qubits).on(*qubits))


class QAOACirqBackendShotBasedSimulator(QAOABaseBackendShotBased, QAOABaseBackendParametric):
    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        n_shots: int,
        prepend_state: Optional[cirq.Circuit],
        append_state: Optional[cirq.Circuit],
        init_hadamard: bool,
        cvar_alpha: float,
        seed_simulator: Optional[int] = None,
    ):
        """
        Initialize the shot-based simulator backend with Cirq.

        Args:
            qaoa_descriptor (QAOADescriptor): The descriptor of the QAOA problem.
            n_shots (int): The number of shots for sampling.
            prepend_state (Optional[cirq.Circuit]): Circuit to prepend to the QAOA circuit.
            append_state (Optional[cirq.Circuit]): Circuit to append to the QAOA circuit.
            init_hadamard (bool): Whether to apply an initial layer of Hadamard gates.
            cvar_alpha (float): The CVaR parameter.
            seed_simulator (Optional[int]): Seed for the simulator's random number generator.
        """
        QAOABaseBackendShotBased.__init__(
            self,
            qaoa_descriptor,
            n_shots,
            prepend_state,
            append_state,
            init_hadamard,
            cvar_alpha,
        )
        self.gate_applicator = CirqGateApplicator()
        self.qubits = cirq.LineQubit.range(self.n_qubits)
        self.simulator = cirq.Simulator(seed=seed_simulator)

    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> cirq.Circuit:
        """
        Construct the QAOA circuit with the given parameters.

        Args:
            params (QAOAVariationalBaseParams): The variational parameters for the QAOA circuit.

        Returns:
            cirq.Circuit: The constructed QAOA circuit.
        """
        angles_list = self.obtain_angles_for_pauli_list(self.abstract_circuit, params)
        parametric_circuit = self.parametric_qaoa_circuit()
        
        for angle, gate in zip(angles_list, self.qiskit_parameter_list):
            parametric_circuit = cirq.resolve_parameters(parametric_circuit, {gate: angle})

        if self.append_state:
            parametric_circuit += self.append_state
        parametric_circuit += cirq.measure(*self.qubits, key='result')

        return parametric_circuit

    def parametric_qaoa_circuit(self) -> cirq.Circuit:
        """
        Create a parametric QAOA circuit with symbolic parameters.

        Returns:
            cirq.Circuit: The parametric QAOA circuit.
        """
        parametric_circuit = cirq.Circuit()
        if self.prepend_state:
            parametric_circuit += self.prepend_state
        if self.init_hadamard:
            parametric_circuit.append(cirq.H(q) for q in self.qubits)

        self.qiskit_parameter_list = []
        for each_gate in self.abstract_circuit:
            if each_gate.gate_label.type.value in ["MIXER", "COST"]:
                angle_param = cirq.Symbol(each_gate.gate_label.__repr__())
                self.qiskit_parameter_list.append(angle_param)
                each_gate.angle_value = angle_param
            if type(each_gate) in QAOACirqBackendShotBasedSimulator.QISKIT_GATEMAP_LIBRARY:
                decomposition = each_gate.decomposition("trivial")
            else:
                decomposition = each_gate.decomposition("standard")
            for gate_func, qubits in decomposition:
                gate_func(self.gate_applicator, parametric_circuit, qubits)

        return parametric_circuit

    def get_counts(self, params: QAOAVariationalBaseParams, n_shots=None) -> dict:
        """
        Run the QAOA circuit and get the measurement counts.

        Args:
            params (QAOAVariationalBaseParams): The variational parameters for the QAOA circuit.
            n_shots (Optional[int]): The number of shots for sampling.

        Returns:
            dict: The measurement counts.
        """
        self.job_id = generate_uuid()
        n_shots = self.n_shots if n_shots is None else n_shots

        qaoa_circuit = self.qaoa_circuit(params)
        result = self.simulator.run(qaoa_circuit, repetitions=n_shots)
        counts = result.histogram(key='result', fold_func=flip_counts)
        self.measurement_outcomes = counts
        return counts

    def circuit_to_qasm(self):
        """
        Convert the QAOA circuit to QASM format.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError()

    def reset_circuit(self):
        """
        Reset the QAOA circuit.

        Raises:
            NotImplementedError: This method is not implemented.
        """
        raise NotImplementedError()

class QAOACirqBackendStatevecSimulator(QAOABaseBackendStatevector, QAOABaseBackendParametric):
    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        prepend_state: Optional[Union[np.ndarray, cirq.Circuit]],
        append_state: Optional[Union[np.ndarray, cirq.Circuit]],
        init_hadamard: bool,
        cvar_alpha: float = 1,
    ):
        """
        Initialize the statevector simulator backend with Cirq.

        Args:
            qaoa_descriptor (QAOADescriptor): The descriptor of the QAOA problem.
            prepend_state (Optional[Union[np.ndarray, cirq.Circuit]]): State or circuit to prepend to the QAOA circuit.
            append_state (Optional[Union[np.ndarray, cirq.Circuit]]): State or circuit to append to the QAOA circuit.
            init_hadamard (bool): Whether to apply an initial layer of Hadamard gates.
            cvar_alpha (float): The CVaR parameter.
        """
        QAOABaseBackendStatevector.__init__(
            self,
            qaoa_descriptor,
            prepend_state,
            append_state,
            init_hadamard,
            cvar_alpha,
        )
        self.qubits = cirq.LineQubit.range(self.n_qubits)
        self.gate_applicator = CirqGateApplicator()

    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> cirq.Circuit:
        """
        Construct the QAOA circuit with the given parameters.

        Args:
            params (QAOAVariationalBaseParams): The variational parameters for the QAOA circuit.

        Returns:
            cirq.Circuit: The constructed QAOA circuit.
        """
        angles_list = self.obtain_angles_for_pauli_list(self.abstract_circuit, params)
        parametric_circuit = self.parametric_qaoa_circuit()
        
        for angle, gate in zip(angles_list, self.qiskit_parameter_list):
            parametric_circuit = cirq.resolve_parameters(parametric_circuit, {gate: angle})

        return parametric_circuit

    def parametric_qaoa_circuit(self) -> cirq.Circuit:
        """
        Create a parametric QAOA circuit with symbolic parameters.

        Returns:
            cirq.Circuit: The parametric QAOA circuit.
        """
        parametric_circuit = cirq.Circuit()
        if self.prepend_state:
            parametric_circuit += self.prepend_state
        if self.init_hadamard:
            parametric_circuit.append(cirq.H(q) for q in self.qubits)

        self.qiskit_parameter_list = []
        for each_gate in self.abstract_circuit:
            if each_gate.gate_label.type.value in ["MIXER", "COST"]:
                angle_param = cirq.Symbol(each_gate.gate_label.__repr__())
                self.qiskit_parameter_list.append(angle_param)
                each_gate.angle_value = angle_param
            if type(each_gate) in QAOACirqBackendStatevecSimulator.QISKIT_GATEMAP_LIBRARY:
                decomposition = each_gate.decomposition("trivial")
            else:
                decomposition = each_gate.decomposition("standard")
            for gate_func, qubits in decomposition:
                gate_func(self.gate_applicator, parametric_circuit, qubits)

        if self.append_state:
            parametric_circuit += self.append_state

        return parametric_circuit

    def wavefunction(self, params: QAOAVariationalBaseParams) -> Union[List[complex], np.ndarray]:
        ckt = self.qaoa_circuit(params)
        result = self.simulator.simulate(ckt)
        wf = result.final_state_vector
        self.measurement_outcomes = wf
        return wf

    @round_value
    def expectation(self, params: QAOAVariationalBaseParams) -> float:
        ckt = self.qaoa_circuit(params)
        result = self.simulator.simulate(ckt)
        self.measurement_outcomes = result.final_state_vector
        cost = np.real(cirq.expectation_value(ckt, self.qiskit_cost_hamiltonian))
        return cost

    @round_value
    def expectation_w_uncertainty(self, params: QAOAVariationalBaseParams) -> Tuple[float, float]:
        ckt = self.qaoa_circuit(params)
        result = self.simulator.simulate(ckt)
        self.measurement_outcomes = result.final_state_vector
        cost = np.real(cirq.expectation_value(ckt, self.qiskit_cost_hamiltonian))
        cost_sq = np.real(cirq.expectation_value(ckt, self.qiskit_cost_hamiltonian_sq))
        uncertainty = np.sqrt(cost_sq - cost**2)
        return cost, uncertainty

    def reset_circuit(self):
        raise NotImplementedError()

    def circuit_to_qasm(self):
        raise NotImplementedError()
