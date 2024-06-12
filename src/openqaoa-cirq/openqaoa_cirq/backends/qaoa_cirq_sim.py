import numpy as np
import sympy
import cirq
from cirq.ops import PauliSum
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


class QAOACirqBackendShotBasedSimulator(
    QAOABaseBackendShotBased, QAOABaseBackendParametric
):

    CIRQ_GATEMAP_LIBRARY = [
        RXGateMap,
        RYGateMap,
        RZGateMap,
        RXXGateMap,
        RYYGateMap,
        RZZGateMap,
        RZXGateMap,
    ]

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
            prepend_state (Optional[np.ndarray, cirq.Circuit]): Circuit to prepend to the QAOA circuit.
            append_state (Optional[np.ndarray, cirq.Circuit]): Circuit to append to the QAOA circuit.
            init_hadamard (bool): Whether to apply an initial layer of Hadamard gates.
            cvar_alpha (float): The value of alpha for the CVaR cost function.
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
        assert (
            cvar_alpha == 1
        ), "Please use the shot-based simulator for simulations with cvar_alpha < 1"

        self.qubits = cirq.LineQubit.range(self.n_qubits)
        self.gate_applicator = CirqGateApplicator()

        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.all_qubits()), (
                "Cannot attach a bigger circuit " "to the QAOA routine"
            )

        # For parametric circuits
        self.parametric_circuit = self.parametric_qaoa_circuit
        self.cirq_cost_hamil = self.cirq_cost_hamiltonian
        self.cirq_cost_hamil_sq = self.cirq_cost_hamil * self.cirq_cost_hamil

    @property
    def cirq_cost_hamiltonian(self):
        """
        The Cirq cost hamiltonian for the QAOA circuit represented
        as a `PauliSum` object.
        """
        cost_hamil = self.cost_hamiltonian
        n_qubits = cost_hamil.n_qubits
        pauli_strings_list = ["I" * n_qubits] * len(cost_hamil.terms)
        for i, pauli_op in enumerate(cost_hamil.terms):
            pauli_term = list(pauli_strings_list[i])
            for pauli, qubit in zip(pauli_op.pauli_str, pauli_op.qubit_indices):
                pauli_term[qubit] = pauli
            pauli_strings_list[i] = "".join(str(term) for term in pauli_term)

        pauli_strings_list.append("I" * n_qubits)
        pauli_coeffs = cost_hamil.coeffs

        cirq_pauli_op = [
            [pauli_strings, coeff]
            for pauli_strings, coeff in zip(pauli_strings_list, pauli_coeffs)
        ]
        cirq_pauli_op.append(["I" * n_qubits, cost_hamil.constant])
        cirq_cost_hamil = PauliSum.from_terms(cirq_pauli_op)
        return cirq_cost_hamil

    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> cirq.Circuit:
        """
        Construct the QAOA circuit with the given parameters.

        Args:
            params (QAOAVariationalBaseParams): The variational parameters for the QAOA circuit.

        Returns:
            cirq.Circuit: The constructed QAOA circuit.
        """
        # generate a job id for the wavefunction evaluation
        self.job_id = generate_uuid()

        angles_list = self.obtain_angles_for_pauli_list(self.abstract_circuit, params)
        circuit_with_angles = cirq.resolve_parameters(
            self.parametric_circuit, dict(zip(self.cirq_parameter_list, angles_list))
        )
        return circuit_with_angles

    @property
    def parametric_qaoa_circuit(self) -> cirq.Circuit:
        """
        Create a parametric QAOA circuit, given the qubit pairs, single qubits with biases,
        and a set of circuit angles.

        Args:
            params (QAOAVariationalBaseParams): The variational parameters for the QAOA circuit.

        Returns:
            cirq.Circuit: The parametric QAOA circuit.
        """
        parametric_circuit = cirq.Circuit()

        if self.prepend_state:
            parametric_circuit.append(self.prepend_state)

        # Initial state is all |+>
        if self.init_hadamard:
            parametric_circuit.append(cirq.H.on_each(self.qubits))

        self.cirq_parameter_list = []
        for each_gate in self.abstract_circuit:
            # if gate is of type mixer or cost gate, assign parameter to it
            if each_gate.gate_label.type.value in ["MIXER", "COST"]:
                angle_param = sympy.Symbol(each_gate.gate_label.__repr__())
                self.cirq_parameter_list.append(angle_param)
                each_gate.angle_value = angle_param
            decomposition = each_gate.decomposition("standard")
            # if (
            #     type(each_gate)
            #     in QAOACirqBackendShotBasedSimulator.CIRQ_GATEMAP_LIBRARY
            # ):
            #     decomposition = each_gate.decomposition("trivial")
            # else:
            #     decomposition = each_gate.decomposition("standard")

            # Create Circuit
            for each_tuple in decomposition:
                gate = each_tuple[0](self.gate_applicator, *each_tuple[1])
                gate.apply_gate(parametric_circuit)

        if self.append_state:
            parametric_circuit.append(self.append_state)

        return parametric_circuit

    def wavefunction(
        self, params: QAOAVariationalBaseParams
    ) -> Union[List[complex], np.ndarray]:
        """
        Get the wavefunction of the state produced by the parametric circuit.

        Args:
            params: `QAOAVariationalBaseParams`

        Returns:
            wf: `List[complex]` or `np.ndarray[complex]`
                A list of the wavefunction amplitudes.
        """
        ckt = self.qaoa_circuit(params)
        simulator = cirq.Simulator()
        result = simulator.simulate(ckt)
        wf = result.final_state
        self.measurement_outcomes = wf
        return wf

    @round_value
    def expectation(self, params: QAOAVariationalBaseParams) -> float:
        """
        Compute the expectation value w.r.t the Cost Hamiltonian

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters - an object of one of the parameter classes, containing
            variable parameters.

        Returns
        -------
        `float`
            expectation value of cost operator wrt to quantum state produced by QAOA circuit
        """
        ckt = self.qaoa_circuit(params)
        simulator = cirq.Simulator()
        result = simulator.simulate(ckt)
        output_wf = result.final_state
        self.measurement_outcomes = output_wf
        cost = np.real(
            cirq.expectation_from_state_vector(
                output_wf, self.cirq_cost_hamil, self.qubits
            )
        )
        return cost

    @round_value
    def expectation_w_uncertainty(
        self, params: QAOAVariationalBaseParams
    ) -> Tuple[float, float]:
        """
        Compute the expectation value w.r.t the Cost Hamiltonian and its uncertainty

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters - an object of one of the parameter classes, containing
            variable parameters.

        Returns
        -------
        `Tuple[float]`
            expectation value and its uncertainty of cost operator wrt
            to quantum state produced by QAOA circuit.
        """
        ckt = self.qaoa_circuit(params)
        simulator = cirq.Simulator()
        result = simulator.simulate(ckt)
        output_wf = result.final_state
        self.measurement_outcomes = output_wf
        cost = np.real(
            cirq.expectation_from_state_vector(
                output_wf, self.cirq_cost_hamil, self.qubits
            )
        )
        cost_sq = np.real(
            cirq.expectation_from_state_vector(
                output_wf, self.cirq_cost_hamil_sq, self.qubits
            )
        )

        uncertainty = np.sqrt(cost_sq - cost**2)
        return (cost, uncertainty)

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


class QAOACirqBackendStatevecSimulator(
    QAOABaseBackendStatevector, QAOABaseBackendParametric
):
    CIRQ_GATEMAP_LIBRARY = [
        RXGateMap,
        RYGateMap,
        RZGateMap,
        RXXGateMap,
        RYYGateMap,
        RZZGateMap,
        RZXGateMap,
    ]

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
            parametric_circuit = cirq.resolve_parameters(
                parametric_circuit, {gate: angle}
            )

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
                angle_param = sympy.Symbol(each_gate.gate_label.__repr__())
                self.qiskit_parameter_list.append(angle_param)
                each_gate.angle_value = angle_param
            if type(each_gate) in QAOACirqBackendStatevecSimulator.CIRQ_GATEMAP_LIBRARY:
                decomposition = each_gate.decomposition("trivial")
            else:
                decomposition = each_gate.decomposition("standard")
            for gate_func, qubits in decomposition:
                gate_func(self.gate_applicator, *qubits)

        if self.append_state:
            parametric_circuit += self.append_state

        return parametric_circuit

    def wavefunction(
        self, params: QAOAVariationalBaseParams
    ) -> Union[List[complex], np.ndarray]:
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
    def expectation_w_uncertainty(
        self, params: QAOAVariationalBaseParams
    ) -> Tuple[float, float]:
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
