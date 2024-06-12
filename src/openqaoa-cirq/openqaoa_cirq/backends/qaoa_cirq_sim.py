import numpy as np
import sympy
import cirq
from cirq.ops import PauliSum
from typing import Union, List, Tuple, Optional

from .gates_cirq import CirqGateApplicator

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
    """
    Local Shot-based simulators offered by Cirq

    Parameters
    ----------
    qaoa_descriptor: `QAOADescriptor`
        An object of the class ``QAOADescriptor`` which contains information on
        circuit construction and depth of the circuit.

    n_shots: `int`
        The number of shots to be taken for each circuit.

    prepend_state: `cirq.Circuit`
        The state prepended to the circuit.

    append_state: `cirq.Circuit`
        The state appended to the circuit.

    init_hadamard: `bool`
        Whether to apply a Hadamard gate to the beginning of the
        QAOA part of the circuit.

    cvar_alpha: `float`
        The value of alpha for the CVaR cost function.
    """

    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        n_shots: int,
        prepend_state: Optional[cirq.Circuit],
        append_state: Optional[cirq.Circuit],
        init_hadamard: bool,
        cvar_alpha: float,
    ):
        QAOABaseBackendShotBased.__init__(
            self,
            qaoa_descriptor,
            n_shots,
            prepend_state,
            append_state,
            init_hadamard,
            cvar_alpha,
        )
        self.qubits = cirq.LineQubit.range(self.n_qubits)
        self.gate_applicator = CirqGateApplicator()

        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.all_qubits()), (
                "Cannot attach a bigger circuit " "to the QAOA routine"
            )
        self.simulator = cirq.Simulator()

        # For parametric circuits
        self.parametric_circuit = self.parametric_qaoa_circuit

    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> cirq.Circuit:
        """
        The final QAOA circuit to be executed on the simulator.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`

        Returns
        -------
        qaoa_circuit: `cirq.Circuit`
            The final QAOA circuit after binding angles from variational parameters.
        """
        angles_list = self.obtain_angles_for_pauli_list(self.abstract_circuit, params)
        circuit_with_angles = cirq.resolve_parameters(
            self.parametric_circuit, dict(zip(self.cirq_parameter_list, angles_list))
        )
        # circuit_with_angles.append(cirq.measure(*self.qubits, key="z"))

        return circuit_with_angles

    @property
    def parametric_qaoa_circuit(self) -> cirq.Circuit:
        """
        Creates a parametric QAOA circuit, given the qubit pairs, single qubits with biases,
        and a set of circuit angles.
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
            if (
                type(each_gate)
                in QAOACirqBackendShotBasedSimulator.CIRQ_GATEMAP_LIBRARY
            ):
                decomposition = each_gate.decomposition("trivial")
            else:
                decomposition = each_gate.decomposition("standard")
            # Create Circuit
            for each_tuple in decomposition:
                gate = each_tuple[0](self.gate_applicator, *each_tuple[1])
                gate.apply_gate(parametric_circuit)

        if self.append_state:
            parametric_circuit.append(self.append_state)

        return parametric_circuit

    def get_counts(self, params: QAOAVariationalBaseParams, n_shots=None) -> dict:
        """
        Returns the counts of the final QAOA circuit after binding angles from variational parameters.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`
            The QAOA parameters - an object of one of the parameter classes, containing variable parameters.
        n_shots: `int`
            The number of times to run the circuit. If None, n_shots is set to the default: self.n_shots

        Returns
        -------
        counts: `dict`
            The counts of the final QAOA circuit after binding angles from variational parameters.
        """
        # set the number of shots, if not specified take the default
        n_shots = self.n_shots if n_shots is None else n_shots

        qaoa_circuit = self.qaoa_circuit(params)
        result = self.simulator.run(qaoa_circuit, repetitions=n_shots)
        counts = result.histogram(key="z")

        final_counts = flip_counts(counts)
        self.measurement_outcomes = final_counts
        return final_counts

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
    """
    Local Statevector-based simulators using Cirq

    Parameters
    ----------
    qaoa_descriptor: `QAOADescriptor`
        An object of the class ``QAOADescriptor`` which contains information on
        circuit construction and depth of the circuit.

    prepend_state: `np.ndarray` or `cirq.Circuit`
        The state prepended to the circuit.

    append_state: `cirq.Circuit or np.ndarray`
        The state appended to the circuit.

    init_hadamard: `bool`
        Whether to apply a Hadamard gate to the beginning of the
        QAOA part of the circuit.

    cvar_alpha: `float`
        The value of alpha for the CVaR cost function.
    """

    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        prepend_state: Optional[Union[np.ndarray, cirq.Circuit]],
        append_state: Optional[Union[np.ndarray, cirq.Circuit]],
        init_hadamard: bool,
        cvar_alpha: float = 1,
    ):
        QAOABaseBackendStatevector.__init__(
            self,
            qaoa_descriptor,
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

        cirq_pauli_strings = []
        string_2_cirq_op = {"I": cirq.I, "X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}
        for pauli_string, coeff in zip(pauli_strings_list, pauli_coeffs):
            qubit_paulis = []
            for qubit, pauli in enumerate(pauli_string):
                q = cirq.LineQubit(qubit)
                qubit_paulis.append(string_2_cirq_op[pauli](q))
            cirq_pauli_strings.append(cirq.PauliString(*qubit_paulis) * coeff)
        cirq_pauli_strings.append(cirq.PauliString() * cost_hamil.constant)

        cirq_cost_hamil = cirq.PauliSum.from_pauli_strings(cirq_pauli_strings)
        return cirq_cost_hamil

    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> cirq.Circuit:
        """
        The final QAOA circuit to be executed on the QPU.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`

        Returns
        -------
        qaoa_circuit: `cirq.Circuit`
            The final QAOA circuit after binding angles from variational parameters.
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
        Creates a parametric QAOA circuit, given the qubit pairs, single qubits with biases,
        and a set of circuit angles.

        Parameters
        ----------
            params:
                Object of type QAOAVariationalBaseParams
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
            if (
                type(each_gate)
                in QAOACirqBackendShotBasedSimulator.CIRQ_GATEMAP_LIBRARY
            ):
                decomposition = each_gate.decomposition("trivial")
            else:
                decomposition = each_gate.decomposition("standard")
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

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`

        Returns
        -------
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

    def reset_circuit(self):
        """
        Reset self.circuit after performing a computation
        """
        raise NotImplementedError()

    def circuit_to_qasm(self):
        """ """
        raise NotImplementedError()
