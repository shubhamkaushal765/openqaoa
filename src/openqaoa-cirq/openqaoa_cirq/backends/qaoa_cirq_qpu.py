import time
from typing import Optional, List
import warnings
import sympy

import cirq
from cirq import Circuit, LineQubit, MeasurementGate

from .devices import DeviceCirq
from .gates_cirq import CirqGateApplicator
from openqaoa.backends.basebackend import (
    QAOABaseBackendShotBased,
    QAOABaseBackendCloud,
    QAOABaseBackendParametric,
)
from openqaoa.qaoa_components import QAOADescriptor
from openqaoa.qaoa_components.variational_parameters.variational_baseparams import (
    QAOAVariationalBaseParams,
)
from openqaoa.utilities import flip_counts


class QAOACirqQPUBackend(
    QAOABaseBackendParametric, QAOABaseBackendCloud, QAOABaseBackendShotBased
):
    """
    A QAOA simulator as well as for real QPU using cirq as the backend

    Parameters
    ----------
    device: `DeviceCirq`
        An object of the class ``DeviceCirq`` which contains the credentials
        for accessing the QPU via cloud and the name of the device.
    qaoa_descriptor: `QAOADescriptor`
        An object of the class ``QAOADescriptor`` which contains information on
        circuit construction and depth of the circuit.
    n_shots: `int`
        The number of shots to be taken for each circuit.
    prepend_state: `Circuit`
        The state prepended to the circuit.
    append_state: `Circuit`
        The state appended to the circuit.
    init_hadamard: `bool`
        Whether to apply a Hadamard gate to the beginning of the
        QAOA part of the circuit.
    cvar_alpha: `float`
        The value of alpha for the CVaR method.
    """

    def __init__(
        self,
        qaoa_descriptor: QAOADescriptor,
        device: DeviceCirq,
        n_shots: int,
        prepend_state: Optional[Circuit],
        append_state: Optional[Circuit],
        init_hadamard: bool,
        initial_qubit_mapping: Optional[List[int]] = None,
        cirq_optimization_level: int = 1,
        cvar_alpha: float = 1,
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
        QAOABaseBackendCloud.__init__(self, device)

        self.qubits = [LineQubit(i) for i in range(self.n_qubits)]
        self.problem_qubits = self.qubits[0: self.problem_qubits]

        if cirq_optimization_level in [0, 1, 2, 3]:
            self.cirq_optimization_level = cirq_optimization_level
        else:
            raise ValueError(
                f"cirq_optimization_level cannot be {cirq_optimization_level}. Choose between 0 to 3"
            )
        self.gate_applicator = CirqGateApplicator()

        if self.initial_qubit_mapping is None:
            self.initial_qubit_mapping = (
                initial_qubit_mapping
                if initial_qubit_mapping is not None
                else list(range(self.n_qubits))
            )
        else:
            if isinstance(initial_qubit_mapping, list):
                warnings.warn(
                    "Ignoring the initial_qubit_mapping since the routing algorithm chose one"
                )

        if self.prepend_state:
            assert self.n_qubits >= len(prepend_state.all_qubits()), (
                "Cannot attach a bigger circuit" "to the QAOA routine"
            )

        if not (self.device.provider_connected and self.device.qpu_connected):
            raise Exception(
                "Error connecting to {}.".format(self.device.device_location.upper())
            )

        if self.device.n_qubits < self.n_qubits:
            raise Exception(
                "There are fewer qubits on the device than the number of qubits required for the circuit."
            )
        # For parametric circuits
        self.parametric_circuit = self.parametric_qaoa_circuit

    def qaoa_circuit(self, params: QAOAVariationalBaseParams) -> Circuit:
        """
        The final QAOA circuit to be executed on the QPU.

        Parameters
        ----------
        params: `QAOAVariationalBaseParams`

        Returns
        -------
        qaoa_circuit: `Circuit`
            The final QAOA circuit after binding angles from variational parameters.
        """
        angles_list = self.obtain_angles_for_pauli_list(self.abstract_circuit, params)
        param_dict = dict(zip(self.cirq_parameter_list, angles_list))
        circuit_with_angles = self.parametric_circuit.transform_qubits(
            lambda q: self.qubits[param_dict[q.name]]
        )

        if self.append_state:
            circuit_with_angles += self.append_state

        # only measure the problem qubits
        if self.final_mapping is None:
            circuit_with_angles += [MeasurementGate(num_qubits=len(self.problem_qubits)).on(*self.problem_qubits)]
        else:
            for idx, qubit in enumerate(self.final_mapping[0: len(self.problem_qubits)]):
                cbit = self.problem_qubits[idx]
                circuit_with_angles += [MeasurementGate(num_qubits=1).on(qubit).with_key(str(cbit))]

        return circuit_with_angles

    @property
    def parametric_qaoa_circuit(self) -> Circuit:
        """
        Creates a parametric QAOA circuit, given the qubit pairs, single qubits with biases,
        and a set of circuit angles. Note that this function does not actually run
        the circuit. To do this, you will need to subsequently execute the command self.eng.flush().

        Parameters
        ----------
            params:
                Object of type QAOAVariationalBaseParams
        """
        parametric_circuit = Circuit()

        if self.prepend_state:
            parametric_circuit += self.prepend_state
        # Initial state is all |+>
        if self.init_hadamard:
            parametric_circuit += [cirq.H.on(q) for q in self.problem_qubits]

        self.cirq_parameter_list = []
        for each_gate in self.abstract_circuit:
            # if gate is of type mixer or cost gate, assign parameter to it
            if each_gate.gate_label.type.value in ["MIXER", "COST"]:
                angle_param = sympy.Symbol(each_gate.gate_label.__repr__())
                self.cirq_parameter_list.append(angle_param)
                each_gate.angle_value = angle_param
            decomposition = each_gate.decomposition("standard")
            # using the list above, construct the circuit
            for each_tuple in decomposition:
                gate = each_tuple[0](self.gate_applicator, *each_tuple[1])
                gate.apply_gate(parametric_circuit)

        return parametric_circuit

    def get_counts(self, params: QAOAVariationalBaseParams, n_shots=None) -> dict:
        """
        Execute the circuit and obtain the counts

        Parameters
        ----------
        params: QAOAVariationalBaseParams
            The QAOA parameters - an object of one of the parameter classes, containing
            variable parameters.
        n_shots: int
            The number of times to run the circuit. If None, n_shots is set to the default: self.n_shots

        Returns
        -------
            A dictionary with the bitstring as the key and the number of counts
            as its value.
        """

        n_shots = self.n_shots if n_shots is None else n_shots

        circuit = self.qaoa_circuit(params)

        job_state = False
        no_of_job_retries = 0
        max_job_retries = 5

        while job_state is False:
            job = self.device.backend_device.run(circuit, repetitions=n_shots)

            api_contact = False
            no_of_api_retries = 0
            max_api_retries = 5

            while api_contact is False:
                try:
                    self.job_id = job.job_id()
                    counts = job.result().histogram(key='result', fold_func=flip_counts)
                    api_contact = True
                    job_state = True
                except Exception as e:
                    print(f"There was an error when trying to contact the Cirq API: {e}")
                    job_state = True
                    no_of_api_retries += 1
                    time.sleep(5)

                if no_of_api_retries >= max_api_retries:
                    raise ConnectionError(
                        "Number of API Retries exceeded Maximum allowed."
                    )

            if no_of_job_retries >= max_job_retries:
                raise ConnectionError("An Error Occurred with the Job(s) sent to Cirq.")

        # Expose counts
        final_counts = flip_counts(counts)
        self.measurement_outcomes = final_counts
        return final_counts

    def circuit_to_qasm(self, params: QAOAVariationalBaseParams) -> str:
        """
        A method to convert the entire QAOA `Circuit` object into
        an OpenQASM string
        """
        raise NotImplementedError()
        # qasm_string = self.qaoa_circuit(params).to_qasm()
        # return qasm_string

    def reset_circuit(self):
        """
        Reset self.circuit after performing a computation
        """
        raise NotImplementedError()
