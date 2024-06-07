from .backends import (
    DeviceCirq,
    QAOACirqQPUBackend,
    QAOACirqBackendShotBasedSimulator,
    QAOACirqBackendStatevecSimulator,
)

# Mapping device access
device_access = {DeviceCirq: QAOACirqQPUBackend}

# Mapping device names to corresponding backend classes
device_name_to_obj = {
    "cirq.qasm_simulator": QAOACirqBackendShotBasedSimulator,
    "cirq.shot_simulator": QAOACirqBackendShotBasedSimulator,
    "cirq.statevector_simulator": QAOACirqBackendStatevecSimulator,
}

# Configuration details
device_location = "google"  # or any other provider you are using with Cirq
device_plugin = DeviceCirq
device_args = {DeviceCirq: ["project_id"]}  # Assuming you need a project ID for Cirq

# Backend arguments
backend_args = {
    QAOACirqBackendStatevecSimulator: [],
    QAOACirqBackendShotBasedSimulator: [
        "n_shots",
        "seed_simulator",
        # Add more arguments specific to your Cirq implementation if any
    ],
    QAOACirqQPUBackend: [
        "n_shots",
        # Add more arguments specific to your Cirq QPU backend if any
    ],
}
