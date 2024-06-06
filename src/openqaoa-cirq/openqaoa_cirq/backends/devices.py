import cirq
from typing import List
from openqaoa.backends.basedevice import DeviceBase


class DeviceCirq(DeviceBase):
    """
    Contains the required information and methods needed to access remote
    Cirq QPUs.

    Attributes
    ----------
    available_qpus: `list`
      When connection to a provider is established, this attribute contains a list
      of backend names which can be used to access the selected backend by reinitialising
      the Access Object with the name of the available backend as input to the
      device_name parameter.
    n_qubits: `int`
        The maximum number of qubits available for the selected backend. Only
        available if check_connection method is executed and a connection to the
        qpu and provider is established.
    """

    def __init__(
        self,
        device_name: str = "Simulator",
        as_emulator: bool = True,
    ):
        """
        Parameters
        ----------
        device_name: `str`
            The name of the Cirq device to be used.
            Note: Access to QPUs is currently restricted to those in an approved group.
            Source: https://quantumai.google/cirq/tutorials/google/start
        as_emulator: `bool`
            Whether to use the device as an emulator.
        """

        self.device_name = device_name
        self.device_location = "cirq"
        self.as_emulator = as_emulator

        self.provider_connected = None
        self.qpu_connected = None

    def check_connection(self) -> bool:
        """
        This method should allow a user to easily check if the credentials
        provided to access the remote QPU is valid.

        If no backend was specified in initialisation of object, just runs
        a test connection without a specific backend.
        If backend was specified, checks if connection to that backend
        can be established.

        Returns
        -------
        bool
            True if successfully connected to Cirq or Cirq and the QPU backend
            if it was specified. False if unable to connect to Cirq or failure
            in the attempt to connect to the specified backend.
        """

        self.provider_connected = self._check_provider_connection()

        if self.provider_connected == False:
            return self.provider_connected

        self.available_qpus = [
            "Bristlecone",
            "Simulator",
        ]

        if self.device_name == "":
            return self.provider_connected

        self.qpu_connected = self._check_backend_connection()

        if self.provider_connected and self.qpu_connected:
            return True
        else:
            return False

    def _check_backend_connection(self) -> bool:
        """Private method for checking connection with backend(s)."""

        if self.device_name in self.available_qpus:
            if self.device_name == "Bristlecone":
                self.backend_device = cirq.google.Bristlecone
            self.n_qubits = len(self.backend_device.qubits)
            if self.as_emulator is True:
                self.backend_device = cirq.Simulator()
            return True
        else:
            print(f"Please choose from {self.available_qpus} for this provider")
            return False

    def _check_provider_connection(self) -> bool:
        """
        Private method for checking connection with provider.
        """

        try:
            # Simulating provider connection in Cirq
            # Cirq does not have a direct equivalent to Qiskit's IBMProvider
            # But we can check if the backend exists
            self.provider = "cirq_provider"  # Placeholder for actual provider handling
            return True
        except Exception as e:
            print(
                "An Exception has occurred when trying to connect with the provider."
                "Please make sure that you have set up your Cirq environment correctly. \n {}".format(
                    e
                )
            )
            return False

    def connectivity(self) -> List[List[int]]:
        # Returns a simplified coupling map for the example backend
        return [
            [i, j] for i in range(self.n_qubits) for j in range(i + 1, self.n_qubits)
        ]
