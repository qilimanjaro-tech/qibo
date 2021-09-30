
from qibo.backends.numpy import NumpyBackend


class QilimanjaroBackend(NumpyBackend):  # pragma: no cover
    # remote backend is not tested until `qilimanjaroq` is available

    description = "Uses remote devices controlled by Qilimanjaro."

    def __init__(self):
        super().__init__()
        self.name = "qilimanjaroq"
        import qilimanjaroq  # pylint: disable=E0401
        self.is_hardware = True
        self.hardware_circuit = qilimanjaroq.circuit.RemoteCircuit
