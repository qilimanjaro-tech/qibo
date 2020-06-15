"""
Testing tensorflow circuit and gates.
"""
import numpy as np
import pytest
import qibo
from qibo.models import Circuit
from qibo import gates

_BACKENDS = ["custom", "defaulteinsum", "matmuleinsum"]
_DEVICE_BACKENDS = [("custom", None), ("matmuleinsum", None),
                    ("custom", {"/GPU:0": 1, "/GPU:1": 1})]


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_circuit_addition_result(backend, accelerators):
    """Check if circuit addition works properly on Tensorflow circuit."""
    qibo.set_backend(backend)
    c1 = Circuit(2, accelerators)
    c1.add(gates.H(0))
    c1.add(gates.H(1))

    c2 = Circuit(2, accelerators)
    c2.add(gates.CNOT(0, 1))

    c3 = c1 + c2

    c = Circuit(2, accelerators)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.CNOT(0, 1))

    np.testing.assert_allclose(c3.execute().numpy(), c.execute().numpy())


@pytest.mark.parametrize("backend", _BACKENDS)
def test_hadamard(backend):
    """Check Hadamard gate is working properly."""
    qibo.set_backend(backend)
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
    final_state = c.execute().numpy()
    target_state = np.ones_like(final_state) / 2
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_flatten(backend):
    """Check ``Flatten`` gate works in circuits ."""
    qibo.set_backend(backend)
    target_state = np.ones(4) / 2.0
    c = Circuit(2)
    c.add(gates.Flatten(target_state))
    final_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_xgate(backend, accelerators):
    """Check X gate is working properly."""
    qibo.set_backend(backend)
    c = Circuit(2, accelerators)
    c.add(gates.X(0))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_ygate(backend):
    """Check Y gate is working properly."""
    qibo.set_backend(backend)
    c = Circuit(2)
    c.add(gates.Y(1))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[1] = 1j
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_zgate(backend, accelerators):
    """Check Z gate is working properly."""
    qibo.set_backend(backend)
    c = Circuit(2, accelerators)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.Z(0))
    final_state = c.execute().numpy()
    target_state = np.ones_like(final_state) / 2.0
    target_state[2] *= -1.0
    target_state[3] *= -1.0
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_multicontrol_xgate(backend):
    """Check that fallback method for X works for one or two controls."""
    qibo.set_backend(backend)
    c1 = Circuit(3)
    c1.add(gates.X(0))
    c1.add(gates.X(2).controlled_by(0))
    final_state = c1.execute().numpy()
    c2 = Circuit(3)
    c2.add(gates.X(0))
    c2.add(gates.CNOT(0, 2))
    target_state = c2.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)

    c1 = Circuit(3)
    c1.add(gates.X(0))
    c1.add(gates.X(2))
    c1.add(gates.X(1).controlled_by(0, 2))
    final_state = c1.execute().numpy()
    c2 = Circuit(3)
    c2.add(gates.X(0))
    c2.add(gates.X(2))
    c2.add(gates.TOFFOLI(0, 2, 1))
    target_state = c2.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_multicontrol_xgate_more_controls(backend, accelerators):
    """Check that fallback method for X works for more than two controls."""
    qibo.set_backend(backend)
    c = Circuit(4, accelerators)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.X(2))
    c.add(gates.X(3).controlled_by(0, 1, 2))
    c.add(gates.X(0))
    c.add(gates.X(2))
    final_state = c.execute().numpy()

    c = Circuit(4)
    c.add(gates.X(1))
    c.add(gates.X(3))
    target_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_rz_phase0(backend):
    """Check RZ gate is working properly when qubit is on |0>."""
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.RZ(0, theta))
    final_state = c.execute().numpy()

    target_state = np.zeros_like(final_state)
    target_state[0] = np.exp(-1j * theta / 2.0)
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_rz_phase1(backend):
    """Check RZ gate is working properly when qubit is on |1>."""
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.RZ(0, theta))
    final_state = c.execute().numpy()

    target_state = np.zeros_like(final_state)
    target_state[2] = np.exp(1j * theta / 2.0)
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_rx(backend):
    """Check RX gate is working properly."""
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(1)
    c.add(gates.H(0))
    c.add(gates.RX(0, theta=theta))
    final_state = c.execute().numpy()

    phase = np.exp(1j * theta / 2.0)
    gate = np.array([[phase.real, -1j * phase.imag],
                    [-1j * phase.imag, phase.real]])
    target_state = gate.dot(np.ones(2)) / np.sqrt(2)
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_ry(backend):
    """Check RY gate is working properly."""
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(1)
    c.add(gates.H(0))
    c.add(gates.RY(0, theta))
    final_state = c.execute().numpy()

    phase = np.exp(1j * theta / 2.0)
    gate = np.array([[phase.real, -phase.imag],
                     [phase.imag, phase.real]])
    target_state = gate.dot(np.ones(2)) / np.sqrt(2)
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cnot_no_effect(backend):
    """Check CNOT gate is working properly on |00>."""
    qibo.set_backend(backend)
    c = Circuit(2)
    c.add(gates.CNOT(0, 1))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[0] = 1.0
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cnot(backend):
    """Check CNOT gate is working properly on |10>."""
    qibo.set_backend(backend)
    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.CNOT(0, 1))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[3] = 1.0
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_cz(backend):
    """Check CZ gate is working properly on random state."""
    qibo.set_backend(backend)
    init_state = np.random.random(4) + 1j * np.random.random(4)
    matrix = np.eye(4)
    matrix[3, 3] = -1
    target_state = matrix.dot(init_state)
    c = Circuit(2)
    c.add(gates.CZ(0, 1))
    final_state = c.execute(np.copy(init_state)).numpy()
    np.testing.assert_allclose(final_state, target_state)

    c = Circuit(2)
    c.add(gates.Z(1).controlled_by(0))
    final_state = c.execute(np.copy(init_state)).numpy()
    assert c.queue[0].name == "cz"
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_czpow(backend, accelerators):
    """Check CZPow gate is working properly on |11>."""
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(2, accelerators)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.CZPow(0, 1, theta))
    final_state = c.execute().numpy()

    phase = np.exp(1j * theta)
    target_state = np.zeros_like(final_state)
    target_state[3] = phase
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_fsim(backend):
    """Check fSim gate is working properly on |++>."""
    qibo.set_backend(backend)
    theta = 0.1234
    phi = 0.4321

    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.fSim(0, 1, theta, phi))
    final_state = c.execute().numpy()

    target_state = np.ones_like(final_state) / 2.0
    rotation = np.array([[np.cos(theta), -1j * np.sin(theta)],
                         [-1j * np.sin(theta), np.cos(theta)]])
    matrix = np.eye(4, dtype=target_state.dtype)
    matrix[1:3, 1:3] = rotation
    matrix[3, 3] = np.exp(-1j * phi)
    target_state = matrix.dot(target_state)
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_controlled_by_fsim(backend, accelerators):
    """Check ``controlled_by`` for fSim gate."""
    qibo.set_backend(backend)
    theta = 0.1234
    phi = 0.4321

    c = Circuit(6, accelerators)
    c.add((gates.H(i) for i in range(6)))
    c.add(gates.fSim(5, 3, theta, phi).controlled_by(0, 2, 1))
    final_state = c.execute().numpy()

    target_state = np.ones_like(final_state) / np.sqrt(2 ** 6)
    rotation = np.array([[np.cos(theta), -1j * np.sin(theta)],
                         [-1j * np.sin(theta), np.cos(theta)]])
    matrix = np.eye(4, dtype=target_state.dtype)
    matrix[1:3, 1:3] = rotation
    matrix[3, 3] = np.exp(-1j * phi)
    ids = [56, 57, 60, 61]
    target_state[ids] = matrix.dot(target_state[ids])
    ids = [58, 59, 62, 63]
    target_state[ids] = matrix.dot(target_state[ids])
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_generalized_fsim(backend, accelerators):
    """Check GeneralizedfSim gate is working properly on |++>."""
    qibo.set_backend(backend)
    phi = np.random.random()
    rotation = np.random.random((2, 2)) + 1j * np.random.random((2, 2))

    c = Circuit(3, accelerators)
    c.add((gates.H(i) for i in range(3)))
    c.add(gates.GeneralizedfSim(1, 2, rotation, phi))
    final_state = c.execute().numpy()

    target_state = np.ones_like(final_state) / np.sqrt(8)
    matrix = np.eye(4, dtype=target_state.dtype)
    matrix[1:3, 1:3] = rotation
    matrix[3, 3] = np.exp(-1j * phi)
    target_state[:4] = matrix.dot(target_state[:4])
    target_state[4:] = matrix.dot(target_state[4:])
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_generalized_fsim_error(backend):
    """Check GenerelizedfSim gate raises error for wrong unitary shape."""
    qibo.set_backend(backend)
    phi = np.random.random()
    rotation = np.random.random((4, 4)) + 1j * np.random.random((4, 4))
    c = Circuit(2)
    with pytest.raises(ValueError):
        c.add(gates.GeneralizedfSim(0, 1, rotation, phi))


@pytest.mark.parametrize("backend", _BACKENDS)
def test_doubly_controlled_by_rx_no_effect(backend):
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(3)
    c.add(gates.X(0))
    c.add(gates.RX(2, theta).controlled_by(0, 1))
    c.add(gates.X(0))
    final_state = c.execute().numpy()

    target_state = np.zeros_like(final_state)
    target_state[0] = 1.0

    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_doubly_controlled_by_rx(backend, accelerators):
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(3, accelerators)
    c.add(gates.RX(2, theta))
    target_state = c.execute().numpy()

    c = Circuit(3)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.RX(2, theta).controlled_by(0, 1))
    c.add(gates.X(0))
    c.add(gates.X(1))
    final_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_swap(backend):
    """Check SWAP gate is working properly on |01>."""
    qibo.set_backend(backend)
    c = Circuit(2)
    c.add(gates.X(1))
    c.add(gates.SWAP(0, 1))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_multiple_swap(backend):
    """Check SWAP gate is working properly when called multiple times."""
    qibo.set_backend(backend)
    c = Circuit(4)
    c.add(gates.X(0))
    c.add(gates.X(2))
    c.add(gates.SWAP(0, 1))
    c.add(gates.SWAP(2, 3))
    final_state = c.execute().numpy()

    c = Circuit(4)
    c.add(gates.X(1))
    c.add(gates.X(3))
    target_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_controlled_by_swap_small(backend):
    """Check controlled SWAP using controlled by for ``nqubits=3``."""
    qibo.set_backend(backend)
    c = Circuit(3)
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    c.add(gates.SWAP(1, 2).controlled_by(0))
    final_state = c.execute().numpy()
    c = Circuit(3)
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    target_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)

    c = Circuit(3)
    c.add(gates.X(0))
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    c.add(gates.SWAP(1, 2).controlled_by(0))
    c.add(gates.X(0))
    final_state = c.execute().numpy()
    c = Circuit(3)
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    c.add(gates.SWAP(1, 2))
    target_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_controlled_by_swap(backend):
    """Check controlled SWAP using controlled by for ``nqubits=4``."""
    qibo.set_backend(backend)
    c = Circuit(4)
    c.add(gates.RX(2, theta=0.1234))
    c.add(gates.RY(3, theta=0.4321))
    c.add(gates.SWAP(2, 3).controlled_by(0))
    final_state = c.execute().numpy()
    c = Circuit(4)
    c.add(gates.RX(2, theta=0.1234))
    c.add(gates.RY(3, theta=0.4321))
    target_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)

    c = Circuit(4)
    c.add(gates.X(0))
    c.add(gates.RX(2, theta=0.1234))
    c.add(gates.RY(3, theta=0.4321))
    c.add(gates.SWAP(2, 3).controlled_by(0))
    c.add(gates.X(0))
    final_state = c.execute().numpy()
    c = Circuit(4)
    c.add(gates.RX(2, theta=0.1234))
    c.add(gates.RY(3, theta=0.4321))
    c.add(gates.SWAP(2, 3))
    target_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_doubly_controlled_by_swap(backend, accelerators):
    """Check controlled SWAP using controlled by two qubits."""
    qibo.set_backend(backend)
    c = Circuit(4, accelerators)
    c.add(gates.X(0))
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    c.add(gates.SWAP(1, 2).controlled_by(0, 3))
    c.add(gates.X(0))
    final_state = c.execute().numpy()
    c = Circuit(4)
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    target_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)

    c = Circuit(4, accelerators)
    c.add(gates.X(0))
    c.add(gates.X(3))
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    c.add(gates.SWAP(1, 2).controlled_by(0, 3))
    c.add(gates.X(0))
    c.add(gates.X(3))
    final_state = c.execute().numpy()
    c = Circuit(4)
    c.add(gates.RX(1, theta=0.1234))
    c.add(gates.RY(2, theta=0.4321))
    c.add(gates.SWAP(1, 2))
    target_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_toffoli_no_effect(backend):
    """Check Toffoli gate is working properly on |010>."""
    qibo.set_backend(backend)
    c = Circuit(3)
    c.add(gates.X(1))
    c.add(gates.TOFFOLI(0, 1, 2))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[2] = 1.0
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_toffoli(backend):
    """Check Toffoli gate is working properly on |110>."""
    qibo.set_backend(backend)
    c = Circuit(3)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.TOFFOLI(0, 1, 2))
    final_state = c.execute().numpy()
    target_state = np.zeros_like(final_state)
    target_state[-1] = 1.0
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_unitary_common_gates(backend):
    """Check that `Unitary` gate can create common gates."""
    qibo.set_backend(backend)
    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.H(1))
    target_state = c.execute().numpy()
    c = Circuit(2)
    c.add(gates.Unitary(np.array([[0, 1], [1, 0]]), 0))
    c.add(gates.Unitary(np.array([[1, 1], [1, -1]]) / np.sqrt(2), 1))
    final_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)

    thetax = 0.1234
    thetay = 0.4321
    c = Circuit(2)
    c.add(gates.RX(0, theta=thetax))
    c.add(gates.RY(1, theta=thetay))
    c.add(gates.CNOT(0, 1))
    target_state = c.execute().numpy()
    c = Circuit(2)
    rx = np.array([[np.cos(thetax / 2), -1j * np.sin(thetax / 2)],
                   [-1j * np.sin(thetax / 2), np.cos(thetax / 2)]])
    ry = np.array([[np.cos(thetay / 2), -np.sin(thetay / 2)],
                   [np.sin(thetay / 2), np.cos(thetay / 2)]])
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    c.add(gates.Unitary(rx, 0))
    c.add(gates.Unitary(ry, 1))
    c.add(gates.Unitary(cnot, 0, 1))
    final_state = c.execute().numpy()
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_unitary_random_onequbit_gate(backend):
    """Check that ``Unitary`` gate can apply random 2x2 matrices."""
    qibo.set_backend(backend)
    init_state = np.ones(4) / 2.0
    matrix = np.random.random([2, 2])
    target_state = np.kron(np.eye(2), matrix).dot(init_state)

    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.Unitary(matrix, 1, name="random"))
    final_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_unitary_random_twoqubit_gate(backend):
    """Check that ``Unitary`` gate can apply random 4x4 matrices."""
    qibo.set_backend(backend)
    init_state = np.ones(8) / np.sqrt(8)
    matrix = np.random.random([4, 4])
    target_state = np.kron(np.eye(2), matrix).dot(init_state)

    c = Circuit(3)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.H(2))
    c.add(gates.Unitary(matrix, 1, 2, name="random"))
    final_state = c.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_unitary_controlled_by(backend, accelerators):
    """Check that `controlled_by` works as expected with `Unitary`."""
    qibo.set_backend(backend)
    matrix = np.random.random([2, 2])
    c = Circuit(2, accelerators)
    c.add(gates.H(0))
    c.add(gates.H(1))
    c.add(gates.Unitary(matrix, 1).controlled_by(0))
    final_state = c.execute().numpy()
    target_state = np.ones_like(final_state) / 2.0
    target_state[2:] = matrix.dot(target_state[2:])
    np.testing.assert_allclose(final_state, target_state)

    matrix = np.random.random([4, 4])
    c = Circuit(4, accelerators)
    c.add((gates.H(i) for i in range(4)))
    c.add(gates.Unitary(matrix, 1, 3).controlled_by(0, 2))
    final_state = c.execute().numpy()
    target_state = np.ones_like(final_state) / 4.0
    ids = [10, 11, 14, 15]
    target_state[ids] = matrix.dot(target_state[ids])
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_unitary_bad_shape(backend):
    qibo.set_backend(backend)
    matrix = np.random.random((8, 8))
    with pytest.raises(ValueError):
        gate = gates.Unitary(matrix, (0, 1))


@pytest.mark.parametrize("gates", ["custom", "native"])
def test_construct_unitary(gates):
    if gates == "custom":
        from qibo.tensorflow import cgates as gates
    else:
        from qibo.tensorflow import gates
    target_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    np.testing.assert_allclose(gates.H.construct_unitary().numpy(), target_matrix)
    target_matrix = np.array([[0, 1], [1, 0]])
    np.testing.assert_allclose(gates.X.construct_unitary().numpy(), target_matrix)
    target_matrix = np.array([[0, -1j], [1j, 0]])
    np.testing.assert_allclose(gates.Y.construct_unitary().numpy(), target_matrix)
    target_matrix = np.array([[1, 0], [0, -1]])
    np.testing.assert_allclose(gates.Z.construct_unitary().numpy(), target_matrix)
    target_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0, 0, 1], [0, 0, 1, 0]])
    np.testing.assert_allclose(gates.CNOT.construct_unitary().numpy(), target_matrix)
    target_matrix = np.diag([1, 1, 1, -1])
    np.testing.assert_allclose(gates.CZ.construct_unitary().numpy(), target_matrix)
    target_matrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                              [0, 1, 0, 0], [0, 0, 0, 1]])
    np.testing.assert_allclose(gates.SWAP.construct_unitary().numpy(), target_matrix)
    target_matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 0, 0, 1, 0]])
    np.testing.assert_allclose(gates.TOFFOLI.construct_unitary().numpy(), target_matrix)

    theta = 0.1234
    target_matrix = np.array([[np.cos(theta / 2.0), -1j * np.sin(theta / 2.0)],
                              [-1j * np.sin(theta / 2.0), np.cos(theta / 2.0)]])
    np.testing.assert_allclose(gates.RX.construct_unitary(theta).numpy(), target_matrix)
    target_matrix = np.array([[np.cos(theta / 2.0), -np.sin(theta / 2.0)],
                              [np.sin(theta / 2.0), np.cos(theta / 2.0)]])
    np.testing.assert_allclose(gates.RY.construct_unitary(theta).numpy(), target_matrix)
    target_matrix = np.diag([np.exp(-1j * theta / 2.0), np.exp(1j * theta / 2.0)])
    np.testing.assert_allclose(gates.RZ.construct_unitary(theta).numpy(), target_matrix)
    target_matrix = np.diag([1, 1, 1, np.exp(1j * theta)])
    np.testing.assert_allclose(gates.CZPow.construct_unitary(theta).numpy(), target_matrix)

    with pytest.raises(NotImplementedError):
        gates.fSim.construct_unitary()


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("nqubits", [4, 5, 6, 7, 10])
def test_variational_one_layer(backend, nqubits):
    qibo.set_backend(backend)
    theta = 2 * np.pi * np.random.random(nqubits)
    c = Circuit(nqubits)
    c.add((gates.RY(i, t) for i, t in enumerate(theta)))
    c.add((gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2)))
    target_state = c().numpy()

    c = Circuit(nqubits)
    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    thetas = {i: theta[i] for i in range(nqubits)}
    c.add(gates.VariationalLayer(pairs, gates.RY, gates.CZ, thetas))
    final_state = c().numpy()
    np.testing.assert_allclose(target_state, final_state)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("nqubits", [4, 5, 6, 7, 10])
def test_variational_two_layers(backend, nqubits):
    qibo.set_backend(backend)
    theta = 2 * np.pi * np.random.random(2 * nqubits)
    theta_iter = iter(theta)
    c = Circuit(nqubits)
    c.add((gates.RY(i, next(theta_iter)) for i in range(nqubits)))
    c.add((gates.CZ(i, i + 1) for i in range(0, nqubits - 1, 2)))
    c.add((gates.RY(i, next(theta_iter)) for i in range(nqubits)))
    c.add((gates.CZ(i, i + 1) for i in range(1, nqubits - 2, 2)))
    c.add(gates.CZ(0, nqubits - 1))
    target_state = c().numpy()

    c = Circuit(nqubits)
    theta = theta.reshape((2, nqubits))
    pairs1 = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    pairs2 = list((i, i + 1) for i in range(1, nqubits - 2, 2))
    pairs2.append((0, nqubits - 1))
    thetas0 = {i: theta[0, i] for i in range(nqubits)}
    thetas1 = {i: theta[1, i] for i in range(nqubits)}
    c.add(gates.VariationalLayer(pairs1, gates.RY, gates.CZ, thetas0))
    c.add(gates.VariationalLayer(pairs2, gates.RY, gates.CZ, thetas1))
    final_state = c().numpy()
    np.testing.assert_allclose(target_state, final_state)

    c = Circuit(nqubits)
    theta = theta.reshape((2, nqubits))
    pairs = list((i, i + 1) for i in range(0, nqubits - 1, 2))
    thetas0 = {i: theta[0, i] for i in range(nqubits)}
    thetas1 = {i: theta[1, i] for i in range(nqubits)}
    c.add(gates.VariationalLayer(pairs, gates.RY, gates.CZ, thetas0, thetas1))
    c.add((gates.CZ(i, i + 1) for i in range(1, nqubits - 2, 2)))
    c.add(gates.CZ(0, nqubits - 1))
    final_state = c().numpy()
    np.testing.assert_allclose(target_state, final_state)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_variational_layer_errors(backend):
    qibo.set_backend(backend)
    c = Circuit(6)
    pairs = list((i, i + 1) for i in range(0, 5, 2))
    thetas0 = {i: 0 for i in range(6)}
    thetas1 = {i: 0 for i in range(6)}
    thetas1[6] = 1
    with pytest.raises(ValueError):
        c.add(gates.VariationalLayer(pairs, gates.RY, gates.CZ, thetas0, thetas1))
    thetas0 = {i: 0 for i in range(8)}
    thetas1 = {i: 0 for i in range(8)}
    with pytest.raises(ValueError):
        c.add(gates.VariationalLayer(pairs, gates.RY, gates.CZ, thetas0, thetas1))


@pytest.mark.parametrize("backend", _BACKENDS)
def test_custom_circuit(backend):
    """Check consistency between Circuit and custom circuits"""
    import tensorflow as tf
    qibo.set_backend(backend)
    theta = 0.1234

    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.X(1))
    c.add(gates.CZPow(0, 1, theta))
    r1 = c.execute().numpy()

    # custom circuit
    def custom_circuit(initial_state, theta):
        l1 = gates.X(0)(initial_state)
        l2 = gates.X(1)(l1)
        o = gates.CZPow(0, 1, theta)(l2)
        return o

    init2 = c._default_initial_state()
    init3 = c._default_initial_state()
    if backend != "custom":
        init2 = tf.reshape(init2, (2, 2))
        init3 = tf.reshape(init3, (2, 2))

    r2 = custom_circuit(init2, theta).numpy().ravel()
    np.testing.assert_allclose(r1, r2)

    tf_custom_circuit = tf.function(custom_circuit)
    if backend == "custom":
        with pytest.raises(NotImplementedError):
            r3 = tf_custom_circuit(init3, theta).numpy().ravel()
    else:
        r3 = tf_custom_circuit(init3, theta).numpy().ravel()
        np.testing.assert_allclose(r2, r3)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_compiled_circuit(backend):
    """Check that compiling with `Circuit.compile` does not break results."""
    qibo.set_backend(backend)
    def create_circuit(theta = 0.1234):
        c = Circuit(2)
        c.add(gates.X(0))
        c.add(gates.X(1))
        c.add(gates.CZPow(0, 1, theta))
        return c

    # Run eager circuit
    c1 = create_circuit()
    r1 = c1.execute().numpy()

    # Run compiled circuit
    c2 = create_circuit()
    if backend == "custom":
        with pytest.raises(RuntimeError):
            c2.compile()
    else:
        c2.compile()
        r2 = c2.execute().numpy()
        np.testing.assert_allclose(r1, r2)


def test_compiling_twice_exception():
    """Check that compiling a circuit a second time raises error."""
    from qibo.tensorflow import gates
    c = Circuit(2)
    c.add([gates.H(0), gates.H(1)])
    c.compile()
    with pytest.raises(RuntimeError):
        c.compile()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_circuit_custom_compilation(backend):
    qibo.set_backend(backend)
    theta = 0.1234
    init_state = np.ones(4) / 2.0

    def run_circuit(initial_state):
        c = Circuit(2)
        c.add(gates.X(0))
        c.add(gates.X(1))
        c.add(gates.CZPow(0, 1, theta))
        return c.execute(initial_state)

    r1 = run_circuit(init_state).numpy()
    import tensorflow as tf
    compiled_circuit = tf.function(run_circuit)
    if backend == "custom":
        with pytest.raises(NotImplementedError):
            r2 = compiled_circuit(init_state)
    else:
        r2 = compiled_circuit(init_state)
        np.testing.assert_allclose(r1, r2)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_bad_initial_state(backend, accelerators):
    """Check that errors are raised when bad initial state is passed."""
    qibo.set_backend(backend)
    import tensorflow as tf
    c = Circuit(2, accelerators)
    c.add([gates.H(0), gates.H(1)])
    with pytest.raises(ValueError):
        final_state = c(initial_state=np.zeros(2**3))
    with pytest.raises(ValueError):
        final_state = c(initial_state=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        final_state = c(initial_state=np.zeros((2, 2, 2)))
    with pytest.raises(TypeError):
        final_state = c(initial_state=0)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_final_state_property(backend, accelerators):
    """Check accessing final state using the circuit's property."""
    qibo.set_backend(backend)
    import tensorflow as tf
    c = Circuit(2, accelerators)
    c.add([gates.H(0), gates.H(1)])

    with pytest.raises(RuntimeError):
        final_state = c.final_state

    _ = c()
    final_state = c.final_state.numpy()
    target_state = np.ones(4) / 2
    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_variable_theta(backend, accelerators):
    """Check that parametrized gates accept `tf.Variable` parameters."""
    qibo.set_backend(backend)
    import tensorflow as tf
    from qibo.config import DTYPES
    theta1 = tf.Variable(0.1234, dtype=DTYPES.get('DTYPE'))
    theta2 = tf.Variable(0.4321, dtype=DTYPES.get('DTYPE'))

    cvar = Circuit(2, accelerators)
    cvar.add(gates.RX(0, theta1))
    cvar.add(gates.RY(1, theta2))
    final_state = cvar().numpy()

    c = Circuit(2)
    c.add(gates.RX(0, 0.1234))
    c.add(gates.RY(1, 0.4321))
    target_state = c().numpy()

    np.testing.assert_allclose(final_state, target_state)


@pytest.mark.parametrize(("backend", "accelerators"), _DEVICE_BACKENDS)
def test_circuit_copy(backend, accelerators):
    """Check that circuit copy execution is equivalent to original circuit."""
    qibo.set_backend(backend)
    theta = 0.1234

    c1 = Circuit(2, accelerators)
    c1.add([gates.X(0), gates.X(1), gates.CZPow(0, 1, theta)])
    c2 = c1.copy()

    target_state = c1.execute().numpy()
    final_state = c2.execute().numpy()

    np.testing.assert_allclose(final_state, target_state)
