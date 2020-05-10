import pytest
from qibo.models import Circuit
from qibo import gates, __version__


def assert_strings_equal(a, b):
    """Asserts that two strings are identical character by character."""
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x == y


def test_empty():
    c = Circuit(2)
    target = f"""// Generated by QIBO {__version__}
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];"""
    assert_strings_equal(c.to_qasm(), target)


def test_simple():
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.H(1))
    target = f"""// Generated by QIBO {__version__}
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
h q[1];"""
    assert_strings_equal(c.to_qasm(), target)


def test_unknown_gate_error():
    """"Check that using `to_qasm` with not supported gates raises error."""
    c = Circuit(2)
    c.add(gates.Flatten(4 * [0]))
    with pytest.raises(ValueError):
        c.to_qasm()


def test_controlled_by_error():
    """Check that using `to_qasm` with controlled by gates raises error."""
    c = Circuit(3)
    c.add(gates.H(0))
    c.add(gates.Y(1).controlled_by(0, 2))
    with pytest.raises(ValueError):
        c.to_qasm()


def test_multiqubit_gates():
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.CNOT(0, 1))
    c.add(gates.X(1))
    c.add(gates.SWAP(0, 1))
    c.add(gates.X(0).controlled_by(1))
    # `controlled_by` here falls back to CNOT and should work
    target = f"""// Generated by QIBO {__version__}
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
x q[1];
swap q[0],q[1];
cx q[1],q[0];"""
    assert_strings_equal(c.to_qasm(), target)


def test_toffoli():
    c = Circuit(3)
    c.add(gates.Y(0))
    c.add(gates.TOFFOLI(0, 1, 2))
    c.add(gates.X(1))
    c.add(gates.TOFFOLI(0, 2, 1))
    c.add(gates.Z(2))
    c.add(gates.TOFFOLI(1, 2, 0))
    target = f"""// Generated by QIBO {__version__}
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
y q[0];
ccx q[0],q[1],q[2];
x q[1];
ccx q[0],q[2],q[1];
z q[2];
ccx q[1],q[2],q[0];"""
    assert_strings_equal(c.to_qasm(), target)


def test_parametrized_gate():
    c = Circuit(2)
    c.add(gates.Y(0))
    c.add(gates.RY(1, 0.1234))
    target = f"""// Generated by QIBO {__version__}
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
y q[0];
ry(0.1234) q[1];"""
    assert_strings_equal(c.to_qasm(), target)


def test_czpow():
    c = Circuit(2)
    c.add(gates.RX(0, 0.1234))
    c.add(gates.RZ(1, 0.4321))
    c.add(gates.CZPow(0, 1, 0.567))
    target = f"""// Generated by QIBO {__version__}
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rx(0.1234) q[0];
rz(0.4321) q[1];
crz(0.567) q[0],q[1];"""
    assert_strings_equal(c.to_qasm(), target)


def test_measurements():
    c = Circuit(2)
    c.add(gates.X(0))
    c.add(gates.Y(1))
    c.add(gates.M(0, 1))
    target = f"""// Generated by QIBO {__version__}
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg register0[2];
x q[0];
y q[1];
measure q[0] -> register0[0];
measure q[1] -> register0[1];"""
    assert_strings_equal(c.to_qasm(), target)


def test_multiple_measurements():
    c = Circuit(5)
    c.add(gates.M(0, 2, 4, register_name="a"))
    c.add(gates.M(1, 3, register_name="b"))
    target = f"""// Generated by QIBO {__version__}
OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg a[3];
creg b[2];
measure q[0] -> a[0];
measure q[2] -> a[1];
measure q[4] -> a[2];
measure q[1] -> b[0];
measure q[3] -> b[1];"""
    assert_strings_equal(c.to_qasm(), target)


def test_capital_in_register_name_error():
    """Check that using capital letter in register name raises error."""
    c = Circuit(2)
    c.add(gates.M(0, 1, register_name="Abc"))
    with pytest.raises(NameError):
        c.to_qasm()


def test_from_qasm_simple():
    target = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
h q[1];"""
    c = Circuit.from_qasm(target)
    assert c.nqubits == 2
    assert c.depth == 2
    assert isinstance(c.queue[0], gates.H)
    assert isinstance(c.queue[1], gates.H)


def test_from_qasm_evaluation():
    import numpy as np
    target = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
h q[1];"""
    c = Circuit.from_qasm(target)
    final_state = c().numpy()
    target_state = np.ones(4) / 2.0
    np.testing.assert_allclose(target_state, final_state)


def test_from_qasm_multiqubit_gates():
    target = f"""// Generated by QIBO {__version__}
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
cx q[0],q[2];
x q[1];
swap q[0],q[1];
cx q[1],q[0];
ccx q[1],q[2],q[0];"""
    c = Circuit.from_qasm(target)
    assert c.nqubits == 3
    assert c.depth == 5
    assert isinstance(c.queue[0], gates.CNOT)
    assert c.queue[0].qubits == (0, 2)
    assert isinstance(c.queue[1], gates.X)
    assert c.queue[1].qubits == (1,)
    assert isinstance(c.queue[2], gates.SWAP)
    assert c.queue[2].qubits == (0, 1)
    assert isinstance(c.queue[3], gates.CNOT)
    assert c.queue[3].qubits == (1, 0)
    assert isinstance(c.queue[4], gates.TOFFOLI)
    assert c.queue[4].qubits == (1, 2, 0)


def test_from_qasm_multiple_qregs():
    target = f"""// Generated by QIBO {__version__}
OPENQASM 2.0;
include "qelib1.inc";
qreg a[2],b[1];
cx a[0],b[0];
// random comment
x a[1];
qreg c[2];
// random comment 2
swap a[0],c[1];
ccx b[0],c[1],c[0];"""
    c = Circuit.from_qasm(target)
    assert c.nqubits == 5
    assert c.depth == 4
    assert isinstance(c.queue[0], gates.CNOT)
    assert c.queue[0].qubits == (0, 2)
    assert isinstance(c.queue[1], gates.X)
    assert c.queue[1].qubits == (1,)
    assert isinstance(c.queue[2], gates.SWAP)
    assert c.queue[2].qubits == (0, 4)
    assert isinstance(c.queue[3], gates.TOFFOLI)
    assert c.queue[3].qubits == (2, 4, 3)


def test_from_qasm_invalid_script():
    # Missing starting line
    target = """qreg q[2];
x q[1];"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)

    # Undefined qubit
    target = """OPENQASM 2.0;
qreg q[2];
x q[2];
"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)

    # Undefined qubit
    target = """OPENQASM 2.0;
qreg q[2];
x a[0];
"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)

    # Invalid command `test`
    target = """OPENQASM 2.0;
test q[2];
"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)


def test_from_qasm_measurements():
    target = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg a[3];
creg b[2];
measure q[0] -> a[0];
x q[3];
measure q[1] -> b[0];
measure q[2] -> a[1];
measure q[4] -> a[2];
measure q[3] -> b[1];"""
    c = Circuit.from_qasm(target)
    assert c.depth == 1
    assert isinstance(c.queue[0], gates.X)
    assert isinstance(c.measurement_gate, gates.M)
    assert c.measurement_tuples == {"a": (0, 2, 4), "b": (1, 3)}


def test_from_qasm_invalid_measurements():
    # Undefined qubit
    target = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg a[2];
measure q[2] -> a[0];"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)

    # Undefined register
    target = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg a[2];
measure q[0] -> b[0];"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)

    # Register index out of range
    target = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg a[2];
measure q[0] -> a[2];"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)

    # Reuse measured qubit
    target = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg a[2];
measure q[0] -> a[0];
x q[1];
measure q[1] -> a[1];"""
    # Note that in this example the full register measurement is added during
    # the first `measurement` call
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)
