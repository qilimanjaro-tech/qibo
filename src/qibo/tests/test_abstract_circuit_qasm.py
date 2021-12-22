"""Tests creating abstract Qibo circuits from OpenQASM code."""
import pytest
import qibo
from qibo import __version__
from qibo.abstractions import gates
from qibo.tests.test_abstract_circuit import Circuit


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

def test_singlequbit_gates():
    c = Circuit(2)
    c.add(gates.H(0))
    c.add(gates.X(1))
    c.add(gates.Y(0))
    c.add(gates.Z(1))
    c.add(gates.S(0))
    c.add(gates.SDG(1))
    c.add(gates.T(0))
    c.add(gates.TDG(1))
    c.add(gates.I(0))
    target = f"""// Generated by QIBO {__version__}
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
x q[1];
y q[0];
z q[1];
s q[0];
sdg q[1];
t q[0];
tdg q[1];
id q[0];"""
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


def test_cu1():
    c = Circuit(2)
    c.add(gates.RX(0, 0.1234))
    c.add(gates.RZ(1, 0.4321))
    c.add(gates.CU1(0, 1, 0.567))
    target = f"""// Generated by QIBO {__version__}
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
rx(0.1234) q[0];
rz(0.4321) q[1];
cu1(0.567) q[0],q[1];"""
    assert_strings_equal(c.to_qasm(), target)


def test_ugates():
    c = Circuit(3)
    c.add(gates.RX(0, 0.1))
    c.add(gates.RZ(1, 0.4))
    c.add(gates.U2(2, 0.5, 0.6))
    c.add(gates.CU1(0, 1, 0.7))
    c.add(gates.CU3(2, 1, 0.2, 0.3, 0.4))
    target = f"""// Generated by QIBO {__version__}
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
rx(0.1) q[0];
rz(0.4) q[1];
u2(0.5, 0.6) q[2];
cu1(0.7) q[0],q[1];
cu3(0.2, 0.3, 0.4) q[2],q[1];"""
    assert_strings_equal(c.to_qasm(), target)

    c = Circuit(2)
    c.add(gates.CU2(0, 1, 0.1, 0.2))
    with pytest.raises(ValueError):
        target = c.to_qasm()


def test_crotations():
    c = Circuit(3)
    c.add(gates.RX(0, 0.1))
    c.add(gates.RZ(1, 0.4))
    c.add(gates.CRX(0, 2, 0.5))
    c.add(gates.RY(1, 0.3).controlled_by(2))
    target = f"""// Generated by QIBO {__version__}
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
rx(0.1) q[0];
rz(0.4) q[1];
crx(0.5) q[0],q[2];
cry(0.3) q[2],q[1];"""
    assert_strings_equal(c.to_qasm(), target)

    c = Circuit(2)
    c.add(gates.CU2(0, 1, 0.1, 0.2))
    with pytest.raises(ValueError):
        target = c.to_qasm()


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
    assert c.depth == 4
    assert c.ngates == 5
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


def test_from_qasm_singlequbit_gates():
    target = f"""// Generated by QIBO {__version__}
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
x q[1];
y q[0];
z q[1];
s q[0];
sdg q[1];
t q[0];
tdg q[1];
id q[0];"""
    c = Circuit.from_qasm(target)
    assert c.nqubits == 2
    assert c.depth == 5
    assert c.ngates == 9
    assert isinstance(c.queue[0], gates.H)
    assert c.queue[0].qubits == (0,)
    assert isinstance(c.queue[1], gates.X)
    assert c.queue[1].qubits == (1,)
    assert isinstance(c.queue[2], gates.Y)
    assert c.queue[2].qubits == (0,)
    assert isinstance(c.queue[3], gates.Z)
    assert c.queue[3].qubits == (1,)
    assert isinstance(c.queue[4], gates.S)
    assert c.queue[4].qubits == (0,)
    assert isinstance(c.queue[5], gates.SDG)
    assert c.queue[5].qubits == (1,)
    assert isinstance(c.queue[6], gates.T)
    assert c.queue[6].qubits == (0,)
    assert isinstance(c.queue[7], gates.TDG)
    assert c.queue[7].qubits == (1,)
    assert isinstance(c.queue[8], gates.I)
    assert c.queue[8].qubits == (0,)


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
    assert c.depth == 3
    assert isinstance(c.queue[0], gates.CNOT)
    assert c.queue[0].qubits == (0, 2)
    assert isinstance(c.queue[1], gates.X)
    assert c.queue[1].qubits == (1,)
    assert isinstance(c.queue[2], gates.SWAP)
    assert c.queue[2].qubits == (0, 4)
    assert isinstance(c.queue[3], gates.TOFFOLI)
    assert c.queue[3].qubits == (2, 4, 3)


def test_from_qasm_ugates():
    target = """OPENQASM 2.0;
qreg q[2];
u1(0.1) q[0];
u2(0.2,0.6) q[1];
cu3(0.3,0.4,0.5) q[0],q[1];"""
    c = Circuit.from_qasm(target)
    assert c.depth == 2
    assert isinstance(c.queue[0], gates.U1)
    assert isinstance(c.queue[1], gates.U2)
    assert isinstance(c.queue[2], gates.CU3)
    assert c.queue[0].parameters == 0.1
    assert c.queue[1].parameters == (0.2, 0.6)
    assert c.queue[2].parameters == (0.3, 0.4, 0.5)


def test_from_qasm_crotations():
    target = """OPENQASM 2.0;
qreg q[2];
crx(0.1) q[0],q[1];
crz(0.3) q[1],q[0];
cry(0.2) q[0],q[1];"""
    c = Circuit.from_qasm(target)
    assert c.depth == 3
    assert isinstance(c.queue[0], gates.CRX)
    assert isinstance(c.queue[1], gates.CRZ)
    assert isinstance(c.queue[2], gates.CRY)
    assert c.queue[0].parameters == 0.1
    assert c.queue[1].parameters == 0.3
    assert c.queue[2].parameters == 0.2


def test_from_qasm_parametrized_gates():
    target = """OPENQASM 2.0;
qreg q[2];
rx(0.1234) q[0];
rz(0.4321) q[1];
cu1(0.567) q[0],q[1];"""
    c = Circuit.from_qasm(target)
    assert c.depth == 2
    assert isinstance(c.queue[0], gates.RX)
    assert isinstance(c.queue[1], gates.RZ)
    assert isinstance(c.queue[2], gates.CU1)
    assert c.queue[0].parameters == 0.1234
    assert c.queue[1].parameters == 0.4321
    assert c.queue[2].parameters == 0.567


def test_from_qasm_invalid_script():
    # Missing starting line
    target = """qreg q[2];
x q[1];"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)

    # Qubit index out of range
    target = """OPENQASM 2.0;
qreg q[2];
x q[2];
"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)

    # Invalid qubit index
    target = """OPENQASM 2.0;
qreg q[a];
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


def test_from_qasm_measurements(backend):
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


def test_from_qasm_measurements_order(backend):
    target = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg a[3];
creg b[2];
measure q[2] -> b[1];
measure q[3] -> a[1];
measure q[4] -> a[0];
measure q[1] -> a[2];
measure q[0] -> b[0];
"""
    c = Circuit.from_qasm(target)
    assert c.measurement_tuples == {"a": (4, 3, 1), "b": (0, 2)}


def test_from_qasm_invalid_measurements(backend):
    # Undefined qubit
    target = """OPENQASM 2.0;
qreg q[2];
creg a[2];
measure q[2] -> a[0];"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)

    # Undefined register
    target = """OPENQASM 2.0;
qreg q[2];
creg a[2];
measure q[0] -> b[0];"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)

    # Register index out of range
    target = """OPENQASM 2.0;
qreg q[2];
creg a[2];
measure q[0] -> a[2];"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)

    # Reuse measured qubit
    target = """OPENQASM 2.0;
qreg q[2];
creg a[2];
measure q[0] -> a[0];
x q[1];
measure q[1] -> a[1];"""
    # Note that in this example the full register measurement is added during
    # the first `measurement` call
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)

    # Reuse classical register
    target = """OPENQASM 2.0;
qreg q[2];
creg a[2];
measure q[0] -> a[1];
measure q[1] -> a[1];"""
    with pytest.raises(KeyError):
        c = Circuit.from_qasm(target)

    # Invalid measurement command
    target = """OPENQASM 2.0;
qreg q[2];
creg a[2];
measure q[0] -> a[1] -> a[0];"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)


def test_from_qasm_invalid_parametrized_gates():
    # Parametrize non-parametrized gate
    target = """OPENQASM 2.0;
qreg q[2];
x(0.1234) q[0];
"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)

    # Failure to give theta value for parametrized gate
    target = """OPENQASM 2.0;
qreg q[2];
rx q[0];
"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)

    # Invalid parameter value
    target = """OPENQASM 2.0;
qreg q[2];
rx(0.123a) q[0];
"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)

    # Invalid parenthesis use
    target = """OPENQASM 2.0;
qreg q[2];
rx(0.123)(0.25)(0) q[0];
"""
    with pytest.raises(ValueError):
        c = Circuit.from_qasm(target)


def test_from_qasm_empty_spaces(backend):
    target = """OPENQASM 2.0; qreg q[2];
creg a[2]; h q[0];x q[1]; cx q[0], q[1];
measure q[0] -> a[0];measure q[1]->a[1]"""
    c = Circuit.from_qasm(target)
    assert c.nqubits == 2
    assert c.depth == 2
    assert isinstance(c.queue[0], gates.H)
    assert isinstance(c.queue[1], gates.X)
    assert isinstance(c.queue[2], gates.CNOT)
    assert c.measurement_tuples == {"a": (0, 1)}
