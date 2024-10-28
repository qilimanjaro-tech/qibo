import pytest

from qibo import gates
from qibo.models.circuit import Circuit
from qibo.transpiler._exceptions import (
    ConnectivityError,
    DecompositionError,
    PlacementError,
    TranspilerPipelineError,
)
from qibo.transpiler.pipeline import restrict_connectivity_qubits
from qibo.transpiler.unroller import NativeGates
from qibo.transpiler.utils import (
    assert_circuit_equivalence,
    assert_connectivity,
    assert_decomposition,
    assert_mapping_consistency,
    assert_placement,
    assert_qubit_match,
)


def test_assert_qubit_match(star_connectivity):
    circuit = Circuit(3)
    with pytest.raises(TranspilerPipelineError):
        assert_qubit_match(circuit, star_connectivity())

    circuit = Circuit(5)
    assert_qubit_match(circuit, star_connectivity())


def test_assert_circuit_equivalence_equal():
    circ1 = Circuit(2)
    circ2 = Circuit(2)
    circ1.add(gates.X(0))
    circ1.add(gates.CZ(0, 1))
    circ2.add(gates.X(0))
    circ2.add(gates.CZ(0, 1))
    final_map = {0: 0, 1: 1}
    assert_circuit_equivalence(circ1, circ2, final_map=final_map)


def test_assert_circuit_equivalence_swap():
    circ1 = Circuit(2)
    circ2 = Circuit(2)
    circ1.add(gates.X(0))
    circ2.add(gates.SWAP(0, 1))
    circ2.add(gates.X(1))
    final_map = {0: 1, 1: 0}
    assert_circuit_equivalence(circ1, circ2, final_map=final_map)


def test_assert_circuit_equivalence_false():
    circ1 = Circuit(2)
    circ2 = Circuit(2)
    circ1.add(gates.X(0))
    circ2.add(gates.SWAP(0, 1))
    circ2.add(gates.X(1))
    final_map = {0: 0, 1: 1}
    with pytest.raises(TranspilerPipelineError):
        assert_circuit_equivalence(circ1, circ2, final_map=final_map)


@pytest.mark.parametrize("names", [["A", "B", "C", "D", "E"], [0, 1, 2, 3, 4]])
def test_mapping_consistency(names, star_connectivity):
    assert_mapping_consistency(names, star_connectivity(names))


def test_mapping_consistency_error(star_connectivity):
    with pytest.raises(PlacementError):
        assert_mapping_consistency(["A", "B", "C", "D", "F"], star_connectivity())


@pytest.mark.parametrize("names", [["A", "B", "C", "D", "E"], [0, 1, 2, 3, 4]])
def test_mapping_consistency_restricted(names, star_connectivity):
    connectivity = star_connectivity(names)
    on_qubit = [names[0], names[2]]
    restricted_connectivity = restrict_connectivity_qubits(connectivity, on_qubit)
    assert_mapping_consistency(on_qubit, restricted_connectivity)


@pytest.mark.parametrize("names", [["A", "B", "C", "D", "E"], [0, 1, 2, 3, 4]])
def test_mapping_consistency_restricted_error(names, star_connectivity):
    connectivity = star_connectivity(names)
    on_qubit = [names[0], names[2]]
    restricted_connectivity = restrict_connectivity_qubits(connectivity, on_qubit)
    with pytest.raises(PlacementError):
        assert_mapping_consistency([names[3], names[4]], restricted_connectivity)


def test_assert_placement_true(star_connectivity):
    circuit = Circuit(5)
    assert_placement(circuit, connectivity=star_connectivity())


@pytest.mark.parametrize(
    "qubits, names", [(5, ["A", "B", "C", "D", "F"]), (3, ["A", "B", "C"])]
)
def test_assert_placement_false(qubits, names, star_connectivity):
    connectivity = star_connectivity()
    circuit = Circuit(qubits, wire_names=names)
    with pytest.raises(PlacementError):
        assert_placement(circuit, connectivity)


@pytest.mark.parametrize("qubits", [10, 1])
def test_assert_placement_error(qubits, star_connectivity):
    connectivity = star_connectivity()
    circuit = Circuit(qubits)
    with pytest.raises(PlacementError):
        assert_placement(circuit, connectivity)


def test_assert_decomposition():
    circuit = Circuit(2)
    circuit.add(gates.CZ(0, 1))
    circuit.add(gates.Z(0))
    circuit.add(gates.M(1))
    assert_decomposition(circuit, native_gates=NativeGates.default())


def test_assert_decomposition_fail_1q():
    circuit = Circuit(1)
    circuit.add(gates.X(0))
    with pytest.raises(DecompositionError):
        assert_decomposition(circuit, native_gates=NativeGates.default())


@pytest.mark.parametrize("gate", [gates.CNOT(0, 1), gates.iSWAP(0, 1)])
def test_assert_decomposition_fail_2q(gate):
    circuit = Circuit(2)
    circuit.add(gate)
    with pytest.raises(DecompositionError):
        assert_decomposition(circuit, native_gates=NativeGates.default())


def test_assert_decomposition_fail_3q():
    circuit = Circuit(3)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    with pytest.raises(DecompositionError):
        assert_decomposition(circuit, native_gates=NativeGates.default())


def test_assert_connectivity(star_connectivity):
    names = ["A", "B", "C", "D", "E"]
    circuit = Circuit(5, wire_names=names)
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.CZ(2, 1))
    assert_connectivity(star_connectivity(names), circuit)


def test_assert_connectivity_false(star_connectivity):
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 1))
    with pytest.raises(ConnectivityError):
        assert_connectivity(star_connectivity(), circuit)


def test_assert_connectivity_3q(star_connectivity):
    circuit = Circuit(5)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    with pytest.raises(ConnectivityError):
        assert_connectivity(star_connectivity(), circuit)
