import itertools

import networkx as nx
import numpy as np
import pytest

from qibo import gates
from qibo.backends import NumpyBackend
from qibo.models import Circuit
from qibo.quantum_info.random_ensembles import random_unitary
from qibo.transpiler._exceptions import ConnectivityError
from qibo.transpiler.blocks import Block
from qibo.transpiler.optimizer import Preprocessing
from qibo.transpiler.pipeline import (
    assert_circuit_equivalence,
    restrict_connectivity_qubits,
)
from qibo.transpiler.placer import (
    Custom,
    Random,
    StarConnectivityPlacer,
    Subgraph,
    Trivial,
    assert_placement,
)
from qibo.transpiler.router import (
    CircuitMap,
    Sabre,
    ShortestPaths,
    StarConnectivityRouter,
    assert_connectivity,
)


def star_connectivity(names=["q0", "q1", "q2", "q3", "q4"], middle_qubit_idx=2):
    chip = nx.Graph()
    chip.add_nodes_from(names)
    graph_list = [
        (names[i], names[middle_qubit_idx])
        for i in range(len(names))
        if i != middle_qubit_idx
    ]
    chip.add_edges_from(graph_list)
    return chip


def grid_connectivity(names=["q0", "q1", "q2", "q3", "q4"]):
    chip = nx.Graph()
    chip.add_nodes_from(names)
    graph_list = [
        (names[0], names[1]),
        (names[1], names[2]),
        (names[2], names[3]),
        (names[3], names[0]),
        (names[0], names[4]),
    ]
    chip.add_edges_from(graph_list)
    return chip


def line_connectivity(n, names):
    if names is None:
        names = [f"q{i}" for i in range(n)]
    chip = nx.Graph()
    chip.add_nodes_from(names)
    graph_list = [(names[i], names[i + 1]) for i in range(n - 1)]
    chip.add_edges_from(graph_list)
    return chip


def generate_random_circuit(nqubits, ngates, names=None, seed=42):
    """Generate a random circuit with RX and CZ gates."""
    np.random.seed(seed)
    one_qubit_gates = [gates.RX, gates.RY, gates.RZ, gates.X, gates.Y, gates.Z, gates.H]
    two_qubit_gates = [
        gates.CNOT,
        gates.CZ,
        gates.SWAP,
        gates.iSWAP,
        gates.CRX,
        gates.CRY,
        gates.CRZ,
    ]
    n1, n2 = len(one_qubit_gates), len(two_qubit_gates)
    n = n1 + n2 if nqubits > 1 else n1
    circuit = Circuit(nqubits, wire_names=names)
    for _ in range(ngates):
        igate = int(np.random.randint(0, n))
        if igate >= n1:
            q = tuple(np.random.randint(0, nqubits, 2))
            while q[0] == q[1]:
                q = tuple(np.random.randint(0, nqubits, 2))
            gate = two_qubit_gates[igate - n1]
        else:
            q = (np.random.randint(0, nqubits),)
            gate = one_qubit_gates[igate]
        if issubclass(gate, gates.ParametrizedGate):
            theta = 2 * np.pi * np.random.random()
            circuit.add(gate(*q, theta=theta, trainable=False))
        else:
            circuit.add(gate(*q))
    return circuit


def star_circuit():
    circuit = Circuit(5)
    for i in range(1, 5):
        circuit.add(gates.CNOT(i, 0))
    return circuit


def matched_circuit(names):
    """Return a simple circuit that can be executed on star connectivity"""
    circuit = Circuit(5, wire_names=names)
    circuit.add(gates.CZ(0, 2))
    circuit.add(gates.CZ(1, 2))
    circuit.add(gates.Z(1))
    circuit.add(gates.CZ(2, 1))
    circuit.add(gates.M(0))
    return circuit


def test_assert_connectivity():
    names = ["A", "B", "C", "D", "E"]
    assert_connectivity(star_connectivity(names), matched_circuit(names))


def test_assert_connectivity_false():
    circuit = Circuit(5)
    circuit.add(gates.CZ(0, 1))
    with pytest.raises(ConnectivityError):
        assert_connectivity(star_connectivity(), circuit)


def test_assert_connectivity_3q():
    circuit = Circuit(5)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    with pytest.raises(ConnectivityError):
        assert_connectivity(star_connectivity(), circuit)


def test_bell_state_3q():
    from qibo.transpiler.pipeline import _transpose_qubits

    circuit = Circuit(3)
    circuit.add(gates.H(0))
    circuit.add(gates.CNOT(0, 2))
    circuit.add(gates.X(0))
    circuit.add(gates.M(0, 1, 2))

    c = circuit.copy()
    connectivity = line_connectivity(3, None)
    router = Sabre(connectivity=connectivity)
    routed_circuit, final_map = router(c)

    backend = NumpyBackend()
    state = np.array([1, 0, 0, 0, 0, 0, 0, 0])
    target_state = backend.execute_circuit(routed_circuit, state).state()
    target_state = _transpose_qubits(target_state, list(final_map.values()))
    assert np.all(
        np.isclose(
            np.real(target_state).round(1), np.array([0, 0.7, 0, 0, 0.7, 0, 0, 0])
        )
    )


@pytest.mark.parametrize("ngates", [5, 25])
@pytest.mark.parametrize("placer", [Trivial, Random])
@pytest.mark.parametrize("connectivity", [star_connectivity(), grid_connectivity()])
def test_random_circuits_5q(ngates, placer, connectivity):
    placer = placer(connectivity=connectivity)
    transpiler = ShortestPaths(connectivity=connectivity)

    circuit = generate_random_circuit(nqubits=5, ngates=ngates)
    original_circuit = circuit.copy()
    placer(circuit)
    transpiled_circuit, final_qubit_map = transpiler(circuit)

    assert transpiler.added_swaps >= 0
    assert_connectivity(connectivity, transpiled_circuit)
    assert_placement(transpiled_circuit, connectivity)
    assert ngates + transpiler.added_swaps == transpiled_circuit.ngates
    assert_circuit_equivalence(
        original_circuit=original_circuit,
        transpiled_circuit=transpiled_circuit,
        final_map=final_qubit_map,
    )


@pytest.mark.parametrize("nqubits", [11, 12, 13, 14, 15])
@pytest.mark.parametrize("ngates", [30, 50])
def test_random_circuits_15q_50g(nqubits, ngates):
    connectivity = line_connectivity(nqubits, None)
    placer = Random(connectivity=connectivity)
    transpiler = Sabre(connectivity=connectivity)
    circuit = generate_random_circuit(nqubits=nqubits, ngates=ngates)
    original_circuit = circuit.copy()

    placer(circuit)
    transpiled_circuit, final_qubit_map = transpiler(circuit)

    assert transpiler.added_swaps >= 0
    assert_connectivity(connectivity, transpiled_circuit)
    assert_placement(transpiled_circuit, connectivity)
    assert ngates + transpiler.added_swaps == transpiled_circuit.ngates
    assert_circuit_equivalence(
        original_circuit=original_circuit,
        transpiled_circuit=transpiled_circuit,
        final_map=final_qubit_map,
    )


def test_star_circuit():
    connectivity = star_connectivity()
    circuit = star_circuit()
    placer = Subgraph(connectivity=connectivity)
    transpiler = ShortestPaths(connectivity=connectivity)

    placer(circuit)
    transpiled_circuit, final_qubit_map = transpiler(circuit)

    assert transpiler.added_swaps == 0
    assert_connectivity(star_connectivity(), transpiled_circuit)
    assert_placement(transpiled_circuit, connectivity)
    assert_circuit_equivalence(
        original_circuit=star_circuit(),
        transpiled_circuit=transpiled_circuit,
        final_map=final_qubit_map,
    )


def test_star_circuit_custom_map():
    connectivity = star_connectivity()
    circuit = star_circuit()
    placer = Custom(
        initial_map=["q1", "q0", "q2", "q3", "q4"], connectivity=connectivity
    )
    transpiler = ShortestPaths(connectivity=connectivity)

    placer(circuit)
    transpiled_circuit, final_qubit_map = transpiler(circuit)

    assert transpiler.added_swaps == 1
    assert_connectivity(star_connectivity(), transpiled_circuit)
    assert_placement(transpiled_circuit, connectivity)
    assert_circuit_equivalence(
        original_circuit=star_circuit(),
        transpiled_circuit=transpiled_circuit,
        final_map=final_qubit_map,
    )


def test_routing_with_measurements():
    connectivity = star_connectivity()
    circuit = Circuit(5)
    circuit.add(gates.CNOT(0, 1))
    circuit.add(gates.M(0, 2, 3))
    placer = Trivial(connectivity)
    transpiler = ShortestPaths(connectivity)

    placer(circuit=circuit)
    transpiled_circuit, final_qubit_map = transpiler(circuit)

    assert transpiled_circuit.ngates == 3
    measured_qubits = transpiled_circuit.queue[2].qubits
    assert measured_qubits == (0, 1, 3)
    assert_circuit_equivalence(
        original_circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        final_map=final_qubit_map,
    )


def test_sabre_looping():
    # Setup where the looping occurs
    # Line connectivity, gates with gate_array, Trivial placer

    connectivity = line_connectivity(10, None)
    gate_array = [(7, 2), (6, 0), (5, 6), (4, 8), (3, 5), (9, 1)]
    loop_circ = Circuit(10)
    for qubits in gate_array:
        loop_circ.add(gates.CZ(*qubits))

    placer = Trivial(connectivity)
    router_no_threshold = Sabre(
        connectivity=connectivity, swap_threshold=np.inf
    )  # Without reset
    router_threshold = Sabre(connectivity=connectivity)  # With reset

    placer(loop_circ)
    routed_no_threshold, final_mapping_no_threshold = router_no_threshold(loop_circ)
    routed_threshold, final_mapping_threshold = router_threshold(loop_circ)

    count_no_threshold = router_no_threshold.added_swaps
    count_threshold = router_threshold.added_swaps

    assert count_no_threshold > count_threshold
    assert_circuit_equivalence(
        original_circuit=loop_circ,
        transpiled_circuit=routed_no_threshold,
        final_map=final_mapping_no_threshold,
    )
    assert_circuit_equivalence(
        original_circuit=loop_circ,
        transpiled_circuit=routed_threshold,
        final_map=final_mapping_threshold,
    )


def test_sabre_shortest_path_routing():
    gate_array = [(0, 9), (5, 9), (2, 8)]  # The gate (2, 8) should be routed next

    loop_circ = Circuit(10)
    for qubits in gate_array:
        loop_circ.add(gates.CZ(*qubits))

    connectivity = line_connectivity(10, None)

    placer = Trivial(connectivity)
    router = Sabre(connectivity)

    placer(loop_circ)
    router._preprocessing(circuit=loop_circ)
    router._shortest_path_routing()  # q2 should be moved adjacent to q8

    gate_28 = router.circuit_map.circuit_blocks.block_list[2]
    gate_28_qubits = router.circuit_map.get_physical_qubits(gate_28)

    # Check if the physical qubits of the gate (2, 8) are adjacent
    assert gate_28_qubits[1] in list(router.connectivity.neighbors(gate_28_qubits[0]))
    assert gate_28_qubits[0] in list(router.connectivity.neighbors(gate_28_qubits[1]))


def test_circuit_map():
    circ = Circuit(4)
    circ.add(gates.H(1))
    circ.add(gates.H(0))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.H(0))
    circ.add(gates.CZ(1, 2))
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(2, 3))
    circ.wire_names = ["q1", "q2", "q0", "q3"]

    circuit_map = CircuitMap(circuit=circ)
    block_list = circuit_map.circuit_blocks
    # test blocks_qubits_pairs
    assert circuit_map.blocks_logical_qubits_pairs() == [(0, 1), (1, 2), (0, 1), (2, 3)]
    # test execute_block and routed_circuit
    circuit_map.execute_block(block_list.search_by_index(0))
    routed_circuit = circuit_map.routed_circuit(circuit_kwargs=circ.init_kwargs)
    assert isinstance(routed_circuit.queue[0], gates.H)
    assert len(routed_circuit.queue) == 4
    qubits = routed_circuit.queue[2].qubits
    assert (
        routed_circuit.wire_names[qubits[0]] == "q1"
        and routed_circuit.wire_names[qubits[1]] == "q2"
    )

    # test update 1
    circuit_map.update((0, 2))
    routed_circuit = circuit_map.routed_circuit(circuit_kwargs=circ.init_kwargs)
    assert isinstance(routed_circuit.queue[4], gates.SWAP)
    qubits = routed_circuit.queue[4].qubits
    assert (
        routed_circuit.wire_names[qubits[0]] == "q1"
        and routed_circuit.wire_names[qubits[1]] == "q0"
    )
    assert circuit_map._swaps == 1
    assert circuit_map.physical_to_logical == [2, 1, 0, 3]
    assert circuit_map.logical_to_physical == [2, 1, 0, 3]

    # test update 2
    circuit_map.update((1, 2))
    routed_circuit = circuit_map.routed_circuit(circuit_kwargs=circ.init_kwargs)
    assert isinstance(routed_circuit.queue[5], gates.SWAP)
    qubits = routed_circuit.queue[5].qubits
    assert (
        routed_circuit.wire_names[qubits[0]] == "q2"
        and routed_circuit.wire_names[qubits[1]] == "q1"
    )
    assert circuit_map._swaps == 2
    assert circuit_map.physical_to_logical == [1, 2, 0, 3]
    assert circuit_map.logical_to_physical == [2, 0, 1, 3]

    # test execute_block after multiple swaps
    circuit_map.execute_block(block_list.search_by_index(1))
    circuit_map.execute_block(block_list.search_by_index(2))
    circuit_map.execute_block(block_list.search_by_index(3))
    routed_circuit = circuit_map.routed_circuit(circuit_kwargs=circ.init_kwargs)
    assert isinstance(routed_circuit.queue[6], gates.CZ)

    qubits = routed_circuit.queue[6].qubits
    assert (
        routed_circuit.wire_names[qubits[0]] == "q1"
        and routed_circuit.wire_names[qubits[1]] == "q2"
    )
    qubits = routed_circuit.queue[7].qubits
    assert (
        routed_circuit.wire_names[qubits[0]] == "q0"
        and routed_circuit.wire_names[qubits[1]] == "q1"
    )
    qubits = routed_circuit.queue[8].qubits
    assert (
        routed_circuit.wire_names[qubits[0]] == "q2"
        and routed_circuit.wire_names[qubits[1]] == "q3"
    )
    assert len(circuit_map.circuit_blocks()) == 0
    # test final layout
    assert circuit_map.final_layout() == {"q0": 1, "q1": 2, "q2": 0, "q3": 3}


@pytest.mark.parametrize(
    "names",
    [["q0", "q1", "q2", "q3", "q4"], [0, 1, 2, 3, 4], ["A", "B", "C", "D", "E"]],
)
def test_sabre_matched(names):
    connectivity = star_connectivity(names=names)
    circuit = matched_circuit(names)
    original_circuit = circuit.copy()
    placer = Trivial()
    router = Sabre(connectivity=connectivity)

    placer(circuit)
    routed_circuit, final_map = router(circuit)

    assert router.added_swaps == 0
    assert_connectivity(circuit=routed_circuit, connectivity=connectivity)
    assert_circuit_equivalence(
        original_circuit=original_circuit,
        transpiled_circuit=routed_circuit,
        final_map=final_map,
    )


@pytest.mark.parametrize("seed", [42])
def test_sabre_simple(seed):
    connectivity = star_connectivity()
    circ = Circuit(5)
    circ.add(gates.CZ(0, 1))
    original_circuit = circ.copy()
    placer = Trivial()
    router = Sabre(connectivity=connectivity, seed=seed)

    placer(circ)
    routed_circuit, final_map = router(circ)

    assert router.added_swaps == 1
    assert final_map == {"q0": 2, "q1": 1, "q2": 0, "q3": 3, "q4": 4}
    assert routed_circuit.queue[0].qubits == (0, 2)
    assert isinstance(routed_circuit.queue[0], gates.SWAP)
    assert isinstance(routed_circuit.queue[1], gates.CZ)
    assert_connectivity(circuit=routed_circuit, connectivity=connectivity)
    assert_circuit_equivalence(
        original_circuit=original_circuit,
        transpiled_circuit=routed_circuit,
        final_map=final_map,
    )


@pytest.mark.parametrize("n_gates", [10, 40])
@pytest.mark.parametrize("look", [0, 5])
@pytest.mark.parametrize("decay", [0.5, 1.0])
@pytest.mark.parametrize("placer_param", [Trivial, Random])
@pytest.mark.parametrize("connectivity", [star_connectivity(), grid_connectivity()])
def test_sabre_random_circuits(n_gates, look, decay, placer_param, connectivity):
    circuit = generate_random_circuit(nqubits=5, ngates=n_gates)
    measurement = gates.M(*range(5))
    circuit.add(measurement)
    original_circuit = circuit.copy()
    placer = placer_param(connectivity=connectivity)
    router = Sabre(connectivity=connectivity, lookahead=look, decay_lookahead=decay)

    placer(circuit)
    transpiled_circuit, final_qubit_map = router(circuit)

    assert router.added_swaps >= 0
    assert_connectivity(connectivity, transpiled_circuit)
    assert_placement(transpiled_circuit, connectivity)
    assert n_gates + router.added_swaps + 1 == transpiled_circuit.ngates
    assert_circuit_equivalence(
        original_circuit=original_circuit,
        transpiled_circuit=transpiled_circuit,
        final_map=final_qubit_map,
    )
    assert transpiled_circuit.queue[-1].register_name == measurement.register_name


def test_sabre_memory_map():
    placer = Trivial()
    layout_circ = Circuit(5)
    placer(layout_circ)
    router = Sabre(connectivity=star_connectivity())
    router._preprocessing(circuit=star_circuit())
    router._memory_map = [[1, 0, 2, 3, 4]]
    value = router._compute_cost((0, 1))
    assert value == float("inf")


def test_sabre_intermediate_measurements():
    measurement = gates.M(1)
    circ = Circuit(3, density_matrix=True)
    circ.add(gates.H(0))
    circ.add(measurement)
    circ.add(gates.CNOT(0, 2))
    connectivity = nx.Graph()
    connectivity.add_nodes_from([0, 1, 2])
    connectivity.add_edges_from([(0, 1), (1, 2)])
    router = Sabre(connectivity=connectivity)
    routed_circ, _ = router(circuit=circ)
    assert routed_circ.queue[3].register_name == measurement.register_name


@pytest.mark.parametrize("router_algorithm", [Sabre, ShortestPaths])
def test_restrict_qubits(router_algorithm):
    circ = Circuit(3)
    circ.add(gates.CZ(0, 1))
    circ.add(gates.CZ(0, 2))
    circ.add(gates.CZ(2, 1))
    circ.wire_names = ["q0", "q3", "q2"]
    connectivity = star_connectivity()
    restricted_connectivity = restrict_connectivity_qubits(
        connectivity, ["q0", "q2", "q3"]
    )
    router = router_algorithm(connectivity=restricted_connectivity)
    routed_circ, final_layout = router(circuit=circ)
    assert_circuit_equivalence(
        original_circuit=circ,
        transpiled_circuit=routed_circ,
        final_map=final_layout,
    )
    assert_connectivity(restricted_connectivity, routed_circ)
    assert_placement(routed_circ, connectivity=restricted_connectivity)
    assert routed_circ.wire_names == ["q0", "q3", "q2"]


def test_star_error_multi_qubit():
    circuit = Circuit(3)
    circuit.add(gates.TOFFOLI(0, 1, 2))
    connectivity = star_connectivity(middle_qubit_idx=2)
    transpiler = StarConnectivityRouter(connectivity)
    with pytest.raises(ConnectivityError):
        transpiled, hardware_qubits = transpiler(circuit=circuit)


# @pytest.mark.parametrize("nqubits", [1, 3, 5])
@pytest.mark.parametrize("nqubits", [5])
@pytest.mark.parametrize("middle_qubit", [0, 2, 4])
@pytest.mark.parametrize("depth", [2, 10])
@pytest.mark.parametrize("measurements", [True, False])
@pytest.mark.parametrize("unitaries", [True, False])
@pytest.mark.parametrize(
    "names",
    [["q0", "q1", "q2", "q3", "q4"], [0, 1, 2, 3, 4], ["A", "B", "C", "D", "E"]],
)
def test_star_router(nqubits, depth, middle_qubit, measurements, unitaries, names):
    unitary_dim = min(2, nqubits)
    connectivity = star_connectivity(names=names, middle_qubit_idx=middle_qubit)
    if unitaries:
        circuit = Circuit(nqubits, wire_names=names)
        pairs = list(itertools.combinations(range(nqubits), unitary_dim))
        for _ in range(depth):
            qubits = pairs[int(np.random.randint(len(pairs)))]
            circuit.add(
                gates.Unitary(
                    random_unitary(2**unitary_dim, backend=NumpyBackend()), *qubits
                )
            )
    else:
        circuit = generate_random_circuit(nqubits, depth, names)
    if measurements:
        circuit.add(gates.M(0))
    original_circuit = circuit.copy()
    transpiler = StarConnectivityRouter(connectivity)
    placer = StarConnectivityPlacer(connectivity)

    placer(circuit)
    transpiled_circuit, final_qubit_map = transpiler(circuit)
    assert_connectivity(connectivity, transpiled_circuit)
    assert_placement(transpiled_circuit, connectivity)
    assert_circuit_equivalence(
        original_circuit=original_circuit,
        transpiled_circuit=transpiled_circuit,
        final_map=final_qubit_map,
    )


def test_undo():
    circ = Circuit(4)
    circ.wire_names = ["q0", "q1", "q2", "q3"]
    circuit_map = CircuitMap(circuit=circ)

    # Two SWAP gates are added
    circuit_map.update((1, 2))
    circuit_map.update((2, 3))
    assert circuit_map.physical_to_logical == [0, 3, 1, 2]
    assert circuit_map.logical_to_physical == [0, 2, 3, 1]
    assert len(circuit_map._routed_blocks.block_list) == 2
    assert circuit_map._swaps == 2

    # Undo the last SWAP gate
    circuit_map.undo()
    assert circuit_map.physical_to_logical == [0, 2, 1, 3]
    assert circuit_map.logical_to_physical == [0, 2, 1, 3]
    assert circuit_map._swaps == 1
    assert len(circuit_map._routed_blocks.block_list) == 1

    # Undo the first SWAP gate
    circuit_map.undo()
    assert circuit_map.physical_to_logical == [0, 1, 2, 3]
    assert circuit_map.logical_to_physical == [0, 1, 2, 3]
    assert circuit_map._swaps == 0
    assert len(circuit_map._routed_blocks.block_list) == 0


def test_circuitmap_no_circuit():
    # If a `CircuitMap` is not a temporary instance and is created without a circuit, it should raise an error.
    with pytest.raises(ValueError):
        circuit_map = CircuitMap()


def test_logical_to_physical_setter():
    circ = Circuit(4)
    circ.wire_names = ["q0", "q3", "q2", "q1"]
    circuit_map = CircuitMap(circuit=circ)
    circuit_map.logical_to_physical = [2, 0, 1, 3]
    assert circuit_map.logical_to_physical == [2, 0, 1, 3]
    assert circuit_map.physical_to_logical == [1, 2, 0, 3]
