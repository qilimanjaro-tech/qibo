import numpy as np
import pytest
from scipy.linalg import sqrtm

from qibo.config import PRECISION_TOL
from qibo.quantum_info.entropies import (
    classical_relative_entropy,
    classical_relative_renyi_entropy,
    classical_renyi_entropy,
    classical_tsallis_entropy,
    entanglement_entropy,
    entropy,
    relative_entropy,
    relative_renyi_entropy,
    renyi_entropy,
    shannon_entropy,
    tsallis_entropy,
)
from qibo.quantum_info.random_ensembles import (
    random_density_matrix,
    random_statevector,
    random_unitary,
)


def test_shannon_entropy_errors(backend):
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, -2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([[1.0], [0.0]])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, -1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.1, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([0.5, 0.4999999])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = shannon_entropy(prob, backend=backend)


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_shannon_entropy(backend, base):
    prob_array = [1.0, 0.0]
    result = shannon_entropy(prob_array, base, backend=backend)
    backend.assert_allclose(result, 0.0)

    if base == 2:
        prob_array = np.array([0.5, 0.5])
        prob_array = backend.cast(prob_array, dtype=prob_array.dtype)
        result = shannon_entropy(prob_array, base, backend=backend)
        backend.assert_allclose(result, 1.0)


@pytest.mark.parametrize("kind", [None, list])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_classical_relative_entropy(backend, base, kind):
    with pytest.raises(TypeError):
        prob = np.random.rand(1, 2)
        prob_q = np.random.rand(1, 5)
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_entropy(prob, prob_q, backend=backend)
    with pytest.raises(TypeError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_entropy(prob, prob_q, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([-1, 2.0])
        prob_q = np.random.rand(1, 5)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_entropy(prob, prob_q, backend=backend)
    with pytest.raises(ValueError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_entropy(prob, prob_q, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.random.rand(1, 2)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_entropy(prob, prob_q, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.array([0.0, 1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_entropy(prob, prob_q, base=-2, backend=backend)

    prob_p = np.random.rand(10)
    prob_q = np.random.rand(10)
    prob_p /= np.sum(prob_p)
    prob_q /= np.sum(prob_q)

    target = np.sum(prob_p * np.log(prob_p) / np.log(base)) - np.sum(
        prob_p * np.log(prob_q) / np.log(base)
    )

    if kind is not None:
        prob_p, prob_q = kind(prob_p), kind(prob_q)

    divergence = classical_relative_entropy(prob_p, prob_q, base=base, backend=backend)

    backend.assert_allclose(divergence, target, atol=1e-5)


@pytest.mark.parametrize("kind", [None, list])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 1, 2, 3, np.inf])
def test_classical_renyi_entropy(backend, alpha, base, kind):
    with pytest.raises(TypeError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha="2", backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha=-2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha, base="2", backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha, base=-2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([[1.0], [0.0]])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, -1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.1, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([0.5, 0.4999999])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_renyi_entropy(prob, alpha, backend=backend)

    prob_dist = np.random.rand(10)
    prob_dist /= np.sum(prob_dist)

    if alpha == 0.0:
        target = np.log2(len(prob_dist)) / np.log2(base)
    elif alpha == 1:
        target = shannon_entropy(prob_dist, base=base, backend=backend)
    elif alpha == 2:
        target = -1 * np.log2(np.sum(prob_dist**2)) / np.log2(base)
    elif alpha == np.inf:
        target = -1 * np.log2(max(prob_dist)) / np.log2(base)
    else:
        target = (1 / (1 - alpha)) * np.log2(np.sum(prob_dist**alpha)) / np.log2(base)

    if kind is not None:
        prob_dist = kind(prob_dist)

    renyi_ent = classical_renyi_entropy(prob_dist, alpha, base=base, backend=backend)

    backend.assert_allclose(renyi_ent, target, atol=1e-5)


@pytest.mark.parametrize("kind", [None, list])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 1 / 2, 1, 2, 3, np.inf])
def test_classical_relative_renyi_entropy(backend, alpha, base, kind):
    with pytest.raises(TypeError):
        prob = np.random.rand(1, 2)
        prob_q = np.random.rand(1, 5)
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_renyi_entropy(
            prob, prob_q, alpha, base, backend=backend
        )
    with pytest.raises(TypeError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_renyi_entropy(
            prob, prob_q, alpha, base, backend=backend
        )
    with pytest.raises(ValueError):
        prob = np.array([-1, 2.0])
        prob_q = np.random.rand(1, 5)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_renyi_entropy(
            prob, prob_q, alpha, base, backend=backend
        )
    with pytest.raises(ValueError):
        prob = np.random.rand(1, 2)[0]
        prob_q = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_renyi_entropy(
            prob, prob_q, alpha, base, backend=backend
        )
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.random.rand(1, 2)[0]
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_renyi_entropy(
            prob, prob_q, alpha, base, backend=backend
        )
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.array([0.0, 1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_renyi_entropy(
            prob, prob_q, alpha, base=-2, backend=backend
        )
    with pytest.raises(TypeError):
        prob = np.array([1.0, 0.0])
        prob_q = np.array([0.0, 1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_renyi_entropy(
            prob, prob_q, alpha="1", base=base, backend=backend
        )
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob_q = np.array([0.0, 1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        prob_q = backend.cast(prob_q, dtype=prob_q.dtype)
        test = classical_relative_renyi_entropy(
            prob, prob_q, alpha=-2, base=base, backend=backend
        )

    prob_p = np.random.rand(10)
    prob_q = np.random.rand(10)
    prob_p /= np.sum(prob_p)
    prob_q /= np.sum(prob_q)

    if alpha == 0.5:
        target = -2 * np.log2(np.sum(np.sqrt(prob_p * prob_q))) / np.log2(base)
    elif alpha == 1.0:
        target = classical_relative_entropy(prob_p, prob_q, base=base, backend=backend)
    elif alpha == np.inf:
        target = np.log2(max(prob_p / prob_q)) / np.log2(base)
    else:
        target = (
            (1 / (alpha - 1))
            * np.log2(np.sum(prob_p**alpha * prob_q ** (1 - alpha)))
            / np.log2(base)
        )

    if kind is not None:
        prob_p, prob_q = kind(prob_p), kind(prob_q)

    divergence = classical_relative_renyi_entropy(
        prob_p, prob_q, alpha=alpha, base=base, backend=backend
    )

    backend.assert_allclose(divergence, target, atol=1e-5)


@pytest.mark.parametrize("kind", [None, list])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 1, 2, 3])
def test_classical_tsallis_entropy(backend, alpha, base, kind):
    with pytest.raises(TypeError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha="2", backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha=-2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha, base="2", backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha, base=-2, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([[1.0], [0.0]])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha, backend=backend)
    with pytest.raises(TypeError):
        prob = np.array([])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.0, -1.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([1.1, 0.0])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha, backend=backend)
    with pytest.raises(ValueError):
        prob = np.array([0.5, 0.4999999])
        prob = backend.cast(prob, dtype=prob.dtype)
        test = classical_tsallis_entropy(prob, alpha, backend=backend)

    prob_dist = np.random.rand(10)
    prob_dist /= np.sum(prob_dist)

    if alpha == 1.0:
        target = shannon_entropy(prob_dist, base=base, backend=backend)
    else:
        target = (1 / (1 - alpha)) * (np.sum(prob_dist**alpha) - 1)

    if kind is not None:
        prob_dist = kind(prob_dist)

    backend.assert_allclose(
        classical_tsallis_entropy(prob_dist, alpha=alpha, base=base, backend=backend),
        target,
        atol=1e-5,
    )


@pytest.mark.parametrize("check_hermitian", [False, True])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_entropy(backend, base, check_hermitian):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = entropy(
            state, base=base, check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(ValueError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        test = entropy(state, base=0, check_hermitian=check_hermitian, backend=backend)
    with pytest.raises(TypeError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        test = entropy(state, base=base, check_hermitian="False", backend=backend)

    if backend.__class__.__name__ in ["CupyBackend", "CuQuantumBackend"]:
        with pytest.raises(NotImplementedError):
            state = random_unitary(4, backend=backend)
            test = entropy(state, base=base, check_hermitian=True, backend=backend)
    else:
        state = random_unitary(4, backend=backend)
        test = entropy(state, base=base, check_hermitian=True, backend=backend)

    state = np.array([1.0, 0.0])
    state = backend.cast(state, dtype=state.dtype)
    backend.assert_allclose(entropy(state, backend=backend), 0.0, atol=PRECISION_TOL)

    state = np.array([1.0, 0.0, 0.0, 0.0])
    state = np.outer(state, state)
    state = backend.cast(state, dtype=state.dtype)

    nqubits = 2
    state = backend.identity_density_matrix(nqubits)
    if base == 2:
        test = 2.0
    elif base == 10:
        test = 0.6020599913279624
    elif base == np.e:
        test = 1.3862943611198906
    else:
        test = 0.8613531161467861

    backend.assert_allclose(
        entropy(state, base, check_hermitian=check_hermitian, backend=backend), test
    )


@pytest.mark.parametrize("check_hermitian", [False, True])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
def test_relative_entropy(backend, base, check_hermitian):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        target = random_density_matrix(2, pure=True, backend=backend)
        test = relative_entropy(
            state, target, base=base, check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(TypeError):
        target = np.random.rand(2, 3)
        target = backend.cast(target, dtype=target.dtype)
        state = random_density_matrix(2, pure=True, backend=backend)
        test = relative_entropy(
            state, target, base=base, check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(ValueError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        target = np.array([0.0, 1.0])
        target = backend.cast(target, dtype=target.dtype)
        test = relative_entropy(
            state, target, base=0, check_hermitian=check_hermitian, backend=backend
        )
    with pytest.raises(TypeError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        target = np.array([0.0, 1.0])
        target = backend.cast(target, dtype=target.dtype)
        test = relative_entropy(
            state, target, base=base, check_hermitian="False", backend=backend
        )

    nqubits = 2
    dims = 2**nqubits

    state = random_density_matrix(dims, backend=backend)
    target = backend.identity_density_matrix(nqubits, normalize=True)

    backend.assert_allclose(
        relative_entropy(state, target, base, check_hermitian, backend),
        np.log(dims) / np.log(base)
        - entropy(state, base=base, check_hermitian=check_hermitian, backend=backend),
        atol=1e-5,
    )

    state = backend.cast([1.0, 0.0], dtype=np.float64)
    target = backend.cast([0.0, 1.0], dtype=np.float64)

    assert relative_entropy(state, target, backend=backend) == 0.0

    # for coverage when GPUs are present
    if backend.__class__.__name__ in ["CupyBackend", "CuQuantumBackend"]:
        with pytest.raises(NotImplementedError):
            state = random_unitary(4, backend=backend)
            target = random_density_matrix(4, backend=backend)
            test = relative_entropy(
                state, target, base=base, check_hermitian=True, backend=backend
            )
        with pytest.raises(NotImplementedError):
            target = random_unitary(4, backend=backend)
            state = random_density_matrix(4, backend=backend)
            test = relative_entropy(
                state, target, base=base, check_hermitian=True, backend=backend
            )
    else:
        state = random_unitary(4, backend=backend)
        target = random_unitary(4, backend=backend)
        test = relative_entropy(
            state, target, base=base, check_hermitian=True, backend=backend
        )


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 1, 2, 3, np.inf])
def test_renyi_entropy(backend, alpha, base):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = renyi_entropy(state, alpha=alpha, base=base, backend=backend)
    with pytest.raises(TypeError):
        state = random_statevector(4, backend=backend)
        test = renyi_entropy(state, alpha="2", base=base, backend=backend)
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        test = renyi_entropy(state, alpha=-1, base=base, backend=backend)
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        test = renyi_entropy(state, alpha=alpha, base=0, backend=backend)

    state = random_density_matrix(4, backend=backend)

    if alpha == 0.0:
        target = np.log2(len(state)) / np.log2(base)
    elif alpha == 1.0:
        target = entropy(state, base=base, backend=backend)
    elif alpha == np.inf:
        target = backend.calculate_norm_density_matrix(state, order=2)
        target = -1 * np.log2(target) / np.log2(base)
    else:
        target = np.log2(np.trace(np.linalg.matrix_power(state, alpha)))
        target = (1 / (1 - alpha)) * target / np.log2(base)

    backend.assert_allclose(
        renyi_entropy(state, alpha=alpha, base=base, backend=backend), target, atol=1e-5
    )

    # test pure state
    state = random_density_matrix(4, pure=True, backend=backend)
    backend.assert_allclose(
        renyi_entropy(state, alpha=alpha, base=base, backend=backend), 0.0, atol=1e-8
    )


@pytest.mark.parametrize(
    ["state_flag", "target_flag"], [[True, True], [False, True], [True, False]]
)
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 1, 2, 3, np.inf])
def test_relative_renyi_entropy(backend, alpha, base, state_flag, target_flag):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        target = random_density_matrix(4, backend=backend)
        test = relative_renyi_entropy(
            state, target, alpha=alpha, base=base, backend=backend
        )
    with pytest.raises(TypeError):
        target = np.random.rand(2, 3)
        target = backend.cast(target, dtype=target.dtype)
        state = random_density_matrix(4, backend=backend)
        test = relative_renyi_entropy(
            state, target, alpha=alpha, base=base, backend=backend
        )
    with pytest.raises(TypeError):
        state = random_statevector(4, backend=backend)
        target = random_statevector(4, backend=backend)
        test = relative_renyi_entropy(
            state, target, alpha="2", base=base, backend=backend
        )
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        target = random_statevector(4, backend=backend)
        test = relative_renyi_entropy(
            state, target, alpha=-1, base=base, backend=backend
        )
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        target = random_statevector(4, backend=backend)
        test = relative_renyi_entropy(
            state, target, alpha=alpha, base=0, backend=backend
        )

    state = (
        random_statevector(4, backend=backend)
        if state_flag
        else random_density_matrix(4, backend=backend)
    )
    target = (
        random_statevector(4, backend=backend)
        if target_flag
        else random_density_matrix(4, backend=backend)
    )

    if alpha == 1.0:
        log = relative_entropy(state, target, base, backend=backend)
    elif alpha == np.inf:
        if backend.__class__.__name__ in ["CupyBackend", "CuQuantumBackend"]:
            eigenvalues_state, eigenvectors_state = np.linalg.eigh(state)
            new_state = np.zeros_like(state, dtype=complex)
            new_state = backend.cast(new_state, dtype=new_state.dtype)
            for eigenvalue, eigenstate in zip(
                eigenvalues_state, np.transpose(eigenvectors_state)
            ):
                new_state += np.sqrt(eigenvalue) * np.outer(
                    eigenstate, np.conj(eigenstate)
                )

            eigenvalues_target, eigenvectors_target = np.linalg.eigh(target)
            new_target = np.zeros_like(target, dtype=complex)
            new_target = backend.cast(new_target, dtype=new_target.dtype)
            for eigenvalue, eigenstate in zip(
                eigenvalues_target, np.transpose(eigenvectors_target)
            ):
                new_target += np.sqrt(eigenstate) * np.outer(
                    eigenstate, np.conj(eigenstate)
                )
        else:
            if len(state.shape) == 1:
                state = np.outer(state, np.conj(state))

            if len(target.shape) == 1:
                target = np.outer(target, np.conj(target))

            new_state, new_target = sqrtm(state).astype("complex128"), sqrtm(
                target
            ).astype("complex128")
            new_state = backend.cast(new_state, dtype=new_state.dtype)
            new_target = backend.cast(new_target, dtype=new_target.dtype)

        log = np.log2(
            backend.calculate_norm_density_matrix(new_state @ new_target, order=1)
        )

        log = -2 * log / np.log2(base)

    else:
        if len(state.shape) == 1:
            state = np.outer(state, np.conj(state))

        if len(target.shape) == 1:
            target = np.outer(target, np.conj(target))

        log = np.linalg.matrix_power(state, alpha)
        log = log @ np.linalg.matrix_power(target, 1 - alpha)
        log = np.log2(np.trace(log))

        log = (1 / (alpha - 1)) * log / np.log2(base)

    backend.assert_allclose(
        relative_renyi_entropy(state, target, alpha=alpha, base=base, backend=backend),
        log,
        atol=1e-5,
    )

    # test pure states
    state = random_density_matrix(4, pure=True, backend=backend)
    target = random_density_matrix(4, pure=True, backend=backend)
    backend.assert_allclose(
        relative_renyi_entropy(state, target, alpha=alpha, base=base, backend=backend),
        0.0,
        atol=1e-8,
    )


@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("alpha", [0, 1, 2, 3, np.inf])
def test_tsallis_entropy(backend, alpha, base):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = renyi_entropy(state, alpha=alpha, base=base, backend=backend)
    with pytest.raises(TypeError):
        state = random_statevector(4, backend=backend)
        test = renyi_entropy(state, alpha="2", base=base, backend=backend)
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        test = renyi_entropy(state, alpha=-1, base=base, backend=backend)
    with pytest.raises(ValueError):
        state = random_statevector(4, backend=backend)
        test = renyi_entropy(state, alpha=alpha, base=0, backend=backend)

    state = random_density_matrix(4, backend=backend)

    if alpha == 1.0:
        target = entropy(state, base=base, backend=backend)
    else:
        target = (1 / (1 - alpha)) * (
            np.trace(np.linalg.matrix_power(state, alpha)) - 1
        )

    backend.assert_allclose(
        tsallis_entropy(state, alpha=alpha, base=base, backend=backend),
        target,
        atol=1e-5,
    )


@pytest.mark.parametrize("check_hermitian", [False, True])
@pytest.mark.parametrize("base", [2, 10, np.e, 5])
@pytest.mark.parametrize("bipartition", [[0], [1]])
def test_entanglement_entropy(backend, bipartition, base, check_hermitian):
    with pytest.raises(TypeError):
        state = np.random.rand(2, 3)
        state = backend.cast(state, dtype=state.dtype)
        test = entanglement_entropy(
            state,
            bipartition=bipartition,
            base=base,
            check_hermitian=check_hermitian,
            backend=backend,
        )
    with pytest.raises(ValueError):
        state = np.array([1.0, 0.0])
        state = backend.cast(state, dtype=state.dtype)
        test = entanglement_entropy(
            state,
            bipartition=bipartition,
            base=0,
            check_hermitian=check_hermitian,
            backend=backend,
        )
    if backend.__class__.__name__ == "CupyBackend":
        with pytest.raises(NotImplementedError):
            state = random_unitary(4, backend=backend)
            test = entanglement_entropy(
                state,
                bipartition=bipartition,
                base=base,
                check_hermitian=True,
                backend=backend,
            )

    # Bell state
    state = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
    state = backend.cast(state, dtype=state.dtype)

    entang_entrop = entanglement_entropy(
        state,
        bipartition=bipartition,
        base=base,
        check_hermitian=check_hermitian,
        backend=backend,
    )

    if base == 2:
        test = 1.0
    elif base == 10:
        test = 0.30102999566398125
    elif base == np.e:
        test = 0.6931471805599454
    else:
        test = 0.4306765580733931

    backend.assert_allclose(entang_entrop, test, atol=PRECISION_TOL)

    # Product state
    state = np.kron(
        random_statevector(2, backend=backend), random_statevector(2, backend=backend)
    )

    entang_entrop = entanglement_entropy(
        state,
        bipartition=bipartition,
        base=base,
        check_hermitian=check_hermitian,
        backend=backend,
    )

    backend.assert_allclose(entang_entrop, 0.0, atol=PRECISION_TOL)
