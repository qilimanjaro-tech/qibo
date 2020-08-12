from qibo import gates


class Hamiltonian(object):
    """This class implements the abstract Hamiltonian operator.

    Args:
        nqubits (int): number of quantum bits.
        matrix (np.ndarray): Matrix representation of the Hamiltonian in the
            computational basis as an array of shape
            ``(2 ** nqubits, 2 ** nqubits)``.
    """

    NUMERIC_TYPES = None

    def __init__(self, nqubits, matrix):
        if not isinstance(nqubits, int):
            raise RuntimeError(f'nqubits must be an integer')
        shape = tuple(matrix.shape)
        if shape != 2 * (2 ** nqubits,):
            raise ValueError(f"The Hamiltonian is defined for {nqubits} qubits "
                              "while the given matrix has shape {shape}.")

        self.nqubits = nqubits
        self.matrix = matrix
        self._eigenvalues = None
        self._eigenvectors = None
        self._exp = {"a": None, "result": None}

    def _calculate_eigenvalues(self): # pragma: no cover
        raise NotImplementedError

    def _calculate_eigenvectors(self): # pragma: no cover
        raise NotImplementedError

    def _calculate_exp(self, a): # pragma: no cover
        raise NotImplementedError

    def _eye(self, n=None): # pragma: no cover
        raise NotImplementedError

    def eigenvalues(self):
        """Computes the eigenvalues for the Hamiltonian."""
        if self._eigenvalues is None:
            self._eigenvalues = self._calculate_eigenvalues()
        return self._eigenvalues

    def eigenvectors(self):
        """Computes a tensor with the eigenvectors for the Hamiltonian."""
        if self._eigenvectors is None:
            self._eigenvalues, self._eigenvectors = self._calculate_eigenvectors()
        return self._eigenvectors

    def exp(self, a):
        """Computes a tensor corresponding to exp(-1j * a * H).

        Args:
            a (complex): Complex number to multiply Hamiltonian before
                exponentiation.
        """
        if self._exp["a"] != a:
            self._exp["a"] = a
            self._exp["result"] = self._calculate_exp(a)
        return self._exp["result"]

    def expectation(self, state, normalize=False):
        """Computes the real expectation value for a given state.

        Args:
            state (array): the expectation state.
            normalize (bool): If ``True`` the expectation value is divided
                with the state's norm squared.

        Returns:
            Real number corresponding to the expectation value.
        """
        raise NotImplementedError

    def __add__(self, o):
        """Add operator."""
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise RuntimeError('Only hamiltonians with the same '
                                   'number of qubits can be added.')
            new_matrix = self.matrix + o.matrix
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, self.NUMERIC_TYPES):
            return self.__class__(self.nqubits, self.matrix + o * self._eye())
        else:
            raise NotImplementedError(f'Hamiltonian addition to {type(o)} '
                                      'not implemented.')

    def __radd__(self, o): # pragma: no cover
        """Right operator addition."""
        return self.__add__(o)

    def __sub__(self, o):
        """Subtraction operator."""
        if isinstance(o, self.__class__):
            if self.nqubits != o.nqubits:
                raise RuntimeError('Only hamiltonians with the same '
                                   'number of qubits can be added.')
            new_matrix = self.matrix - o.matrix
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, self.NUMERIC_TYPES): # pragma: no cover
            return self.__class__(self.nqubits, self.matrix - o * self._eye())
        else:
            raise NotImplementedError(f'Hamiltonian subtraction to {type(o)} '
                                      'not implemented.')

    def __rsub__(self, o):
        """Right subtraction operator."""
        if isinstance(o, self.__class__): # pragma: no cover
            if self.nqubits != o.nqubits:
                raise RuntimeError('Only hamiltonians with the same '
                                   'number of qubits can be added.')
            new_matrix = o.matrix - self.matrix
            return self.__class__(self.nqubits, new_matrix)
        elif isinstance(o, self.NUMERIC_TYPES):
            return self.__class__(self.nqubits, o * self._eye() - self.matrix)
        else:
            raise NotImplementedError(f'Hamiltonian subtraction to {type(o)} '
                                      'not implemented.')

    def __mul__(self, o):
        """Multiplication to scalar operator."""
        if isinstance(o, self.NUMERIC_TYPES):
            new_matrix = self.matrix * o
            r = self.__class__(self.nqubits, new_matrix)
            if self._eigenvalues is not None:
                if o.real >= 0:
                    r._eigenvalues = o * self._eigenvalues
                else:
                    r._eigenvalues = o * self._eigenvalues[::-1]
            if self._eigenvectors is not None:
                if o.real > 0:
                    r._eigenvectors = self._eigenvectors
                elif o == 0:
                    r._eigenvectors = self._eye(int(self._eigenvectors.shape[0]))
            return r
        else:
            raise NotImplementedError(f'Hamiltonian multiplication to {type(o)} '
                                      'not implemented.')

    def __rmul__(self, o):
        """Right scalar multiplication."""
        return self.__mul__(o)

    def __matmul__(self, o): # pragma: no cover
        """Matrix multiplication with other Hamiltonians or state vectors."""
        raise NotImplementedError


class LocalHamiltonian(object):

    def __init__(self, terms):
        for term in terms:
            if not issubclass(type(term), Hamiltonian):
                raise TypeError("Invalid term type {}.".format(type(term)))
            if term.nqubits not in {1, 2}:
                raise ValueError("LocalHamiltonian terms should target one or "
                                 "two qubits but {} was given."
                                 "".format(term.nqubits))
        self.nqubits = len(terms)
        self.terms = terms
        self._dt = None
        self._gates = []

    def trotter(self, dt):
        if self._dt != dt:
            # TODO: Use a Circuit instead of a gate list
            self._dt = dt
            self._gates = []
            n = len(self.terms) - len(self.terms) % 2
            terms = iter(self.terms[:n])
            for i, term in enumerate(terms):
                i1, i2, i3 = 2 * i, 2 * i + 1, (2 * i + 2) % self.nqubits
                self._gates.append(gates.Unitary(term.exp(dt / 2.0), i1, i2))
                self._gates.append(gates.Unitary(next(terms).exp(dt), i2, i3))
                self._gates.append(gates.Unitary(term.exp(dt / 2.0), i1, i2))
            if len(self.terms) % 2:
                e = self.terms[-1].exp(dt)
                self._gates.append(gates.Unitary(e, self.nqubits - 1, 0))
        return self._gates
