"""Compute NTO occupations and left/right NTO orbitals from the transition density matrix."""

import numpy as np

SPIN_MAP = {"2": (1, 1), "u": (1, 0), "d": (0, 1), "0": (0, 0)}


def occ_array(det: str) -> np.ndarray:
    """'[2u0d]' -> [1,1, 1,0, 0,0, 0,1]: interleaved (α1, β1, α2, β2, ...), as in Dyson class."""
    return np.array([b for c in det.strip("[]") for b in SPIN_MAP[c]], dtype=np.int8)


def _flip(occ, p, to):
    """Apply an annihilation (to=0) or creation (to=1) operation.

    Returns the fermionic phase and the resulting occupation array.
    """
    new = occ.copy()
    new[p] = to
    sign = -1 if occ[:p].sum() % 2 else 1
    return sign, new


class TransitionDensity:
    def __init__(self, ci_bra: dict, ci_ket: dict, norb: int):
        self.bra = {occ_array(det).tobytes(): coeff for det, coeff in ci_bra.items()}
        self.ket = {det.strip("[]"): coeff for det, coeff in ci_ket.items()}
        self.norb = norb

    @property
    def parsed_norms(self):
        """Return the squared norms of the bra and ket CI vectors."""
        bra_norm = sum(abs(coeff) ** 2 for coeff in self.bra.values())
        ket_norm = sum(abs(coeff) ** 2 for coeff in self.ket.values())
        return bra_norm, ket_norm

    def gamma(self) -> np.ndarray:
        """Compute the transition 1-RDM between two CAS CI vectors.

        The returned matrix is defined as

            gamma[p, q] = <bra | E_pq | ket>,

        where E_pq = a†_p a_q and p, q refer to active spatial orbitals.
        """
        gamma = np.zeros((self.norb, self.norb))
        for det, c_ket in self.ket.items():
            occ = occ_array(det)
            for q in np.flatnonzero(occ):  # a_q for occupied
                s_q, occ_q = _flip(occ, q, 0)  # remove electron from q
                for p in np.flatnonzero(occ_q == 0):  # a†_p for unoccupied
                    if (p - q) % 2:
                        continue  # same spin, cannot excite
                    s_p, occ_p = _flip(occ_q, p, 1)  # add electron to p
                    c_bra = self.bra.get(occ_p.tobytes())
                    if c_bra is not None:
                        gamma[p // 2, q // 2] += s_q * s_p * c_bra * c_ket
        return gamma

    def to_ao(self, c_active: np.ndarray) -> np.ndarray:
        """Transform the transition density matrix to the AO basis."""
        gamma = self.gamma()
        if gamma is None:
            gamma = np.zeros((self.norb, self.norb))
        return c_active @ gamma @ c_active.T

    @staticmethod
    def ntos(gamma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute NTO occupations and left/right NTO orbitals from gamma."""
        U, singular_values, Vt = np.linalg.svd(gamma)
        return singular_values**2, U, Vt.T
