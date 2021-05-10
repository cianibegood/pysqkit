import pytest

from pysqkit.bases import FockBasis, fock_basis


class TestFockBasis:
    def test_init(self):
        """
        test_init Tests the initialization of the system
        """
        basis = FockBasis(10)
        assert basis.dim_hilbert == 10

    def test_truncate(self):
        """
        test_truncate Tests whether basis.truncate correctly truncates
        the basis to the desired value
        """
        basis = FockBasis(10)
        basis.truncate(5)

        assert basis.is_truncated
        assert basis.truncated_dim == 5
        assert basis.dim_hilbert == 10

    @pytest.mark.parametrize("trunc_dim", ["string", 0, -20, 3.14, 100])
    def test_truncate_args(self, trunc_dim):
        """
        test_truncate_args Tests whether basis.truncate correctly raises
        Errors in case of wrong dimension given (non integer, negative or
        equal to 0 or greater than hilbert dimension)
        """
        basis = FockBasis(10)

        with pytest.raises(ValueError):
            basis.truncate(trunc_dim)


def test_fock_basis():
    """
    test_fock_basis Tests whether bases.fock_basis() correctly initializes
    a FockBasis object of the correct dimension
    """
    basis = fock_basis(20)
    assert isinstance(basis, FockBasis)
    assert basis.dim_hilbert == 20


@pytest.mark.parametrize("init_arg", ["string", 0, -20, 3.14])
def test_fock_basis_arg(init_arg):
    """
    test_fock_basis Tests whether bases.fock_basis() raises ValueError in case
    of incorrect dimension provided (non integer, negative or equal to 0 or
    greater than hilbert dimension)
    """
    with pytest.raises(ValueError):
        basis = fock_basis(init_arg)
