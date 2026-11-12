"""Independent physical invariants found during the textbook audit."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from vbb_study.vector_arm_chain import hwp45_matrix, qwp45_matrix


@pytest.mark.parametrize('error', [-0.3, 0.0, 0.23])
@pytest.mark.parametrize('matrix', [hwp45_matrix, qwp45_matrix])
def test_lossless_waveplate_is_unitary(matrix, error):
    j = matrix(error)
    np.testing.assert_allclose(j.conj().T @ j, np.eye(2), atol=1e-14)


def test_hwp_retardance_error_leaves_unconverted_component():
    error = 0.2
    out = hwp45_matrix(error) @ np.array([1., 0.])
    assert abs(out[0])**2 == pytest.approx(np.sin(error/2)**2)
    assert abs(out[1])**2 == pytest.approx(np.cos(error/2)**2)


def _retrieval():
    path = Path(__file__).resolve().parents[1] / 'notebooks/experimental/axicon_aberration_correction/miao_full_retrieval.py'
    spec = importlib.util.spec_from_file_location('textbook_miao', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('reference', [np.ones((3, 64)), np.zeros((3, 64)), np.full((3, 64), np.nan)])
def test_uninformative_reference_never_resolves_branch(reference):
    m = _retrieval()
    theta = np.arange(64)*2*np.pi/64
    fields = np.tile(1+.3*np.cos(theta), (3, 1)).astype(complex)
    assert m.resolve_conjugate_branch(fields, reference)[0] == 'unresolved'


def test_reference_rows_follow_sorted_annuli():
    m = _retrieval()
    theta = np.arange(64)*2*np.pi/64
    z = np.array([.03, .02])
    fields = np.array([1+.4*np.cos(theta), 1-.4*np.cos(theta)], complex)
    planes = [m.PlaneRetrieval(i, zi, 0, 0, 4e5, 2, 0, 1, 0,
                              np.arange(-2, 3), np.zeros(5), theta, fields[i])
              for i, zi in enumerate(z)]
    full = m.assemble_full_aperture(planes, z, 1030e-9,
                                   reference_intensity_rows=abs(fields)**2)
    assert full.branch == 'direct'
