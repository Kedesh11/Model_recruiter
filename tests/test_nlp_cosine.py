import numpy as np
from seeg_core.nlp import cosine_sim


def test_cosine_basic_same_vector():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert abs(cosine_sim(a, b) - 1.0) < 1e-9


def test_cosine_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(cosine_sim(a, b) - 0.0) < 1e-9


def test_cosine_different_lengths():
    a = np.array([1.0, 1.0, 0.0])
    b = np.array([1.0, 1.0])
    # After truncation, vectors are identical -> cosine = 1
    assert abs(cosine_sim(a, b) - 1.0) < 1e-9
