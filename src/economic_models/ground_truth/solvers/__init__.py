"""Numerical solvers the structural ground-truth models are built on.

Model-agnostic linear algebra, kept out of the model packages so a model file
stays a statement of its economics. :mod:`~economic_models.ground_truth.solvers.gensys`
solves linear rational-expectations systems; the pysolve-based models need
nothing from here.
"""

from economic_models.ground_truth.solvers.gensys import GensysResult, gensys

__all__ = ["gensys", "GensysResult"]
