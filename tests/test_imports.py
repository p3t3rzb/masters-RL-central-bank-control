"""Smoke-import every module in the package.

Catches rename drift (a module importing a path that no longer exists) that
nothing else exercises until a script is actually run.
"""

import importlib
import pkgutil

import economic_models


def test_every_module_imports() -> None:
    prefix = economic_models.__name__ + "."
    for module in pkgutil.walk_packages(economic_models.__path__, prefix):
        importlib.import_module(module.name)
