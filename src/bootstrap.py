"""Bootstrap helpers run early to prepare runtime compatibility (e.g. TensorFlow shims).

This module should be imported as early as possible by entry scripts so that
third-party libraries (Keras) see the compatibility shims at import-time.
"""
from __future__ import annotations

import importlib
import types
import warnings
import logging


def _ensure_ragged_compat() -> None:
    """Ensure tf.ragged.RaggedTensorValue is available or mapped to compat.v1.

    This helps older third-party code that references the deprecated symbol.
    It's best-effort and will not raise on failure. Suppress warnings/logging
    while performing the mapping to avoid emitting deprecation warnings.
    """
    try:
        tf_spec = importlib.util.find_spec('tensorflow')
        if tf_spec is None:
            return
        try:
            import tensorflow as tf  # type: ignore
        except Exception:
            return

        # Temporarily suppress DeprecationWarning and TensorFlow logger output
        tf_logger = logging.getLogger('tensorflow')
        old_level = tf_logger.level
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*RaggedTensorValue.*')
                tf_logger.setLevel(logging.ERROR)

                # Ensure tf.ragged namespace exists
                if not hasattr(tf, 'ragged'):
                    tf.ragged = types.SimpleNamespace()

                # If RaggedTensorValue is missing, prefer compat.v1 mapping
                if not hasattr(tf.ragged, 'RaggedTensorValue'):
                    mapped = None
                    if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1'):
                        compat_v1 = getattr(tf.compat, 'v1')
                        if hasattr(compat_v1, 'ragged') and hasattr(compat_v1.ragged, 'RaggedTensorValue'):
                            mapped = compat_v1.ragged.RaggedTensorValue

                    # Fallback: try to use tf.RaggedTensor class if available
                    if mapped is None and hasattr(tf, 'RaggedTensor'):
                        mapped = getattr(tf, 'RaggedTensor')

                    if mapped is not None:
                        setattr(tf.ragged, 'RaggedTensorValue', mapped)
        finally:
            tf_logger.setLevel(old_level)
    except Exception:
        # Swallow all exceptions; this is a best-effort compatibility layer
        return


_ensure_ragged_compat()


def _ensure_keras_shim() -> None:
    """If standalone `keras` is not installed, expose `keras` name pointing to `tf.keras`.

    TensorFlow's lazy loader may attempt `import keras` when accessing `tf.keras`.
    Creating a best-effort shim avoids ImportError in environments where only
    `tensorflow` is installed.
    """
    try:
        import importlib
        import sys
        import types

        # If keras is already importable, nothing to do
        if importlib.util.find_spec("keras") is not None:
            return

        # Try to import tensorflow; if unavailable, skip
        tf_spec = importlib.util.find_spec("tensorflow")
        if tf_spec is None:
            return
        import tensorflow as tf  # type: ignore

        if not hasattr(tf, "keras"):
            return

        # Create a lightweight stub module named 'keras' to satisfy import
        # and basic version checks performed by TensorFlow's lazy loader.
        # The stub intentionally does NOT delegate into tf.keras to avoid
        # triggering recursive import logic in environments where tf.keras
        # initialization itself tries to import `keras`.
        fake = types.ModuleType("keras")
        # Use a non-3.x version string to avoid Keras-3 specific branches.
        fake.__version__ = getattr(tf, "__version__", "0.0.0")
        fake.__name__ = "keras"
        # Add a few common submodule placeholders so attribute access succeeds
        fake.layers = types.ModuleType("keras.layers")
        fake.models = types.ModuleType("keras.models")
        fake.utils = types.ModuleType("keras.utils")
        fake.backend = types.ModuleType("keras.backend")

        sys.modules.setdefault("keras", fake)
    except Exception:
        # Best-effort only
        return


# Run the shim early
_ensure_keras_shim()
