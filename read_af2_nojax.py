#!/usr/bin/env python3
"""
JAX-Free AlphaFold2 PKL Reader

Extracts metrics from AlphaFold2 result PKL files without requiring JAX.
Uses module mocking to completely bypass JAX imports during pickle loading.
Works regardless of JAX version installed or no JAX at all.

Usage (standalone):
    python read_af2_nojax.py --pkl result.pkl
    python read_af2_nojax.py --pkl result.pkl --json metrics.json
    python read_af2_nojax.py --pkl result.pkl --keys

Usage (as importable module):
    from read_af2_nojax import load_pkl_without_jax, extract_metrics
    prediction = load_pkl_without_jax("result.pkl")
    metrics = extract_metrics(prediction)
"""

import sys
from typing import Any, Optional, Union

# ── JAX Module Mocking ──────────────────────────────────────────────
# Mock JAX modules BEFORE any other imports.
# This prevents pickle from trying to import real JAX modules.
# Runs once at import time - safe for use across multiprocessing workers
# when each worker imports this module independently.

class _MockJaxArray:
    """Mock JAX array that passes through numpy arrays during unpickling."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.data = args[0] if args else None

    def __array__(self) -> Any:
        return self.data


class _MockJaxModule:
    """Mock module that returns passthrough functions for all JAX calls.
    
    Python's dunder/special methods bypass __getattr__ entirely - the
    interpreter looks them up directly on the *type*, not the instance.
    Pickle's object-reconstruction machinery may call __call__, iterate
    with __iter__, index with __getitem__, or test truthiness with
    __bool__ on the mock objects it receives.  Without explicit dunder
    definitions those operations raise TypeError instead of gracefully
    passing through, producing errors like:
        '_MockJaxModule' object is not iterable
        '_MockJaxModule' object is not callable
    The methods below cover every protocol pickle (and numpy) might use.
    """

    def __getattr__(self, name: str) -> Any:
        if name == '_reconstruct_array':
            return lambda x, *args, **kwargs: x
        if name == 'Array':
            return _MockJaxArray
        return _MockJaxModule()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return args[0] if args else self

    def __iter__(self):
        return iter([])

    def __len__(self) -> int:
        return 0

    def __getitem__(self, key: Any) -> Any:
        return _MockJaxModule()

    def __bool__(self) -> bool:
        return False


_JAX_MODULES_TO_MOCK = [
    'jax', 'jax._src', 'jax._src.core',
    'jax.numpy', 'jax.lax', 'jax.random',
    'jax._src.device_array',
    'jax._src.array',
    'jaxlib', 'jaxlib.xla_extension', 'jaxlib.xla_client',
]

for _module_name in _JAX_MODULES_TO_MOCK:
    sys.modules[_module_name] = _MockJaxModule()

# Now safe to import other modules
import pickle
import argparse
import json
import gzip
import bz2
from pathlib import Path

import numpy as np

# ── Constants ────────────────────────────────────────────────────────

MAX_RECURSION_DEPTH = 100

# pLDDT confidence band boundaries (Ångströms)
PLDDT_VERY_HIGH_THRESHOLD = 90
PLDDT_HIGH_THRESHOLD = 70
PLDDT_LOW_THRESHOLD = 50


# ── PKL Loading ──────────────────────────────────────────────────────

def load_pkl_without_jax(filepath: Union[str, Path]) -> dict:
    """
    Load an AlphaFold2 PKL file without using JAX.

    Args:
        filepath: Path to .pkl, .pkl.gz, or .pkl.bz2 file.

    Returns:
        Prediction results dictionary with pure NumPy arrays.
    """
    filepath = Path(filepath)
    suffix = ''.join(filepath.suffixes).lower()

    if '.bz2' in suffix:
        opener = lambda path: bz2.BZ2File(path, 'rb')
    elif '.gz' in suffix:
        opener = lambda path: gzip.open(path, 'rb')
    else:
        opener = lambda path: open(path, 'rb')

    with opener(filepath) as file_handle:
        result = pickle.load(file_handle)

    return _convert_to_numpy(result)


def _convert_to_numpy(obj: Any, depth: int = 0) -> Any:
    """
    Recursively ensure all array-like objects are pure NumPy arrays.

    Args:
        obj: Any Python object, potentially containing JAX arrays.
        depth: Current recursion depth for safety limiting.

    Returns:
        The same structure with all arrays converted to np.ndarray.
    """
    if depth > MAX_RECURSION_DEPTH:
        return obj

    if isinstance(obj, dict):
        return {key: _convert_to_numpy(value, depth + 1) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_numpy(item, depth + 1) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_to_numpy(item, depth + 1) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj
    elif hasattr(obj, '__array__'):
        try:
            return np.asarray(obj)
        except Exception:
            return obj
    else:
        return obj


# ── Metric Extraction ────────────────────────────────────────────────

def extract_scalar(value: Any) -> Optional[float]:
    """
    Extract a Python float from various scalar or array types.

    Args:
        value: A scalar, numpy array of size 1, or array-like with .item().

    Returns:
        A Python float, or None if conversion is not possible.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.flat[0])
        return None
    if hasattr(value, 'item'):
        return float(value.item())
    if hasattr(value, '__float__'):
        return float(value)
    return None


def extract_metrics(prediction_result: dict) -> dict:
    """
    Extract key quality metrics from AlphaFold2 prediction results.

    Args:
        prediction_result: Dictionary returned by load_pkl_without_jax().

    Returns:
        Dictionary of metrics with Python-native types (JSON-serialisable).
    """
    metrics: dict = {}

    scalar_keys = ['iptm', 'ptm', 'ranking_confidence', 'max_predicted_aligned_error']
    for key in scalar_keys:
        if key in prediction_result:
            scalar_value = extract_scalar(prediction_result[key])
            if scalar_value is not None:
                metrics[key] = scalar_value

    if 'plddt' in prediction_result:
        plddt_scores = np.asarray(prediction_result['plddt']).flatten()

        metrics['plddt_mean'] = float(np.mean(plddt_scores))
        metrics['plddt_median'] = float(np.median(plddt_scores))
        metrics['plddt_min'] = float(np.min(plddt_scores))
        metrics['plddt_max'] = float(np.max(plddt_scores))
        metrics['plddt_std'] = float(np.std(plddt_scores))
        metrics['num_residues'] = int(len(plddt_scores))

        metrics['plddt_very_high'] = int(np.sum(plddt_scores >= PLDDT_VERY_HIGH_THRESHOLD))
        metrics['plddt_high'] = int(np.sum(
            (plddt_scores >= PLDDT_HIGH_THRESHOLD) & (plddt_scores < PLDDT_VERY_HIGH_THRESHOLD)
        ))
        metrics['plddt_low'] = int(np.sum(
            (plddt_scores >= PLDDT_LOW_THRESHOLD) & (plddt_scores < PLDDT_HIGH_THRESHOLD)
        ))
        metrics['plddt_very_low'] = int(np.sum(plddt_scores < PLDDT_LOW_THRESHOLD))

    if 'predicted_aligned_error' in prediction_result:
        pae_matrix = np.asarray(prediction_result['predicted_aligned_error'])
        metrics['pae_mean'] = float(np.mean(pae_matrix))
        metrics['pae_max'] = float(np.max(pae_matrix))
        metrics['pae_shape'] = list(pae_matrix.shape)

    if 'num_recycles' in prediction_result:
        recycle_value = extract_scalar(prediction_result['num_recycles'])
        if recycle_value is not None:
            metrics['num_recycles'] = int(recycle_value)

    return metrics


def list_keys(prediction_result: dict) -> dict[str, str]:
    """
    List all keys in the prediction result with human-readable type descriptions.

    Args:
        prediction_result: Dictionary returned by load_pkl_without_jax().

    Returns:
        Dictionary mapping each key to a string describing its type and shape.
    """
    key_descriptions: dict[str, str] = {}
    max_value_preview_length = 50

    for key, value in prediction_result.items():
        if isinstance(value, np.ndarray):
            key_descriptions[key] = f"ndarray{list(value.shape)} dtype={value.dtype}"
        elif isinstance(value, dict):
            key_descriptions[key] = f"dict[{len(value)} keys]"
        elif isinstance(value, (list, tuple)):
            key_descriptions[key] = f"{type(value).__name__}[{len(value)}]"
        else:
            value_preview = str(value)
            if len(value_preview) > max_value_preview_length:
                value_preview = value_preview[:max_value_preview_length] + "..."
            key_descriptions[key] = f"{type(value).__name__}: {value_preview}"

    return key_descriptions


# ── CLI Entry Point ──────────────────────────────────────────────────

def main() -> dict:
    """
    Parse CLI arguments, load a PKL file, and output extracted metrics.

    Returns:
        Dictionary of extracted metrics.
    """
    parser = argparse.ArgumentParser(
        description="Read AlphaFold2 PKL files without requiring JAX"
    )
    parser.add_argument("--pkl", required=True, help="PKL file path")
    parser.add_argument("--keys", action="store_true", help="List all keys in the PKL file")
    parser.add_argument("--json", help="Save metrics to a JSON file")
    parser.add_argument("--extract-pae", help="Save PAE matrix to .npy file")
    parser.add_argument("--extract-plddt", help="Save pLDDT array to .npy file")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()

    if not args.quiet:
        print(f"Loading: {args.pkl}", file=sys.stderr)

    try:
        prediction_result = load_pkl_without_jax(args.pkl)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)

    if args.keys:
        print("\n=== Available Keys ===")
        for key, description in sorted(list_keys(prediction_result).items()):
            print(f"  {key}: {description}")
        print()

    metrics = extract_metrics(prediction_result)

    if not args.quiet:
        print("\n=== Key Metrics ===")
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    if args.json:
        with open(args.json, 'w') as json_file:
            json.dump(metrics, json_file, indent=2)
        print(f"\nSaved: {args.json}")

    if args.extract_pae and 'predicted_aligned_error' in prediction_result:
        np.save(args.extract_pae, prediction_result['predicted_aligned_error'])
        print(f"Saved PAE: {args.extract_pae}")

    if args.extract_plddt and 'plddt' in prediction_result:
        np.save(args.extract_plddt, prediction_result['plddt'])
        print(f"Saved pLDDT: {args.extract_plddt}")

    return metrics


if __name__ == "__main__":
    main()
