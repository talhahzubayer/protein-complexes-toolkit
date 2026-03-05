"""
Diagnostic tests for multiprocessing with ProcessPoolExecutor on Windows.

These tests isolate whether the parallel mode in toolkit.py can:
1. Pickle the worker function and work items
2. Import toolkit.py successfully in a subprocess
3. Process a single complex through ProcessPoolExecutor
4. Produce results matching sequential mode
5. Handle batch submission of multiple complexes
"""

import pickle
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pytest

PROJECT_ROOT = Path(r"C:\Users\Talhah Zubayer\Documents\protein-complexes-toolkit")


@pytest.mark.multiprocessing
class TestMultiprocessing:
    """Diagnostic tests for parallel processing via ProcessPoolExecutor."""

    def test_worker_function_is_picklable(self):
        """_worker_process_complex must survive pickle round-trip."""
        from toolkit import _worker_process_complex

        pickled = pickle.dumps(_worker_process_complex)
        restored = pickle.loads(pickled)
        assert callable(restored)

    def test_work_item_is_picklable(self, ref_pdb_1, ref_pkl_1):
        """Work item tuple (name, paths, kwargs) must survive pickle round-trip."""
        item = (
            'A0A0B4J2C3_P24534',
            {'pdb': ref_pdb_1, 'pkl': ref_pkl_1},
            {
                'run_interface': True,
                'run_interface_pae': True,
                'export_interfaces': False,
                'verbose': False,
            },
        )
        pickled = pickle.dumps(item)
        restored = pickle.loads(pickled)
        assert restored[0] == 'A0A0B4J2C3_P24534'
        assert restored[1]['pdb'] == ref_pdb_1

    def test_toolkit_import_in_subprocess(self):
        """Importing toolkit.py in a fresh subprocess must complete within 30s.

        On Windows, ProcessPoolExecutor uses 'spawn' — each worker re-imports
        the module. If this hangs, workers will never start.
        """
        result = subprocess.run(
            [sys.executable, '-c', 'import toolkit; print("OK")'],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, (
            f"Subprocess import failed.\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        assert 'OK' in result.stdout

    @pytest.mark.slow
    def test_single_complex_via_executor(self, ref_pdb_1, ref_pkl_1):
        """One complex processed through ProcessPoolExecutor returns a valid result.

        Uses max_workers=1 and a 60s timeout to detect hangs. If this test
        times out, workers are failing silently during startup.
        """
        from toolkit import _worker_process_complex

        item = (
            'A0A0B4J2C3_P24534',
            {'pdb': ref_pdb_1, 'pkl': ref_pkl_1},
            {
                'run_interface': True,
                'run_interface_pae': True,
                'export_interfaces': False,
                'verbose': False,
            },
        )
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_worker_process_complex, item)
            result = future.result(timeout=60)

        assert result['complex_name'] == 'A0A0B4J2C3_P24534'
        assert result['quality_tier'] in ('High', 'Medium', 'Low')
        assert 'iptm' in result

    @pytest.mark.slow
    def test_parallel_matches_sequential(self, ref_pdb_1, ref_pkl_1):
        """Results from ProcessPoolExecutor must match sequential processing."""
        from toolkit import _worker_process_complex, process_single_complex

        file_paths = {'pdb': ref_pdb_1, 'pkl': ref_pkl_1}
        kwargs = {
            'run_interface': True,
            'run_interface_pae': True,
            'export_interfaces': False,
            'verbose': False,
        }

        # Sequential
        seq_result = process_single_complex(
            'A0A0B4J2C3_P24534', file_paths, **kwargs,
        )

        # Parallel (1 worker to isolate multiprocessing from concurrency)
        item = ('A0A0B4J2C3_P24534', file_paths, kwargs)
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_worker_process_complex, item)
            par_result = future.result(timeout=60)

        # Compare all keys
        for key in seq_result:
            assert seq_result[key] == par_result[key], (
                f"Mismatch on '{key}': sequential={seq_result[key]} "
                f"vs parallel={par_result[key]}"
            )

    @pytest.mark.slow
    def test_batch_submission_completes(self, test_data_dir):
        """All test complexes complete via ProcessPoolExecutor with 2 workers.

        This mimics the real batch path in run_batch_parallel() but on the
        61-complex test set. A 5-minute timeout prevents an indefinite hang.
        """
        from toolkit import _worker_process_complex, find_paired_data_files

        pairs = find_paired_data_files(str(test_data_dir))
        kwargs = {
            'run_interface': False,
            'run_interface_pae': False,
            'export_interfaces': False,
            'verbose': False,
        }
        work_items = [
            (name, paths, kwargs)
            for name, paths in sorted(pairs.items())
        ]

        results = []
        with ProcessPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(_worker_process_complex, item): item[0]
                for item in work_items
            }
            for future in as_completed(futures, timeout=300):
                row = future.result()
                results.append(row)

        assert len(results) == len(work_items)
        # Every result should have a complex_name
        names = {r['complex_name'] for r in results}
        assert len(names) == len(work_items)
