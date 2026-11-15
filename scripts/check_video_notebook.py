"""Execute the generated notebook in a real local CPU Jupyter kernel.

This changes configuration values only; it does not emulate CUDA or replace
the model/pipeline with mocks. An executed notebook and first-error report are
saved even when a cell fails.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def configure_cell(source, values):
    lines = source.splitlines(keepends=True)
    replacements = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in values:
                replacements.append((node.lineno-1, node.end_lineno, f'{name} = {values[name]!r}\n'))
    for start, end, line in reversed(replacements):
        lines[start:end] = [line]
    return ''.join(lines)


def run(notebook, source, weights, output, max_frames=300, quoted_empty_paths=False):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    notebook, source, weights, output = map(lambda p: Path(p).resolve(), (notebook, source, weights, output))
    for path in (notebook, source, weights):
        if not path.is_file():
            raise ValueError('Missing input: '+str(path))
    output.mkdir(parents=True, exist_ok=False)
    runtime = output/'runtime'
    runtime.mkdir()
    original_source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    frozen_video = runtime/('source'+source.suffix.lower())
    shutil.copy2(source, frozen_video)
    if hashlib.sha256(frozen_video.read_bytes()).hexdigest() != original_source_hash:
        raise RuntimeError('The input video changed while preparing the notebook test')
    shutil.copy2(weights, runtime/'best.pt')
    document = nbformat.read(notebook, as_version=4)
    overrides = {'PREFERRED_DEVICE': 'cpu', 'INSTALL_DEPENDENCIES': False,
                 'PROCESSING_HEIGHT': 1080,
                 'MAX_FRAMES': max_frames, 'REPEATS': 1, 'COMPARE_CPU': False,
                 'COMPARE_VIEWS': False, 'MAKE_BROWSER_PREVIEW': True, 'RUN_FULL_VIDEO': False}
    if quoted_empty_paths:
        overrides.update(WEIGHTS_PATH='""', VIDEO_PATH='""')
    for cell in document.cells:
        if cell.cell_type == 'code':
            cell.source = configure_cell(cell.source, overrides)
    document.metadata['ppe_local_validation'] = {'configuration': overrides, 'cuda_emulated': False,
                                                 'runtime_root': str(runtime),
                                                 'original_notebook_sha256': hashlib.sha256(notebook.read_bytes()).hexdigest()}
    kernel_root = output/'kernels'
    kernel = kernel_root/'ppe-cpu-check'
    kernel.mkdir(parents=True)
    (kernel/'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'PPE CPU validation', 'language': 'python',
        'env': {'PPE_NOTEBOOK_ROOT': str(runtime), 'PYTHONIOENCODING': 'utf-8',
                'IPYTHONDIR': str(output/'ipython'), 'JUPYTER_RUNTIME_DIR': str(output/'connections'),
                'MPLBACKEND': 'Agg'}
    }, indent=2), encoding='utf-8')
    manager = KernelManager(kernel_name='ppe-cpu-check',
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))

    class LoggedClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            if msg['msg_type'] == 'stream':
                print(msg['content']['text'], end='', flush=True)
            return super().process_message(msg, cell, cell_index)

    def starting(cell, cell_index):
        print(f"Executing cell {cell_index}: {cell.source.splitlines()[0]}", flush=True)

    client = LoggedClient(document, km=manager, timeout=900, allow_errors=False,
                          resources={'metadata': {'path': str(ROOT)}}, on_cell_execute=starting)
    started = perf_counter()
    error = None
    try:
        client.execute()
    except Exception as failure:
        error = {'type': type(failure).__name__, 'message': str(failure)[-3000:]}
    finally:
        nbformat.write(document, output/'executed.ipynb')
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
    failures = []
    for index, cell in enumerate(document.cells):
        for item in cell.get('outputs', []):
            if item.output_type == 'error':
                failures.append({'cell_index': index, 'title': cell.source.splitlines()[0],
                                 'type': item.ename, 'message': item.evalue, 'traceback': item.traceback})
    report = {'status': 'failed' if error else 'completed', 'device': 'cpu', 'cuda_emulated': False,
              'source': str(source), 'frozen_video': str(frozen_video), 'source_sha256': original_source_hash,
              'configuration': overrides, 'elapsed_seconds': perf_counter()-started,
              'executed_code_cells': sum(c.cell_type == 'code' and c.execution_count is not None for c in document.cells),
              'first_error': failures[0] if failures else error,
              'executed_notebook': str(output/'executed.ipynb')}
    benchmarks = list(runtime.glob('ppe-video-ready-results/*/benchmark-*/benchmark.json'))
    if benchmarks:
        benchmark = json.loads(benchmarks[-1].read_text())
        report['benchmark_status'] = benchmark['status']
        report['exports'] = [{'frames': r['frames'], 'device': r['device'], 'readback': r['verification'],
                              'detection_validation': r['detection_validation']} for r in benchmark['runs']]
    (output/'execution-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if error:
        raise RuntimeError('Notebook execution failed; first error saved to '+str(output/'execution-report.json'))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', type=Path, default=ROOT/'notebooks/PPE_Video_GPU_Ready.ipynb')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT/'models/best.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=300)
    parser.add_argument('--quoted-empty-paths', action='store_true')
    args = parser.parse_args()
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.platform == 'win32':
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    run(args.notebook, args.source, args.weights, args.output, args.max_frames, args.quoted_empty_paths)
