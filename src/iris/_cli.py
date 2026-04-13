"""IRIS command-line interface."""
import argparse
import importlib
import sys

from . import __version__
from ._logging import get_main_logger

MODULES = {
    'cluster': ('iris.cluster_analysis', 'Basic Scanpy clustering'),
    'dual-cluster': (
        'iris.dual_layer_clustering',
        'Gene + isoform dual-layer clustering'),
    'dtu': (
        'iris.differential_transcript_usage',
        'Differential transcript usage testing'),
    'novel-isoforms': (
        'iris.novel_isoform_discovery',
        'Cluster-specific novel isoform discovery'),
    'trajectory': (
        'iris.isoform_trajectory',
        'Isoform-aware trajectory analysis'),
    'ase': (
        'iris.allele_specific_expression',
        'Allele-specific expression analysis'),
    'annotate': (
        'iris.cell_type_annotation',
        'Cell type annotation'),
    'export': (
        'iris.export_anndata',
        'Export unified AnnData h5ad'),
    'benchmark-dtu': (
        'iris.benchmark_dtu',
        'DTU detection benchmarking'),
    'report': (
        'iris.report',
        'Generate HTML report'),
    'plot': (
        'iris.plot',
        'Generate publication-quality figures'),
    'compare': (
        'iris.multi_sample',
        'Cross-sample comparison of IRIS results'),
    'validate': (
        'iris.validate',
        'Validate pipeline inputs'),
    'run': (
        'iris.pipeline',
        'Run full IRIS pipeline'),
}


def cli():
    """IRIS CLI entry point."""
    parser = argparse.ArgumentParser(
        'iris',
        description=(
            'IRIS: Isoform-Resolved Inference for Single-cells. '
            'Isoform-aware scRNA-seq analysis for long-read data.'),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--version', action='version',
        version=f'iris {__version__}')

    subparsers = parser.add_subparsers(
        title='commands', dest='command',
        description='Available analysis modules:')

    # Register all modules
    for cmd_name, (module_path, description) in MODULES.items():
        try:
            mod = importlib.import_module(module_path)
            sub_parser = mod.argparser()
            subparsers.add_parser(
                cmd_name, parents=[sub_parser],
                description=description,
                help=description,
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        except (ImportError, AttributeError):
            # Module not available or no argparser
            pass

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    logger = get_main_logger("iris")

    module_path = MODULES[args.command][0]
    mod = importlib.import_module(module_path)
    mod.main(args)
