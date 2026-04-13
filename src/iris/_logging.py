"""Logging and argument parsing utilities for IRIS."""
import argparse
import logging
import sys

_LOG_NAME = "iris"
_INITIALIZED = False


def get_main_logger(name=None, log_file=None, level=logging.INFO):
    """Create the root logger for IRIS.

    :param name: logger name (default: 'iris').
    :param log_file: optional path to log file.
    :param level: logging level.
    :returns: configured logger.
    """
    global _LOG_NAME, _INITIALIZED
    name = name or _LOG_NAME
    _LOG_NAME = name

    if not _INITIALIZED:
        fmt = '[%(asctime)s - %(name)s] %(message)s'
        datefmt = '%H:%M:%S'
        logging.basicConfig(
            format=fmt, datefmt=datefmt, level=level,
            stream=sys.stderr)
        _INITIALIZED = True

    logger = logging.getLogger(name)

    if log_file is not None:
        fh = logging.FileHandler(log_file, mode='a')
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(
            '[%(asctime)s - %(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(fh)

    return logger


def get_named_logger(name):
    """Create a named child logger."""
    name = name.ljust(10)[:10]
    return logging.getLogger(f'{_LOG_NAME}.{name}')


def wf_parser(name):
    """Create a standard argument parser for an IRIS module."""
    return argparse.ArgumentParser(
        name,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)
