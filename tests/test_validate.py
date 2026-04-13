"""Tests for IRIS input validation module."""
import gzip
import shutil
from pathlib import Path

import numpy as np
import pytest
import scipy.io
import scipy.sparse


@pytest.fixture
def valid_mex_dir(tmp_path):
    """Create a valid MEX directory."""
    d = tmp_path / "gene_matrix"
    d.mkdir()
    n_cells, n_genes = 50, 100
    data = np.random.poisson(3, (n_genes, n_cells)).astype(np.float32)
    sparse_mat = scipy.sparse.csc_matrix(data)
    scipy.io.mmwrite(str(d / "matrix.mtx"), sparse_mat)

    with open(d / "barcodes.tsv", "w") as f:
        for i in range(n_cells):
            f.write(f"CELL{i:04d}-1\n")
    with open(d / "features.tsv", "w") as f:
        for i in range(n_genes):
            f.write(f"GENE{i:04d}\tGENE{i:04d}\tGene Expression\n")

    for fname in ["matrix.mtx", "barcodes.tsv", "features.tsv"]:
        with open(d / fname, 'rb') as fi:
            with gzip.open(d / f"{fname}.gz", 'wb') as fo:
                shutil.copyfileobj(fi, fo)
        (d / fname).unlink()
    return d


class TestValidateMexDirectory:
    def test_valid(self, valid_mex_dir):
        from iris.validate import validate_mex_directory
        result = validate_mex_directory(valid_mex_dir)
        assert result['valid']
        assert result['n_barcodes'] == 50
        assert result['n_features'] == 100

    def test_missing_dir(self, tmp_path):
        from iris.validate import validate_mex_directory, ValidationError
        with pytest.raises(ValidationError, match="does not exist"):
            validate_mex_directory(tmp_path / "nonexistent")

    def test_missing_files(self, tmp_path):
        from iris.validate import validate_mex_directory, ValidationError
        empty = tmp_path / "empty_mex"
        empty.mkdir()
        with pytest.raises(ValidationError, match="Missing"):
            validate_mex_directory(empty)


class TestValidateTsvFile:
    def test_valid(self, tmp_path):
        from iris.validate import validate_tsv_file
        f = tmp_path / "test.tsv"
        f.write_text("col_a\tcol_b\n1\t2\n3\t4\n")
        result = validate_tsv_file(f, required_columns=["col_a"])
        assert result['valid']
        assert result['n_rows'] == 2

    def test_missing_column(self, tmp_path):
        from iris.validate import validate_tsv_file, ValidationError
        f = tmp_path / "test.tsv"
        f.write_text("col_a\tcol_b\n1\t2\n")
        with pytest.raises(ValidationError, match="missing required"):
            validate_tsv_file(f, required_columns=["col_c"])

    def test_file_not_found(self, tmp_path):
        from iris.validate import validate_tsv_file, ValidationError
        with pytest.raises(ValidationError, match="not found"):
            validate_tsv_file(tmp_path / "nope.tsv")


class TestValidateBarcodeOverlap:
    def test_full_overlap(self, valid_mex_dir):
        from iris.validate import validate_barcode_overlap
        result = validate_barcode_overlap(valid_mex_dir, valid_mex_dir)
        assert result['pct_overlap'] == 100.0

    def test_partial_overlap(self, tmp_path):
        from iris.validate import validate_barcode_overlap

        d1 = tmp_path / "m1"
        d1.mkdir()
        d2 = tmp_path / "m2"
        d2.mkdir()
        with gzip.open(d1 / "barcodes.tsv.gz", 'wt') as f:
            f.write("A\nB\nC\nD\n")
        with gzip.open(d2 / "barcodes.tsv.gz", 'wt') as f:
            f.write("C\nD\nE\nF\n")

        result = validate_barcode_overlap(d1, d2)
        assert result['n_overlap'] == 2
        assert result['pct_overlap'] == 50.0


class TestValidationError:
    def test_is_exception(self):
        from iris.validate import ValidationError
        assert issubclass(ValidationError, Exception)
