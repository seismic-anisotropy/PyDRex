"""> PyDRex: tests for the SCSV plain text file format."""
import tempfile

import numpy as np
import pytest
from numpy import testing as nt

from pydrex import io as _io
from pydrex import logger as _log
from pydrex import exceptions as _err


def test_validate_schema():
    """Test SCSV schema validation."""
    schema_nofill = {
        "delimiter": ",",
        "missing": "-",
        "fields": [{"name": "nofill", "type": "float"}],
    }
    schema_nomissing = {
        "delimiter": ",",
        "fields": [{"name": "nomissing", "type": "float", "fill": "NaN"}],
    }
    schema_nofields = {"delimiter": ",", "missing": "-"}
    schema_badfieldname = {
        "delimiter": ",",
        "missing": "-",
        "fields": [{"name": "bad name", "type": "float", "fill": "NaN"}],
    }
    schema_delimiter_eq_missing = {
        "delimiter": ",",
        "missing": ",",
        "fields": [{"name": "baddelim", "type": "float", "fill": "NaN"}],
    }
    schema_delimiter_in_missing = {
        "delimiter": ",",
        "missing": "-,",
        "fields": [{"name": "baddelim", "type": "float", "fill": "NaN"}],
    }
    schema_long_delimiter = {
        "delimiter": ",,",
        "missing": "-",
        "fields": [{"name": "baddelim", "type": "float", "fill": "NaN"}],
    }

    with pytest.raises(_err.SCSVError):
        temp = tempfile.NamedTemporaryFile()
        _io.save_scsv(temp.name, schema_nofill, [[0.1]])
    with pytest.raises(_err.SCSVError):
        temp = tempfile.NamedTemporaryFile()
        _io.save_scsv(temp.name, schema_nomissing, [[0.1]])
    with pytest.raises(_err.SCSVError):
        temp = tempfile.NamedTemporaryFile()
        _io.save_scsv(temp.name, schema_nofields, [[0.1]])
    with pytest.raises(_err.SCSVError):
        temp = tempfile.NamedTemporaryFile()
        _io.save_scsv(temp.name, schema_badfieldname, [[0.1]])
    with pytest.raises(_err.SCSVError):
        temp = tempfile.NamedTemporaryFile()
        _io.save_scsv(temp.name, schema_delimiter_eq_missing, [[0.1]])
    with pytest.raises(_err.SCSVError):
        temp = tempfile.NamedTemporaryFile()
        _io.save_scsv(temp.name, schema_delimiter_in_missing, [[0.1]])
    # CSV module already raises a TypeError on long delimiters.
    with pytest.raises(TypeError):
        temp = tempfile.NamedTemporaryFile()
        _io.save_scsv(temp.name, schema_long_delimiter, [[0.1]])


def test_read_specfile(data_specs):
    """Test SCSV spec file parsing."""
    data = _io.read_scsv(data_specs / "spec.scsv")
    assert data._fields == (
        "first_column",
        "second_column",
        "third_column",
        "float_column",
        "bool_column",
        "complex_column",
    )
    nt.assert_equal(data.first_column, ["s1", "MISSING", "s3"])
    nt.assert_equal(data.second_column, ["A", "B, b", ""])
    nt.assert_equal(data.third_column, [999999, 10, 1])
    nt.assert_equal(data.float_column, [1.1, np.nan, 1.0])
    nt.assert_equal(data.bool_column, [True, False, True])
    nt.assert_equal(data.complex_column, [0.1 + 0 * 1j, np.nan + 0 * 1j, 1.0 + 1 * 1j])


def test_save_specfile(outdir):
    """Test SCSV spec file reproduction."""
    schema = {
        "delimiter": ",",
        "missing": "-",
        "fields": [
            {
                "name": "first_column",
                "type": "string",
                "fill": "MISSING",
                "unit": "percent",
            },
            {"name": "second_column"},
            {"name": "third_column", "type": "integer", "fill": "999999"},
            {"name": "float_column", "type": "float", "fill": "NaN"},
            {"name": "bool_column", "type": "boolean"},
            {"name": "complex_column", "type": "complex", "fill": "NaN"},
        ],
    }
    schema_alt = {
        "delimiter": ",",
        "missing": "-",
        "fields": [
            {
                "name": "first_column",
                "type": "string",
                "unit": "percent",
            },
            {"name": "second_column"},
            {"name": "third_column", "type": "integer", "fill": "999991"},
            {"name": "float_column", "type": "float", "fill": "0.0"},
            {"name": "bool_column", "type": "boolean"},
            {"name": "complex_column", "type": "complex", "fill": "NaN"},
        ],
    }

    data = [
        ["s1", "MISSING", "s3"],
        ["A", "B, b", ""],
        [999999, 10, 1],
        [1.1, np.nan, 1.0],
        [True, False, True],
        [0.1 + 0 * 1j, np.nan + 0 * 1j, 1.0 + 1 * 1j],
    ]
    data_alt = [
        ["s1", "", "s3"],
        ["A", "B, b", ""],
        [999991, 10, 1],
        [1.1, 0.0, 1.0],
        [True, False, True],
        [0.1 + 0 * 1j, np.nan + 0 * 1j, 1.0 + 1 * 1j],
    ]

    # The test writes two variants of the file, with identical CSV contents but
    # different YAML header specs. Contents after thhe header must match.
    if outdir is not None:
        _io.save_scsv(f"{outdir}/spec_out.scsv", schema, data)
        _io.save_scsv(f"{outdir}/spec_out_alt.scsv", schema_alt, data_alt)

    temp = tempfile.NamedTemporaryFile()
    temp_alt = tempfile.NamedTemporaryFile()
    _io.save_scsv(temp.name, schema, data)
    _io.save_scsv(temp_alt.name, schema_alt, data_alt)
    raw_read = []
    raw_read_alt = []
    with open(temp.name) as stream:
        raw_read = stream.readlines()[23:]  # Extra spec for first column 'fill' value.
    with open(temp_alt.name) as stream:
        raw_read_alt = stream.readlines()[22:]
    _log.debug("\n  first file: %s\n  second file: %s", raw_read, raw_read_alt)
    nt.assert_equal(raw_read, raw_read_alt)


def test_read_Kaminski2002(scsvfiles_thirdparty):
    data = _io.read_scsv(scsvfiles_thirdparty / "Kaminski2002_ISAtime.scsv")
    assert data._fields == ("time_ISA", "vorticity")
    # fmt: off
    nt.assert_equal(
        data.time_ISA,
        np.array(
            [2.48, 2.50, 2.55, 2.78, 3.07, 3.58, 4.00, 4.88, 4.01, 3.79,
             3.72, 3.66, 3.71, 4.22, 4.73, 3.45, 1.77, 0.51]
        ),
    )
    nt.assert_equal(
        data.vorticity,
        np.array(
            [0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
             0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
        ),
    )
    # fmt: on


def test_read_Kaminski2004(scsvfiles_thirdparty):
    data = _io.read_scsv(scsvfiles_thirdparty / "Kaminski2004_AaxisDynamicShear.scsv")
    assert data._fields == ("time", "meanA_X0", "meanA_X02", "meanA_X04")
    # fmt: off
    nt.assert_equal(
        data.time,
        np.array(
            [-0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
             1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
             2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1,
             3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2,
             4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]
        ),
    )
    nt.assert_equal(
        data.meanA_X02,
        np.array(
            [-0.54, -0.54, -0.27, 0.13, 0.94, 2.82, 5.37, 9.53, 14.77, 20.40,
             26.58, 32.89, 39.73, 47.25, 53.69, 58.66, 60.81, 60.81, 59.73, 58.52, 58.12,
             56.64, 54.09, 53.69, 55.57, 57.05, 58.66, 60.54, 60.81, 61.21, 61.21, 61.61,
             61.48, 61.61, 61.61, 61.61, 61.21, 61.21, 61.07, 60.81, 60.81, 60.54, 60.00,
             59.60, 59.33, 58.52, 58.12, 57.85, 57.45, 57.05, 57.05]
        ),
    )
    # fmt: on


def test_read_Skemer2016(scsvfiles_thirdparty):
    data = _io.read_scsv(scsvfiles_thirdparty / "Skemer2016_ShearStrainAngles.scsv")
    assert data._fields == (
        "study",
        "sample_id",
        "shear_strain",
        "angle",
        "fabric",
        "M_index",
    )
    # fmt: off
    nt.assert_equal(
        data.study,
        ["Z&K 1200 C", "Z&K 1200 C", "Z&K 1200 C", "Z&K 1200 C", "Z&K 1200 C",
         "Z&K 1300 C", "Z&K 1300 C", "Z&K 1300 C", "Z&K 1300 C", "Z&K 1300 C",
         "Z&K 1300 C", "Z&K 1300 C", "Bystricky 2000", "Bystricky 2000", "Bystricky 2000",
         "Bystricky 2000", "Warren 2008", "Warren 2008", "Warren 2008", "Warren 2008",
         "Warren 2008", "Warren 2008", "Warren 2008", "Warren 2008", "Warren 2008",
         "Skemer 2011", "Skemer 2011", "Skemer 2011", "Webber 2010", "Webber 2010",
         "Webber 2010", "Webber 2010", "Webber 2010", "Webber 2010", "Webber 2010",
         "Katayama 2004", "Katayama 2004", "Katayama 2004", "Katayama 2004", "Katayama 2004",
         "Katayama 2004", "Skemer 2010", "Skemer 2010", "Skemer 2010", "Skemer 2010",
         "Skemer 2010", "Jung 2006", "Jung 2006", "Jung 2006", "Jung 2006",
         "Jung 2006", "Jung 2006", "Hansen 2014", "Hansen 2014", "Hansen 2014",
         "Hansen 2014", "Hansen 2014", "Hansen 2014", "Hansen 2014", "Hansen 2014",
         "Hansen 2014", "Hansen 2014", "Hansen 2014", "Hansen 2014", "Hansen 2014",
         "Hansen 2014", "Hansen 2014", "Hansen 2014", "Hansen 2014", "Hansen 2014",
         "Hansen 2014", "Hansen 2014", "Hansen 2014", "Hansen 2014", "Hansen 2014",
         "Hansen 2014", "Hansen 2014", "Hansen 2014", "H&W 2015", "H&W 2015",
         "H&W 2015", "H&W 2015", "H&W 2015", "H&W 2015", "H&W 2015",
         "H&W 2015", "H&W 2015", "H&W 2015", "H&W 2015", "H&W 2015",
         "H&W 2015", "H&W 2015", "H&W 2015", "H&W 2015"],
    )
    nt.assert_equal(
        data.sample_id,
        ["PI-148", "PI-150", "PI-154", "PI-158", "PI-284", "MIT-5", "MIT-20", "MIT-6",
         "MIT-21", "MIT-17", "MIT-18", "MIT-19", "", "", "", "", "3923J01",
         "3923J11", "3923J13", "3923J14", "3924J06", "3924J09a", "3924J09b", "3924J08",
         "3924J10", "PIP-20", "PIP-21", "PT-484", "", "", "", "", "", "", "", "GA10",
         "GA38", "GA23", "GA12", "GA45", "GA25", "3925G08", "3925G05", "3925G02",
         "3925G01", "PT-248_4", "JK43", "JK21B", "JK26", "JK11", "JK18", "JK23",
         "PT0535", "PT0248_1", "", "PT0655", "PT0248_2", "PI-284", "PT0248_3", "PT0503_5",
         "PT0248_4", "PT0248_5", "PT0503_4", "PT0640", "PT0503_3", "", "PT0494",
         "PT0538", "PT0503_2", "PT0541", "PT0503_1", "PT0633", "PT0552", "PT0505",
         "PT0651", "PT0499", "PT619", "PT0484", "3923J06", "3923J07", "3923J09",
         "3923J08", "3923J13", "3923J12", "3924J06", "3924J05", "JP13-D07", "JP13-D06",
         "3924J03a", "3924J03b", "3924J09a", "3924J09b", "3924J08", "3924J07"],
    )
    nt.assert_equal(
        data.shear_strain,
        np.array(
            [
                17, 30, 45, 65, 110, 11, 7, 65, 58, 100,
                115, 150, 50, 200, 400, 500, 0, 65, 118, 131, 258, 386,
                386, 525, 168, 120, 180, 350, 0, 25, 100, 130, 168, 330,
                330, 60, 60, 80, 120, 260, 630, 60, 150, 1000, 1000, 290,
                120, 400, 100, 110, 120, 400, 0, 30, 50, 100, 100, 110,
                210, 270, 290, 390, 390, 400, 490, 500, 590, 680, 680, 760,
                820, 870, 880, 1020, 1060, 1090, 1420, 1870, 32, 32, 81, 81,
                118, 118, 258, 258, 286, 286, 337, 337, 386, 386, 525, 525
            ]
        ),
    )
    nt.assert_equal(
        data.angle,
        np.array(
            [
                43, 37, 38, 24, 20, 36, 28, 18, 10, 5,
                10, 0, 0, 0, 0, 0, 62, 37, 49, 61, 4, 11, 0, 1, 33, 55,
                53, 45, 55, 35, 47, 29, 37, 47, 45, -49, -26, -10, -35,
                -10, -15, -52, -30, -14, -11, 0, 26, 15, 27, 25, 16, 36,
                -71, -78, 71, 29, 39, 24, 12, 7.3, -3.7, -1.3, -6.1, 0.4,
                -8.6, 5.4, 0.6, -5.5, -4.3, -8.4, -1.9, -0.9, -8.9, 2.8,
                -2, -1.5, -4.1, -5.8, 48, 51, 44, 35, 35, 40, 1, 15, 25,
                28, 28, 39, 1, 8, 4, 11
            ]
        ),
    )
    nt.assert_equal(
        data.fabric,
        [
            "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A",
            "D", "D", "D", "D", "A", "A", "A", "A", "A", "A", "A", "A",
            "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "E",
            "E", "E", "E", "E", "E", "E", "E", "E", "E", "D", "A", "B",
            "C", "C", "C", "C", "D", "D", "D", "D", "D", "D", "D", "D",
            "D", "D", "D", "D", "D", "A", "A", "A", "A", "A", "A", "A",
            "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A",
            "A", "A", "A", "A", "A", "A", "A", "A", "A", "A",
        ],
    )
    nt.assert_equal(
        data.M_index,
        np.array(
            [
                np.nan, np.nan, np.nan, np.nan, 0.09, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan, np.nan, 0.05, np.nan,
                np.nan, 0.17, 0.08, 0.12, 0.16, 0.2, 0.17, 0.06, 0.13,
                0.14, 0.16, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan, np.nan, 0.17, np.nan,
                np.nan, 0.28, np.nan, 0.44, 0.03, 0.1, 0.47, 0.37,
                0.36, 0.06, 0.21, 0.21, 0.23, 0.27, 0.1, 0.01, 0.06,
                0.05, 0.06, 0.06, 0.09, 0.2, 0.26, 0.36, 0.35, 0.34,
                0.36, 0.45, 0.17, 0.4, 0.46, 0.46, 0.44, 0.35, 0.39,
                0.4, 0.69, 0.55, 0.61, 0.51, 0.51, 0.08, 0.2, 0.16,
                0.12, 0.12, 0.17, 0.18, 0.13, 0.15, 0.09, 0.19, 0.27,
                0.12, 0.18, 0.14, 0.26,
            ]
        ),
    )
