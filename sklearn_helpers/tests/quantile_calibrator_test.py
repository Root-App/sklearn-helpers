from sklearn_helpers.quantile_calibrator import QuantileCalibrator


# TODO: Real tests.
def test_quantile_calibrator():
    qc = QuantileCalibrator()

    assert qc is not None
