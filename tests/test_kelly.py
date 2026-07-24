from backend.kelly import american_to_decimal, compute_kelly_row, kelly_fraction


def test_american_to_decimal():
    assert american_to_decimal(100) == 2.0
    assert american_to_decimal(200) == 3.0
    assert abs(american_to_decimal(-150) - 1.6667) < 0.001
    assert american_to_decimal(-100) == 2.0


def test_kelly_fraction_and_floor():
    assert abs(kelly_fraction(0.55, 2.0) - 0.10) < 1e-9
    assert kelly_fraction(0.40, 2.0) == 0.0


def test_compute_kelly_row():
    full, qk = compute_kelly_row(0.55, 100)
    assert abs(full - 0.10) < 1e-9
    assert abs(qk - 0.025) < 1e-6
    favorite_full, favorite_qk = compute_kelly_row(0.65, -150)
    assert favorite_full > 0
    assert abs(favorite_qk - favorite_full * 0.25) < 1e-6
