from datetime import date

from neon_radar.application.services.walk_forward_analyzer import _add_months


def test_add_months():
    d = date(2024, 1, 15)

    d2 = _add_months(d, 3)
    assert d2 == date(2024, 4, 15)

    d3 = _add_months(d, 12)
    assert d3 == date(2025, 1, 15)

    # leap year
    d_leap = date(2024, 2, 29)
    d4 = _add_months(d_leap, 12)
    assert d4 == date(2025, 2, 28)

    d5 = _add_months(d_leap, 24)
    assert d5 == date(2026, 2, 28)

    d6 = _add_months(date(2024, 1, 31), 1)
    assert d6 == date(2024, 2, 29)
