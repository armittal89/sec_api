"""Revenue tag selection when candidate tags disagree."""

from sec_financial_data.sec_financial_data import (
    _get_revenue_excluding_derivatives,
    _get_revenue_value,
)

EXCL = "RevenueFromContractWithCustomerExcludingAssessedTax"
INCL = "RevenueFromContractWithCustomerIncludingAssessedTax"
PRICE_RISK = "GainLossOnPriceRiskDerivativeInstrumentsNotDesignatedAsHedgingInstruments"
B = 1_000_000_000


def test_identity_picks_revenues_over_partial_contract_revenue():
    # HUM FY2024: contract revenue is only the services line.
    data = {
        INCL: 4.43 * B,
        "Revenues": 117.76 * B,
        "OperatingIncomeLoss": 2.56 * B,
        "BenefitsLossesAndExpenses": 115.20 * B,
    }
    assert _get_revenue_value(data) == 117.76 * B


def test_identity_picks_contract_over_gross_revenues():
    # HAS FY2024: `Revenues` is a gross pre-elimination figure.
    data = {
        EXCL: 4.136 * B,
        "Revenues": 4.746 * B,
        "OperatingIncomeLoss": 0.69 * B,
        "CostsAndExpenses": 3.446 * B,
    }
    assert _get_revenue_value(data) == 4.136 * B


def test_identity_small_gap_still_overrides_tag_order():
    # LYFT FY2024: contract revenue is 7% short of the reported total.
    data = {
        EXCL: 5.37 * B,
        "Revenues": 5.786 * B,
        "OperatingIncomeLoss": -0.12 * B,
        "CostsAndExpenses": 5.906 * B,
    }
    assert _get_revenue_value(data) == 5.786 * B


def test_sub_line_preferred_tag_loses_to_largest_without_identity():
    # Banks/REITs: contract revenue is fee income only.
    assert _get_revenue_value({EXCL: 1.64 * B, "Revenues": 8.25 * B}) == 8.25 * B


def test_close_candidates_without_identity_keep_tag_order():
    # RSG FY2024: no total-cost tag; `Revenues` is gross of intercompany.
    assert _get_revenue_value({EXCL: 16.032 * B, "Revenues": 18.418 * B}) == 16.032 * B
    # BLK/GIS: the contract total is first and largest.
    assert _get_revenue_value({EXCL: 17.86 * B, "Revenues": 11.01 * B}) == 17.86 * B


def test_unmatched_identity_falls_through():
    # A sub-line OperatingExpenses (KMI) matches no candidate.
    data = {
        EXCL: 13.48 * B,
        "Revenues": 15.10 * B,
        "OperatingIncomeLoss": 4.38 * B,
        "OperatingExpenses": 0.74 * B,
    }
    assert _get_revenue_value(data) == 13.48 * B


def test_operating_expenses_gross_profit_identity_is_ignored():
    # DELL FY2018: OperatingExpenses excludes cost of sales, so op income +
    # opex is gross profit, which sits within 1% of the services line.
    data = {
        "SalesRevenueNet": 78.66 * B,
        "SalesRevenueGoodsNet": 58.80 * B,
        "SalesRevenueServicesNet": 19.86 * B,
        "OperatingIncomeLoss": -3.33 * B,
        "OperatingExpenses": 23.39 * B,
    }
    assert _get_revenue_value(data) == 78.66 * B


def test_tax_excluded_contract_tag_wins_over_tax_included():
    assert _get_revenue_value({EXCL: 20 * B, INCL: 24 * B}) == 20 * B
    # PM FY2022: tax-included is gross of $48.9B excise, far past the sub-line ratio.
    assert _get_revenue_value({EXCL: 31.76 * B, INCL: 80.67 * B}) == 31.76 * B


def test_single_and_fallback_tags():
    assert _get_revenue_value({EXCL: 0, INCL: 7_000_000}) == 7_000_000
    assert _get_revenue_value({"SalesRevenueNet": 53.99 * B}) == 53.99 * B
    assert _get_revenue_value({"Revenue": 15.67 * B}) == 15.67 * B


def test_no_candidate_returns_default():
    assert _get_revenue_value({"NetIncomeLoss": 5}) == 0
    assert _get_revenue_value({"Revenues": 0, EXCL: None}, default=None) is None


def test_derivative_loss_inside_revenue_is_added_back():
    # EQT FY2022: total revenues include a $4.64B hedging loss.
    revenue = 7.497689 * B
    data = {EXCL: 12.114168 * B, "Revenues": revenue, PRICE_RISK: -4.642932 * B}
    assert _get_revenue_excluding_derivatives(data, revenue) == revenue + 4.642932 * B


def test_derivative_outside_revenue_is_left_alone():
    # Adding the loss back moves revenue away from the contract total.
    data = {EXCL: 10 * B, "Revenues": 10.05 * B, PRICE_RISK: -0.5 * B}
    assert _get_revenue_excluding_derivatives(data, 10.05 * B) == 10.05 * B


def test_derivative_without_comparable_candidate_is_left_alone():
    # EQT FY2024 tags only `Revenues`, so inclusion can't be confirmed.
    data = {"Revenues": 5.273309 * B, PRICE_RISK: 0.051117 * B}
    assert _get_revenue_excluding_derivatives(data, 5.273309 * B) == 5.273309 * B


def test_no_revenue_passes_through():
    assert _get_revenue_excluding_derivatives({PRICE_RISK: 1 * B}, 0) == 0
