"""Pre-tax income and D&A tag resolution."""

import pytest

from sec_financial_data.sec_financial_data import (
    TAG_MAPPINGS,
    _annual_fact_values,
    _get_depreciation_value,
    _get_financial_statement_data,
    _get_value_from_tags,
    _sum_fiscal_year_quarters,
)

STANDARD_PRETAX = "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"
B = 1_000_000_000


def test_standard_pretax_tag_resolves():
    # GOOGL FY2025: pre-tax was read as operating income ($129.04B).
    data = {STANDARD_PRETAX: 158.826 * B, "OperatingIncomeLoss": 129.039 * B}
    assert _get_value_from_tags(data, TAG_MAPPINGS["incomeBeforeTax"]) == 158.826 * B


def test_combined_da_tag_wins():
    data = {"DepreciationDepletionAndAmortization": 5 * B, "Depreciation": 4 * B}
    assert _get_depreciation_value(data) == 5 * B


def test_depreciation_only_filer():
    # GOOGL tags only `Depreciation` on the cash-flow statement.
    assert _get_depreciation_value({"Depreciation": 21.136 * B}) == 21.136 * B


def test_depreciation_plus_intangible_amortization():
    data = {"Depreciation": 3 * B, "AmortizationOfIntangibleAssets": 1 * B}
    assert _get_depreciation_value(data) == 4 * B


def test_ifrs_components_summed():
    data = {
        "AdjustmentsForDepreciationExpense": 3 * B,
        "AdjustmentsForAmortisationExpense": 2 * B,
    }
    assert _get_depreciation_value(data) == 5 * B
    assert (
        _get_depreciation_value({"AdjustmentsForDepreciationExpense": 3 * B}) == 3 * B
    )


def test_no_da_returns_default():
    assert _get_depreciation_value({}, default=0) == 0


def test_annual_report_drops_quarter_only_fact():
    # COO FY2019 10-K: only a Q4 (91-day) pre-tax fact.
    assert _annual_fact_values([(91, 128.5)]) == []
    assert _annual_fact_values([(91, 128.5), (364, 477.4)]) == [(364, 477.4)]


def test_misdated_annual_fact_kept_when_near_prior_year():
    # LHX FY2025 revenue ($21.87B) is dated from the Q4 start.
    assert _annual_fact_values([(91, 21.865 * B)], 21.325 * B) == [(91, 21.865 * B)]
    # A genuine Q4 stays dropped.
    assert _annual_fact_values([(91, 5.5 * B)], 21.325 * B) == []


def test_annual_report_keeps_instant_facts():
    assert _annual_fact_values([(0, 5.0)]) == [(0, 5.0)]


def test_fiscal_year_summed_from_contiguous_quarters():
    # AAPL FY2012 10-K tags dividends paid only per quarter.
    quarters = [
        ("2011-09-25", "2011-12-31", 0),
        ("2012-01-01", "2012-03-31", 0),
        ("2012-04-01", "2012-06-30", 0),
        ("2012-07-01", "2012-09-29", 2.5 * B),
    ]
    assert _sum_fiscal_year_quarters(quarters, "2011-09-25", "2012-09-29") == 2.5 * B
    # A missing or non-adjacent quarter gives no annual value.
    assert _sum_fiscal_year_quarters(quarters[1:], "2011-09-25", "2012-09-29") is None
    gap = quarters[:2] + [("2012-04-08", "2012-06-30", 0)] + quarters[3:]
    assert _sum_fiscal_year_quarters(gap, "2011-09-25", "2012-09-29") is None
    # Four quarters ending mid-year are a trailing twelve months, not a fiscal year.
    assert _sum_fiscal_year_quarters(quarters, "2011-07-01", "2012-09-29") is None


CIK = "0000000001"


def _fact(val, start, end, form, filed, fy, fp):
    return {
        "val": val,
        "start": start,
        "end": end,
        "form": form,
        "filed": filed,
        "fy": fy,
        "fp": fp,
    }


def _statement(facts, report_type):
    company_facts = {
        "facts": {
            "us-gaap": {tag: {"units": {"USD": rows}} for tag, rows in facts.items()}
        }
    }
    return _get_financial_statement_data(
        CIK,
        "income_statement",
        5,
        report_type,
        {},
        lambda _: CIK,
        lambda _: company_facts,
    )


def test_projection_and_footnote_only_periods_are_dropped():
    k = ("2024-01-01", "2024-12-31", "10-K", "2025-02-15", 2024, "FY")
    q = ("2025-01-01", "2025-03-31", "10-Q", "2025-05-01", 2025, "Q1")
    facts = {
        "Revenues": [_fact(100 * B, *k), _fact(30 * B, *q)],
        "NetIncomeLoss": [_fact(10 * B, *k), _fact(3 * B, *q)],
        STANDARD_PRETAX: [_fact(14 * B, *k), _fact(4 * B, *q)],
        "AmortizationOfIntangibleAssets": [
            # IBKR: amortization expected for a future year.
            _fact(1 * B, "2026-01-01", "2026-12-31", "10-K", "2025-02-15", 2024, "FY"),
            # MO: full-year D&A footnote in a 10-Q, at a non-quarter date.
            _fact(2 * B, "2024-01-01", "2024-12-31", "10-Q", "2025-05-01", 2025, "Q1"),
        ],
    }
    annual = _statement(facts, "10-K")
    assert [(r["date"], r["incomeBeforeTax"]) for r in annual] == [
        ("2024-12-31", 14 * B)
    ]
    quarterly = _statement(facts, "10-Q")
    assert [r["date"] for r in quarterly] == ["2025-03-31"]


def test_pretax_fallback_does_not_double_count_interest():
    # RCL FY2020: no pre-tax tag; "other income (expense)" already includes interest.
    k = ("2020-01-01", "2020-12-31", "10-K", "2021-02-26", 2020, "FY")
    facts = {
        "Revenues": [_fact(2.209 * B, *k)],
        "OperatingIncomeLoss": [_fact(-4.602 * B, *k)],
        "InterestExpense": [_fact(0.844 * B, *k)],
        "NonoperatingIncomeExpense": [_fact(-1.174 * B, *k)],
        "NetIncomeLoss": [_fact(-5.797 * B, *k)],
        "NetIncomeLossAttributableToNoncontrollingInterest": [_fact(0.022 * B, *k)],
    }
    [row] = _statement(facts, "10-K")
    assert row["incomeBeforeTax"] == -5.797 * B + 0.022 * B


def test_operating_fallback_does_not_add_interest_to_nonoperating_total():
    # No pre-tax tag and no net income: operating + NonoperatingIncomeExpense,
    # which already includes interest.
    k = ("2024-01-01", "2024-12-31", "10-K", "2025-02-15", 2024, "FY")
    facts = {
        "Revenues": [_fact(50 * B, *k)],
        "OperatingIncomeLoss": [_fact(10 * B, *k)],
        "InterestExpense": [_fact(1 * B, *k)],
        "NonoperatingIncomeExpense": [_fact(-2 * B, *k)],
    }
    [row] = _statement(facts, "10-K")
    assert row["incomeBeforeTax"] == 8 * B


def test_discontinued_operations_reported_tag():
    k = ("2024-01-01", "2024-12-31", "10-K", "2025-02-15", 2024, "FY")
    facts = {
        "Revenues": [_fact(100 * B, *k)],
        "NetIncomeLoss": [_fact(10 * B, *k)],
        "IncomeTaxExpenseBenefit": [_fact(2 * B, *k)],
        "IncomeLossFromDiscontinuedOperationsNetOfTax": [_fact(3 * B, *k)],
    }
    [row] = _statement(facts, "10-K")
    assert row["netIncomeFromDiscontinuedOperations"] == 3 * B
    assert row["netIncomeFromContinuingOperations"] == 7 * B
    assert row["incomeBeforeTax"] == 9 * B


def test_q4_only_revenue_filled_from_later_comparative():
    # MPC FY2016 10-K tags Revenues only per quarter; the FY2018 10-K
    # discloses the full year under the ASC 606 tag.
    original = ("2016-01-01", "2016-12-31", "10-K", "2017-02-28", 2016, "FY")
    q4 = ("2016-10-01", "2016-12-31", "10-K", "2017-02-28", 2016, "FY")
    later = ("2016-01-01", "2016-12-31", "10-K", "2019-02-28", 2018, "FY")
    facts = {
        "Revenues": [_fact(17.155 * B, *q4)],
        "RevenueFromContractWithCustomerIncludingAssessedTax": [
            _fact(63.277 * B, *later)
        ],
        "NetIncomeLoss": [_fact(1.174 * B, *original), _fact(1.174 * B, *later)],
        "OperatingIncomeLoss": [_fact(2.4 * B, *original)],
        "EarningsPerShareBasic": [_fact(0.50, *q4), _fact(2.21, *later)],
    }
    [row] = _statement(facts, "10-K")
    assert (row["fiscalYear"], row["revenue"]) == ("2016", 63.277 * B)
    assert row["eps"] == 0  # a later filing may be post-split


def test_quarterly_footnote_only_flow_summed_but_eps_not():
    # GIS FY2016 10-K tags GrossProfit only in the quarterly-data footnote.
    filing = ("10-K", "2016-06-30", 2016, "FY")
    k = ("2015-06-01", "2016-05-29", *filing)
    quarters = [
        ("2015-06-01", "2015-08-30", 1.5546),
        ("2015-08-31", "2015-11-29", 1.5406),
        ("2015-11-30", "2016-02-28", 1.3575),
        ("2016-02-29", "2016-05-29", 1.3768),
    ]
    facts = {
        "Revenues": [_fact(16.5631 * B, *k)],
        "NetIncomeLoss": [_fact(1.697 * B, *k)],
        "GrossProfit": [
            _fact(v * B, start, end, *filing) for start, end, v in quarters
        ],
        "EarningsPerShareBasic": [
            _fact(0.7, start, end, *filing) for start, end, _ in quarters
        ],
    }
    [row] = _statement(facts, "10-K")
    assert row["grossProfit"] == pytest.approx(5.8295 * B)
    assert row["eps"] == 0


def test_quarter_sum_does_not_compete_with_reported_group_total():
    # GE: annual Revenues plus contract revenue tagged only per quarter.
    filing = ("10-K", "2023-02-10", 2022, "FY")
    k = ("2022-01-01", "2022-12-31", *filing)
    quarters = [
        ("2022-01-01", "2022-03-31"),
        ("2022-04-01", "2022-06-30"),
        ("2022-07-01", "2022-09-30"),
        ("2022-10-01", "2022-12-31"),
    ]
    facts = {
        "Revenues": [_fact(76.555 * B, *k)],
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            _fact(18.4 * B, start, end, *filing) for start, end in quarters
        ],
        "NetIncomeLoss": [_fact(0.225 * B, *k)],
    }
    [row] = _statement(facts, "10-K")
    assert row["revenue"] == 76.555 * B


def test_no_cost_of_revenue_leaves_gross_and_operating_profit_unknown():
    # TJX FY2010: no GrossProfit or cost-of-sales fact, only SG&A.
    k = ("2009-02-01", "2010-01-30", "10-K", "2010-03-30", 2010, "FY")
    facts = {
        "Revenues": [_fact(20.288 * B, *k)],
        "NetIncomeLoss": [_fact(1.214 * B, *k)],
        "SellingGeneralAndAdministrativeExpense": [_fact(3.329 * B, *k)],
    }
    [row] = _statement(facts, "10-K")
    assert row["grossProfit"] == 0
    assert row["operatingIncome"] == 0


def test_sparse_report_fills_missing_groups_from_later_reports():
    # TJX FY2010: no report for the year has two key items; interest
    # expense is only in the later 10-Ks.
    end = "2010-01-30"
    original = ("2009-02-01", end, "10-K", "2010-03-30", 2010, "FY")
    first_later = ("2009-02-01", end, "10-K", "2011-03-30", 2011, "FY")
    second_later = ("2009-02-01", end, "10-K", "2012-03-27", 2012, "FY")
    facts = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": [
            _fact(20.0 * B, *original)
        ],
        "Revenues": [_fact(20.288 * B, *first_later), _fact(20.288 * B, *second_later)],
        "InterestExpense": [
            _fact(0.049 * B, *first_later),
            _fact(0.050 * B, *second_later),
        ],
        "EarningsPerShareBasic": [_fact(1.45, *first_later)],
        "SellingGeneralAndAdministrativeExpense": [_fact(3.329 * B, *original)],
    }
    [row] = _statement(facts, "10-K")
    assert (row["fiscalYear"], row["interestExpense"]) == ("2010", 0.049 * B)
    assert row["revenue"] == 20.0 * B  # the original's own revenue group stays
    assert row["eps"] == 0  # a later filing may be post-split


def test_operating_income_from_revenue_minus_total_costs():
    # NEM 2019: no operating-income or standard pre-tax tag; gross profit
    # minus the expense lines gave -118M.
    k = ("2019-01-01", "2019-12-31", "10-K", "2020-02-20", 2019, "FY")
    facts = {
        "Revenues": [_fact(9.740 * B, *k)],
        "CostsAndExpenses": [_fact(8.463 * B, *k)],
        "GrossProfit": [_fact(2.5 * B, *k)],
        "GeneralAndAdministrativeExpense": [_fact(0.313 * B, *k)],
        "DepreciationDepletionAndAmortization": [_fact(2.305 * B, *k)],
        "NetIncomeLoss": [_fact(2.805 * B, *k)],
    }
    [row] = _statement(facts, "10-K")
    assert row["operatingIncome"] == pytest.approx(1.277 * B)


def test_sub_line_total_costs_not_used_for_operating_income():
    # GE 2007: a 40B CostsAndExpenses fact against 172B revenue is a segment's.
    k = ("2007-01-01", "2007-12-31", "10-K", "2008-02-15", 2007, "FY")
    facts = {
        "Revenues": [_fact(172.488 * B, *k)],
        "CostsAndExpenses": [_fact(40.173 * B, *k)],
        "NetIncomeLoss": [_fact(22.208 * B, *k)],
    }
    [row] = _statement(facts, "10-K")
    assert row["operatingIncome"] == 0


def test_negative_interest_expense_fact_is_sign_corrected():
    # DHI's FY2013 10-K tags FY2013 interest expense as -7.1M.
    k = ("2012-10-01", "2013-09-30", "10-K", "2013-11-26", 2013, "FY")
    facts = {
        "Revenues": [_fact(6.259 * B, *k)],
        "NetIncomeLoss": [_fact(0.462 * B, *k)],
        "InterestExpense": [_fact(-0.0071 * B, *k)],
    }
    [row] = _statement(facts, "10-K")
    assert row["interestExpense"] == pytest.approx(0.0071 * B)
