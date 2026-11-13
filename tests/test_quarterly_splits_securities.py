"""10-Q discrete quarters, cumulative stock splits, AFS debt securities."""

from sec_financial_data.sec_financial_data import (
    _NO_QUARTER_VALUE,
    _get_financial_statement_data,
    _ten_q_value,
)
from sec_financial_data.stock_splits import StockSplitDetector

B = 1_000_000_000
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


def _statement(facts, statement_type, report_type, units="USD"):
    company_facts = {
        "facts": {
            "us-gaap": {tag: {"units": {units: rows}} for tag, rows in facts.items()}
        }
    }
    return _get_financial_statement_data(
        CIK,
        statement_type,
        10,
        report_type,
        {},
        lambda _: CIK,
        lambda _: company_facts,
    )


# ANET FY2026 10-Qs: three-month and year-to-date facts side by side.
Q1 = ("2026-01-01", "2026-03-31", "10-Q", "2026-05-06", 2026, "Q1")
Q2_3M = ("2026-04-01", "2026-06-30", "10-Q", "2026-08-05", 2026, "Q2")
Q2_YTD = ("2026-01-01", "2026-06-30", "10-Q", "2026-08-05", 2026, "Q2")
Q3_3M = ("2026-07-01", "2026-09-30", "10-Q", "2026-11-04", 2026, "Q3")
Q3_YTD = ("2026-01-01", "2026-09-30", "10-Q", "2026-11-04", 2026, "Q3")


def test_ten_q_income_uses_three_month_facts_not_year_to_date():
    facts = {
        "Revenues": [
            _fact(2.7090 * B, *Q1),
            _fact(3.0357 * B, *Q2_3M),
            _fact(5.7447 * B, *Q2_YTD),
            _fact(3.6576 * B, *Q3_3M),
            _fact(9.4023 * B, *Q3_YTD),
        ],
        "NetIncomeLoss": [
            _fact(1.0229 * B, *Q1),
            _fact(1.2129 * B, *Q2_3M),
            _fact(2.2358 * B, *Q2_YTD),
            _fact(1.5604 * B, *Q3_3M),
            _fact(3.7962 * B, *Q3_YTD),
        ],
    }
    rows = _statement(facts, "income_statement", "10-Q")
    by_date = {r["date"]: r for r in rows}
    assert by_date["2026-09-30"]["revenue"] == 3.6576 * B
    assert by_date["2026-06-30"]["revenue"] == 3.0357 * B
    assert by_date["2026-09-30"]["netIncome"] == 1.5604 * B


def test_ten_q_eps_uses_three_month_fact():
    facts = {
        "Revenues": [_fact(3.0357 * B, *Q2_3M), _fact(5.7447 * B, *Q2_YTD)],
        "EarningsPerShareBasic": [_fact(0.96, *Q2_3M), _fact(1.78, *Q2_YTD)],
    }
    company_facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": facts["Revenues"]}},
                "EarningsPerShareBasic": {
                    "units": {"USD/shares": facts["EarningsPerShareBasic"]}
                },
            }
        }
    }
    rows = _get_financial_statement_data(
        CIK, "income_statement", 5, "10-Q", {}, lambda _: CIK, lambda _: company_facts
    )
    assert rows[0]["eps"] == 0.96


def test_ten_q_cash_flow_year_to_date_is_differenced_to_the_quarter():
    # 10-Q cash-flow statements are year-to-date only.
    facts = {
        "NetCashProvidedByUsedInOperatingActivities": [
            _fact(1.6935 * B, *Q1),
            _fact(2.7765 * B, *Q2_YTD),
            _fact(3.9774 * B, *Q3_YTD),
        ],
    }
    rows = _statement(facts, "cash_flow", "10-Q")
    by_date = {r["date"]: r for r in rows}
    assert by_date["2026-03-31"]["operatingCashFlow"] == 1.6935 * B
    assert abs(by_date["2026-06-30"]["operatingCashFlow"] - 1.0830 * B) < 1
    assert abs(by_date["2026-09-30"]["operatingCashFlow"] - 1.2009 * B) < 1


def test_ten_q_value_rules():
    ytd = {"2026-01-01": {"2026-03-31": ("2026-05-06", 10), "2026-06-30": ("2026-08-05", 25)}}
    # Three-month fact wins over YTD.
    assert _ten_q_value([("2026-04-01", 90, 15), ("2026-01-01", 180, 25)], ytd, "2026-06-30", False, False) == 15
    # YTD only: minus the prior quarter's YTD from the same start.
    assert _ten_q_value([("2026-01-01", 180, 25)], ytd, "2026-06-30", False, False) == 15
    # Prior quarter missing (gap is six months): no value rather than YTD.
    assert (
        _ten_q_value([("2026-01-01", 272, 40)], {"2026-01-01": {"2026-03-31": ("x", 10)}}, "2026-09-30", False, False)
        is _NO_QUARTER_VALUE
    )
    # Per-share values are never differenced.
    assert _ten_q_value([("2026-01-01", 180, 1.78)], ytd, "2026-06-30", True, False) is _NO_QUARTER_VALUE
    # A post-IPO stub period shorter than a quarter is the whole quarter.
    assert _ten_q_value([("2026-02-15", 44, 7)], {}, "2026-03-31", False, False) == 7


def test_six_k_semiannual_value_kept():
    h1 = ("2026-01-01", "2026-06-30", "6-K", "2026-08-20", 2026, "Q2")
    rows = _statement({"Revenues": [_fact(4 * B, *h1)]}, "income_statement", "10-Q")
    assert rows[0]["revenue"] == 4 * B


ANET_SPLITS = [
    {"date": "2024-12-04", "filing_date": "2024-12-04", "ratio": 4},
    {"date": "2021-11-18", "filing_date": "2021-11-18", "ratio": 4},
]


def test_all_splits_after_the_filing_are_applied():
    rows = [
        {"date": "2020-12-31", "filingDate": "2021-02-19", "eps": 8.35, "weightedAverageShsOut": 76_000_000},
        {"date": "2023-12-31", "filingDate": "2024-02-13", "eps": 6.75, "weightedAverageShsOut": 309_000_000},
        {"date": "2025-12-31", "filingDate": "2026-02-17", "eps": 2.79, "weightedAverageShsOut": 1_258_000_000},
    ]
    StockSplitDetector({}).adjust_financials_for_splits(rows, ANET_SPLITS)
    assert abs(rows[0]["eps"] - 8.35 / 16) < 1e-9
    assert rows[0]["weightedAverageShsOut"] == 76_000_000 * 16
    assert abs(rows[1]["eps"] - 6.75 / 4) < 1e-9
    assert rows[2]["eps"] == 2.79 and "_split_adjusted" not in rows[2]


def test_period_before_split_but_filed_after_is_not_adjusted():
    # Filings after a split already report post-split figures, so only the
    # later 2024 split applies to this row filed after the 2021 split.
    rows = [{"date": "2021-09-30", "filingDate": "2021-11-30", "eps": 0.6}]
    StockSplitDetector({}).adjust_financials_for_splits(rows, ANET_SPLITS)
    assert abs(rows[0]["eps"] - 0.6 / 4) < 1e-9


def test_duplicate_split_and_announcement_cutoff():
    # Same event from two sources counts once; an announcement 8-K before
    # the effective date uses the effective date.
    splits = [
        {"date": "2024-06-10", "filing_date": "2024-05-22", "ratio": 10},
        {"date": "2024-06-10", "filing_date": "2024-06-10", "ratio": 10},
    ]
    rows = [{"date": "2024-04-28", "filingDate": "2024-05-29", "eps": 6.0}]
    StockSplitDetector({}).adjust_financials_for_splits(rows, splits)
    assert abs(rows[0]["eps"] - 0.6) < 1e-9


def test_single_split_api_still_works():
    rows = [{"date": "2020-12-31", "filingDate": "2021-02-19", "eps": 8.0}]
    StockSplitDetector({}).adjust_financials_for_latest_split(rows, ANET_SPLITS[0])
    assert rows[0]["eps"] == 2.0


def test_available_for_sale_debt_securities_count_as_short_term_investments():
    # ANET tags its marketable securities only with this ASU 2016-01 tag.
    end = (None, "2026-09-30", "10-Q", "2026-11-04", 2026, "Q3")
    facts = {
        "CashAndCashEquivalentsAtCarryingValue": [_fact(2.2881 * B, *end)],
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent": [_fact(12.0709 * B, *end)],
        "Assets": [_fact(25.2641 * B, *end)],
        "Liabilities": [_fact(8.8174 * B, *end)],
    }
    rows = _statement(facts, "balance_sheet", "10-Q")
    assert rows[0]["shortTermInvestments"] == 12.0709 * B
    assert abs(rows[0]["cashAndShortTermInvestments"] - 14.359 * B) < 1
