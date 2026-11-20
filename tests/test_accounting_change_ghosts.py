"""Sparse accounting-change instants must not replace a real fiscal-year sheet."""

from sec_financial_data.sec_financial_data import _get_financial_statement_data

CIK = "0000000001"
M = 1_000_000


def _fact(val, end, filed, fy):
    return {
        "val": val,
        "end": end,
        "form": "10-K",
        "filed": filed,
        "fy": fy,
        "fp": "FY",
    }


def _balance_sheets(facts_by_tag, limit=10):
    company_facts = {
        "facts": {
            "us-gaap": {
                tag: {"units": {"USD": facts}} for tag, facts in facts_by_tag.items()
            }
        }
    }
    return _get_financial_statement_data(
        CIK,
        "balance_sheet",
        limit,
        "10-K",
        {},
        lambda _: CIK,
        lambda _: company_facts,
    )


def _sheet(end, filed, fy, assets):
    """A full balance sheet as its own 10-K and the next 10-K report it."""
    return {
        "Assets": [_fact(assets, end, filed, fy)],
        "AssetsCurrent": [_fact(assets // 3, end, filed, fy)],
        "LiabilitiesCurrent": [_fact(assets // 4, end, filed, fy)],
        "Liabilities": [_fact(assets // 2, end, filed, fy)],
        "StockholdersEquity": [_fact(assets // 2, end, filed, fy)],
    }


def _merge(*sheets):
    merged = {}
    for sheet in sheets:
        for tag, facts in sheet.items():
            merged.setdefault(tag, []).extend(facts)
    return merged


# ADBE FY2019 10-K: ASC 606 opening balances one day after the FY2018 sheet.
ASC_606_GHOST = {
    "AccountsReceivableNetCurrent": [
        _fact(1358606000, "2018-12-01", "2020-01-21", 2019)
    ],
    "RetainedEarningsAccumulatedDeficit": [
        _fact(12257916000, "2018-12-01", "2020-01-21", 2019)
    ],
    "DeferredRevenueCurrent": [_fact(2863132000, "2018-12-01", "2020-01-21", 2019)],
}


def test_52_53_week_year_keeps_real_sheet_over_adoption_date_ghost():
    # Year-ends wobble too much for the Step 4.1 filter (ADBE): the ghost is
    # relabeled FY2018 and collides with the real sheet in Step 6.1.
    year_ends = {
        2013: "2013-11-29",
        2014: "2014-11-28",
        2015: "2015-11-27",
        2016: "2016-12-02",
        2017: "2017-12-01",
        2018: "2018-11-30",
        2019: "2019-11-29",
        2020: "2020-11-27",
        2021: "2021-12-03",
        2022: "2022-12-02",
    }
    facts = _merge(
        *(
            _sheet(end, f"{year + 1}-01-20", year, (10 + year - 2013) * 1000 * M)
            for year, end in year_ends.items()
        ),
        ASC_606_GHOST,
    )
    rows = _balance_sheets(facts, limit=20)
    by_year = {r["fiscalYear"]: r for r in rows}
    assert by_year["2018"]["date"] == "2018-11-30"
    assert by_year["2018"]["totalAssets"] == 15000 * M
    assert "2018-12-01" not in {r["date"] for r in rows}
    assert len(rows) == len(year_ends)


def test_ghost_on_dominant_year_end_does_not_evict_real_sheet():
    # Year-end is "12-01" every year, so the ghost is the exact match and the
    # real FY2018 sheet a day earlier is the tolerant one (Step 4.1).
    facts = _merge(
        *(
            _sheet(
                f"{year}-12-01",
                f"{year + 1}-01-20",
                year,
                (10 + year - 2014) * 1000 * M,
            )
            for year in (2014, 2015, 2016, 2017, 2019, 2020)
        ),
        _sheet("2018-11-30", "2019-01-25", 2018, 15000 * M),
        ASC_606_GHOST,
    )
    rows = _balance_sheets(facts, limit=20)
    dates = {r["date"] for r in rows}
    assert "2018-11-30" in dates
    assert "2018-12-01" not in dates


def test_long_term_debt_and_lease_obligations_is_noncurrent_debt():
    facts = _merge(
        _sheet("2018-11-30", "2019-01-25", 2018, 18769 * M),
        {
            "LongTermDebtAndCapitalLeaseObligations": [
                _fact(4124800000, "2018-11-30", "2019-01-25", 2018)
            ],
            "DebtCurrent": [_fact(0, "2018-11-30", "2019-01-25", 2018)],
        },
    )
    [row] = _balance_sheets(facts)
    assert row["longTermDebt"] == 4124800000
    assert row["totalDebt"] == 4124800000
