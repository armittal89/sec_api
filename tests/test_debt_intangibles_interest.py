"""Current debt maturities, finite-lived intangibles, interest-expense tags."""

from sec_financial_data.sec_financial_data import _get_financial_statement_data

CIK = "0000000001"
M = 1_000_000
# INTU FY2026 10-K (fiscal year ended 2026-07-31).
FY_INSTANT = (None, "2026-07-31", "10-K", "2026-09-09", 2026, "FY")
FY_DURATION = ("2025-08-01", "2026-07-31", "10-K", "2026-09-09", 2026, "FY")


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


def _row(facts, statement_type, period):
    company_facts = {
        "facts": {
            "us-gaap": {
                tag: {"units": {"USD": [_fact(val, *period)]}}
                for tag, val in facts.items()
            }
        }
    }
    [row] = _get_financial_statement_data(
        CIK,
        statement_type,
        5,
        "10-K",
        {},
        lambda _: CIK,
        lambda _: company_facts,
    )
    return row


INTU_BALANCE = {
    "Assets": 36786 * M,
    "AssetsCurrent": 15311 * M,
    "Liabilities": 17794 * M,
    "LiabilitiesCurrent": 10163 * M,
    "StockholdersEquity": 18992 * M,
    "CashAndCashEquivalentsAtCarryingValue": 4705 * M,
    "Goodwill": 13981 * M,
    "FiniteLivedIntangibleAssetsNet": 4642 * M,
    "LongTermDebt": 7669 * M,
    "LongTermDebtCurrent": 1249 * M,
    "LongTermDebtNoncurrent": 6420 * M,
    "OperatingLeaseLiabilityCurrent": 100 * M,
    "OperatingLeaseLiabilityNoncurrent": 652 * M,
}


def test_current_debt_maturities_count_as_short_term_debt():
    row = _row(INTU_BALANCE, "balance_sheet", FY_INSTANT)
    assert row["shortTermDebt"] == 1249 * M
    assert row["longTermDebt"] == 6420 * M
    # Total debt keeps its lease component (aminvested strips leases).
    assert row["totalDebt"] == 7669 * M + 752 * M


def test_long_term_debt_total_fallback_excludes_current_maturities():
    facts = dict(INTU_BALANCE)
    del facts["LongTermDebtNoncurrent"]
    row = _row(facts, "balance_sheet", FY_INSTANT)
    assert row["shortTermDebt"] == 1249 * M
    assert row["longTermDebt"] == 6420 * M


def test_finite_lived_intangibles_used_without_total_tag():
    row = _row(INTU_BALANCE, "balance_sheet", FY_INSTANT)
    assert row["intangibleAssets"] == 4642 * M
    assert row["goodwillAndIntangibleAssets"] == (13981 + 4642) * M


def test_indefinite_lived_intangibles_added_to_finite_lived():
    facts = dict(INTU_BALANCE, IndefiniteLivedIntangibleAssetsExcludingGoodwill=50 * M)
    row = _row(facts, "balance_sheet", FY_INSTANT)
    assert row["intangibleAssets"] == 4692 * M


def test_intangible_total_tag_wins_over_components():
    facts = dict(INTU_BALANCE, IntangibleAssetsNetExcludingGoodwill=4700 * M)
    row = _row(facts, "balance_sheet", FY_INSTANT)
    assert row["intangibleAssets"] == 4700 * M


INTU_INCOME = {
    "Revenues": 21448 * M,
    "OperatingIncomeLoss": 5884 * M,
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": 6017
    * M,
    "IncomeTaxExpenseBenefit": 1451 * M,
    "NetIncomeLoss": 4566 * M,
    "InterestExpenseDebt": 256 * M,
}


def test_interest_expense_debt_tag_is_read():
    row = _row(INTU_INCOME, "income_statement", FY_DURATION)
    assert row["interestExpense"] == 256 * M


def test_interest_expense_nonoperating_tag_is_read():
    facts = dict(INTU_INCOME)
    del facts["InterestExpenseDebt"]
    facts["InterestExpenseNonoperating"] = 300 * M
    row = _row(facts, "income_statement", FY_DURATION)
    assert row["interestExpense"] == 300 * M
