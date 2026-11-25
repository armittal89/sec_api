"""CMCSA audit: missing balance-sheet components and intangible reinvestment."""

from sec_financial_data.sec_financial_data import _get_financial_statement_data

M = 1_000_000


def _row(values, statement, form="10-Q"):
    duration = statement == "cash_flow"
    facts = {
        "facts": {"us-gaap": {
            tag: {"units": {"USD": [{
                "val": val, "end": "2026-09-30", "form": form,
                "filed": "2026-10-22", "fy": 2026, "fp": "Q3",
                **({"start": "2026-07-01"} if duration else {}),
            }]}}
            for tag, val in values.items()
        }}
    }
    [row] = _get_financial_statement_data(
        "0001166691", statement, 5, form, {},
        lambda _: "0001166691", lambda _: facts,
    )
    return row


BALANCE = {
    "Assets": 257769 * M,
    "LiabilitiesCurrent": 30719 * M,
    "LongTermDebtNoncurrent": 83806 * M,
    "DeferredIncomeTaxLiabilitiesNet": 29022 * M,
    "OtherLiabilitiesNoncurrent": 21554 * M,
    "StockholdersEquity": 92529 * M,
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": 92497 * M,
    "RedeemableNoncontrollingInterestEquityCarryingAmount": 171 * M,
    "Goodwill": 53239 * M,
    "OtherIntangibleAssetsNet": 19073 * M,
    "IndefiniteLivedFranchiseRights": 59365 * M,
}


def test_franchise_and_other_intangibles_are_included():
    row = _row(BALANCE, "balance_sheet")
    assert row["goodwillAndIntangibleAssets"] == 131677 * M
    assert row["intangibleAssets"] == (19073 + 59365) * M


def test_other_intangibles_do_not_double_count_broader_indefinite_total():
    row = _row(dict(BALANCE, IndefiniteLivedIntangibleAssetsExcludingGoodwill=62000 * M), "balance_sheet")
    assert row["goodwillAndIntangibleAssets"] == 131677 * M


def test_liabilities_exclude_consolidated_and_redeemable_equity():
    row = _row(BALANCE, "balance_sheet")
    assert row["totalLiabilities"] == 165101 * M


def test_liability_component_fallback_keeps_deferred_taxes_and_other_liabilities():
    facts = dict(BALANCE)
    del facts["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]
    row = _row(facts, "balance_sheet")
    assert row["totalLiabilities"] == 165101 * M


def test_assets_and_equity_total_is_not_a_liabilities_alias():
    row = _row(dict(BALANCE, LiabilitiesAndStockholdersEquity=257769 * M), "balance_sheet")
    assert row["totalLiabilities"] == 165101 * M


def test_explicit_intangible_and_liability_totals_take_priority():
    row = _row(dict(BALANCE, IntangibleAssetsNetExcludingGoodwill=80000 * M, Liabilities=165000 * M), "balance_sheet")
    assert row["goodwillAndIntangibleAssets"] == (53239 + 80000) * M
    assert row["totalLiabilities"] == 165000 * M


CASH_FLOW = {
    "NetCashProvidedByUsedInOperatingActivities": 7903 * M,
    "PaymentsToAcquirePropertyPlantAndEquipment": 3058 * M,
    "PaymentsToAcquireIntangibleAssets": 633 * M,
    "ShareBasedCompensation": 348 * M,
}


def test_fcf_deducts_separate_intangible_spending():
    row = _row(CASH_FLOW, "cash_flow")
    assert row["capitalExpenditure"] == 3058 * M
    assert row["intangibleCapitalExpenditure"] == 633 * M
    assert row["freeCashFlow"] == 4212 * M


def test_broader_productive_assets_total_is_not_deducted_twice():
    facts = dict(CASH_FLOW)
    del facts["PaymentsToAcquirePropertyPlantAndEquipment"]
    facts["PaymentsToAcquireProductiveAssets"] = 3691 * M
    row = _row(facts, "cash_flow")
    assert row["intangibleCapitalExpenditure"] == 0
    assert row["freeCashFlow"] == 4212 * M


def test_negative_capex_is_normalized_before_fcf_subtraction():
    row = _row(dict(CASH_FLOW, PaymentsToAcquirePropertyPlantAndEquipment=-3058 * M), "cash_flow")
    assert row["freeCashFlow"] == 4212 * M


def test_separate_software_additions_are_reinvestment_alongside_ppe():
    facts = dict(CASH_FLOW)
    del facts["PaymentsToAcquireIntangibleAssets"]
    facts["CapitalizedComputerSoftwareAdditions"] = 633 * M
    row = _row(facts, "cash_flow")
    assert row["freeCashFlow"] == 4212 * M


def test_software_only_capex_is_not_counted_twice():
    row = _row({
        "NetCashProvidedByUsedInOperatingActivities": 1000 * M,
        "CapitalizedComputerSoftwareAdditions": 200 * M,
    }, "cash_flow")
    assert row["freeCashFlow"] == 800 * M
