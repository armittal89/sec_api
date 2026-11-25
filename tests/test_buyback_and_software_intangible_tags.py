"""Buyback fallbacks and goodwill/intangible fallback tags (FISV, MS, LRCX, ALL)."""

from sec_financial_data.sec_financial_data import _get_financial_statement_data

CIK = "0000000001"
M = 1_000_000
# FISV FY2025 10-K.
FY_INSTANT = (None, "2025-12-31", "10-K", "2026-02-20", 2025, "FY")
FY_DURATION = ("2025-01-01", "2025-12-31", "10-K", "2026-02-20", 2025, "FY")


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


FISV_CASH_FLOW = {
    "NetIncomeLoss": 3480 * M,
    "NetCashProvidedByUsedInOperatingActivities": 6058 * M,
    "PaymentsToAcquirePropertyPlantAndEquipment": 1763 * M,
}

FISV_BALANCE = {
    "Assets": 80133 * M,
    "AssetsCurrent": 24654 * M,
    "Liabilities": 54324 * M,
    "LiabilitiesCurrent": 23890 * M,
    "StockholdersEquity": 25792 * M,
    "Goodwill": 37703 * M,
}


def test_treasury_stock_purchases_used_without_cash_flow_tag():
    # FISV's 2019+ cash-flow buyback line is a custom tag.
    facts = dict(FISV_CASH_FLOW, TreasuryStockValueAcquiredCostMethod=5601 * M)
    row = _row(facts, "cash_flow", FY_DURATION)
    assert row["commonStockRepurchased"] == 5601 * M


def test_payments_for_repurchase_of_equity_is_read():
    facts = dict(FISV_CASH_FLOW, PaymentsForRepurchaseOfEquity=900 * M)
    row = _row(facts, "cash_flow", FY_DURATION)
    assert row["commonStockRepurchased"] == 900 * M


def test_cash_flow_buyback_tag_wins_over_equity_statement():
    facts = dict(
        FISV_CASH_FLOW,
        PaymentsForRepurchaseOfCommonStock=1946 * M,
        TreasuryStockValueAcquiredCostMethod=1915 * M,
    )
    row = _row(facts, "cash_flow", FY_DURATION)
    assert row["commonStockRepurchased"] == 1946 * M


def test_intangibles_including_software_tag_is_read():
    # FISV Q2 2026 onward: no FiniteLivedIntangibleAssetsNet.
    facts = dict(
        FISV_BALANCE,
        IntangibleAssetFiniteLivedAndCapitalizedCostSoftwareToBeSoldLeasedOrMarketedAfterAccumulatedAmortization=10161
        * M,
    )
    row = _row(facts, "balance_sheet", FY_INSTANT)
    assert row["intangibleAssets"] == 10161 * M
    assert row["goodwillAndIntangibleAssets"] == 47864 * M


def test_finite_lived_tag_wins_over_software_inclusive_tag():
    facts = dict(
        FISV_BALANCE,
        FiniteLivedIntangibleAssetsNet=10000 * M,
        IntangibleAssetFiniteLivedAndCapitalizedCostSoftwareToBeSoldLeasedOrMarketedAfterAccumulatedAmortization=10161
        * M,
    )
    row = _row(facts, "balance_sheet", FY_INSTANT)
    assert row["intangibleAssets"] == 10000 * M


def test_retired_share_repurchases_are_read():
    # MS tags buybacks only as retired shares in the equity statement.
    facts = dict(FISV_CASH_FLOW, StockRepurchasedDuringPeriodValue=4585 * M)
    row = _row(facts, "cash_flow", FY_DURATION)
    assert row["commonStockRepurchased"] == 4585 * M


def test_combined_goodwill_tag_fills_missing_components():
    # LRCX 10-Qs: only the combined goodwill + intangibles line.
    facts = dict(FISV_BALANCE, IntangibleAssetsNetIncludingGoodwill=1912 * M)
    del facts["Goodwill"]
    row = _row(facts, "balance_sheet", FY_INSTANT)
    assert row["goodwillAndIntangibleAssets"] == 1912 * M


def test_narrower_combined_goodwill_tag_does_not_replace_components():
    # ALL FY2023: combined tag 966M vs 3.50B goodwill + 0.83B intangibles.
    facts = dict(
        FISV_BALANCE,
        Goodwill=3502 * M,
        IntangibleAssetsNetExcludingGoodwill=827 * M,
        IntangibleAssetsNetIncludingGoodwill=966 * M,
    )
    row = _row(facts, "balance_sheet", FY_INSTANT)
    assert row["goodwillAndIntangibleAssets"] == 4329 * M
