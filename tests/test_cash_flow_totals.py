"""Cash-flow section totals tagged only for continuing operations."""

from sec_financial_data.sec_financial_data import _get_financial_statement_data

CIK = "0000000001"
B = 1_000_000_000
M = 1_000_000
K = ("2016-08-29", "2017-09-03", "10-K", "2017-10-18", 2017, "FY")


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


def _cash_flow(facts):
    company_facts = {
        "facts": {
            "us-gaap": {
                tag: {"units": {"USD": [_fact(val, *K)]}} for tag, val in facts.items()
            }
        }
    }
    [row] = _get_financial_statement_data(
        CIK,
        "cash_flow",
        5,
        "10-K",
        {},
        lambda _: CIK,
        lambda _: company_facts,
    )
    return row


# COST FY2017 10-K: section totals tagged only as ContinuingOperations.
COST_FY2017 = {
    "NetIncomeLoss": 2679 * M,
    "DepreciationDepletionAndAmortization": 1437 * M,
    "ShareBasedCompensation": 555 * M,
    "IncreaseDecreaseInInventoriesNet": -894 * M,
    "IncreaseDecreaseInAccountsPayableCurrent": 2012 * M,
    "PaymentsToAcquirePropertyPlantAndEquipment": 2502 * M,
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations": 6726 * M,
    "NetCashProvidedByUsedInInvestingActivitiesContinuingOperations": -2366 * M,
    "NetCashProvidedByUsedInFinancingActivitiesContinuingOperations": -5180 * M,
}


def test_continuing_operations_totals_used_when_only_ones_tagged():
    row = _cash_flow(COST_FY2017)
    # Was the component sum (4.568B for the real filing).
    assert row["operatingCashFlow"] == 6726 * M
    assert row["netCashProvidedByOperatingActivities"] == 6726 * M
    assert row["freeCashFlow"] == 6726 * M - 2502 * M
    assert row["netCashProvidedByInvestingActivities"] == -2366 * M
    assert row["netCashProvidedByFinancingActivities"] == -5180 * M


def test_total_including_discontinued_operations_wins():
    facts = dict(
        COST_FY2017,
        NetCashProvidedByUsedInOperatingActivities=7 * B,
        NetCashProvidedByUsedInInvestingActivities=-3 * B,
        NetCashProvidedByUsedInFinancingActivities=-4 * B,
    )
    row = _cash_flow(facts)
    assert row["operatingCashFlow"] == 7 * B
    assert row["netCashProvidedByInvestingActivities"] == -3 * B
    assert row["netCashProvidedByFinancingActivities"] == -4 * B
