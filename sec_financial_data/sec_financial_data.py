import requests
import time
from functools import lru_cache
from collections import defaultdict
import copy
import re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from typing import Optional, Dict, Any, List

# Base URLs for SEC EDGAR APIs
SEC_BASE_URL = "https://www.sec.gov"
DATA_SEC_BASE_URL = "https://data.sec.gov"

# Rate limiting: SEC allows up to 10 requests per second.
# We'll implement a simple delay to ensure we don't exceed this.
# A more robust solution for high-volume usage might involve a token
# bucket algorithm.
LAST_REQUEST_TIME = 0
REQUEST_INTERVAL = 0.11  # Approximately 9 requests per second to be safe

logger = logging.getLogger("sec_financial_data")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _rate_limit():
    """
    Ensures that API requests respect SEC's rate limits (max 10 requests per second).
    Introduces a small delay if requests are being made too quickly.
    """
    global LAST_REQUEST_TIME
    current_time = time.time()
    elapsed = current_time - LAST_REQUEST_TIME
    if elapsed < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - elapsed)
    LAST_REQUEST_TIME = time.time()


@lru_cache(maxsize=1)
def _fetch_and_cache_cik_map(headers_tuple):
    """
    Fetches and caches the company ticker to CIK mapping from SEC.gov.
    The CIKs are padded with leading zeros to 10 digits as required by some SEC APIs.
    This function is cached based on the headers_tuple to allow different User-Agents
    to have potentially different cache entries if needed, though typically it's one map.

    Returns:
        dict: A dictionary where keys are uppercase ticker symbols and values
              are 10-digit CIK strings. Returns an empty dict on error.
    """
    _rate_limit()
    headers = dict(headers_tuple)  # Convert back from tuple for requests
    url = f"{SEC_BASE_URL}/files/company_tickers.json"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # The JSON structure is a dict with numeric keys, each value is a dict like {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
        # Create a dictionary mapping ticker to CIK (padded to 10 digits)
        cik_map = {
            item["ticker"].upper(): str(item["cik_str"]).zfill(10)
            for item in data.values()
        }
        return cik_map
    except requests.exceptions.RequestException as e:
        print(
            f"Error fetching CIK map with User-Agent '{headers.get('User-Agent')}': {e}"
        )
        return {}


def _get_cik_from_map(symbol, cik_map):
    """
    Retrieves the Central Index Key (CIK) for a given stock ticker symbol.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL").

    Returns:
        str: The 10-digit CIK as a string, or None if not found.
    """
    # Manual overrides for companies with incorrect CIKs in SEC company tickers
    manual_overrides = {
        "BLK": "0001364742",  # Correct CIK for BlackRock, Inc.
        "DLO": "0001846832",  # Correct CIK for DLocal.
    }

    # Check manual overrides first
    if symbol.upper() in manual_overrides:
        return manual_overrides[symbol.upper()]

    # Fall back to SEC company tickers
    return cik_map.get(symbol.upper())


def _get_company_facts_request(symbol_or_cik, headers, get_cik_func):
    """
    Fetches all company facts (XBRL disclosures) for a given company.
    This provides a comprehensive dataset for a company, including various
    financial concepts and their reported values over different periods.

    Args:
        symbol_or_cik (str): The stock ticker symbol (e.g., "AAPL") or
                             the 10-digit CIK (e.g., "0000320193").
        headers (dict): The HTTP headers to use for the request.
        get_cik_func (callable): A function to resolve a symbol to a CIK.

    Returns:
        dict: A dictionary containing all company facts in JSON format,
              or None if the data cannot be retrieved.
    """
    cik = symbol_or_cik
    if not cik.isdigit() or len(cik) != 10:
        cik = get_cik_func(symbol_or_cik)  # Use the passed function
        if not cik:
            print(f"Error: Could not find CIK for symbol: {symbol_or_cik}")
            return None

    _rate_limit()
    url = f"{DATA_SEC_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching company facts for CIK {cik}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error fetching company facts for CIK {cik}: {e}")
        return None


def _get_company_concept_request(symbol_or_cik, taxonomy, tag, headers, get_cik_func):
    """
    Fetches specific XBRL concept data for a given company.
    Examples of taxonomy: "us-gaap", "ifrs-full", "dei", "srt"
    Examples of tag: "Revenues", "Assets", "NetIncomeLoss", "EarningsPerShareBasic"

    Args:
        symbol_or_cik (str): The stock ticker symbol (e.g., "AAPL") or
                             the 10-digit CIK (e.g., "0000320193").
        taxonomy (str): The XBRL taxonomy (e.g., "us-gaap").
        tag (str): The XBRL tag/concept (e.g., "Revenues").
        headers (dict): The HTTP headers to use for the request.
        get_cik_func (callable): A function to resolve a symbol to a CIK.

    Returns:
        dict: A dictionary containing the concept data in JSON format,
              or None if the data cannot be retrieved.
    """
    cik = symbol_or_cik
    if not cik.isdigit() or len(cik) != 10:
        cik = get_cik_func(symbol_or_cik)  # Use the passed function
        if not cik:
            print(f"Error: Could not find CIK for symbol: {symbol_or_cik}")
            return None

    _rate_limit()
    url = f"{DATA_SEC_BASE_URL}/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{tag}.json"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching concept '{tag}' for CIK {cik}: {e}")
        return None


def _get_frames_data_request(
    taxonomy, tag, unit, year, headers, quarter=None, instantaneous=False
):
    """
    Fetches aggregated XBRL data across reporting entities for a specific concept
    and calendrical period. This API aggregates one fact for each reporting entity
    that is last filed that most closely fits the calendrical period requested.

    Args:
        taxonomy (str): The XBRL taxonomy (e.g., "us-gaap").
        tag (str): The XBRL tag/concept (e.g., "Assets").
        unit (str): The unit of measure (e.g., "USD", "shares").
        year (int): The calendar year (e.g., 2023).
        headers (dict): The HTTP headers to use for the request.
        quarter (int, optional): The quarter (1, 2, 3, or 4). If None, fetches annual data.
        instantaneous (bool, optional): True for instantaneous data (e.g., balance sheet items),
                                        False for duration data (e.g., income statement items).
                                        Defaults to False.

    Returns:
        dict: A dictionary containing the aggregated frame data in JSON format,
              or None if the data cannot be retrieved.
    """
    period = f"CY{year}"
    if quarter:
        period += f"Q{quarter}"
    if instantaneous:
        period += "I"  # Suffix 'I' for instantaneous periods

    _rate_limit()
    url = f"{DATA_SEC_BASE_URL}/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching frames data for {tag} in {period}: {e}")
        return None


def _get_financial_statement_data(
    symbol_or_cik,
    statement_type,
    limit,
    report_type,
    headers,
    get_cik_func,
    get_company_facts_func,
):
    """
    Fetches and formats basic financial statement data for a given symbol.
    This function aims to provide a simplified, JSON structure for
    common financial statements.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL").
        statement_type (str): Type of statement ("income_statement",
                                        "balance_sheet", "cash_flow").
        limit (int): Number of most recent report periods to retrieve.
        report_type (str): The type of report to filter by.
                           Can be "10-K", "10-Q", or "ALL" (default).
        headers (dict): HTTP headers for requests.
        get_cik_func (callable): Function to get CIK.
        get_company_facts_func (callable): Function to get company facts.

    Returns:
        list: A list of dictionaries, where each dictionary represents a period's
              financial data. Returns an empty list if data cannot be retrieved or
              statement type is invalid.
    """
    # Resolve CIK using the provided function
    cik = (
        symbol_or_cik
        if (
            isinstance(symbol_or_cik, str)
            and symbol_or_cik.isdigit()
            and len(symbol_or_cik) == 10
        )
        else get_cik_func(symbol_or_cik)
    )
    if not cik:
        print(f"Error: Could not find CIK for: {symbol_or_cik}")
        return []

    def _get_financial_value(data_dict, primary_tag, alternate_tags=None, default=0):
        """
        Retrieves a financial value from a dictionary of financial data.

        This helper function attempts to find a value associated with a `primary_tag`.
        If the `primary_tag` is not found or its value is None, it will then try
        any `alternate_tags` provided, in the order they are listed.
        If no suitable tag yields a non-None value, the `default` value is returned.

        Args:
            data_dict (dict): The dictionary containing financial data, where keys are
                              XBRL tags and values are the reported financial figures.
            primary_tag (str): The preferred XBRL tag to look for.
            alternate_tags (str or list, optional): A single XBRL tag string or a list of
                                                   XBRL tag strings to try if the `primary_tag`
                                                   is not found or its value is None. Defaults to None.
            default (any, optional): The value to return if no tag provides a non-None value.
                                     Defaults to 0.

        Returns:
            any: The financial value found, or the `default` value.
        """
        tags_to_try = [primary_tag]
        if alternate_tags:
            if isinstance(alternate_tags, str):
                tags_to_try.append(alternate_tags)
            elif isinstance(alternate_tags, list):
                tags_to_try.extend(alternate_tags)

        for tag in tags_to_try:
            if tag in data_dict:
                val = data_dict.get(tag)
                if val is not None:
                    return val
        return default

    company_facts = get_company_facts_func(cik)  # Use the passed function
    if not company_facts:
        return []

    # Define common US GAAP tags for each statement type
    # This is a simplified list; real financial statements have many more tags.
    # Users can extend this list based on their needs by exploring
    # company_facts data.
    statement_tags = {
        "income_statement": [
            "BusinessAcquisitionsProFormaRevenue",
            "BusinessCombinationProFormaInformationRevenueOfAcquireeSinceAcquisitionDateActual",
            "ContractWithCustomerAssetCumulativeCatchUpAdjustmentToRevenueChangeInMeasureOfProgress",
            "ContractWithCustomerLiabilityCumulativeCatchUpAdjustmentToRevenueChangeInMeasureOfProgress",
            "ContractWithCustomerLiabilityRevenueRecognized",
            "CostOfGoodsAndServicesSold",
            "CostOfRevenue",
            "DeferredRevenue",
            "DeferredRevenueAdditions",
            "DeferredRevenueCurrent",
            "DeferredRevenueNoncurrent",
            "DeferredRevenueRevenueRecognized1",
            "DepreciationDepletionAndAmortization",
            "DisposalGroupIncludingDiscontinuedOperationDeferredRevenueCurrent",
            "EarningsPerShareBasic",
            "EarningsPerShareDiluted",
            "FinanceLeaseInterestExpense",
            "FinancialGuaranteeInsuranceContractsAcceleratedPremiumRevenueAmount",
            "GeneralAndAdministrativeExpense",
            "GrossProfit",
            "IncomeBeforeIncomeTax",
            "IncomeBeforeTax",
            "IncomeLossFromContinuingOperationsBeforeIncomeTax",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeTaxExpenseBenefit",
            "IncreaseDecreaseInDeferredRevenue",
            "InsuranceServicesRevenue",
            "InterestExpense",
            "InterestIncome",
            "InterestIncomeOther",
            "InterestIncomeExpenseNet",
            "InterestRevenueExpenseNet",
            "MarketDataRevenue",
            "NetIncomeLoss",
            "NetIncomeLossFromDiscontinuedOperationsNetOfTax",
            "NonoperatingIncomeLoss",
            "OperatingExpenses",
            "OperatingIncomeLoss",
            "OtherCostOfOperatingRevenue",
            "OtherNonoperatingIncomeExpense",
            "OtherOperatingExpenses",
            "ProfitLossBeforeTax",
            "RelatedPartyTransactionOtherRevenuesFromTransactionsWithRelatedParty",
            "ResearchAndDevelopmentExpense",
            "RevenueFromCollaborativeArrangementExcludingRevenueFromContractWithCustomer",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "RevenueFromRelatedParties",
            "RevenueNotFromContractWithCustomer",
            "RevenueRemainingPerformanceObligation",
            "RevenueRemainingPerformanceObligationPercentage",
            "Revenues",
            "RoyaltyIncomeNonoperating",
            "SalesRevenueGoodsNet",
            "SalesRevenueNet",
            "SalesRevenueServicesNet",
            "SellingAndMarketingExpense",
            "SellingGeneralAndAdministrativeExpense",
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "WeightedAverageNumberOfSharesOutstandingBasic",
        ],
        "balance_sheet": [
            "CommonStockSharesOutstanding",  # Added for shares outstanding
            "WeightedAverageNumberOfDilutedSharesOutstanding",  # Added as a fallback
            "AccountsPayableCurrent",
            "AccountsReceivableNetCurrent",
            "AccountsReceivableTradeCurrent",
            "AccountsReceivableNet",  # Alternative tag used by LEN
            "AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment",
            "AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipmentPeriodIncreaseDecrease",
            "AccumulatedDepreciationDepletionAndAmortizationSaleOfPropertyPlantAndEquipment1",
            "AccumulatedOtherComprehensiveIncomeLossNetOfTax",
            "AdditionalPaidInCapital",
            "AccruedLiabilitiesCurrent",
            "Assets",
            "AssetsCurrent",
            "AssetsNoncurrent",
            "CapitalLeaseObligationsCurrent",
            "CapitalLeaseObligationsNoncurrent",
            "CashAndCashEquivalents",
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",  # Alternative tag used by LEN in recent years
            "CommercialPaper",
            "CommonStock",
            "CommonStockValue",
            "CommonStocksIncludingAdditionalPaidInCapital",
            "ContractWithCustomerLiabilityCurrent",
            "ContractWithCustomerLiabilityNoncurrent",
            "Debt",
            "DebtCurrent",
            "DeferredRevenueCurrent",
            "DeferredRevenueNoncurrent",
            "DeferredTaxAssetsNet",
            "DeferredTaxAssetsPropertyPlantAndEquipment",
            "DeferredTaxLiabilitiesNoncurrent",
            "DeferredTaxLiabilitiesPropertyPlantAndEquipment",
            "Depreciation",
            "DepreciationAndAmortization",
            "FinanceLeaseRightOfUseAsset",
            "Goodwill",
            "IncomeTaxesPayable",
            "IncreaseDecreaseInDeferredRevenue",
            "IncreaseDecreaseInAccountsPayableAndAccruedLiabilities",  # Alternative tag used by LEN in recent years
            "IntangibleAssets",
            "IntangibleAssetsNetExcludingGoodwill",
            "Inventory",
            "InventoryNet",
            "InventoryOperativeBuilders",  # Alternative tag used by LEN
            "InventoryAdjustments",  # Alternative tag used by LEN in recent years
            "AccountsAndNotesReceivableNet",  # Alternative tag used by LEN in recent years
            "Liabilities",
            "LiabilitiesCurrent",
            "LiabilitiesNoncurrent",
            "LongTermDebt",  # Alternative tag used by LEN
            "LongTermDebtCurrent",
            "LongTermDebtCurrentMaturities",
            "LongTermDebtCurrentMaturitiesAndOtherShortTermDebt",
            "LongTermDebtNoncurrent",
            "LongTermInvestments",
            "MarketableSecuritiesCurrent",
            "MarketableSecuritiesNoncurrent",
            "MinorityInterest",
            "NoncontrollingInterest",
            "NotesPayableCurrent",
            "OperatingLeaseLiabilityCurrent",
            "OperatingLeaseLiabilityNoncurrent",
            "OtherAccountsPayableCurrent",
            "OtherAssetsCurrent",
            "OtherAssetsNoncurrent",
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PreferredStockValue",
            "PrepaidExpenseCurrent",
            "ProceedsFromSaleOfPropertyPlantAndEquipment",
            "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
            "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAccumulatedDepreciationAndAmortization",
            "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetBeforeAccumulatedDepreciationAndAmortization",
            "PropertyPlantAndEquipmentDisposals",
            "PropertyPlantAndEquipmentGross",
            "PropertyPlantAndEquipmentNet",
            "RedeemablePreferredStockCarryingAmount",
            "RetainedEarnings",
            "RetainedEarningsAccumulatedDeficit",
            "ShortTermBorrowings",
            "ShortTermInvestments",
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "TotalAssets",
            "TotalDebt",
            "TotalLiabilities",
            "TreasuryStockValue",
        ],
        "cash_flow": [
            "Cash",
            "CashAndCashEquivalentsAtBeginningOfPeriod",
            "CashAndCashEquivalentsAtCarryingValue",
            "CashAndCashEquivalentsPeriodIncreaseDecrease",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsAtBeginningOfPeriod",
            "DeferredIncomeTaxExpenseBenefit",
            "DepreciationDepletionAndAmortization",
            "EffectOfExchangeRateOnCashAndCashEquivalents",
            "IncreaseDecreaseInAccountsPayableCurrent",
            "IncreaseDecreaseInAccountsReceivableNetCurrent",
            "IncreaseDecreaseInDeferredIncomeTaxes",
            "IncreaseDecreaseInInventoriesNet",
            "IncreaseDecreaseInOtherOperatingAssetsLiabilitiesNet",
            "IncomeTaxesPaidNet",
            "InterestPaidNet",
            "NetCashProvidedByUsedInFinancingActivities",
            "NetCashProvidedByUsedInInvestingActivities",
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            "NetIncomeLoss",
            "NetOperatingCashFlow",
            "OperatingCashFlow",
            "OtherFinancingActivitiesCashFlows",
            "OtherInvestingActivitiesCashFlows",
            "OtherNoncashIncomeExpense",
            "PaymentsForPurchasesOfInvestments",
            "PaymentsForRepurchaseOfCommonStock",
            "PaymentsForRepurchaseOfPreferredStock",
            "PaymentsOfDividendsCommonStock",
            "PaymentsOfDividendsPreferredStock",
            "PaymentsToAcquireBusinessesNetOfCashAcquired",
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "ProceedsFromIssuanceOfCommonStock",
            "ProceedsFromIssuanceOfLongTermDebt",
            "ProceedsFromIssuanceOfPreferredStock",
            "ProceedsFromSaleAndMaturityOfMarketableSecurities",
            "ProceedsFromShortTermDebt",
            "RepaymentsOfLongTermDebt",
            "RepaymentsOfShortTermDebt",
            "ShareBasedCompensation",
        ],
    }

    if statement_type not in statement_tags:
        print(
            f"Error: Invalid financial statement type: {statement_type}. Choose from: {', '.join(statement_tags.keys())}"
        )
        return []

    required_tags = statement_tags[statement_type]

    # Step 1: Collect data for each unique report instance (filing).
    # Key: (form_group, end_date, filed_at)
    # Value: Dictionary holding report details and its financial data.
    report_instances_data = {}

    facts_section = company_facts.get("facts", {})
    us_gaap_facts = facts_section.get("us-gaap", {})

    # Define EPS and share count tags for special handling
    eps_tags = {
        "EarningsPerShareBasic",
        "EarningsPerShareDiluted",
        "eps",
        "epsDiluted",
        "earningsPerShareBasic",
        "earningsPerShareDiluted",
    }
    share_count_tags = {
        "WeightedAverageShsOut",
        "WeightedAverageShsOutDil",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "CommonStockSharesOutstanding",
    }

    for tag in required_tags:
        concept_data = us_gaap_facts.get(tag, {})
        for unit_type, facts_list in concept_data.get("units", {}).items():
            # Group facts by (form_type, filed_at, end_date, start_date)
            values_by_key = {}
            for fact in facts_list:
                fiscal_year = fact.get("fy")
                fiscal_period = fact.get("fp")
                form_type = fact.get("form")
                filed_at = fact.get("filed")
                value = fact.get("val")
                end_date = fact.get("end")
                start_date = fact.get("start")

                if not (
                    form_type
                    and filed_at
                    and end_date
                    and fiscal_year is not None
                    and fiscal_period
                    and value is not None
                ):
                    continue

                form_group = None
                if form_type.upper().startswith("10-K"):
                    form_group = "10-K"
                elif form_type.upper().startswith("10-Q"):
                    form_group = "10-Q"
                else:
                    continue  # Only process 10-K and 10-Q related forms

                report_instance_key = (form_group, end_date, filed_at)
                if report_instance_key not in report_instances_data:
                    report_instances_data[report_instance_key] = {
                        "symbol": (
                            symbol_or_cik.upper()
                            if isinstance(symbol_or_cik, str)
                            and not symbol_or_cik.isdigit()
                            else "N/A_CIK_USED"
                        ),
                        "fiscalYear": fiscal_year,
                        "fiscalPeriod": fiscal_period,
                        "formType": form_type,
                        "formGroup": form_group,
                        "filedAt": filed_at,
                        "endDate": end_date,
                        "startDate": start_date,
                        "data": {},
                    }
                # Collect all values for this tag/period/form
                values_by_key.setdefault(report_instance_key, []).append(value)

            # After collecting all values for this tag/unit, select the best one
            for report_instance_key, values in values_by_key.items():
                data_dict = report_instances_data[report_instance_key]["data"]
                if tag in eps_tags:
                    # EPS: pick the highest value
                    max_val = max(values)
                    data_dict[tag] = max_val
                elif tag in share_count_tags:
                    # Share count: pick the lowest value
                    min_val = min(values)
                    data_dict[tag] = min_val
                else:
                    # Other tags: pick the first value
                    data_dict[tag] = values[0]

    # Step 2: Determine the canonical (latest filed) report for each (form_group, end_date)
    # If the latest report has insufficient data, fall back to the second most
    # recent
    canonical_reports = {}

    # Group reports by (form_group, end_date) and sort by filedAt (earliest first)
    reports_by_period = {}
    for report_obj in report_instances_data.values():
        key = (report_obj["formGroup"], report_obj["endDate"])
        if key not in reports_by_period:
            reports_by_period[key] = []
        reports_by_period[key].append(report_obj)

    # Sort each group by filedAt (earliest first)
    for key in reports_by_period:
        reports_by_period[key].sort(key=lambda r: r["filedAt"])  # ascending

    # Define key balance sheet items to check for data completeness
    key_balance_sheet_items = [
        "CashAndCashEquivalentsAtCarryingValue",
        "AssetsCurrent",
        "Assets",
        "LiabilitiesCurrent",
        "StockholdersEquity",
    ]

    # Define key income statement items to check for data completeness
    key_income_statement_items = [
        "Revenues",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "OperatingIncomeLoss",
        "NetIncomeLoss",
        "CostOfRevenue",
        "GrossProfit",
    ]

    # Define key cash flow items to check for data completeness
    key_cash_flow_items = [
        "NetCashProvidedByUsedInOperatingActivities",
        "OperatingCashFlow",
        "NetOperatingCashFlow",
        "NetIncomeLoss",
        "DepreciationDepletionAndAmortization",
    ]

    # Select appropriate key items based on statement type
    key_items_to_check = []
    if statement_type == "balance_sheet":
        key_items_to_check = key_balance_sheet_items
    elif statement_type == "income_statement":
        key_items_to_check = key_income_statement_items
    elif statement_type == "cash_flow":
        key_items_to_check = key_cash_flow_items

    for key, reports in reports_by_period.items():
        if not reports:
            continue

        # Start with the earliest report (as-reported, pre-split)
        selected_report = reports[0]

        # Check if the earliest report has sufficient data
        data = selected_report.get("data", {})
        num_nonzero_items = sum(
            1 for item in key_items_to_check if data.get(item, 0) != 0
        )
        has_sufficient_data = num_nonzero_items >= 2

        # If the earliest report has insufficient data, try subsequent reports
        if not has_sufficient_data and len(reports) > 1:
            for i in range(1, len(reports)):
                candidate_report = reports[i]
                candidate_data = candidate_report.get("data", {})
                candidate_num_nonzero_items = sum(
                    1 for item in key_items_to_check if candidate_data.get(item, 0) != 0
                )
                candidate_has_sufficient_data = candidate_num_nonzero_items >= 2

                # If this report has sufficient data, use it
                if candidate_has_sufficient_data:
                    selected_report = candidate_report
                    break

        canonical_reports[key] = selected_report

    all_canonical_reports_list = list(canonical_reports.values())

    # Step 3: Separate into 10-K and 10-Q lists
    ten_k_reports = [r for r in all_canonical_reports_list if r["formGroup"] == "10-K"]
    ten_q_reports = [r for r in all_canonical_reports_list if r["formGroup"] == "10-Q"]

    # Step 4: Sort each list by endDate (primary) and filedAt (secondary, for
    # tie-breaking), most recent first
    def sort_key_func(r):
        return (
            r.get("endDate", "0000-00-00"),
            r.get("filedAt", "0000-00-00T00:00:00Z"),
        )

    ten_k_reports.sort(key=sort_key_func, reverse=True)
    ten_q_reports.sort(key=sort_key_func, reverse=True)

    # Step 4.1: For 10-Ks, determine the fiscal year-end (most common endDate month-day)
    ten_k_end_dates = [r["endDate"] for r in ten_k_reports]
    # Extract month-day (e.g., '12-31')
    ten_k_month_days = [d[5:] for d in ten_k_end_dates if len(d) == 10]
    from collections import Counter

    fiscal_year_end_md = None
    if ten_k_month_days:
        # Get the most common month-day pattern
        month_day_counts = Counter(ten_k_month_days)
        most_common = month_day_counts.most_common(1)[0]
        fiscal_year_end_md = most_common[0]
        most_common_count = most_common[1]
        total_reports = len(ten_k_month_days)

        # Check if there are recent reports with different month-day patterns
        # Sort reports by date to identify recent patterns
        ten_k_reports_sorted = sorted(
            ten_k_reports, key=lambda r: r["endDate"], reverse=True
        )

        # Look at the most recent reports (up to 5) to see if there's a pattern change
        recent_month_days = [r["endDate"][5:] for r in ten_k_reports_sorted[:5]]
        recent_counts = Counter(recent_month_days)

        # If recent reports show a different dominant pattern, use that instead
        if len(recent_counts) > 0:
            recent_most_common = recent_counts.most_common(1)[0]
            recent_pattern = recent_most_common[0]
            recent_count = recent_most_common[1]

            # If recent pattern is different and appears in at least 2 recent reports
            if recent_pattern != fiscal_year_end_md and recent_count >= 2:
                print(
                    f"[DEBUG] Detected recent fiscal year-end change: {fiscal_year_end_md} -> {recent_pattern}"
                )
                fiscal_year_end_md = recent_pattern

        # Check if the fiscal year-end pattern is consistent enough to filter by
        # If the most common pattern represents less than 30% of all reports,
        # or if there are too many different patterns, don't filter
        pattern_consistency_ratio = most_common_count / total_reports
        unique_patterns = len(month_day_counts)

        # Don't filter if:
        # 1. Most common pattern represents less than 30% of reports, OR
        # 2. There are more than 8 unique patterns (indicating inconsistency)
        if pattern_consistency_ratio < 0.3 or unique_patterns > 8:

            fiscal_year_end_md = None

    # Filter 10-Ks to only those matching fiscal year-end (if pattern is consistent)
    if fiscal_year_end_md:
        ten_k_reports = [
            r for r in ten_k_reports if r["endDate"][5:] == fiscal_year_end_md
        ]

    # Step 5: Select reports based on report_type and apply limit
    selected_reports = []
    report_type_upper = report_type.upper()

    if report_type_upper == "10-K":
        selected_reports = ten_k_reports
    elif report_type_upper == "10-Q":
        selected_reports = ten_q_reports
    elif report_type_upper == "ALL":
        # Combine, sort, then limit
        combined_reports = ten_k_reports + ten_q_reports
        # Ensure overall chronological order
        combined_reports.sort(key=sort_key_func, reverse=True)
        selected_reports = combined_reports
    else:
        print(f"Warning: Invalid report_type '{report_type}'. Defaulting to 'ALL'.")
        combined_reports = ten_k_reports + ten_q_reports
        combined_reports.sort(key=sort_key_func, reverse=True)
        selected_reports = combined_reports

    # Step 6: Deduplicate by endDate (period end), patch original with amendments if present
    reports_by_end_date = defaultdict(list)
    for report in selected_reports:
        reports_by_end_date[report["endDate"]].append(report)

    deduped_reports = []
    for end_date, reports in reports_by_end_date.items():
        originals = [
            r for r in reports if r.get("formType", "").upper() in ("10-K", "10-Q")
        ]
        amendments = [
            r for r in reports if r.get("formType", "").upper() in ("10-K/A", "10-Q/A")
        ]
        if originals:
            originals.sort(key=lambda r: r.get("filedAt", ""), reverse=True)
            base = originals[0].copy()
            # Patch the base with all amendments in chronological order (oldest to newest)
            if amendments:
                amendments.sort(key=lambda r: r.get("filedAt", ""))  # oldest to newest
                for amendment in amendments:
                    for k, v in amendment.get("data", {}).items():
                        if v is not None:
                            base["data"][k] = v
                    base["filingDate"] = amendment.get(
                        "filingDate", base.get("filingDate")
                    )
                    base["acceptedDate"] = amendment.get(
                        "acceptedDate", base.get("acceptedDate")
                    )
                    base["_formType_original"] = amendment.get(
                        "formType", base.get("formType")
                    )
            best = base
        elif amendments:
            amendments.sort(key=lambda r: r.get("filedAt", ""), reverse=True)
            best = amendments[0]
        else:
            continue
        # Preserve the original fiscal year from SEC data instead of deriving from end date
        # best["fiscalYear"] = best["endDate"][:4]  # This was incorrect for companies with non-calendar fiscal years
        deduped_reports.append(best)

    # Step 6.1: Additional deduplication for 10-K reports by fiscal year
    # For 10-K reports, we should only keep one report per fiscal year
    if report_type_upper == "10-K":
        reports_by_fiscal_year = defaultdict(list)
        for report in deduped_reports:
            if report["formGroup"] == "10-K":
                fiscal_year = report["fiscalYear"]
                reports_by_fiscal_year[fiscal_year].append(report)

        # For each fiscal year, keep the report with the latest end date
        # If multiple reports have the same end date, keep the one with the latest filing date
        final_deduped_reports = []
        for fiscal_year, reports in reports_by_fiscal_year.items():
            if len(reports) > 1:
                # Sort by end date (descending), then by filing date (descending)
                reports.sort(key=lambda r: (r["endDate"], r["filedAt"]), reverse=True)

            final_deduped_reports.append(reports[0])

        # Add any non-10-K reports (shouldn't happen with report_type="10-K", but just in case)
        for report in deduped_reports:
            if report["formGroup"] != "10-K":
                final_deduped_reports.append(report)

        deduped_reports = final_deduped_reports

    # Step 7: Apply the limit to the deduplicated, patched reports
    deduplicated_reports = deduped_reports[:limit]

    # Step 8: Format results
    formatted_results = []
    for period_details in deduplicated_reports:
        data = period_details.get("data", {})

        # Use _get_financial_value as before
        def get_val(tag, preferred_unit=None, alternate_tags=None, default=0):
            tags_to_try = [tag]
            if alternate_tags:
                if isinstance(alternate_tags, str):
                    tags_to_try.append(alternate_tags)
                elif isinstance(alternate_tags, list):
                    tags_to_try.extend(alternate_tags)
            for t in tags_to_try:
                if t in data:
                    val = _get_financial_value(data, t)
                    if val is not None:
                        return val
            return default

        # Determine period string (FY, Q1, Q2, etc.)
        period_val = period_details["fiscalPeriod"]
        if period_details["formGroup"] == "10-K":
            period_val = "FY"

        # Revenue: Try a list of common tags in order of preference.
        # Ensure these tags are included in statement_tags["income_statement"]
        # above.
        revenue = 0  # Default value
        revenue_possible_tags = [
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
            "SalesRevenueServicesNet",
        ]
        for r_tag in revenue_possible_tags:
            if r_tag in data:  # Check if the tag was found and data collected for it
                revenue = get_val(r_tag)
                # If we found a non-zero revenue, prefer it. If it's 0,
                # continue checking.
                if revenue != 0:
                    break  # Use the first non-zero revenue found from the preferred list
        # If all found tags resulted in 0, revenue remains 0. If no tags were
        # found, revenue remains 0.

        costOfRevenue = 0
        costOfRevenue_possible_tags = ["CostOfRevenue", "CostOfGoodsAndServicesSold"]
        for tag in costOfRevenue_possible_tags:
            if tag in data:
                costOfRevenue = get_val(tag)
                if costOfRevenue != 0:
                    break

        # GrossProfit can be explicitly found or calculated.
        # If GrossProfit tag exists and is non-zero, use it. Otherwise,
        # calculate.
        grossProfit_explicit = get_val("GrossProfit")
        grossProfit = (
            grossProfit_explicit
            if grossProfit_explicit != 0
            else (revenue - costOfRevenue)
        )
        # If GrossProfit was explicitly 0, but revenue and CoR allow calculation, it will be calculated.
        # If GrossProfit tag was not found, it will be calculated.

        generalAndAdministrativeExpenses = 0
        sellingAndMarketingExpenses = 0
        researchAndDevelopmentExpense = get_val("ResearchAndDevelopmentExpense")
        sga_combined = get_val("SellingGeneralAndAdministrativeExpense")
        ga_separate = get_val("GeneralAndAdministrativeExpense")
        sm_separate = get_val("SellingAndMarketingExpense")
        otherOperatingExpenses_val = get_val("OtherOperatingExpenses")

        if ga_separate != 0 or sm_separate != 0:  # Prefer separate tags if available
            generalAndAdministrativeExpenses = ga_separate
            sellingAndMarketingExpenses = sm_separate
        elif sga_combined != 0:  # Use combined SG&A
            generalAndAdministrativeExpenses = sga_combined

        otherExpenses_val = otherOperatingExpenses_val

        # Calculate total operating expenses
        # Use the specific 'OperatingExpenses' tag if available and non-zero,
        # otherwise sum components.
        operatingExpenses_from_tag = get_val("OperatingExpenses")
        calculated_operating_components_sum = (
            researchAndDevelopmentExpense
            + generalAndAdministrativeExpenses
            + sellingAndMarketingExpenses
            + otherExpenses_val
        )

        # Use explicit if non-zero
        operatingExpenses = (
            operatingExpenses_from_tag
            if operatingExpenses_from_tag != 0
            else calculated_operating_components_sum
        )

        # Derived (Cost of Revenue + All Operating Expenses)
        costAndExpenses = costOfRevenue + operatingExpenses

        # Interest Income / Expense
        # Prioritize discrete InterestIncome and InterestExpense. Fallback to
        # InterestIncomeExpenseNet.
        interestIncome_val = get_val("InterestIncome")
        interestExpense_val = get_val("InterestExpense")
        interestIncomeExpenseNet_val = get_val("InterestIncomeExpenseNet")

        if (
            interestIncome_val == 0
            and interestExpense_val == 0
            and interestIncomeExpenseNet_val != 0
        ):
            if interestIncomeExpenseNet_val > 0:
                interestIncome_val = interestIncomeExpenseNet_val
            else:  # interestIncomeExpenseNet_val < 0
                interestExpense_val = abs(interestIncomeExpenseNet_val)
        # If gross values were present, they are used. If only net was present, it's now split.
        # If all were zero, they remain zero.

        netInterestIncome = interestIncome_val - interestExpense_val  # Derived

        depreciationAndAmortization = get_val("DepreciationDepletionAndAmortization")

        operatingIncome = get_val("OperatingIncomeLoss")

        # If operatingIncome is zero from tag, try alternative methods
        if operatingIncome == 0:
            # Method 1: Try working backwards from pre-tax income (most accurate when available)
            preTaxIncome = get_val(
                "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"
            )
            if preTaxIncome != 0:
                # Add back financing costs and non-operating items
                interestExpense = get_val("InterestExpense")
                interestIncome = get_val("InterestIncomeOther")
                otherNonOperating = get_val("OtherNonoperatingIncomeExpense")
                royaltyIncomeNonOperating = get_val("RoyaltyIncomeNonoperating")
                financeLeaseInterest = get_val("FinanceLeaseInterestExpense")

                # Calculate net non-operating expenses to add back
                netInterestExpense = (
                    interestExpense + financeLeaseInterest - interestIncome
                )
                # Note: OtherNonoperatingIncomeExpense is typically negative for expenses
                # RoyaltyIncomeNonoperating is positive income, so we subtract it
                netOtherNonOperating = -otherNonOperating - royaltyIncomeNonOperating

                # Operating Income = Pre-tax Income + Net Interest Expense + Net Other Non-operating Expenses
                operatingIncome = (
                    preTaxIncome + netInterestExpense + netOtherNonOperating
                )

                if operatingIncome != 0:
                    logger.info(
                        f"[DEBUG] Derived operating income from pre-tax income: {operatingIncome} = {preTaxIncome} + {netInterestExpense} + {netOtherNonOperating}"
                    )

            # Method 2: If pre-tax income method didn't work, fall back to gross profit calculation
            if operatingIncome == 0:
                if grossProfit != 0 and (
                    researchAndDevelopmentExpense != 0
                    or generalAndAdministrativeExpenses != 0
                    or sellingAndMarketingExpenses != 0
                    or otherExpenses_val != 0
                ):
                    operatingIncome = grossProfit - (
                        researchAndDevelopmentExpense
                        + generalAndAdministrativeExpenses
                        + sellingAndMarketingExpenses
                        + otherExpenses_val
                        + depreciationAndAmortization
                    )  # More complete OpEx for derivation

                    if operatingIncome != 0:
                        logger.info(
                            f"[DEBUG] Derived operating income from gross profit: {operatingIncome} = {grossProfit} - expenses"
                        )

        ebitda = operatingIncome + depreciationAndAmortization  # Derived
        ebit = operatingIncome  # ebit is Operating Income

        totalOtherIncomeExpensesNet = get_val("NonoperatingIncomeLoss")

        incomeBeforeTax = 0
        incomeBeforeTax_possible_tags = [
            "IncomeLossFromContinuingOperationsBeforeIncomeTax",
            "IncomeBeforeIncomeTax",
            "IncomeBeforeTax",
            "ProfitLossBeforeTax",
        ]
        for tag in incomeBeforeTax_possible_tags:
            if tag in data:
                incomeBeforeTax = get_val(tag)
                if incomeBeforeTax != 0:
                    break
        # Fallback calculation for Income Before Tax (EBT)
        if incomeBeforeTax == 0 and operatingIncome != 0:
            incomeBeforeTax = (
                operatingIncome + netInterestIncome + totalOtherIncomeExpensesNet
            )  # Common derivation

        incomeTaxExpense = get_val("IncomeTaxExpenseBenefit")

        netIncome = get_val("NetIncomeLoss")
        netIncomeFromDiscontinuedOperations = get_val(
            "NetIncomeLossFromDiscontinuedOperationsNetOfTax"
        )
        netIncomeFromContinuingOperations = (
            netIncome - netIncomeFromDiscontinuedOperations
        )  # Derived

        eps = get_val("EarningsPerShareBasic")
        epsDiluted = get_val("EarningsPerShareDiluted")
        weightedAverageShsOut = get_val("WeightedAverageNumberOfSharesOutstandingBasic")
        weightedAverageShsOutDil = get_val(
            "WeightedAverageNumberOfDilutedSharesOutstanding"
        )

        # Base item structure common to all statement types
        base_item = {
            "date": period_details["endDate"],
            # Store original symbol if provided
            "symbol": (
                symbol_or_cik.upper()
                if isinstance(symbol_or_cik, str) and not symbol_or_cik.isdigit()
                else "N/A_CIK_USED"
            ),
            "reportedCurrency": "USD",  # Assuming USD
            "cik": cik,
            "filingDate": period_details["filedAt"][:10],
            "acceptedDate": period_details["filedAt"],
            "fiscalYear": str(period_details["fiscalYear"]),
            "period": period_val,
            "fiscalDateEnding": period_details["endDate"],
            # Internal reference
            "_formType_original": period_details["formType"],
        }

        if statement_type == "income_statement":
            item = {
                **base_item,
                "revenue": revenue,
                "costOfRevenue": costOfRevenue,
                "grossProfit": grossProfit,
                "researchAndDevelopmentExpense": researchAndDevelopmentExpense,
                "generalAndAdministrativeExpenses": generalAndAdministrativeExpenses,
                "sellingAndMarketingExpenses": sellingAndMarketingExpenses,
                "otherExpenses": otherExpenses_val,
                "operatingExpenses": operatingExpenses,
                "costAndExpenses": costAndExpenses,
                "interestIncome": interestIncome_val,
                "interestExpense": interestExpense_val,
                "depreciationAndAmortization": depreciationAndAmortization,
                "ebitda": ebitda,
                "operatingIncome": operatingIncome,
                "totalOtherIncomeExpensesNet": totalOtherIncomeExpensesNet,
                "incomeBeforeTax": incomeBeforeTax,
                "incomeTaxExpense": incomeTaxExpense,
                "netIncome": netIncome,
                "eps": eps,
                "epsDiluted": epsDiluted,
                "weightedAverageShsOut": weightedAverageShsOut,
                "weightedAverageShsOutDil": weightedAverageShsOutDil,
                "ebit": ebit,
                "netIncomeFromContinuingOperations": netIncomeFromContinuingOperations,
                "netIncomeFromDiscontinuedOperations": netIncomeFromDiscontinuedOperations,
                "otherAdjustmentsToNetIncome": 0,
                "netIncomeDeductions": 0,
                "bottomLineNetIncome": netIncome,
            }
        elif statement_type == "balance_sheet":

            # ASSETS
            # Current Assets
            cashAndCashEquivalents = get_val(
                "CashAndCashEquivalentsAtCarryingValue",
                alternate_tags=[
                    "CashAndCashEquivalents",
                    "Cash",
                    "CashAndCashEquivalentsAtCarryingValue",
                    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",  # Alternative tag used by LEN in recent years
                ],
            )

            shortTermInvestments = get_val(
                "MarketableSecuritiesCurrent",
                alternate_tags=[
                    "ShortTermInvestments",
                    "AvailableForSaleSecuritiesCurrent",
                    "MarketableSecuritiesDebtMaturitiesWithinOneYearAmortizedCost",
                ],
            )

            cashAndShortTermInvestments = cashAndCashEquivalents + shortTermInvestments

            netReceivables = get_val(
                "AccountsReceivableNetCurrent",
                alternate_tags=[
                    "ReceivablesNetCurrent",
                    "AccountsReceivableGrossCurrent",
                    "AccountsReceivableNet",  # Alternative tag used by LEN
                    "AccountsAndNotesReceivableNet",  # Alternative tag used by LEN in recent years
                ],
            )

            accountsReceivables = get_val(
                "AccountsReceivableTradeCurrent",
                alternate_tags=["AccountsReceivableGrossCurrent"],
            )
            otherReceivables = get_val(
                "OtherReceivablesCurrent",
                alternate_tags=[
                    "NotesAndLoansReceivableNetCurrent",
                    "ContractReceivableRetainage",
                ],
            )
            inventory = get_val(
                "InventoryNet",
                alternate_tags=[
                    "Inventory",
                    "InventoryFinishedGoods",
                    "InventoryRawMaterials",
                    "InventoryWorkInProcess",
                    "InventoryOperativeBuilders",  # Alternative tag used by LEN
                    "InventoryAdjustments",  # Alternative tag used by LEN in recent years
                ],
            )

            prepaids = get_val(
                "PrepaidExpenseCurrent",
                alternate_tags=["PrepaidExpenseAndOtherAssetsCurrent"],
            )
            otherCurrentAssets = get_val(
                "OtherAssetsCurrent",
                alternate_tags=["OtherAssetsMiscellaneousCurrent"],
            )

            totalCurrentAssets = get_val(
                "AssetsCurrent",
                alternate_tags=[
                    "AssetsHeldForSaleCurrent",
                    "AssetsOfDisposalGroupIncludingDiscontinuedOperationCurrent",
                ],
            )
            if totalCurrentAssets == 0:
                # If the total isn't explicitly stated, sum the components
                totalCurrentAssets = (
                    cashAndShortTermInvestments
                    + netReceivables
                    + inventory
                    + prepaids
                    + otherCurrentAssets
                )

            # Non-Current Assets
            propertyPlantEquipmentNet = get_val(
                "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
                alternate_tags=[
                    "PropertyPlantAndEquipmentNet",
                    "PropertyPlantAndEquipmentGross",
                ],
            )
            # Fallback: try before-accum-depr minus accum-depr if net is missing
            if propertyPlantEquipmentNet == 0:
                ppe_before = get_val(
                    "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetBeforeAccumulatedDepreciationAndAmortization"
                )
                accum_depr = get_val(
                    "AccumulatedDepreciationDepletionAndAmortizationPropertyPlantAndEquipment"
                )
                if ppe_before != 0 and accum_depr != 0:
                    propertyPlantEquipmentNet = ppe_before - accum_depr
            goodwill = get_val(
                "Goodwill",
                alternate_tags=["GoodwillImpairedAccumulatedImpairmentLoss"],
            )
            intangibleAssets = get_val(
                "IntangibleAssetsNetExcludingGoodwill",
                alternate_tags=[
                    "IntangibleAssets",
                    "IntangibleAssetsGrossExcludingGoodwill",
                ],
            )

            goodwillAndIntangibleAssets = get_val("GoodwillAndIntangibleAssets")
            if goodwillAndIntangibleAssets == 0:
                goodwillAndIntangibleAssets = goodwill + intangibleAssets

            longTermInvestments = get_val(
                "MarketableSecuritiesNoncurrent",
                alternate_tags=[
                    "LongTermInvestments",
                    "AvailableForSaleSecuritiesRestrictedNoncurrent",
                ],
            )
            taxAssets = get_val(
                "DeferredTaxAssetsNet",
                alternate_tags=[
                    "DeferredTaxAssetsNetCurrent",
                    "DeferredTaxAssetsNetNoncurrent",
                ],
            )
            otherNonCurrentAssets = get_val(
                "OtherAssetsNoncurrent",
                alternate_tags=["OtherAssetsMiscellaneousNoncurrent"],
            )

            totalNonCurrentAssets = get_val(
                "AssetsNoncurrent", alternate_tags=["NoncurrentAssets"]
            )
            if totalNonCurrentAssets == 0:
                # If the total isn't explicitly stated, sum the components
                totalNonCurrentAssets = (
                    propertyPlantEquipmentNet
                    + goodwillAndIntangibleAssets
                    + longTermInvestments
                    + taxAssets
                    + otherNonCurrentAssets
                )

            # Total Assets
            totalAssets = get_val(
                "Assets",
                alternate_tags=["TotalAssets", "LiabilitiesAndStockholdersEquity"],
            )
            if totalAssets == 0:
                totalAssets = totalCurrentAssets + totalNonCurrentAssets

            # LIABILITIES
            # Current Liabilities
            accountPayables = get_val(
                "AccountsPayableCurrent",
                alternate_tags=[
                    "IncreaseDecreaseInAccountsPayable",
                    "IncreaseDecreaseInAccountsPayableAndAccruedLiabilities",  # Alternative tag used by LEN in recent years
                ],
            )
            otherPayables = get_val("OtherAccountsPayableCurrent")
            totalPayables = accountPayables + otherPayables
            accruedExpenses = get_val(
                "AccruedLiabilitiesCurrent",
                alternate_tags=[
                    "AccruedIncomeTaxesCurrent",
                    "AccruedIncomeTaxesNoncurrent",
                ],
            )

            shortTermDebt = get_val(
                "DebtCurrent",
                alternate_tags=["LongTermDebtAndCapitalLeaseObligationsCurrent"],
            )
            if shortTermDebt == 0:
                component_sum = (
                    get_val("CommercialPaper")
                    + get_val("LongTermDebtCurrentMaturities")
                    + get_val("NotesPayableCurrent")
                    + get_val("ShortTermBorrowings")
                )
                if component_sum == 0:
                    alt_std = get_val(
                        "LongTermDebtCurrentMaturitiesAndOtherShortTermDebt"
                    )
                    if alt_std != 0:
                        component_sum = alt_std
                shortTermDebt = component_sum

            capitalLeaseObligationsCurrent = get_val(
                "OperatingLeaseLiabilityCurrent",
                alternate_tags=["CapitalLeaseObligationsCurrent"],
            )
            taxPayables = get_val("IncomeTaxesPayable")
            deferredRevenue = get_val(
                "DeferredRevenueCurrent",
                alternate_tags=["ContractWithCustomerLiabilityCurrent"],
            )
            otherCurrentLiabilities = get_val("OtherLiabilitiesCurrent")

            totalCurrentLiabilities = get_val(
                "LiabilitiesCurrent",
                alternate_tags=[
                    "LiabilitiesOfDisposalGroupIncludingDiscontinuedOperationCurrent"
                ],
            )
            if totalCurrentLiabilities == 0:
                totalCurrentLiabilities = (
                    accountPayables
                    + otherPayables
                    + accruedExpenses
                    + shortTermDebt
                    + capitalLeaseObligationsCurrent
                    + taxPayables
                    + deferredRevenue
                    + otherCurrentLiabilities
                )

            # Non-Current Liabilities
            longTermDebt_val = get_val(
                "LongTermDebtNoncurrent",
                alternate_tags=["LongTermDebt"],
            )
            capitalLeaseObligationsNonCurrent = get_val(
                "OperatingLeaseLiabilityNoncurrent",
                alternate_tags=["CapitalLeaseObligationsNoncurrent"],
            )
            deferredRevenueNonCurrent = get_val(
                "DeferredRevenueNoncurrent",
                alternate_tags=["ContractWithCustomerLiabilityNoncurrent"],
            )
            deferredTaxLiabilitiesNonCurrent = get_val(
                "DeferredTaxLiabilitiesNoncurrent"
            )
            otherNonCurrentLiabilities = get_val("OtherLiabilitiesNoncurrent")

            totalNonCurrentLiabilities = get_val("LiabilitiesNoncurrent")
            if totalNonCurrentLiabilities == 0:
                totalNonCurrentLiabilities = (
                    longTermDebt_val
                    + capitalLeaseObligationsNonCurrent
                    + deferredRevenueNonCurrent
                    + deferredTaxLiabilitiesNonCurrent
                    + otherNonCurrentLiabilities
                )

            capitalLeaseObligations = (
                capitalLeaseObligationsCurrent + capitalLeaseObligationsNonCurrent
            )

            # Total Liabilities
            totalLiabilities = get_val(
                "Liabilities",
                alternate_tags=["TotalLiabilities", "LiabilitiesAndStockholdersEquity"],
            )
            if totalLiabilities == 0:
                totalLiabilities = totalCurrentLiabilities + totalNonCurrentLiabilities

            # EQUITY
            treasuryStock = get_val("TreasuryStockValue")
            preferredStock = get_val(
                "PreferredStockValue",
                alternate_tags=["RedeemablePreferredStockCarryingAmount"],
            )
            commonStock = get_val(
                "CommonStockValue",
                alternate_tags=[
                    "CommonStock",
                    "CommonStocksIncludingAdditionalPaidInCapital",
                    "CommonStockSharesOutstanding",
                    "CommonStockSharesIssued",
                ],
            )
            additionalPaidInCapital = get_val(
                "AdditionalPaidInCapital",
                alternate_tags=["AdditionalPaidInCapitalCommonStock"],
            )
            if (
                commonStock == get_val("CommonStocksIncludingAdditionalPaidInCapital")
                and additionalPaidInCapital == 0
            ):
                commonStock -= additionalPaidInCapital
                if commonStock < 0:
                    commonStock = get_val("CommonStockValue")
            retainedEarnings = get_val(
                "RetainedEarningsAccumulatedDeficit",
                alternate_tags=["RetainedEarnings"],
            )
            accumulatedOtherComprehensiveIncomeLoss = get_val(
                "AccumulatedOtherComprehensiveIncomeLossNetOfTax"
            )
            minorityInterest = get_val(
                "MinorityInterest", alternate_tags=["NoncontrollingInterest"]
            )

            # Total Equity
            totalStockholdersEquity = get_val(
                "StockholdersEquity",
                alternate_tags=[
                    "TotalStockholdersEquity",
                    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                ],
            )

            if (
                minorityInterest == 0
                and get_val("StockholdersEquity") != totalStockholdersEquity
            ):
                minorityInterest = totalStockholdersEquity - get_val(
                    "StockholdersEquity"
                )

            if totalStockholdersEquity == 0:
                # This is a common calculation, but note that treasury stock is
                # negative equity.
                totalStockholdersEquity = (
                    commonStock
                    + additionalPaidInCapital
                    + retainedEarnings
                    + accumulatedOtherComprehensiveIncomeLoss
                    - treasuryStock
                )

            # DERIVED & SUMMARY METRICS
            totalLiabilitiesAndTotalEquity = totalLiabilities + totalStockholdersEquity
            totalInvestments = shortTermInvestments + longTermInvestments

            totalDebt = get_val("TotalDebt", alternate_tags=["Debt"])
            if totalDebt == 0:
                totalDebt = shortTermDebt + longTermDebt_val + capitalLeaseObligations

            netDebt = totalDebt - cashAndCashEquivalents

            item = {
                **base_item,
                "cashAndCashEquivalents": cashAndCashEquivalents,
                "shortTermInvestments": shortTermInvestments,
                "cashAndShortTermInvestments": cashAndShortTermInvestments,
                "netReceivables": netReceivables,
                # May need refinement based on company specifics
                "accountsReceivables": accountsReceivables,
                "otherReceivables": otherReceivables,  # May need refinement
                "inventory": inventory,
                "prepaids": prepaids,
                "otherCurrentAssets": otherCurrentAssets,
                "totalCurrentAssets": totalCurrentAssets,
                "propertyPlantEquipmentNet": propertyPlantEquipmentNet,
                "goodwill": goodwill,
                "intangibleAssets": intangibleAssets,
                "goodwillAndIntangibleAssets": goodwillAndIntangibleAssets,
                "longTermInvestments": longTermInvestments,
                "taxAssets": taxAssets,
                "otherNonCurrentAssets": otherNonCurrentAssets,
                "totalNonCurrentAssets": totalNonCurrentAssets,
                "otherAssets": 0,  # Default
                "totalAssets": totalAssets,
                "totalPayables": totalPayables,  # Derived
                "accountPayables": accountPayables,
                "otherPayables": otherPayables,
                "accruedExpenses": accruedExpenses,
                "shortTermDebt": shortTermDebt,
                "capitalLeaseObligationsCurrent": capitalLeaseObligationsCurrent,
                "taxPayables": taxPayables,
                "deferredRevenue": deferredRevenue,
                "otherCurrentLiabilities": otherCurrentLiabilities,
                "totalCurrentLiabilities": totalCurrentLiabilities,
                "longTermDebt": longTermDebt_val,
                "capitalLeaseObligationsNonCurrent": capitalLeaseObligationsNonCurrent,
                "deferredRevenueNonCurrent": deferredRevenueNonCurrent,
                "deferredTaxLiabilitiesNonCurrent": deferredTaxLiabilitiesNonCurrent,
                "otherNonCurrentLiabilities": otherNonCurrentLiabilities,
                "totalNonCurrentLiabilities": totalNonCurrentLiabilities,
                "otherLiabilities": 0,  # Default
                "capitalLeaseObligations": capitalLeaseObligations,  # Derived
                "totalLiabilities": totalLiabilities,
                "treasuryStock": treasuryStock,
                "preferredStock": preferredStock,
                "commonStock": commonStock,
                "retainedEarnings": retainedEarnings,
                "additionalPaidInCapital": additionalPaidInCapital,
                "accumulatedOtherComprehensiveIncomeLoss": accumulatedOtherComprehensiveIncomeLoss,
                "otherTotalStockholdersEquity": 0,  # Default
                "totalStockholdersEquity": totalStockholdersEquity,
                "totalEquity": totalStockholdersEquity,
                "minorityInterest": minorityInterest,
                "totalLiabilitiesAndTotalEquity": totalLiabilitiesAndTotalEquity,
                "totalInvestments": totalInvestments,
                "totalDebt": totalDebt,
                "netDebt": netDebt,
                "commonStockSharesOutstanding": get_val(
                    "CommonStockSharesOutstanding",
                    alternate_tags=["WeightedAverageNumberOfDilutedSharesOutstanding"],
                ),
            }
        elif statement_type == "cash_flow":
            # Operating Activities
            netIncome_cf = get_val("NetIncomeLoss")  # Often starting point
            depreciationAndAmortization_cf = get_val(
                "DepreciationDepletionAndAmortization"
            )
            deferredIncomeTax_cf = get_val(
                "DeferredIncomeTaxExpenseBenefit",
                alternate_tags=["IncreaseDecreaseInDeferredIncomeTaxes"],
            )
            stockBasedCompensation_cf = get_val("ShareBasedCompensation")

            accountsReceivables_flow = get_val(
                "IncreaseDecreaseInAccountsReceivableNetCurrent"
            )
            inventory_flow = get_val("IncreaseDecreaseInInventoriesNet")
            accountsPayables_flow = get_val("IncreaseDecreaseInAccountsPayableCurrent")
            otherWorkingCapital_flow = get_val(
                "IncreaseDecreaseInOtherOperatingAssetsLiabilitiesNet"
            )
            changeInWorkingCapital = (
                accountsReceivables_flow
                + inventory_flow
                + accountsPayables_flow
                + otherWorkingCapital_flow
            )  # Sum of individual flow components

            otherNonCashItems_cf = get_val("OtherNoncashIncomeExpense")
            netCashProvidedByOperatingActivities = get_val(
                "NetCashProvidedByUsedInOperatingActivities",
                alternate_tags=["OperatingCashFlow", "NetOperatingCashFlow"],
            )
            # Fallback: sum up main components if all are present and netCashProvidedByOperatingActivities is 0
            if not netCashProvidedByOperatingActivities:
                net_income = get_val("NetIncomeLoss")
                depreciation = get_val("DepreciationDepletionAndAmortization")
                deferred_tax = get_val(
                    "DeferredIncomeTaxExpenseBenefit",
                    alternate_tags=["IncreaseDecreaseInDeferredIncomeTaxes"],
                )
                stock_comp = get_val("ShareBasedCompensation")
                change_wc = (
                    get_val("IncreaseDecreaseInAccountsReceivableNetCurrent")
                    + get_val("IncreaseDecreaseInInventoriesNet")
                    + get_val("IncreaseDecreaseInAccountsPayableCurrent")
                    + get_val("IncreaseDecreaseInOtherOperatingAssetsLiabilitiesNet")
                )
                other_non_cash = get_val("OtherNoncashIncomeExpense")
                # Only use fallback if at least net income is present
                if net_income != 0:
                    netCashProvidedByOperatingActivities = (
                        net_income
                        + depreciation
                        + deferred_tax
                        + stock_comp
                        + change_wc
                        + other_non_cash
                    )

            # Investing Activities
            investmentsInPropertyPlantAndEquipment = get_val(
                "PaymentsToAcquirePropertyPlantAndEquipment"
            )  # Typically negative
            acquisitionsNet_cf = get_val("PaymentsToAcquireBusinessesNetOfCashAcquired")
            purchasesOfInvestments_cf = get_val("PaymentsForPurchasesOfInvestments")
            salesMaturitiesOfInvestments_cf = get_val(
                "ProceedsFromSaleAndMaturityOfMarketableSecurities"
            )
            otherInvestingActivities_cf = get_val("OtherInvestingActivitiesCashFlows")
            netCashProvidedByInvestingActivities = get_val(
                "NetCashProvidedByUsedInInvestingActivities"
            )

            # Financing Activities
            proceedsFromLongTermDebt = get_val("ProceedsFromIssuanceOfLongTermDebt")
            repaymentsOfLongTermDebt = get_val(
                "RepaymentsOfLongTermDebt"
            )  # Typically negative
            longTermNetDebtIssuance = (
                proceedsFromLongTermDebt + repaymentsOfLongTermDebt
            )

            proceedsFromShortTermDebt = get_val("ProceedsFromShortTermDebt")
            repaymentsOfShortTermDebt = get_val(
                "RepaymentsOfShortTermDebt"
            )  # Typically negative
            shortTermNetDebtIssuance = (
                proceedsFromShortTermDebt + repaymentsOfShortTermDebt
            )

            netDebtIssuance = longTermNetDebtIssuance + shortTermNetDebtIssuance

            proceedsFromCommonStock = get_val("ProceedsFromIssuanceOfCommonStock")
            paymentsForRepurchaseOfCommonStock = get_val(
                "PaymentsForRepurchaseOfCommonStock"
            )  # Typically negative
            netCommonStockIssuance = (
                proceedsFromCommonStock + paymentsForRepurchaseOfCommonStock
            )
            # Matches example's negative convention
            commonStockRepurchased_cf = paymentsForRepurchaseOfCommonStock

            proceedsFromPreferredStock = get_val("ProceedsFromIssuanceOfPreferredStock")
            paymentsForRepurchaseOfPreferredStock = get_val(
                "PaymentsForRepurchaseOfPreferredStock"
            )  # Typically negative
            netPreferredStockIssuance = (
                proceedsFromPreferredStock + paymentsForRepurchaseOfPreferredStock
            )

            netStockIssuance = netCommonStockIssuance + netPreferredStockIssuance

            commonDividendsPaid_cf = get_val(
                "PaymentsOfDividendsCommonStock"
            )  # Typically negative
            preferredDividendsPaid_cf = get_val(
                "PaymentsOfDividendsPreferredStock"
            )  # Typically negative
            netDividendsPaid_cf = commonDividendsPaid_cf + preferredDividendsPaid_cf

            otherFinancingActivities_cf = get_val("OtherFinancingActivitiesCashFlows")
            netCashProvidedByFinancingActivities = get_val(
                "NetCashProvidedByUsedInFinancingActivities"
            )

            # Summary
            effectOfForexChangesOnCash_cf = get_val(
                "EffectOfExchangeRateOnCashAndCashEquivalents"
            )
            netChangeInCash = get_val("CashAndCashEquivalentsPeriodIncreaseDecrease")
            cashAtEndOfPeriod_cf = get_val("CashAndCashEquivalentsAtCarryingValue")
            cashAtBeginningOfPeriod_cf = get_val(
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsAtBeginningOfPeriod",
                alternate_tags=["CashAndCashEquivalentsAtBeginningOfPeriod"],
            )
            if (
                cashAtBeginningOfPeriod_cf == 0
                and cashAtEndOfPeriod_cf != 0
                and netChangeInCash != 0
            ):  # Try to derive if not found
                cashAtBeginningOfPeriod_cf = (
                    cashAtEndOfPeriod_cf
                    - netChangeInCash
                    - effectOfForexChangesOnCash_cf
                )  # Adjust for forex too

            # Derived & Other
            operatingCashFlow_cf = netCashProvidedByOperatingActivities  # Alias
            capitalExpenditure_cf = (
                investmentsInPropertyPlantAndEquipment * -1
                if investmentsInPropertyPlantAndEquipment < 0
                else investmentsInPropertyPlantAndEquipment
            )  # Make positive
            freeCashFlow_cf = (
                netCashProvidedByOperatingActivities
                - investmentsInPropertyPlantAndEquipment
            )  # Capex is negative

            incomeTaxesPaid_cf = get_val("IncomeTaxesPaidNet")
            interestPaid_cf = get_val("InterestPaidNet")

            item = {
                **base_item,
                "netIncome": netIncome_cf,
                "depreciationAndAmortization": depreciationAndAmortization_cf,
                "deferredIncomeTax": deferredIncomeTax_cf,
                "stockBasedCompensation": stockBasedCompensation_cf,
                "changeInWorkingCapital": changeInWorkingCapital,
                "accountsReceivables": accountsReceivables_flow,  # Note: this is the flow amount
                "inventory": inventory_flow,  # Note: this is the flow amount
                "accountsPayables": accountsPayables_flow,  # Note: this is the flow amount
                "otherWorkingCapital": otherWorkingCapital_flow,
                "otherNonCashItems": otherNonCashItems_cf,
                "netCashProvidedByOperatingActivities": netCashProvidedByOperatingActivities,
                "investmentsInPropertyPlantAndEquipment": investmentsInPropertyPlantAndEquipment,
                "acquisitionsNet": acquisitionsNet_cf,
                "purchasesOfInvestments": purchasesOfInvestments_cf,
                "salesMaturitiesOfInvestments": salesMaturitiesOfInvestments_cf,
                "otherInvestingActivities": otherInvestingActivities_cf,
                "netCashProvidedByInvestingActivities": netCashProvidedByInvestingActivities,
                "netDebtIssuance": netDebtIssuance,
                "longTermNetDebtIssuance": longTermNetDebtIssuance,
                "shortTermNetDebtIssuance": shortTermNetDebtIssuance,
                "netStockIssuance": netStockIssuance,
                "netCommonStockIssuance": netCommonStockIssuance,
                "commonStockIssuance": proceedsFromCommonStock,
                "commonStockRepurchased": commonStockRepurchased_cf,
                "netPreferredStockIssuance": netPreferredStockIssuance,
                "netDividendsPaid": netDividendsPaid_cf,
                "commonDividendsPaid": commonDividendsPaid_cf,
                "preferredDividendsPaid": preferredDividendsPaid_cf,
                "otherFinancingActivities": otherFinancingActivities_cf,
                "netCashProvidedByFinancingActivities": netCashProvidedByFinancingActivities,
                "effectOfForexChangesOnCash": effectOfForexChangesOnCash_cf,
                "netChangeInCash": netChangeInCash,
                "cashAtEndOfPeriod": cashAtEndOfPeriod_cf,
                "cashAtBeginningOfPeriod": cashAtBeginningOfPeriod_cf,
                "operatingCashFlow": operatingCashFlow_cf,
                "capitalExpenditure": capitalExpenditure_cf,
                "freeCashFlow": freeCashFlow_cf,
                "incomeTaxesPaid": incomeTaxesPaid_cf,
                "interestPaid": interestPaid_cf,
            }
        else:
            item = {**base_item, "error": "Unknown statement type for formatting"}
        formatted_results.append(item)

    return formatted_results


# Main class to be exposed as the public interface of the package


class SECHelper:
    def __init__(self, user_agent_string=None):
        """
        Initializes the SECHelper.

        Args:
            user_agent_string (str, optional): A custom user-agent string for API requests.
                                            It's highly recommended to provide a descriptive
                                            user-agent (e.g., "YourAppName/1.0 (your-email@example.com)")
                                            to identify your application to the SEC.
                                            If None, a default will be used.
        """
        if user_agent_string:
            self.user_agent = user_agent_string
        else:
            # Default User-Agent if none provided
            self.user_agent = "PythonSECHelper/0.1.0 (contact@example.com)"

        self.headers = {"User-Agent": self.user_agent}
        print(f"SECHelper initialized. Using User-Agent: {self.user_agent}")
        self._latest_split_cache = {}

    def _get_cik_map(self):
        # Make headers hashable for the cache key
        headers_tuple = tuple(sorted(self.headers.items()))
        return _fetch_and_cache_cik_map(headers_tuple)

    def get_cik_for_symbol(self, symbol):
        """
        Retrieves the Central Index Key (CIK) for a given stock ticker symbol.

        Args:
            symbol (str): The stock ticker symbol (e.g., "AAPL").

        Returns:
            str: The 10-digit CIK as a string, or None if not found.
        """
        cik_map = self._get_cik_map()
        return _get_cik_from_map(symbol, cik_map)

    def get_company_all_facts(self, symbol_or_cik):
        """
        Fetches all company facts (XBRL disclosures) for a given company.
        This provides a comprehensive dataset for a company, including various
        financial concepts and their reported values over different periods.

        Args:
            symbol_or_cik (str): The stock ticker symbol (e.g., "AAPL") or
                                 the 10-digit CIK (e.g., "0000320193").

        Returns:
            dict: A dictionary containing all company facts in JSON format,
                  or None if the data cannot be retrieved.
        """
        result = _get_company_facts_request(
            symbol_or_cik, self.headers, self.get_cik_for_symbol
        )

        # Handle companies without XBRL data
        if result is None:
            print(f"Warning: No XBRL data available for {symbol_or_cik}")
            return {"facts": {}}  # Return empty facts structure

        return result

    def get_company_specific_concept(self, symbol_or_cik, taxonomy, tag):
        """
        Fetches specific XBRL concept data for a given company.

        Args:
            symbol_or_cik (str): The stock ticker symbol (e.g., "AAPL") or
                                 the 10-digit CIK (e.g., "0000320193").
            taxonomy (str): The XBRL taxonomy (e.g., "us-gaap").
            tag (str): The XBRL tag/concept (e.g., "Revenues").

        Returns:
            dict: A dictionary containing the concept data in JSON format,
                  or None if the data cannot be retrieved.
        """
        return _get_company_concept_request(
            symbol_or_cik, taxonomy, tag, self.headers, self.get_cik_for_symbol
        )

    def get_aggregated_frames_data(
        self, taxonomy, tag, unit, year, quarter=None, instantaneous=False
    ):
        """
        Fetches aggregated XBRL data across reporting entities for a specific concept
        and calendrical period. Useful for comparing a single metric across multiple
        companies or for a specific period (e.g., 'Total Assets' for Q1 2023 across all filers).

        Args:
            taxonomy (str): The XBRL taxonomy (e.g., "us-gaap").
            tag (str): The XBRL tag/concept (e.g., "Assets").
            unit (str): The unit of measure (e.g., "USD", "shares").
            year (int): The calendar year (e.g., 2023).
            quarter (int, optional): The quarter (1, 2, 3, or 4). If None, fetches annual data.
            instantaneous (bool, optional): True for instantaneous data (e.g., balance sheet items),
                                            False for duration data (e.g., income statement items).
                                            Defaults to False.

        Returns:
            dict: A dictionary containing the aggregated frame data in JSON format,
                  or None if the data cannot be retrieved.
        """
        return _get_frames_data_request(
            taxonomy, tag, unit, year, self.headers, quarter, instantaneous
        )

    def select_better_report(self, report1, report2, stmt_type):
        """
        Compares two reports for the same period and selects the one with better data quality.
        Simple approach: prefer reports with non-zero values for key metrics.
        """
        data1 = report1.get("data", {})
        data2 = report2.get("data", {})

        # Simple key metrics to check
        key_metrics = ["Revenues", "OperatingIncomeLoss", "NetIncomeLoss"]

        # Count non-zero values for each report
        non_zero_count1 = sum(1 for metric in key_metrics if data1.get(metric, 0) != 0)
        non_zero_count2 = sum(1 for metric in key_metrics if data2.get(metric, 0) != 0)

        # Prefer report with more non-zero values
        if non_zero_count1 > non_zero_count2:
            return report1
        elif non_zero_count2 > non_zero_count1:
            return report2
        else:
            # If equal, prefer the more recent filing
            return report1 if report1["filedAt"] >= report2["filedAt"] else report2

    def get_income_statement(self, symbol, limit=5, report_type="ALL", split_data=None):
        """
        Fetches and formats recent income statement data for a given symbol.
        Automatically adjusts for the latest stock split.
        Args:
            symbol (str): The stock ticker symbol.
            limit (int): The number of recent periods to retrieve.
            report_type (str): The type of report to filter by ("10-K", "10-Q", "ALL").
                               Defaults to "ALL".
            split_data (dict, optional): Pre-fetched split data to use. If None, will fetch split data.
        Returns:
            list: A list of dictionaries with income statement data.
        """
        logger.info(f"[DEBUG] get_income_statement called for {symbol}")
        data = _get_financial_statement_data(
            symbol,
            "income_statement",
            limit,
            report_type,
            self.headers,
            self.get_cik_for_symbol,
            self.get_company_all_facts,  # Pass the instance method
        )
        logger.info(
            f"[DEBUG] _get_financial_statement_data returned {len(data) if data else 0} items"
        )
        logger.info(f"[DEBUG] Calling adjust_financials_for_latest_split for {symbol}")
        return self.adjust_financials_for_latest_split(
            data, symbol_or_cik=symbol, split=split_data
        )

    def get_balance_sheet(self, symbol, limit=5, report_type="ALL", split_data=None):
        """
        Fetches and formats recent balance sheet data for a given symbol.
        Automatically adjusts for the latest stock split.
        Args:
            symbol (str): The stock ticker symbol.
            limit (int): The number of recent periods to retrieve.
            report_type (str): The type of report to filter by ("10-K", "10-Q", "ALL").
                               Defaults to "ALL".
            split_data (dict, optional): Pre-fetched split data to use. If None, will fetch split data.
        Returns:
            list: A list of dictionaries with balance sheet data.
        """
        logger.info(f"[DEBUG] get_balance_sheet called for {symbol}")
        data = _get_financial_statement_data(
            symbol,
            "balance_sheet",
            limit,
            report_type,
            self.headers,
            self.get_cik_for_symbol,
            self.get_company_all_facts,
        )
        logger.info(
            f"[DEBUG] _get_financial_statement_data returned {len(data) if data else 0} items"
        )
        logger.info(f"[DEBUG] Calling adjust_financials_for_latest_split for {symbol}")
        return self.adjust_financials_for_latest_split(
            data, symbol_or_cik=symbol, split=split_data
        )

    def get_cash_flow_statement(
        self, symbol, limit=5, report_type="ALL", split_data=None
    ):
        """
        Fetches and formats recent cash flow statement data for a given symbol.
        Automatically adjusts for the latest stock split.
        Args:
            symbol (str): The stock ticker symbol.
            limit (int): The number of recent periods to retrieve.
            report_type (str): The type of report to filter by ("10-K", "10-Q", "ALL").
                               Defaults to "ALL".
            split_data (dict, optional): Pre-fetched split data to use. If None, will fetch split data.
        Returns:
            list: A list of dictionaries with cash flow statement data.
        """
        logger.info(f"[DEBUG] get_cash_flow_statement called for {symbol}")
        data = _get_financial_statement_data(
            symbol,
            "cash_flow",
            limit,
            report_type,
            self.headers,
            self.get_cik_for_symbol,
            self.get_company_all_facts,
        )
        logger.info(
            f"[DEBUG] _get_financial_statement_data returned {len(data) if data else 0} items"
        )
        logger.info(f"[DEBUG] Calling adjust_financials_for_latest_split for {symbol}")
        return self.adjust_financials_for_latest_split(
            data, symbol_or_cik=symbol, split=split_data
        )

    def find_stock_splits(self, symbol_or_cik, max_filings=50, max_workers=5):
        """
        Find stock split events for a company by parsing SEC filings, using threads for speed.
        Returns the most relevant/accurate split event (most recent by date, then earliest filing_date).
        """
        logger.info(f"[DEBUG] find_stock_splits called for {symbol_or_cik}")

        import requests
        import re
        from datetime import datetime, timedelta
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Step 1: Resolve CIK
        cik = symbol_or_cik
        if not cik.isdigit() or len(cik) != 10:
            logger.info(f"[DEBUG] Resolving CIK for {symbol_or_cik}")
            cik = self.get_cik_for_symbol(symbol_or_cik)
            if not cik:
                logger.info(f"[DEBUG] Could not find CIK for {symbol_or_cik}")
                print(f"Could not find CIK for {symbol_or_cik}")
                return None

        # Step 2: Get recent filings from SEC submissions endpoint
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        headers = self.headers
        try:
            resp = requests.get(submissions_url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"Error fetching submissions: {e}")
            return None

        # Step 3: Filter for relevant filings (last 5 years only)
        today = datetime.today()
        five_years_ago = today - timedelta(days=5 * 365)
        filings = data.get("filings", {}).get("recent", {})
        accession_numbers = filings.get("accessionNumber", [])
        forms = filings.get("form", [])
        filing_dates = filings.get("filingDate", [])

        # Only include filing types that actually announce or report EXECUTED stock splits
        relevant_forms = {
            "8-K",  # Immediate announcements of executed splits
            "10-K",  # Annual reports that may mention recent splits
            "10-Q",  # Quarterly reports that may mention recent splits
            # Removed DEF 14A, DEF 14C, 424B3, 424B4, 424B5 as they are unreliable
            # DEF 14A/C are proxy statements that may mention historical or proposed splits, not executed ones
            # 424B* are prospectus filings that are not reliable for split announcements
        }
        split_candidates = [
            (acc, form, date)
            for acc, form, date in zip(accession_numbers, forms, filing_dates)
            if form in relevant_forms and date >= five_years_ago.strftime("%Y-%m-%d")
        ][:max_filings]

        from .patterns import NUMBER_WORDS

        number_words = NUMBER_WORDS

        # Strong announcement keywords
        from .patterns import get_announcement_keywords

        announcement_keywords = get_announcement_keywords()

        def is_announcement_context(context):
            context_lower = context.lower()
            return any(word in context_lower for word in announcement_keywords)

        def _is_valid_split_context(context):
            """
            Additional validation for split context to catch more cases.
            This helps with announcements that might not use traditional announcement keywords.
            Only returns True for contexts that indicate EXECUTED stock splits.
            """
            context_lower = context.lower()

            # Check for execution/completion language (highest priority)
            from .patterns import get_execution_keywords

            execution_keywords = get_execution_keywords()
            if any(word in context_lower for word in execution_keywords):
                return True

            # Check for historical/proposed indicators (reject these)
            from .patterns import get_historical_indicators

            historical_indicators = get_historical_indicators()
            if any(indicator in context_lower for indicator in historical_indicators):
                return False

            # Check for dividend-related language (stock splits are often effected as dividends)
            dividend_keywords = ["dividend", "special dividend", "stock dividend"]
            if any(word in context_lower for word in dividend_keywords):
                return True

            # Check for specific execution date patterns (not just any date mention)
            execution_date_patterns = [
                r"effective\s+\w+\s+\d{1,2},\s+\d{4}",  # "effective July 1, 2022"
                r"record\s+date\s+of\s+\w+\s+\d{1,2},\s+\d{4}",  # "record date of July 1, 2022"
                r"ex-dividend\s+date\s+\w+\s+\d{1,2},\s+\d{4}",  # "ex-dividend date July 1, 2022"
                r"distribution\s+date\s+\w+\s+\d{1,2},\s+\d{4}",  # "distribution date July 1, 2022"
            ]
            for pattern in execution_date_patterns:
                if re.search(pattern, context_lower):
                    return True

            # Reject if it's just a mention without execution language
            return False

        # Use the stock splits module for adjustment
        from .patterns import FORM_PRIORITY

        form_priority = FORM_PRIORITY

        def process_filing(args):
            acc, form, date = args
            acc_nodash = acc.replace("-", "")
            txt_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{acc}.txt"
            try:
                filing_text = requests.get(txt_url, headers=headers, timeout=10).text
            except Exception:
                return None

            # Look for split keywords and ratio
            if re.search(r"(stock|share) split", filing_text, re.IGNORECASE):
                ratio = None
                context_window = 60  # Increased context window for better detection
                found_context = None
                from .patterns import get_numeric_ratio_patterns

                ratio_patterns = get_numeric_ratio_patterns()
                for pat in ratio_patterns:
                    for match in re.finditer(pat, filing_text, re.IGNORECASE):
                        groups = match.groups()
                        num_str, denom_str = groups

                        # Handle hybrid patterns where first is numeric, second might be word
                        # Also handle reverse splits (one-for-X) and special dividends
                        try:
                            num = int(num_str)
                        except ValueError:
                            # Check if it's "one" for reverse splits
                            if num_str.lower() == "one":
                                num = 1
                            else:
                                continue

                        # Try to convert denominator to int, if it fails, try word lookup
                        try:
                            denom = int(denom_str)
                        except ValueError:
                            # It's a word, look it up in number_words
                            denom = number_words.get(denom_str.lower())
                            if not denom:
                                continue

                        logger.info(
                            f"[DEBUG] Pattern matched: {pat} -> {num_str}:{denom_str} = {num}:{denom}"
                        )

                        start, end = match.span()
                        context = filing_text[
                            max(0, start - context_window) : min(
                                len(filing_text), end + context_window
                            )
                        ]
                        if (
                            denom != 0
                            and 0 < num < 100
                            and 0 < denom < 100
                            and re.search(r"split", context, re.IGNORECASE)
                        ):
                            # More flexible context validation - check if it's a valid split context
                            if is_announcement_context(
                                context
                            ) or _is_valid_split_context(context):
                                ratio = num / denom
                                found_context = context
                                logger.info(
                                    f"[DEBUG] Found valid split ratio {num}:{denom} = {ratio} in filing {acc} for {symbol_or_cik}"
                                )
                                break
                    if ratio:
                        break

                # Word-based ratio patterns (e.g., 'six-for-one split')
                from .patterns import get_word_ratio_patterns

                word_ratio_patterns = get_word_ratio_patterns()
                for pat in word_ratio_patterns:
                    for match in re.finditer(pat, filing_text, re.IGNORECASE):
                        num_word, denom_word = match.groups()
                        num = number_words.get(num_word.lower())
                        denom = number_words.get(denom_word.lower())
                        start, end = match.span()
                        context = filing_text[
                            max(0, start - context_window) : min(
                                len(filing_text), end + context_window
                            )
                        ]
                        if (
                            num
                            and denom
                            and denom != 0
                            and 0 < num < 100
                            and 0 < denom < 100
                            and re.search(r"split", context, re.IGNORECASE)
                        ):
                            # More flexible context validation - check if it's a valid split context
                            if is_announcement_context(
                                context
                            ) or _is_valid_split_context(context):
                                ratio = num / denom
                                found_context = context
                                break
                    if ratio:
                        break

                # Try to extract date (look for Month DD, YYYY or MM/DD/YYYY)
                from .patterns import get_date_patterns

                date_patterns = get_date_patterns()
                date_match = re.search(
                    date_patterns["month_day_year"],
                    filing_text,
                )
                if date_match:
                    try:
                        split_date = datetime.strptime(
                            date_match.group(0), "%B %d, %Y"
                        ).strftime("%Y-%m-%d")
                    except Exception:
                        split_date = date
                else:
                    # Try alternative date formats
                    alt_date_match = re.search(date_patterns["mm_dd_yyyy"], filing_text)
                    if alt_date_match:
                        try:
                            month, day, year = alt_date_match.groups()
                            split_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        except Exception:
                            split_date = date
                    else:
                        split_date = date

                if ratio:
                    return {
                        "date": split_date,
                        "ratio": ratio,
                        "filing": form,
                        "filing_date": date,
                        "accession": acc,
                        "type": "forward",
                        "context": found_context,
                    }
            return None
            return None

        splits = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_filing, args) for args in split_candidates
            ]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    splits.append(result)

        # For each ratio, only keep the split with the earliest announcement date and best form
        best_split_for_ratio = {}
        best_form_for_ratio = {}
        for split in splits:
            ratio = split["ratio"]
            key = ratio
            pri = form_priority.get(split["filing"], 4)
            if key not in best_split_for_ratio:
                best_split_for_ratio[key] = (
                    split["date"],
                    pri,
                    split["filing_date"],
                    split,
                )
                best_form_for_ratio[key] = pri
            else:
                old_date, old_pri, old_filing_date, _ = best_split_for_ratio[key]
                # Prioritize earliest announcement date, then best form, then earliest filing date
                if (
                    split["date"] < old_date
                    or (split["date"] == old_date and pri < old_pri)
                    or (
                        split["date"] == old_date
                        and pri == old_pri
                        and split["filing_date"] < old_filing_date
                    )
                ):
                    best_split_for_ratio[key] = (
                        split["date"],
                        pri,
                        split["filing_date"],
                        split,
                    )
                    best_form_for_ratio[key] = pri
            if best_form_for_ratio[key] == 1:
                continue
        filtered_splits = [v[3] for v in best_split_for_ratio.values()]
        if not filtered_splits:
            logger.info(f"[DEBUG] No splits found for {symbol_or_cik}")
            return None

        # Return the split with the EARLIEST announcement date (not latest)
        # This is crucial for proper financial data adjustment
        #
        # Explanation of the logic:
        # 1. We want the EARLIEST announcement date because:
        #    - Financial data from dates BEFORE the split announcement should be adjusted
        #    - The 2021-12-31 financial record needs to be adjusted for a split announced in 2021
        #    - If we used the latest date (2023-06-02), the 2021 data wouldn't be adjusted
        #
        # 2. Among splits with the same announcement date, we prioritize:
        #    - 8-K filings (immediate announcements) over 10-K filings (annual reports)
        #    - Earlier filing dates when form types are the same
        #
        # 3. This ensures that when someone looks at 2021 financial data, it's properly
        #    adjusted for the stock split that was announced during that period
        #
        # IMPORTANT: For financial data adjustment, we use the FILING DATE, not the announcement date
        # because the filing date represents when the split information was actually made public
        # Sort splits by earliest announcement date, then form priority, then filing date
        filtered_splits.sort(
            key=lambda s: (
                s["date"],  # Earliest announcement date first
                form_priority.get(
                    s["filing"], 4
                ),  # Lower priority number first (8-K before 10-K)
                s["filing_date"],  # Earliest filing date first
            ),
            reverse=False,  # Changed from True to False to get earliest date
        )

        logger.info(
            f"[DEBUG] Selected split: ratio={filtered_splits[0]['ratio']}, announcement_date={filtered_splits[0]['date']}, filing_date={filtered_splits[0]['filing_date']}, form={filtered_splits[0]['filing']}"
        )
        return filtered_splits[0]

    def _get_latest_split(self, symbol_or_cik):
        """
        Returns the latest split for the symbol, using cache if available.
        """
        logger.info(f"[DEBUG] _get_latest_split called for {symbol_or_cik}")

        if symbol_or_cik in self._latest_split_cache:
            logger.info(
                f"[DEBUG] Found {symbol_or_cik} in cache: {self._latest_split_cache[symbol_or_cik]}"
            )
            return self._latest_split_cache[symbol_or_cik]

        logger.info(f"[DEBUG] {symbol_or_cik} not in cache, calling StockSplitDetector")

        try:
            # Use the StockSplitDetector class for better decimal ratio support
            from .stock_splits import StockSplitDetector

            detector = StockSplitDetector(self.headers)
            split = detector.find_stock_splits(symbol_or_cik)
            logger.info(f"[DEBUG] StockSplitDetector returned: {split}")
        except ImportError:
            logger.warning(
                "[DEBUG] StockSplitDetector not available, falling back to legacy method"
            )
            split = self.find_stock_splits(symbol_or_cik)
            logger.info(f"[DEBUG] Legacy find_stock_splits returned: {split}")
        except Exception as e:
            logger.error(
                f"[DEBUG] Error using StockSplitDetector: {e}, falling back to legacy method"
            )
            split = self.find_stock_splits(symbol_or_cik)
            logger.info(f"[DEBUG] Legacy find_stock_splits returned: {split}")

        self._latest_split_cache[symbol_or_cik] = split
        return split

    def adjust_financials_for_latest_split(self, data, symbol_or_cik=None, split=None):
        """
        Adjusts financial data for the latest stock split. Only data prior to the split date is adjusted.
        Args:
            data (list): List of financial data dicts (e.g., from get_income_statement).
            symbol_or_cik (str, optional): Symbol or CIK. Required if split is not provided.
            split (dict, optional): Split info dict as returned by find_stock_splits. If not provided, will be fetched.
        Returns:
            list: Adjusted data.
        """
        logger.info(
            f"[DEBUG] adjust_financials_for_latest_split called with data length: {len(data) if data else 0}"
        )
        logger.info(f"[DEBUG] symbol_or_cik: {symbol_or_cik}")
        logger.info(f"[DEBUG] split provided: {split is not None}")

        if not data:
            logger.info("[DEBUG] No data provided, returning empty data")
            return data

        if not split:
            logger.info("[DEBUG] No split provided, checking symbol_or_cik")
            if not symbol_or_cik:
                logger.info(
                    "[DEBUG] No symbol_or_cik provided, returning unadjusted data"
                )
                return data
            logger.info(f"[DEBUG] Calling _get_latest_split for {symbol_or_cik}")
            split = self._get_latest_split(symbol_or_cik)
            logger.info(f"[DEBUG] _get_latest_split returned: {split}")

        if not split or "ratio" not in split or "date" not in split:
            logger.info(f"[DEBUG] Invalid split data: {split}")
            return data

        # Use the stock splits module for adjustment
        from .stock_splits import StockSplitDetector

        detector = StockSplitDetector(self.headers)
        return detector.adjust_financials_for_latest_split(data, split)

    def get_stock_splits(
        self, symbol_or_cik: str, max_filings: int = 50, max_workers: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Get stock split information for a company.

        Args:
            symbol_or_cik: Company symbol or CIK
            max_filings: Maximum number of filings to analyze
            max_workers: Number of worker threads for parallel processing

        Returns:
            Stock split information dict or None if no splits found
        """
        try:
            from .stock_splits import StockSplitDetector

            # Create stock split detector instance
            detector = StockSplitDetector(self.headers)

            # Find stock splits
            split_info = detector.find_stock_splits(
                symbol_or_cik, max_filings, max_workers
            )

            if split_info:
                logger.info(
                    f"[DEBUG] Found stock split for {symbol_or_cik}: {split_info}"
                )
            else:
                logger.info(f"[DEBUG] No stock split found for {symbol_or_cik}")

            return split_info

        except ImportError:
            logger.warning(
                "[DEBUG] Stock split module not available, falling back to legacy method"
            )
            return self.find_stock_splits(symbol_or_cik, max_filings, max_workers)
        except Exception as e:
            logger.error(f"[DEBUG] Error getting stock splits: {e}")
            return None

    def get_all_stock_splits(
        self, symbol_or_cik: str, max_filings: int = 50, max_workers: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get all stock split events for a company.

        Args:
            symbol_or_cik: Company symbol or CIK
            max_filings: Maximum number of filings to analyze
            max_workers: Number of worker threads for parallel processing

        Returns:
            List of stock split information dicts
        """
        try:
            from .stock_splits import StockSplitDetector

            # Create stock split detector instance
            detector = StockSplitDetector(self.headers)

            # Find all stock splits
            all_splits = detector.get_all_stock_splits(
                symbol_or_cik, max_filings, max_workers
            )

            if all_splits:
                logger.info(
                    f"[DEBUG] Found {len(all_splits)} stock splits for {symbol_or_cik}"
                )
            else:
                logger.info(f"[DEBUG] No stock splits found for {symbol_or_cik}")

            return all_splits

        except ImportError:
            logger.warning(
                "[DEBUG] Stock split module not available, falling back to legacy method"
            )
            # Legacy fallback - return single split as list
            split_info = self.find_stock_splits(symbol_or_cik, max_filings, max_workers)
            return [split_info] if split_info else []
        except Exception as e:
            logger.error(f"[DEBUG] Error getting all stock splits: {e}")
            return []
