# filename: sec_financial_data/sec_financial_data.py

import requests
import time
from functools import lru_cache

# Base URLs for SEC EDGAR APIs
SEC_BASE_URL = "https://www.sec.gov"
DATA_SEC_BASE_URL = "https://data.sec.gov"

# Rate limiting: SEC allows up to 10 requests per second.
# We'll implement a simple delay to ensure we don't exceed this.
# A more robust solution for high-volume usage might involve a token
# bucket algorithm.
LAST_REQUEST_TIME = 0
REQUEST_INTERVAL = 0.11  # Approximately 9 requests per second to be safe


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

        # The JSON structure is a list of dictionaries like {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}
        # Create a dictionary mapping ticker to CIK (padded to 10 digits)
        cik_map = {item['ticker'].upper(): str(item['cik_str']).zfill(10)
                   for item in data.values()}
        return cik_map
    except requests.exceptions.RequestException as e:
        print(
            f"Error fetching CIK map with User-Agent '{headers.get('User-Agent')}': {e}")
        return {}


def _get_cik_from_map(symbol, cik_map):
    """
    Retrieves the Central Index Key (CIK) for a given stock ticker symbol.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL").

    Returns:
        str: The 10-digit CIK as a string, or None if not found.
    """
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


def _get_company_concept_request(
        symbol_or_cik,
        taxonomy,
        tag,
        headers,
        get_cik_func):
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
        taxonomy,
        tag,
        unit,
        year,
        headers,
        quarter=None,
        instantaneous=False):
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
        get_company_facts_func):
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
    cik = symbol_or_cik if (isinstance(symbol_or_cik, str) and symbol_or_cik.isdigit(
    ) and len(symbol_or_cik) == 10) else get_cik_func(symbol_or_cik)
    if not cik:
        print(f"Error: Could not find CIK for: {symbol_or_cik}")
        return []

    def _get_financial_value(
            data_dict,
            primary_tag,
            alternate_tags=None,
            default=0):
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
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
            "SalesRevenueServicesNet",
            "CostOfRevenue",
            "CostOfGoodsAndServicesSold",  # Common alternative for CostOfRevenue
            "GrossProfit",
            "ResearchAndDevelopmentExpense",
            "SellingGeneralAndAdministrativeExpense",
            "GeneralAndAdministrativeExpense",
            "SellingAndMarketingExpense",
            "OtherOperatingExpenses",
            "OperatingExpenses",
            "OperatingIncomeLoss",
            "InterestIncomeExpenseNet",
            "InterestIncome",
            "InterestExpense",
            "NonoperatingIncomeLoss",
            "IncomeLossFromContinuingOperationsBeforeIncomeTax",
            "IncomeBeforeIncomeTax",
            "IncomeBeforeTax",
            "ProfitLossBeforeTax",
            "IncomeTaxExpenseBenefit",
            "NetIncomeLoss",
            "NetIncomeLossFromDiscontinuedOperationsNetOfTax",
            "EarningsPerShareBasic",
            "EarningsPerShareDiluted",
            "WeightedAverageNumberOfSharesOutstandingBasic",
            "WeightedAverageNumberOfDilutedSharesOutstanding",
            "DepreciationDepletionAndAmortization"
        ],
        "balance_sheet": [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashAndCashEquivalents",  # More general tag
            "MarketableSecuritiesCurrent",  # For shortTermInvestments
            "ShortTermInvestments",  # Alternative for shortTermInvestments
            "AccountsReceivableNetCurrent",  # For netReceivables
            "AccountsReceivableTradeCurrent",
            # For accountsReceivables (trade specific)
            "OtherReceivablesCurrent",  # For otherReceivables
            "InventoryNet",
            "PrepaidExpenseCurrent",  # For prepaids
            "Inventory",  # More general tag
            "OtherAssetsCurrent",
            "AssetsCurrent",  # For totalCurrentAssets
            "PropertyPlantAndEquipmentNet",
            "Goodwill",
            "IntangibleAssetsNetExcludingGoodwill",  # For intangibleAssets
            "IntangibleAssets",  # More general tag
            "MarketableSecuritiesNoncurrent",  # For longTermInvestments
            "LongTermInvestments",  # Alternative for longTermInvestments
            "DeferredTaxAssetsNet",
            # For taxAssets (can be sum of current/noncurrent)
            "OtherAssetsNoncurrent",
            "AssetsNoncurrent",  # For totalNonCurrentAssets
            "Assets",
            "TotalAssets",  # Common alternative for total assets
            "AccountsPayableCurrent",  # For accountPayables
            "LongTermDebtCurrent",
            "OtherAccountsPayableCurrent",  # For otherPayables
            "AccruedLiabilitiesCurrent",  # For accruedExpenses
            "DebtCurrent",  # Preferred total short-term debt
            "CommercialPaper",
            "LongTermDebtCurrentMaturities",
            # Specific for current portion of long-term debt
            "NotesPayableCurrent",
            "ShortTermBorrowings",  # General short-term borrowings
            # Broader fallback for current LTD + other STD
            "LongTermDebtCurrentMaturitiesAndOtherShortTermDebt",
            "OperatingLeaseLiabilityCurrent",
            # For capitalLeaseObligationsCurrent (new standard)
            "CapitalLeaseObligationsCurrent",
            # For capitalLeaseObligationsCurrent (old standard)
            "IncomeTaxesPayable",  # For taxPayables
            "DeferredRevenueCurrent",  # For deferredRevenue (current)
            # Alt for deferredRevenue (current)
            "ContractWithCustomerLiabilityCurrent",
            "OtherLiabilitiesCurrent",
            "LiabilitiesCurrent",  # For totalCurrentLiabilities
            "LongTermDebtNoncurrent",
            "OperatingLeaseLiabilityNoncurrent",
            # For capitalLeaseObligationsNonCurrent (new)
            "CapitalLeaseObligationsNoncurrent",
            # For capitalLeaseObligationsNonCurrent (old)
            "DeferredRevenueNoncurrent",  # For deferredRevenue (non-current)
            # Alt for deferredRevenue (non-current)
            "ContractWithCustomerLiabilityNoncurrent",
            "DeferredTaxLiabilitiesNoncurrent",  # Or DeferredTaxLiabilitiesNet
            "OtherLiabilitiesNoncurrent",
            "LiabilitiesNoncurrent",
            "TotalDebt",  # Common tag for total debt
            "Debt",  # More general debt tag
            "Liabilities",  # For totalLiabilities
            "TotalLiabilities",  # Common alternative for total liabilities
            "TreasuryStockValue",
            "PreferredStockValue",
            "RedeemablePreferredStockCarryingAmount",  # Alt for preferredStock
            "CommonStockValue",
            "CommonStock",  # More general tag
            "CommonStocksIncludingAdditionalPaidInCapital",  # Alt if APIC is combined
            "AdditionalPaidInCapital",
            "RetainedEarningsAccumulatedDeficit",
            "RetainedEarnings",  # More general tag
            "AccumulatedOtherComprehensiveIncomeLossNetOfTax",
            "MinorityInterest",  # For minorityInterest
            "NoncontrollingInterest",  # Alt for minorityInterest
            "StockholdersEquity",  # For totalStockholdersEquity
            # Alt for totalStockholdersEquity
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
        ],
        "cash_flow": [
            # Operating Activities - Reconciliation & Direct
            "NetIncomeLoss",  # Needed for reconciliation if not directly reported in CF
            "DepreciationDepletionAndAmortization",
            "ShareBasedCompensation",
            "DeferredIncomeTaxExpenseBenefit",  # General tag for deferred income tax in CF
            "IncreaseDecreaseInDeferredIncomeTaxes",  # Alternative
            "IncreaseDecreaseInAccountsReceivableNetCurrent",  # For accountsReceivables flow
            "IncreaseDecreaseInInventoriesNet",  # For inventory flow
            "IncreaseDecreaseInAccountsPayableCurrent",  # For accountsPayables flow
            "IncreaseDecreaseInOtherOperatingAssetsLiabilitiesNet",  # For 'otherWorkingCapital'
            "OtherNoncashIncomeExpense",  # For 'otherNonCashItems'
            "NetCashProvidedByUsedInOperatingActivities",

            # Investing Activities
            "PaymentsToAcquirePropertyPlantAndEquipment",
            # For 'investmentsInPropertyPlantAndEquipment' &
            # 'capitalExpenditure'
            "PaymentsToAcquireBusinessesNetOfCashAcquired",  # For 'acquisitionsNet'
            "PaymentsForPurchasesOfInvestments",  # For 'purchasesOfInvestments'
            # For 'salesMaturitiesOfInvestments'
            "ProceedsFromSaleAndMaturityOfMarketableSecurities",
            "OtherInvestingActivitiesCashFlows",  # For 'otherInvestingActivities'
            "NetCashProvidedByUsedInInvestingActivities",

            # Financing Activities
            "ProceedsFromIssuanceOfLongTermDebt",
            "RepaymentsOfLongTermDebt",
            "ProceedsFromShortTermDebt",
            "RepaymentsOfShortTermDebt",
            "ProceedsFromIssuanceOfCommonStock",
            "PaymentsForRepurchaseOfCommonStock",  # For 'commonStockRepurchased'
            "ProceedsFromIssuanceOfPreferredStock",
            "PaymentsForRepurchaseOfPreferredStock",
            "PaymentsOfDividendsCommonStock",  # For 'commonDividendsPaid'
            "PaymentsOfDividendsPreferredStock",  # For 'preferredDividendsPaid'
            "OtherFinancingActivitiesCashFlows",  # For 'otherFinancingActivities'
            "NetCashProvidedByUsedInFinancingActivities",

            # Summary & Other
            "CashAndCashEquivalentsPeriodIncreaseDecrease",
            "EffectOfExchangeRateOnCashAndCashEquivalents",
            "CashAndCashEquivalentsAtCarryingValue",
            # For 'cashAtEndOfPeriod' (also in BS tags)
            # For 'cashAtBeginningOfPeriod'
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsAtBeginningOfPeriod",
            "CashAndCashEquivalentsAtBeginningOfPeriod",  # Alternative for beginning cash
            "IncomeTaxesPaidNet",
            "InterestPaidNet",
        ]
    }

    if statement_type not in statement_tags:
        print(
            f"Error: Invalid financial statement type: {statement_type}. Choose from: {', '.join(statement_tags.keys())}")
        return []

    required_tags = statement_tags[statement_type]

    # Step 1: Collect data for each unique report instance (filing).
    # Key: (form_group, end_date, filed_at)
    # Value: Dictionary holding report details and its financial data.
    # report_instances_data is already initialized before this loop.
    report_instances_data = {}

    facts_section = company_facts.get('facts', {})
    us_gaap_facts = facts_section.get('us-gaap', {})

    for tag in required_tags:
        concept_data = us_gaap_facts.get(tag, {})
        for unit_type, facts_list in concept_data.get('units', {}).items():
            # We are interested in monetary units, typically USD.
            # However, some concepts might be in other units (e.g., shares for EPS).
            # For simplicity, we'll process all units and let the structure of company_facts guide us.
            # If specific unit filtering is needed, it can be added here (e.g.,
            # if unit_type != "USD": continue)
            for fact in facts_list:
                fiscal_year = fact.get('fy')
                fiscal_period = fact.get('fp')
                form_type = fact.get('form')
                filed_at = fact.get('filed')
                value = fact.get('val')
                end_date = fact.get('end')
                # This start_date is per-fact, might not be the period
                # start_date
                start_date = fact.get('start')

                if not (
                        form_type and filed_at and end_date and fiscal_year is not None and fiscal_period and value is not None):
                    continue

                form_group = None
                if form_type.upper().startswith('10-K'):
                    form_group = '10-K'
                elif form_type.upper().startswith('10-Q'):
                    form_group = '10-Q'
                else:
                    continue  # Only process 10-K and 10-Q related forms

                report_instance_key = (form_group, end_date, filed_at)

                if report_instance_key not in report_instances_data:
                    report_instances_data[report_instance_key] = {
                        # Store original symbol if provided
                        "symbol": symbol_or_cik.upper() if isinstance(symbol_or_cik, str) and not symbol_or_cik.isdigit() else "N/A_CIK_USED",
                        "fiscalYear": fiscal_year,
                        "fiscalPeriod": fiscal_period,
                        "formType": form_type,
                        "formGroup": form_group,
                        "filedAt": filed_at,
                        "endDate": end_date,
                        # Use the start_date from the first fact encountered for this report instance.
                        # This might need refinement if a true period start_date is required across all facts.
                        "startDate": start_date,
                        "data": {}
                    }

                # Add the fact to the data dictionary - no need for additional filtering
                # since we're already grouping by report_instance_key which ensures
                # all facts belong to the same report instance
                report_instances_data[report_instance_key]['data'][tag] = value

    # Step 2: Determine the canonical (latest filed) report for each (form_group, end_date)
    # If the latest report has insufficient data, fall back to the second most
    # recent
    canonical_reports = {}

    # Group reports by (form_group, end_date) and sort by filedAt (most recent
    # first)
    reports_by_period = {}
    for report_obj in report_instances_data.values():
        key = (report_obj['formGroup'], report_obj['endDate'])
        if key not in reports_by_period:
            reports_by_period[key] = []
        reports_by_period[key].append(report_obj)

    # Sort each group by filedAt (most recent first)
    for key in reports_by_period:
        reports_by_period[key].sort(key=lambda r: r['filedAt'], reverse=True)

    # Define key balance sheet items to check for data completeness
    key_balance_sheet_items = [
        "CashAndCashEquivalentsAtCarryingValue",
        "AssetsCurrent",
        "Assets",
        "LiabilitiesCurrent",
        "StockholdersEquity"
    ]

    for key, reports in reports_by_period.items():
        if not reports:
            continue

        # Start with the most recent report
        selected_report = reports[0]

        # Check if the most recent report has sufficient data
        data = selected_report.get('data', {})
        num_nonzero_items = sum(
            1 for item in key_balance_sheet_items if data.get(
                item, 0) != 0)
        has_sufficient_data = num_nonzero_items >= 2

        # If the most recent report has insufficient data, try subsequent
        # reports
        if not has_sufficient_data and len(reports) > 1:
            for i in range(1, len(reports)):
                candidate_report = reports[i]
                candidate_data = candidate_report.get('data', {})
                candidate_num_nonzero_items = sum(
                    1 for item in key_balance_sheet_items if candidate_data.get(
                        item, 0) != 0)
                candidate_has_sufficient_data = candidate_num_nonzero_items >= 2

                # If this report has sufficient data, use it
                if candidate_has_sufficient_data:
                    selected_report = candidate_report
                    break

        canonical_reports[key] = selected_report

    all_canonical_reports_list = list(canonical_reports.values())

    # Step 3: Separate into 10-K and 10-Q lists
    ten_k_reports = [
        r for r in all_canonical_reports_list if r['formGroup'] == '10-K']
    ten_q_reports = [
        r for r in all_canonical_reports_list if r['formGroup'] == '10-Q']

    # Step 4: Sort each list by endDate (primary) and filedAt (secondary, for
    # tie-breaking), most recent first
    def sort_key_func(r): return (
        r.get(
            'endDate',
            '0000-00-00'),
        r.get(
            'filedAt',
            '0000-00-00T00:00:00Z'))

    ten_k_reports.sort(key=sort_key_func, reverse=True)
    ten_q_reports.sort(key=sort_key_func, reverse=True)

    # Step 5: Select reports based on report_type and apply limit
    selected_reports = []
    report_type_upper = report_type.upper()

    if report_type_upper == "10-K":
        selected_reports = ten_k_reports[:limit]
    elif report_type_upper == "10-Q":
        selected_reports = ten_q_reports[:limit]
    elif report_type_upper == "ALL":
        # Combine, sort, then limit
        combined_reports = ten_k_reports + ten_q_reports
        # Ensure overall chronological order
        combined_reports.sort(key=sort_key_func, reverse=True)
        selected_reports = combined_reports[:limit]
    else:
        print(
            f"Warning: Invalid report_type '{report_type}'. Defaulting to 'ALL'.")
        combined_reports = ten_k_reports + ten_q_reports
        combined_reports.sort(key=sort_key_func, reverse=True)
        selected_reports = combined_reports[:limit]

    # Step 6 (was Step 7): Format results
    formatted_results = []

    for period_details in selected_reports:
        data = period_details.get("data", {})

        # Determine period string (FY, Q1, Q2, etc.)
        period_val = period_details["fiscalPeriod"]
        if period_details["formGroup"] == '10-K':
            period_val = 'FY'

        # Revenue: Try a list of common tags in order of preference.
        # Ensure these tags are included in statement_tags["income_statement"]
        # above.
        revenue = 0  # Default value
        revenue_possible_tags = [
            'Revenues',
            'RevenueFromContractWithCustomerExcludingAssessedTax',
            'SalesRevenueNet',
            'SalesRevenueGoodsNet',
            'SalesRevenueServicesNet'
        ]
        for r_tag in revenue_possible_tags:
            if r_tag in data:  # Check if the tag was found and data collected for it
                revenue = _get_financial_value(data, r_tag)
                # If we found a non-zero revenue, prefer it. If it's 0,
                # continue checking.
                if revenue != 0:
                    break  # Use the first non-zero revenue found from the preferred list
        # If all found tags resulted in 0, revenue remains 0. If no tags were
        # found, revenue remains 0.

        costOfRevenue = 0
        costOfRevenue_possible_tags = [
            'CostOfRevenue', 'CostOfGoodsAndServicesSold']
        for tag in costOfRevenue_possible_tags:
            if tag in data:
                costOfRevenue = _get_financial_value(data, tag)
                if costOfRevenue != 0:
                    break

        # GrossProfit can be explicitly found or calculated.
        # If GrossProfit tag exists and is non-zero, use it. Otherwise,
        # calculate.
        grossProfit_explicit = _get_financial_value(data, 'GrossProfit')
        grossProfit = grossProfit_explicit if grossProfit_explicit != 0 else (
            revenue - costOfRevenue)
        # If GrossProfit was explicitly 0, but revenue and CoR allow calculation, it will be calculated.
        # If GrossProfit tag was not found, it will be calculated.

        generalAndAdministrativeExpenses = 0
        sellingAndMarketingExpenses = 0
        researchAndDevelopmentExpense = _get_financial_value(
            data, "ResearchAndDevelopmentExpense")
        sga_combined = _get_financial_value(
            data, "SellingGeneralAndAdministrativeExpense")
        ga_separate = _get_financial_value(
            data, "GeneralAndAdministrativeExpense")
        sm_separate = _get_financial_value(data, "SellingAndMarketingExpense")
        otherOperatingExpenses_val = _get_financial_value(
            data, "OtherOperatingExpenses")

        if ga_separate != 0 or sm_separate != 0:  # Prefer separate tags if available
            generalAndAdministrativeExpenses = ga_separate
            sellingAndMarketingExpenses = sm_separate
        elif sga_combined != 0:  # Use combined SG&A
            generalAndAdministrativeExpenses = sga_combined

        otherExpenses_val = otherOperatingExpenses_val

        # Calculate total operating expenses
        # Use the specific 'OperatingExpenses' tag if available and non-zero,
        # otherwise sum components.
        operatingExpenses_from_tag = _get_financial_value(
            data, 'OperatingExpenses')
        calculated_operating_components_sum = researchAndDevelopmentExpense + \
            generalAndAdministrativeExpenses + sellingAndMarketingExpenses + otherExpenses_val

        # Use explicit if non-zero
        operatingExpenses = operatingExpenses_from_tag if operatingExpenses_from_tag != 0 else calculated_operating_components_sum

        # Derived (Cost of Revenue + All Operating Expenses)
        costAndExpenses = costOfRevenue + operatingExpenses

        # Interest Income / Expense
        # Prioritize discrete InterestIncome and InterestExpense. Fallback to
        # InterestIncomeExpenseNet.
        interestIncome_val = _get_financial_value(data, 'InterestIncome')
        interestExpense_val = _get_financial_value(data, 'InterestExpense')
        interestIncomeExpenseNet_val = _get_financial_value(
            data, 'InterestIncomeExpenseNet')

        if interestIncome_val == 0 and interestExpense_val == 0 and interestIncomeExpenseNet_val != 0:
            if interestIncomeExpenseNet_val > 0:
                interestIncome_val = interestIncomeExpenseNet_val
            else:  # interestIncomeExpenseNet_val < 0
                interestExpense_val = abs(interestIncomeExpenseNet_val)
        # If gross values were present, they are used. If only net was present, it's now split.
        # If all were zero, they remain zero.

        netInterestIncome = interestIncome_val - interestExpense_val  # Derived

        depreciationAndAmortization = _get_financial_value(
            data, 'DepreciationDepletionAndAmortization')

        operatingIncome = _get_financial_value(data, 'OperatingIncomeLoss')
        # If operatingIncome is zero from tag, try to derive it
        if operatingIncome == 0 and grossProfit != 0 and (
                researchAndDevelopmentExpense != 0 or generalAndAdministrativeExpenses != 0 or sellingAndMarketingExpenses != 0 or otherExpenses_val != 0):
            operatingIncome = grossProfit - (researchAndDevelopmentExpense + generalAndAdministrativeExpenses + \
                                             sellingAndMarketingExpenses + otherExpenses_val + depreciationAndAmortization) # More complete OpEx for derivation

        ebitda = operatingIncome + depreciationAndAmortization  # Derived
        ebit = operatingIncome  # ebit is Operating Income

        totalOtherIncomeExpensesNet = _get_financial_value(
            data, 'NonoperatingIncomeLoss')

        incomeBeforeTax = 0
        incomeBeforeTax_possible_tags = [
            'IncomeLossFromContinuingOperationsBeforeIncomeTax',
            'IncomeBeforeIncomeTax',
            'IncomeBeforeTax',
            'ProfitLossBeforeTax'
        ]
        for tag in incomeBeforeTax_possible_tags:
            if tag in data:
                incomeBeforeTax = _get_financial_value(data, tag)
                if incomeBeforeTax != 0:
                    break
        # Fallback calculation for Income Before Tax (EBT)
        if incomeBeforeTax == 0 and operatingIncome != 0:
            incomeBeforeTax = operatingIncome + netInterestIncome + \
                totalOtherIncomeExpensesNet  # Common derivation

        incomeTaxExpense = _get_financial_value(
            data, 'IncomeTaxExpenseBenefit')

        netIncome = _get_financial_value(data, 'NetIncomeLoss')
        netIncomeFromDiscontinuedOperations = _get_financial_value(
            data, 'NetIncomeLossFromDiscontinuedOperationsNetOfTax')
        netIncomeFromContinuingOperations = netIncome - \
            netIncomeFromDiscontinuedOperations  # Derived

        eps = _get_financial_value(data, 'EarningsPerShareBasic')
        epsDiluted = _get_financial_value(data, 'EarningsPerShareDiluted')
        weightedAverageShsOut = _get_financial_value(
            data, 'WeightedAverageNumberOfSharesOutstandingBasic')
        weightedAverageShsOutDil = _get_financial_value(
            data, 'WeightedAverageNumberOfDilutedSharesOutstanding')

        # Base item structure common to all statement types
        base_item = {
            "date": period_details["endDate"],
            # Store original symbol if provided
            "symbol": symbol_or_cik.upper() if isinstance(symbol_or_cik, str) and not symbol_or_cik.isdigit() else "N/A_CIK_USED",
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
            cashAndCashEquivalents = _get_financial_value(
                data,
                "CashAndCashEquivalentsAtCarryingValue",
                alternate_tags=[
                    "CashAndCashEquivalents",
                    "Cash",
                    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"])
            shortTermInvestments = _get_financial_value(
                data,
                "MarketableSecuritiesCurrent",
                alternate_tags=[
                    "ShortTermInvestments",
                    "AvailableForSaleSecuritiesCurrent",
                    "MarketableSecuritiesDebtMaturitiesWithinOneYearAmortizedCost"])
            cashAndShortTermInvestments = cashAndCashEquivalents + shortTermInvestments

            netReceivables = _get_financial_value(
                data,
                "AccountsReceivableNetCurrent",
                alternate_tags=[
                    "ReceivablesNetCurrent",
                    "AccountsReceivableGrossCurrent"
                ]
            )
            accountsReceivables = _get_financial_value(
                data,
                "AccountsReceivableTradeCurrent",
                alternate_tags=[
                    "AccountsReceivableGrossCurrent"
                ]
            )
            otherReceivables = _get_financial_value(
                data,
                "OtherReceivablesCurrent",
                alternate_tags=[
                    "NotesAndLoansReceivableNetCurrent",
                    "ContractReceivableRetainage"
                ]
            )
            inventory = _get_financial_value(
                data,
                "InventoryNet",
                alternate_tags=[
                    "Inventory",
                    "InventoryFinishedGoods",
                    "InventoryRawMaterials",
                    "InventoryWorkInProcess"
                ]
            )
            prepaids = _get_financial_value(
                data,
                "PrepaidExpenseCurrent",
                alternate_tags=[
                    "PrepaidExpenseAndOtherAssetsCurrent"
                ]
            )
            otherCurrentAssets = _get_financial_value(
                data,
                "OtherAssetsCurrent",
                alternate_tags=[
                    "OtherAssetsMiscellaneousCurrent"
                ]
            )

            totalCurrentAssets = _get_financial_value(
                data,
                "AssetsCurrent",
                alternate_tags=[
                    "AssetsHeldForSaleCurrent",
                    "AssetsOfDisposalGroupIncludingDiscontinuedOperationCurrent"
                ]
            )
            if totalCurrentAssets == 0:
                # If the total isn't explicitly stated, sum the components
                totalCurrentAssets = cashAndShortTermInvestments + \
                    netReceivables + inventory + prepaids + otherCurrentAssets

            # Non-Current Assets
            propertyPlantEquipmentNet = _get_financial_value(
                data,
                "PropertyPlantAndEquipmentNet",
                alternate_tags=[
                    "PropertyPlantAndEquipmentGross"
                ]
            )
            goodwill = _get_financial_value(
                data,
                "Goodwill",
                alternate_tags=[
                    "GoodwillImpairedAccumulatedImpairmentLoss"
                ]
            )
            intangibleAssets = _get_financial_value(
                data,
                "IntangibleAssetsNetExcludingGoodwill",
                alternate_tags=[
                    "IntangibleAssets",
                    "IntangibleAssetsGrossExcludingGoodwill"
                ]
            )

            goodwillAndIntangibleAssets = _get_financial_value(
                data, "GoodwillAndIntangibleAssets")
            if goodwillAndIntangibleAssets == 0:
                goodwillAndIntangibleAssets = goodwill + intangibleAssets

            longTermInvestments = _get_financial_value(
                data,
                "MarketableSecuritiesNoncurrent",
                alternate_tags=[
                    "LongTermInvestments",
                    "AvailableForSaleSecuritiesRestrictedNoncurrent"
                ]
            )
            taxAssets = _get_financial_value(
                data,
                "DeferredTaxAssetsNet",
                alternate_tags=[
                    "DeferredTaxAssetsNetCurrent",
                    "DeferredTaxAssetsNetNoncurrent"
                ]
            )
            otherNonCurrentAssets = _get_financial_value(
                data,
                "OtherAssetsNoncurrent",
                alternate_tags=[
                    "OtherAssetsMiscellaneousNoncurrent"
                ]
            )

            totalNonCurrentAssets = _get_financial_value(
                data,
                "AssetsNoncurrent",
                alternate_tags=[
                    "NoncurrentAssets"
                ]
            )
            if totalNonCurrentAssets == 0:
                # If the total isn't explicitly stated, sum the components
                totalNonCurrentAssets = propertyPlantEquipmentNet + goodwillAndIntangibleAssets + \
                    longTermInvestments + taxAssets + otherNonCurrentAssets

            # Total Assets
            totalAssets = _get_financial_value(
                data,
                "Assets",
                alternate_tags=[
                    "TotalAssets",
                    "LiabilitiesAndStockholdersEquity"
                ]
            )
            if totalAssets == 0:
                totalAssets = totalCurrentAssets + totalNonCurrentAssets

            # LIABILITIES
            # Current Liabilities
            accountPayables = _get_financial_value(
                data,
                "AccountsPayableCurrent",
                alternate_tags=[
                    "IncreaseDecreaseInAccountsPayable"
                ]
            )
            otherPayables = _get_financial_value(
                data,
                "OtherAccountsPayableCurrent"
            )
            totalPayables = accountPayables + otherPayables
            accruedExpenses = _get_financial_value(
                data,
                "AccruedLiabilitiesCurrent",
                alternate_tags=[
                    "AccruedIncomeTaxesCurrent",
                    "AccruedIncomeTaxesNoncurrent"
                ]
            )

            shortTermDebt = _get_financial_value(
                data,
                "DebtCurrent",
                alternate_tags=[
                    "LongTermDebtAndCapitalLeaseObligationsCurrent"
                ]
            )
            if shortTermDebt == 0:
                component_sum = (
                    _get_financial_value(
                        data,
                        "CommercialPaper") +
                    _get_financial_value(
                        data,
                        "LongTermDebtCurrentMaturities") +
                    _get_financial_value(
                        data,
                        "NotesPayableCurrent") +
                    _get_financial_value(
                        data,
                        "ShortTermBorrowings"))
                if component_sum == 0:
                    alt_std = _get_financial_value(
                        data, "LongTermDebtCurrentMaturitiesAndOtherShortTermDebt")
                    if alt_std != 0:
                        component_sum = alt_std
                shortTermDebt = component_sum

            capitalLeaseObligationsCurrent = _get_financial_value(
                data,
                "OperatingLeaseLiabilityCurrent",
                alternate_tags=["CapitalLeaseObligationsCurrent"]
            )
            taxPayables = _get_financial_value(data, "IncomeTaxesPayable")
            deferredRevenue = _get_financial_value(
                data,
                "DeferredRevenueCurrent",
                alternate_tags=["ContractWithCustomerLiabilityCurrent"]
            )
            otherCurrentLiabilities = _get_financial_value(
                data,
                "OtherLiabilitiesCurrent"
            )

            totalCurrentLiabilities = _get_financial_value(
                data,
                "LiabilitiesCurrent",
                alternate_tags=[
                    "LiabilitiesOfDisposalGroupIncludingDiscontinuedOperationCurrent"
                ]
            )
            if totalCurrentLiabilities == 0:
                totalCurrentLiabilities = (
                    accountPayables +
                    otherPayables +
                    accruedExpenses +
                    shortTermDebt +
                    capitalLeaseObligationsCurrent +
                    taxPayables +
                    deferredRevenue +
                    otherCurrentLiabilities)

            # Non-Current Liabilities
            longTermDebt_val = _get_financial_value(
                data,
                "LongTermDebtNoncurrent",
                alternate_tags=["LongTermDebt"]
            )
            capitalLeaseObligationsNonCurrent = _get_financial_value(
                data,
                "OperatingLeaseLiabilityNoncurrent",
                alternate_tags=["CapitalLeaseObligationsNoncurrent"]
            )
            deferredRevenueNonCurrent = _get_financial_value(
                data,
                "DeferredRevenueNoncurrent",
                alternate_tags=["ContractWithCustomerLiabilityNoncurrent"]
            )
            deferredTaxLiabilitiesNonCurrent = _get_financial_value(
                data, "DeferredTaxLiabilitiesNoncurrent")
            otherNonCurrentLiabilities = _get_financial_value(
                data, "OtherLiabilitiesNoncurrent")

            totalNonCurrentLiabilities = _get_financial_value(
                data,
                "LiabilitiesNoncurrent"
            )
            if totalNonCurrentLiabilities == 0:
                totalNonCurrentLiabilities = (
                    longTermDebt_val +
                    capitalLeaseObligationsNonCurrent +
                    deferredRevenueNonCurrent +
                    deferredTaxLiabilitiesNonCurrent +
                    otherNonCurrentLiabilities)

            capitalLeaseObligations = capitalLeaseObligationsCurrent + \
                capitalLeaseObligationsNonCurrent

            # Total Liabilities
            totalLiabilities = _get_financial_value(
                data, "Liabilities", alternate_tags=[
                    "TotalLiabilities", "LiabilitiesAndStockholdersEquity"])
            if totalLiabilities == 0:
                totalLiabilities = totalCurrentLiabilities + totalNonCurrentLiabilities

            # EQUITY
            treasuryStock = _get_financial_value(data, "TreasuryStockValue")
            preferredStock = _get_financial_value(
                data,
                "PreferredStockValue",
                alternate_tags=["RedeemablePreferredStockCarryingAmount"]
            )
            commonStock = _get_financial_value(
                data,
                "CommonStockValue",
                alternate_tags=[
                    "CommonStock",
                    "CommonStocksIncludingAdditionalPaidInCapital",
                    "CommonStockSharesOutstanding",
                    "CommonStockSharesIssued"
                ]
            )
            additionalPaidInCapital = _get_financial_value(
                data,
                "AdditionalPaidInCapital",
                alternate_tags=["AdditionalPaidInCapitalCommonStock"]
            )
            if commonStock == _get_financial_value(
                    data,
                    "CommonStocksIncludingAdditionalPaidInCapital") and additionalPaidInCapital == 0:
                commonStock -= additionalPaidInCapital
                if commonStock < 0:
                    commonStock = _get_financial_value(
                        data, "CommonStockValue")
            retainedEarnings = _get_financial_value(
                data,
                "RetainedEarningsAccumulatedDeficit",
                alternate_tags=["RetainedEarnings"]
            )
            accumulatedOtherComprehensiveIncomeLoss = _get_financial_value(
                data,
                "AccumulatedOtherComprehensiveIncomeLossNetOfTax"
            )
            minorityInterest = _get_financial_value(
                data,
                "MinorityInterest",
                alternate_tags=["NoncontrollingInterest"]
            )

            # Total Equity
            totalStockholdersEquity = _get_financial_value(
                data,
                "StockholdersEquity",
                alternate_tags=[
                    "TotalStockholdersEquity",
                    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"
                ]
            )

            if minorityInterest == 0 and _get_financial_value(
                    data, "StockholdersEquity") != totalStockholdersEquity:
                minorityInterest = totalStockholdersEquity - \
                    _get_financial_value(data, "StockholdersEquity")

            if totalStockholdersEquity == 0:
                # This is a common calculation, but note that treasury stock is
                # negative equity.
                totalStockholdersEquity = (
                    commonStock + additionalPaidInCapital + retainedEarnings +
                    accumulatedOtherComprehensiveIncomeLoss - treasuryStock
                )

            # DERIVED & SUMMARY METRICS
            totalLiabilitiesAndTotalEquity = totalLiabilities + totalStockholdersEquity
            totalInvestments = shortTermInvestments + longTermInvestments

            totalDebt = _get_financial_value(
                data,
                "TotalDebt",
                alternate_tags=["Debt"]
            )
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
            }
        elif statement_type == "cash_flow":
            # Operating Activities
            netIncome_cf = _get_financial_value(
                data, "NetIncomeLoss")  # Often starting point
            depreciationAndAmortization_cf = _get_financial_value(
                data, "DepreciationDepletionAndAmortization")
            deferredIncomeTax_cf = _get_financial_value(
                data,
                "DeferredIncomeTaxExpenseBenefit",
                alternate_tags=["IncreaseDecreaseInDeferredIncomeTaxes"])
            stockBasedCompensation_cf = _get_financial_value(
                data, "ShareBasedCompensation")

            accountsReceivables_flow = _get_financial_value(
                data, "IncreaseDecreaseInAccountsReceivableNetCurrent")
            inventory_flow = _get_financial_value(
                data, "IncreaseDecreaseInInventoriesNet")
            accountsPayables_flow = _get_financial_value(
                data, "IncreaseDecreaseInAccountsPayableCurrent")
            otherWorkingCapital_flow = _get_financial_value(
                data, "IncreaseDecreaseInOtherOperatingAssetsLiabilitiesNet")
            changeInWorkingCapital = accountsReceivables_flow + inventory_flow + \
                accountsPayables_flow + otherWorkingCapital_flow  # Sum of individual flow components

            otherNonCashItems_cf = _get_financial_value(
                data, "OtherNoncashIncomeExpense")
            netCashProvidedByOperatingActivities = _get_financial_value(
                data, "NetCashProvidedByUsedInOperatingActivities")

            # Investing Activities
            investmentsInPropertyPlantAndEquipment = _get_financial_value(
                data, "PaymentsToAcquirePropertyPlantAndEquipment")  # Typically negative
            acquisitionsNet_cf = _get_financial_value(
                data, "PaymentsToAcquireBusinessesNetOfCashAcquired")
            purchasesOfInvestments_cf = _get_financial_value(
                data, "PaymentsForPurchasesOfInvestments")
            salesMaturitiesOfInvestments_cf = _get_financial_value(
                data, "ProceedsFromSaleAndMaturityOfMarketableSecurities")
            otherInvestingActivities_cf = _get_financial_value(
                data, "OtherInvestingActivitiesCashFlows")
            netCashProvidedByInvestingActivities = _get_financial_value(
                data, "NetCashProvidedByUsedInInvestingActivities")

            # Financing Activities
            proceedsFromLongTermDebt = _get_financial_value(
                data, "ProceedsFromIssuanceOfLongTermDebt")
            repaymentsOfLongTermDebt = _get_financial_value(
                data, "RepaymentsOfLongTermDebt")  # Typically negative
            longTermNetDebtIssuance = proceedsFromLongTermDebt + repaymentsOfLongTermDebt

            proceedsFromShortTermDebt = _get_financial_value(
                data, "ProceedsFromShortTermDebt")
            repaymentsOfShortTermDebt = _get_financial_value(
                data, "RepaymentsOfShortTermDebt")  # Typically negative
            shortTermNetDebtIssuance = proceedsFromShortTermDebt + repaymentsOfShortTermDebt

            netDebtIssuance = longTermNetDebtIssuance + shortTermNetDebtIssuance

            proceedsFromCommonStock = _get_financial_value(
                data, "ProceedsFromIssuanceOfCommonStock")
            paymentsForRepurchaseOfCommonStock = _get_financial_value(
                data, "PaymentsForRepurchaseOfCommonStock")  # Typically negative
            netCommonStockIssuance = proceedsFromCommonStock + \
                paymentsForRepurchaseOfCommonStock
            # Matches example's negative convention
            commonStockRepurchased_cf = paymentsForRepurchaseOfCommonStock

            proceedsFromPreferredStock = _get_financial_value(
                data, "ProceedsFromIssuanceOfPreferredStock")
            paymentsForRepurchaseOfPreferredStock = _get_financial_value(
                data, "PaymentsForRepurchaseOfPreferredStock")  # Typically negative
            netPreferredStockIssuance = proceedsFromPreferredStock + \
                paymentsForRepurchaseOfPreferredStock

            netStockIssuance = netCommonStockIssuance + netPreferredStockIssuance

            commonDividendsPaid_cf = _get_financial_value(
                data, "PaymentsOfDividendsCommonStock")  # Typically negative
            preferredDividendsPaid_cf = _get_financial_value(
                data, "PaymentsOfDividendsPreferredStock")  # Typically negative
            netDividendsPaid_cf = commonDividendsPaid_cf + preferredDividendsPaid_cf

            otherFinancingActivities_cf = _get_financial_value(
                data, "OtherFinancingActivitiesCashFlows")
            netCashProvidedByFinancingActivities = _get_financial_value(
                data, "NetCashProvidedByUsedInFinancingActivities")

            # Summary
            effectOfForexChangesOnCash_cf = _get_financial_value(
                data, "EffectOfExchangeRateOnCashAndCashEquivalents")
            netChangeInCash = _get_financial_value(
                data, "CashAndCashEquivalentsPeriodIncreaseDecrease")
            cashAtEndOfPeriod_cf = _get_financial_value(
                data, "CashAndCashEquivalentsAtCarryingValue")
            cashAtBeginningOfPeriod_cf = _get_financial_value(
                data,
                "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalentsAtBeginningOfPeriod",
                alternate_tags=["CashAndCashEquivalentsAtBeginningOfPeriod"])
            if cashAtBeginningOfPeriod_cf == 0 and cashAtEndOfPeriod_cf != 0 and netChangeInCash != 0:  # Try to derive if not found
                cashAtBeginningOfPeriod_cf = cashAtEndOfPeriod_cf - netChangeInCash - \
                    effectOfForexChangesOnCash_cf  # Adjust for forex too

            # Derived & Other
            operatingCashFlow_cf = netCashProvidedByOperatingActivities  # Alias
            capitalExpenditure_cf = investmentsInPropertyPlantAndEquipment * - \
                1 if investmentsInPropertyPlantAndEquipment < 0 else investmentsInPropertyPlantAndEquipment  # Make positive
            freeCashFlow_cf = netCashProvidedByOperatingActivities + \
                investmentsInPropertyPlantAndEquipment  # Capex is negative

            incomeTaxesPaid_cf = _get_financial_value(
                data, "IncomeTaxesPaidNet")
            interestPaid_cf = _get_financial_value(data, "InterestPaidNet")

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
            item = {
                **base_item,
                "error": "Unknown statement type for formatting"}
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
        return _get_company_facts_request(
            symbol_or_cik, self.headers, self.get_cik_for_symbol)

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
            symbol_or_cik,
            taxonomy,
            tag,
            self.headers,
            self.get_cik_for_symbol)

    def get_aggregated_frames_data(
            self,
            taxonomy,
            tag,
            unit,
            year,
            quarter=None,
            instantaneous=False):
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
            taxonomy,
            tag,
            unit,
            year,
            self.headers,
            quarter,
            instantaneous)

    def get_income_statement(self, symbol, limit=5, report_type="ALL"):
        """
        Fetches and formats recent income statement data for a given symbol.

        Args:
            symbol (str): The stock ticker symbol.
            limit (int): The number of recent periods to retrieve.
            report_type (str): The type of report to filter by ("10-K", "10-Q", "ALL").
                               Defaults to "ALL".

        Returns:
            list: A list of dictionaries with income statement data.
        """
        return _get_financial_statement_data(
            symbol, "income_statement", limit, report_type,
            self.headers,
            self.get_cik_for_symbol,
            self.get_company_all_facts  # Pass the instance method
        )

    def get_balance_sheet(self, symbol, limit=5, report_type="ALL"):
        """
        Fetches and formats recent balance sheet data for a given symbol.

        Args:
            symbol (str): The stock ticker symbol.
            limit (int): The number of recent periods to retrieve.
            report_type (str): The type of report to filter by ("10-K", "10-Q", "ALL").
                               Defaults to "ALL".

        Returns:
            list: A list of dictionaries with balance sheet data.
        """
        return _get_financial_statement_data(
            symbol, "balance_sheet", limit, report_type,
            self.headers,
            self.get_cik_for_symbol,
            self.get_company_all_facts
        )

    def get_cash_flow_statement(self, symbol, limit=5, report_type="ALL"):
        """
        Fetches and formats recent cash flow statement data for a given symbol.

        Args:
            symbol (str): The stock ticker symbol.
            limit (int): The number of recent periods to retrieve.
            report_type (str): The type of report to filter by ("10-K", "10-Q", "ALL").
                               Defaults to "ALL".

        Returns:
            list: A list of dictionaries with cash flow statement data.
        """
        return _get_financial_statement_data(
            symbol, "cash_flow", limit, report_type,
            self.headers,
            self.get_cik_for_symbol,
            self.get_company_all_facts
        )
