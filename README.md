# sec-financial-data

[![PyPI version](https://badge.fury.io/py/sec-financial-data.svg)](https://badge.fury.io/py/sec-financial-data)

This Python package, sec-financial-data, is designed to simplify the process of retrieving and accessing publicly available financial data from the U.S. Securities and Exchange Commission's (SEC) EDGAR database. It acts as a convenient wrapper around the SEC's EDGAR APIs, providing a more Pythonic and user-friendly interface for developers and financial analysts.

Core Purpose:

The primary goal of the package is to enable users to programmatically fetch various types of financial information for publicly traded companies. This includes:

    1. Company identification data (CIK numbers).
    2. Raw XBRL (eXtensible Business Reporting Language) data, which forms the basis of SEC filings.
    3. Formatted financial statements like Income Statements, Balance Sheets, and Cash Flow Statements.
    4. Aggregated financial data points across multiple companies for comparative analysis.

Key Features & Functionality:

    1. User-Agent Management: The package emphasizes responsible API usage by allowing and encouraging users to set a custom User-Agent string, which is a requirement by the SEC for programmatic access.
    2. Rate Limiting: It incorporates a basic rate-limiting mechanism to ensure that requests to the SEC EDGAR APIs do not exceed the allowed limit (10 requests per second), preventing potential IP blocks.
    3. CIK Lookup:
        1. Provides a function (get_cik_for_symbol) to retrieve a company's Central Index Key (CIK) using its stock ticker symbol.
        2. The CIK map is fetched from the SEC and cached (@lru_cache) to improve performance for subsequent lookups.
    4. Access to Raw XBRL Data:
        1. get_company_all_facts(): Fetches the complete set of XBRL data (company facts) for a given CIK or ticker symbol. This is a comprehensive JSON object containing all reported financial concepts and their values over time.
        2. get_company_specific_concept(): Allows users to retrieve data for a specific XBRL concept (e.g., "Revenues," "Assets") within a particular taxonomy (e.g., "us-gaap") for a company.
    5. Aggregated Data (Frames API):
        1. get_aggregated_frames_data(): Interacts with the SEC's "frames" API to fetch aggregated data for a specific XBRL tag, unit, and period (year/quarter) across all reporting entities. This is useful for market-wide analysis or peer comparisons.
    6. Formatted Financial Statements:
        1. The core of its value proposition for many users lies in the get_financial_statement() function and its specific wrappers: get_income_statement(), get_balance_sheet(), and get_cash_flow_statement().
        2. These functions take raw XBRL data (obtained via get_company_facts()) and attempt to parse and structure it into a more conventional and readable JSON format for standard financial statements.
        3. Data Normalization Logic:
            1. It identifies relevant XBRL tags for common line items in each statement type (e.g., "Revenues", "NetIncomeLoss", "Assets", "CashAndCashEquivalentsAtCarryingValue").
            2. It includes a list of primary and alternate XBRL tags for many financial line items, acknowledging that companies might use different tags for similar concepts. The _get_financial_value() helper method is crucial here, trying a primary tag first and then falling back to alternates.
            3. It processes facts from 10-K (annual) and 10-Q (quarterly) filings.
            4. It identifies the "canonical" report for each period (latest filing for a given end date) to avoid using superseded data.
            5. It sorts reports chronologically and allows users to specify a limit for the number of recent periods.
            6. It performs some derivations (e.g., calculating Gross Profit if not directly reported, deriving EBITDA, Net Debt).
    7. SECHelper Class: This class serves as the main public interface for the package, encapsulating all the user-facing methods for easy access.

How it Works (High-Level):

    1. The user initializes the SECHelper class, optionally providing a `user_agent_string`.
    2. When a method is called (e.g., get_income_statement("AAPL")):
        1. If a ticker symbol is provided, it's first converted to a CIK.
        2. The necessary SEC API endpoint is constructed.
        3. A rate-limited HTTP GET request is made to the SEC API using the requests library.
        4. The JSON response from the API is retrieved.
        5. For financial statement functions, the extensive get_company_facts() data is processed:
            1. Facts are filtered and grouped by filing (10-K/10-Q) and period.
            2. The most recent, relevant filings are selected.
            3. Values for predefined financial statement line items are extracted using the primary/alternate tag logic.
            4. Some line items are calculated based on others.
        6. The final data is returned as a Python dictionary or list of dictionaries.

Target Audience:

    1. Developers: Building financial applications, tools, or dashboards.
    2. Financial Analysts & Researchers: Performing quantitative analysis, company valuation, or market research.
    3. Students & Hobbyists: Learning about financial data and programming.

Key Abstractions:

    1. It abstracts away the complexities of directly interacting with the SEC EDGAR APIs (endpoint URLs, rate limits, raw XBRL structure).
    2. It attempts to normalize the varied XBRL tags into a more standardized set of financial line items, though this is inherently challenging due to the flexibility of XBRL.

Important Considerations for Users:

    1. XBRL Tag Variability: Financial data reporting using XBRL can vary between companies and over time. While the package uses a list of common and alternate tags, it might not capture every possible variation for every company. Users might need to inspect the raw company_facts for specific or less common tags.
    2. Data Accuracy: The package relies on the data as reported to the SEC. The parsing and normalization logic aims for accuracy but is based on common US GAAP interpretations.
    3. API Changes: The SEC may update its APIs, which could potentially require updates to the package.
    4. User-Agent: It's crucial to provide a descriptive user_agent for responsible API usage.

In summary, sec-financial-data provides a valuable toolkit for accessing and working with SEC financial data in Python, significantly lowering the barrier to entry for retrieving and making sense of this rich dataset. It balances ease of use for common financial statements with the ability to access raw, detailed XBRL data for more advanced use cases.

## Features

*   Fetch financial statements (10-K, 10-Q) for a given company.
*   Extract specific financial data points.
*   User-friendly API.

## Installation

You can install `sec-financial-data` from PyPI:

```bash
pip install sec-financial-data
```

Requires Python 3.7+.

## Usage
```python
from sec_financial_data.sec_financial_data import SECHelper
import json

# It's highly recommended to provide a descriptive user-agent.
# Replace "YourAppName/1.0" and "your-email@example.com" accordingly.
helper = SECHelper(user_agent_string="MyApp/1.0 (contact@example.com)")

# --- Get CIK for a Ticker Symbol ---
symbol = "AAPL"
cik = helper.get_cik_for_symbol(symbol)
if cik:
    print(f"CIK for {symbol}: {cik}")
else:
    print(f"Could not find CIK for {symbol}.")

# --- Get All Company Facts (XBRL data) ---
# Can use symbol or CIK
company_facts = helper.get_company_all_facts(symbol) # or helper.get_company_all_facts(cik)
if company_facts:
    # This is a large JSON object containing all reported XBRL data for the company.
    # print(json.dumps(company_facts, indent=2)) # Be cautious, can be very large
    print(f"Successfully fetched all facts for {company_facts.get('entityName', symbol)}.")
    # Example: Accessing a specific fact (e.g., Assets for us-gaap)
    # assets_data = company_facts.get('facts', {}).get('us-gaap', {}).get('Assets', {})
    # if assets_data:
    #     print(f"Assets data available: {assets_data.get('label')}")

# --- Get Specific Company Concept ---
taxonomy = "us-gaap"
tag = "Revenues"
concept_data = helper.get_company_specific_concept(symbol, taxonomy, tag) # or CIK
if concept_data:
    print(f"\nData for concept '{tag}' ({taxonomy}) for {symbol}:")
    # print(json.dumps(concept_data, indent=2)) # Raw XBRL concept data
    if 'units' in concept_data and 'USD' in concept_data['units']:
        print(f"Label: {concept_data.get('label')}, Description: {concept_data.get('description')}")
        print("Last 2 reported USD values for Revenues:")
        for fact in concept_data['units']['USD'][-2:]: # Print last 2 facts
            print(f"  Value: {fact.get('val')}, End Date: {fact.get('end')}, Form: {fact.get('form')}")

# --- Get Aggregated Frames Data ---
# Fetches data for a specific tag across many companies for a given period.
# Example: Total Assets for all filers in Q1 2023.
frame_taxonomy = "us-gaap"
frame_tag = "Assets"
frame_unit = "USD"
frame_year = 2023
frame_quarter = 1 # Optional, for quarterly data
frame_instantaneous = True # True for balance sheet items

assets_frame_data = helper.get_aggregated_frames_data(
    frame_taxonomy, frame_tag, frame_unit, frame_year,
    quarter=frame_quarter, instantaneous=frame_instantaneous
)
if assets_frame_data and 'data' in assets_frame_data:
    print(f"\nAggregated '{frame_tag}' data for {frame_year} Q{frame_quarter}:")
    print(f"Total companies reporting: {len(assets_frame_data['data'])}")
    # print(json.dumps(assets_frame_data['data'][:2], indent=2)) # Print data for first 2 companies
else:
    print(f"No aggregated '{frame_tag}' data found for {frame_year} Q{frame_quarter}.")

# --- Get Formatted Financial Statements ---

# Get the last 3 Income Statements (combining 10-K and 10-Q, most recent first)
income_all = helper.get_income_statement(symbol, limit=3)
print(f"--- Income Statements (ALL, limit 3) for {symbol} ---")
# print(json.dumps(income_all, indent=2))

# Get the last 2 annual (10-K) Income Statements
income_10k = helper.get_income_statement(symbol, limit=2, report_type="10-K")
print(f"\n--- Income Statements (10-K, limit 2) for {symbol} ---")
# print(json.dumps(income_10k, indent=2))
if income_10k:
    for report in income_10k:
        print(f"Date: {report['date']}, Form: {report['_formType_original']}, Revenue: {report['revenue']}")

# Get the last 4 quarterly (10-Q) Balance Sheets
balance_sheet_10q = helper.get_balance_sheet(symbol, limit=4, report_type="10-Q")
print(f"\n--- Balance Sheets (10-Q, limit 4) for {symbol} ---")
# print(json.dumps(balance_sheet_10q, indent=2))
if balance_sheet_10q:
    for report in balance_sheet_10q:
        print(f"Date: {report['date']}, Form: {report['_formType_original']}, Total Assets: {report['totalAssets']}")

# Get Cash Flow Statement (last 2 periods)
cash_flow_statement = helper.get_cash_flow_statement(symbol, limit=2)
print(f"\nCash Flow Statement for {symbol} (last {len(cash_flow_statement)} periods):")
# print(json.dumps(cash_flow_statement, indent=2))
```

*(Please replace the example above with a concise, working example of how to use your library.)*

## Dependencies

*   requests (>=2.20.0)

## Contributing
Contributions are welcome! Please see the Bug Tracker for issues or to submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Project Links
*   Homepage
*   Bug Tracker
