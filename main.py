# filename: main.py (or any script in your project)

from sec_financial_data.sec_financial_data import SECHelper
import json

def main():
    # Initialize the helper. It's highly recommended to provide a descriptive user-agent.
    # Replace "YourAppName/1.0" with your application name and "your-email@example.com"
    # with a contact email for responsible usage as per SEC guidelines.
    helper = SECHelper(user_agent_string="sec_api/0.1.0 (arun.mittal.sjsu@gmail.com)")

    symbol = "AAPL" # Example: Apple Inc.

    print(f"\n--- Getting CIK for {symbol} ---")
    cik = helper.get_cik_for_symbol(symbol)
    if cik:
        print(f"CIK for {symbol}: {cik}")
    else:
        print(f"Could not find CIK for {symbol}")
        return # Exit if CIK not found

    # --- Income Statement 10-K Only ---
    print(f"\n--- Getting Income Statement for {symbol} (10-K, limit 5) ---")
    income_statement_10k = helper.get_income_statement(symbol, limit=5, report_type="10-K")
    if income_statement_10k:
        print(f"Found {len(income_statement_10k)} 10-K reports.")
        print(json.dumps(income_statement_10k, indent=2))
    else:
        print(f"No 10-K income statement data found for {symbol}")

    # --- Balance Sheet 10-K Only ---
    print(f"\n--- Getting Balance Sheet for {symbol} (10-K, limit 5) ---")
    balance_sheet_10k = helper.get_balance_sheet(symbol, limit=5, report_type="10-K")
    if balance_sheet_10k:
        print(f"Found {len(balance_sheet_10k)} 10-K reports.")
        print(json.dumps(balance_sheet_10k, indent=2))
    else:
        print(f"No 10-K balance sheet data found for {symbol}")

    # --- Cash Flow Statement 10-K Only ---
    print(f"\n--- Getting Cash Flow Statement for {symbol} (10-K, limit 5) ---")
    cash_flow_10k = helper.get_cash_flow_statement(symbol, limit=5, report_type="10-K")
    if cash_flow_10k:
        print(f"Found {len(cash_flow_10k)} 10-K reports.")
        print(json.dumps(cash_flow_10k, indent=2))
    else:
        print(f"No 10-K cash flow statement data found for {symbol}")

    # --- Income Statement 10-Q Only ---
    print(f"\n--- Getting Income Statement for {symbol} (10-Q, limit 1) ---")
    income_statement_10q = helper.get_income_statement(symbol, limit=1, report_type="10-Q")
    if income_statement_10q:
        print(f"Found {len(income_statement_10q)} 10-Q reports.")
        print(json.dumps(income_statement_10q, indent=2))
    else:
        print(f"No 10-Q income statement data found for {symbol}")

    # --- Balance Sheet 10-Q Only ---
    print(f"\n--- Getting Balance Sheet for {symbol} (10-Q, limit 1) ---")
    balance_sheet_10q = helper.get_balance_sheet(symbol, limit=1, report_type="10-Q")
    if balance_sheet_10q:
        print(f"Found {len(balance_sheet_10q)} 10-Q reports.")
        print(json.dumps(balance_sheet_10q, indent=2))
    else:
        print(f"No 10-Q balance sheet data found for {symbol}")

    # --- Cash Flow Statement 10-Q Only ---
    print(f"\n--- Getting Cash Flow Statement for {symbol} (10-Q, limit 1) ---")
    cash_flow_10q = helper.get_cash_flow_statement(symbol, limit=1, report_type="10-Q")
    if cash_flow_10q:
        print(f"Found {len(cash_flow_10q)} 10-Q reports.")
        print(json.dumps(cash_flow_10q, indent=2))
    else:
        print(f"No 10-Q cash flow statement data found for {symbol}")

    # --- Advanced Usage: Fetching specific XBRL concepts ---
    print(f"\n--- Getting specific concept: 'Revenues' for {symbol} ---")
    revenues_concept = helper.get_company_specific_concept(symbol, "us-gaap", "Revenues")
    if revenues_concept:
        # This will contain raw XBRL facts. You'd typically process this further.
        # For demonstration, we'll just print a small part of it.
        print(f"Concept: {revenues_concept.get('label')}")
        print(f"Description: {revenues_concept.get('description')}")

        # Example of accessing some facts (first 2 for brevity)
        if 'units' in revenues_concept and 'USD' in revenues_concept['units']:
            print("First 2 USD facts for Revenues:")
            for fact in revenues_concept['units']['USD'][:2]:
                print(f"  Value: {fact.get('val')}, Fiscal Year: {fact.get('fy')}, Period: {fact.get('fp')}, End Date: {fact.get('end')}")
        else:
            print("No USD facts found for Revenues.")
    else:
        print(f"No concept data found for 'Revenues' for {symbol}")

    # --- Advanced Usage: Getting aggregated frame data (e.g., Assets for all companies in a specific quarter) ---
    print("\n--- Getting aggregated 'Assets' for Q1 2024 (USD) ---")
    # This might return a very large dataset as it covers many companies.
    # Be mindful of the output size and SEC rate limits.
    # We'll only print a small sample for demonstration.
    # Note: The SEC Frames API data availability can vary. Using a slightly older year for better chance of data.
    year_for_frames = 2023
    assets_q1_frames = helper.get_aggregated_frames_data(
        "us-gaap", "Assets", "USD", year_for_frames, quarter=1, instantaneous=True
    )
    if assets_q1_frames and 'data' in assets_q1_frames: # The 'frames' API returns data under a 'data' key
        print(f"Total number of companies reporting Assets in Q1 {year_for_frames}: {len(assets_q1_frames['data'])}")
        if assets_q1_frames['data']:
            print(f"Label for aggregated data: {assets_q1_frames.get('label', 'N/A')}")
            print(f"Description for aggregated data: {assets_q1_frames.get('description', 'N/A')}")
            print("Sample of Assets data (first 3 companies):")
            for i, item in enumerate(assets_q1_frames['data'][:3]):
                # The 'frames' API returns CIKs directly
                sample_cik = item.get('cik')
                sample_entity_name = item.get('entityName', 'N/A')
                sample_value = item.get('val')
                sample_accn = item.get('accn') # Accession number of the filing
                sample_form = item.get('form') # Form type (e.g., 10-K, 10-Q)
                sample_end = item.get('end')   # End date of the reported fact
                print(
                    f"  Company CIK: {sample_cik}, Name: {sample_entity_name}, "
                    f"Value: {sample_value}, Form: {sample_form}, End Date: {sample_end}, Accn: {sample_accn}"
                )
        else:
            print(f"No company data found within the aggregated Assets data for Q1 {year_for_frames}.")
    else:
        print(f"No aggregated Assets data found for Q1 {year_for_frames}.")


if __name__ == "__main__":
    main()
