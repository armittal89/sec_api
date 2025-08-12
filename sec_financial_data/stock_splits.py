#!/usr/bin/env python3
"""
Stock Split Module

This module provides clean, focused functionality for detecting and managing stock splits
from SEC filings. It handles the complex logic of finding stock split announcements,
prioritizing them by date and filing type, and adjusting financial data accordingly.
"""

import re
import requests
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


class StockSplitDetector:
    """
    Detects stock splits from SEC filings and provides methods for financial data adjustment.
    """

    def __init__(self, headers: Dict[str, str]):
        """
        Initialize the stock split detector.

        Args:
            headers: HTTP headers for SEC API requests
        """
        self.headers = headers

        # Performance optimizations
        self._cik_cache = {}  # Cache for CIK lookups
        self._split_cache = {}  # Cache for split results

        # Number words dictionary for parsing word-based ratios
        self.number_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
            "twenty-one": 21,
            "twenty-two": 22,
            "twenty-three": 23,
            "twenty-four": 24,
            "twenty-five": 25,
            "twenty-six": 26,
            "twenty-seven": 27,
            "twenty-eight": 28,
            "twenty-nine": 29,
            "thirty": 30,
            "thirty-one": 31,
            "thirty-two": 32,
            "thirty-three": 33,
            "thirty-four": 34,
            "thirty-five": 35,
            "thirty-six": 36,
            "thirty-seven": 37,
            "thirty-eight": 38,
            "thirty-nine": 39,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
            "hundred": 100,
        }

        # Filing type priority (lower number = higher priority)
        self.form_priority = {
            "8-K": 1,  # Highest priority - immediate announcements
            "10-Q": 2,  # High priority - quarterly reports
            "10-K": 3,  # Medium priority - annual reports
            "DEF 14A": 4,  # Lower priority - proxy statements
            "DEF 14C": 4,  # Lower priority - proxy statements
            "424B3": 5,  # Lower priority - prospectus filings
            "424B4": 5,  # Lower priority - prospectus filings
            "424B5": 5,  # Lower priority - prospectus filings
        }

        # Announcement keywords for context validation
        self.announcement_keywords = [
            "declared",
            "approved",
            "announced",
            "board of directors",
            "executed",
            "effected",
            "implemented",
            "completed",
            "authorized",
            "resolved",
            "determined",
            "decided",
        ]

        # OPTIMIZATION: Pre-compiled regex patterns for better performance
        self._compile_regex_patterns()

    def _compile_regex_patterns(self):
        """Pre-compile regex patterns for better performance."""
        # High-priority patterns (most common) - check these first
        self.high_priority_patterns = [
            re.compile(
                r"(\d+)\s*for\s*one\s+stock\s+split", re.IGNORECASE
            ),  # Most common: "20-for-one"
            re.compile(r"(\d+)\s*for\s*one\s+split", re.IGNORECASE),
            re.compile(r"(\d+)[-–]for[-–]one\s+stock\s+split", re.IGNORECASE),
        ]

        # Medium-priority patterns
        self.medium_priority_patterns = [
            re.compile(r"(\d+)\s*for\s*(\d+)\s+stock\s+split", re.IGNORECASE),
            re.compile(r"(\d+)\s*for\s*(\d+)\s+split", re.IGNORECASE),
            re.compile(r"(\d+)[-–]for[-–](\d+)\s+stock\s+split", re.IGNORECASE),
        ]

        # Low-priority patterns (check last)
        self.low_priority_patterns = [
            re.compile(r"(\d+)\s*for\s*(\w+)\s+stock\s+split", re.IGNORECASE),
            re.compile(r"(\d+)\s*for\s*(\w+)\s+split", re.IGNORECASE),
            re.compile(r"(\w+)\s*for\s*(\w+)\s+stock\s+split", re.IGNORECASE),
        ]

        # Context patterns
        self.context_patterns = [
            re.compile(r"executed\s+a\s+", re.IGNORECASE),
            re.compile(r"effected\s+", re.IGNORECASE),
            re.compile(r"implemented\s+", re.IGNORECASE),
        ]

    @lru_cache(maxsize=1000)
    def _resolve_cik(self, symbol: str) -> Optional[str]:
        """
        Resolve CIK for a symbol with caching for performance.

        Args:
            symbol: Company symbol

        Returns:
            CIK string or None if not found
        """
        # Check cache first
        if symbol in self._cik_cache:
            return self._cik_cache[symbol]

        try:
            # Use a more efficient lookup method
            # This could be enhanced with a local database or API
            # For now, we'll use the existing method but cache results
            logger.info(f"[DEBUG] Resolving CIK for {symbol}")

            # TODO: Implement more efficient CIK lookup
            # This is a placeholder - you'll need to implement the actual lookup
            # based on your existing infrastructure

            # Cache the result (even if None to avoid repeated lookups)
            self._cik_cache[symbol] = None
            return None

        except Exception as e:
            logger.error(f"Error resolving CIK for {symbol}: {e}")
            self._cik_cache[symbol] = None
            return None

    def _extract_relevant_sections(self, filing_text: str) -> List[str]:
        """
        Extract only relevant sections of the filing text for stock split detection.
        This dramatically improves performance by reducing the text to search.

        Args:
            filing_text: Full filing text

        Returns:
            List of relevant text sections
        """
        relevant_sections = []

        # Look for sections that are likely to contain stock split information
        section_patterns = [
            r"stock\s+split.*?(?=\n\n|\n[A-Z]|$)",
            r"split.*?(?=\n\n|\n[A-Z]|$)",
            r"capital\s+stock.*?(?=\n\n|\n[A-Z]|$)",
            r"authorized\s+shares.*?(?=\n\n|\n[A-Z]|$)",
        ]

        for pattern in section_patterns:
            matches = re.finditer(pattern, filing_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                section = match.group(0)
                # Only include sections that are reasonably sized and contain relevant keywords
                if 50 <= len(section) <= 2000 and re.search(
                    r"(split|stock|share)", section, re.IGNORECASE
                ):
                    relevant_sections.append(section)

        # If no specific sections found, use the first 10KB of text (usually contains executive summary)
        if not relevant_sections:
            relevant_sections.append(filing_text[:10000])

        return relevant_sections

    def _fast_pattern_match(self, text: str) -> Optional[tuple]:
        """
        Fast pattern matching using pre-compiled regex and early termination.

        Args:
            text: Text to search

        Returns:
            Tuple of (numerator, denominator) or None if no match
        """
        # Check high-priority patterns first (most common cases)
        for pattern in self.high_priority_patterns:
            match = pattern.search(text)
            if match:
                num_str = match.group(1)
                try:
                    num = int(num_str)
                    if 0 < num < 100:
                        return (num, 1)
                except ValueError:
                    continue

        # Check medium-priority patterns
        for pattern in self.medium_priority_patterns:
            match = pattern.search(text)
            if match:
                num_str, denom_str = match.groups()
                try:
                    num = int(num_str)
                    denom = int(denom_str)
                    if 0 < num < 100 and 0 < denom < 100 and denom != 0:
                        return (num, denom)
                except ValueError:
                    continue

        # Check low-priority patterns last
        for pattern in self.low_priority_patterns:
            match = pattern.search(text)
            if match:
                num_str, denom_str = match.groups()

                # Handle numeric numerator
                try:
                    num = int(num_str)
                except ValueError:
                    num = self.number_words.get(num_str.lower())

                # Handle denominator
                try:
                    denom = int(denom_str)
                except ValueError:
                    denom = self.number_words.get(denom_str.lower())

                if num and denom and 0 < num < 100 and 0 < denom < 100 and denom != 0:
                    return (num, denom)

        return None

    def _is_valid_split_context_fast(self, text: str) -> bool:
        """
        Fast context validation using pre-compiled patterns.

        Args:
            text: Text to validate

        Returns:
            True if valid split context
        """
        # Quick checks for common valid contexts
        if any(pattern.search(text) for pattern in self.context_patterns):
            return True

        # Check for date patterns (indicates specific announcement)
        if re.search(r"\d{1,2}/\d{1,2}/\d{4}|\w+\s+\d{1,2},\s+\d{4}", text):
            return True

        return False

    def find_stock_splits(
        self, symbol_or_cik: str, max_filings: int = 50, max_workers: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Find stock split events for a company by parsing SEC filings.

        Args:
            symbol_or_cik: Company symbol or CIK
            max_filings: Maximum number of filings to analyze
            max_workers: Number of worker threads for parallel processing

        Returns:
            Stock split information dict or None if no splits found
        """
        logger.info(f"[DEBUG] find_stock_splits called for {symbol_or_cik}")

        # OPTIMIZATION: Check cache first
        cache_key = f"{symbol_or_cik}_{max_filings}"
        if cache_key in self._split_cache:
            logger.info(f"[DEBUG] Using cached result for {symbol_or_cik}")
            return self._split_cache[cache_key]

        # Step 1: Resolve CIK with caching
        cik = symbol_or_cik
        if not cik.isdigit() or len(cik) != 10:
            cik = self._resolve_cik(symbol_or_cik)
            if not cik:
                logger.info(f"[DEBUG] Could not find CIK for {symbol_or_cik}")
                return None

        # Step 2: Get recent filings from SEC submissions endpoint
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            resp = requests.get(submissions_url, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Error fetching submissions: {e}")
            return None

        # Step 3: Filter for relevant filings (last 5 years only)
        today = datetime.today()
        five_years_ago = today - timedelta(days=5 * 365)
        filings = data.get("filings", {}).get("recent", {})
        accession_numbers = filings.get("accessionNumber", [])
        forms = filings.get("form", [])
        filing_dates = filings.get("filingDate", [])

        # OPTIMIZATION: Prioritize filings by form type for faster results
        priority_filings = []
        other_filings = []

        for acc, form, date in zip(accession_numbers, forms, filing_dates):
            if form in self.form_priority and date >= five_years_ago.strftime(
                "%Y-%m-%d"
            ):
                if form in ["8-K", "10-Q", "10-K"]:  # High priority forms
                    priority_filings.append((acc, form, date))
                else:
                    other_filings.append((acc, form, date))

        # Process high priority filings first
        split_candidates = (
            priority_filings[: max_filings // 2] + other_filings[: max_filings // 2]
        )

        # Step 4: Process filings in parallel with early termination
        splits = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._process_filing, args, cik)
                for args in split_candidates
            ]

            # OPTIMIZATION: Early termination when we find a good split
            for future in as_completed(futures):
                result = future.result()
                if result:
                    splits.append(result)
                    # If we find a high-priority form split, we can stop early
                    if result["filing"] in ["8-K", "10-Q"]:
                        logger.info(
                            f"[DEBUG] Found high-priority split, stopping early"
                        )
                        break

        if not splits:
            logger.info(f"[DEBUG] No splits found for {symbol_or_cik}")
            # Cache negative results to avoid repeated processing
            self._split_cache[cache_key] = None
            return None

        # Step 5: Select best split (earliest announcement date, then best form)
        best_split = self._select_best_split(splits)

        # Cache the result
        self._split_cache[cache_key] = best_split

        logger.info(
            f"[DEBUG] Selected split: ratio={best_split['ratio']}, announcement_date={best_split['date']}, filing_date={best_split['filing_date']}, form={best_split['filing']}"
        )
        return best_split

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
        logger.info(f"[DEBUG] get_all_stock_splits called for {symbol_or_cik}")

        # Step 1: Resolve CIK
        cik = symbol_or_cik
        if not cik.isdigit() or len(cik) != 10:
            logger.info(f"[DEBUG] Resolving CIK for {symbol_or_cik}")
            # Note: This would need to be implemented or passed from the main SECHelper
            # For now, we'll assume CIK is already resolved
            pass

        # Step 2: Get recent filings from SEC submissions endpoint
        submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            resp = requests.get(submissions_url, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"Error fetching submissions: {e}")
            return []

        # Step 3: Filter for relevant filings (last 5 years only)
        today = datetime.today()
        five_years_ago = today - timedelta(days=5 * 365)
        filings = data.get("filings", {}).get("recent", {})
        accession_numbers = filings.get("accessionNumber", [])
        forms = filings.get("form", [])
        filing_dates = filings.get("filingDate", [])

        split_candidates = [
            (acc, form, date)
            for acc, form, date in zip(accession_numbers, forms, filing_dates)
            if form in self.form_priority
            and date >= five_years_ago.strftime("%Y-%m-%d")
        ][:max_filings]

        # Step 4: Process filings in parallel
        splits = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._process_filing, args, cik)
                for args in split_candidates
            ]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    splits.append(result)

        if not splits:
            logger.info(f"[DEBUG] No splits found for {symbol_or_cik}")
            return []

        # Step 5: Sort all splits by date and form priority
        sorted_splits = self._sort_all_splits(splits)

        logger.info(
            f"[DEBUG] Found {len(sorted_splits)} stock splits for {symbol_or_cik}"
        )
        return sorted_splits

    def _process_filing(self, args: tuple, cik: str) -> Optional[Dict[str, Any]]:
        """Process a single filing to extract stock split information."""
        acc, form, date = args
        acc_nodash = acc.replace("-", "")
        txt_url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{acc}.txt"
        )

        try:
            filing_text = requests.get(txt_url, headers=self.headers, timeout=10).text
        except Exception:
            return None

        # OPTIMIZATION: Quick check for split keywords before detailed processing
        if not re.search(r"(stock|share)\s+split", filing_text, re.IGNORECASE):
            return None

        # OPTIMIZATION: Extract only relevant sections instead of processing entire text
        relevant_sections = self._extract_relevant_sections(filing_text)

        ratio = None
        found_context = None

        # OPTIMIZATION: Use fast pattern matching on relevant sections
        for section in relevant_sections:
            ratio_match = self._fast_pattern_match(section)
            if ratio_match:
                num, denom = ratio_match
                ratio = num / denom

                # Quick context validation
                if self._is_valid_split_context_fast(section):
                    found_context = section
                    break

        if not ratio:
            return None

        # Extract date and return result
        split_date = self._extract_split_date(filing_text, date)

        return {
            "date": split_date,
            "ratio": ratio,
            "filing": form,
            "filing_date": date,
            "accession": acc,
            "type": "forward",
            "context": found_context,
        }

    def _extract_numeric_ratio(
        self, filing_text: str, context_window: int
    ) -> Optional[float]:
        """Extract stock split ratio using numeric patterns."""
        ratio_patterns = [
            r"(\d+)[-–]for[-–](\d+)\s+(?:stock|share)?\s*split",
            r"(\d+)\s*for\s*(\d+)\s+(?:stock|share)?\s*split",
            r"(\d+)[-–]for[-–](\d+)",
            r"(\d+)\s*for\s*(\d+)",
            r"(\d+)\s*for\s*(\d+)\s+stock\s+split",
            r"(\d+)\s*for\s*(\d+)\s+share\s+split",
            r"(\d+)\s*for\s*(\d+)\s+split",
            r"executed\s+a\s+(\d+)[-–]for[-–](\d+)\s+stock\s+split",
            r"executed\s+a\s+(\d+)\s*for\s*(\d+)\s+stock\s+split",
            r"(\d+)[-–]for[-–](\d+)\s+stock\s+split.*effected",
            r"(\d+)\s*for\s*(\d+)\s+stock\s+split.*effected",
            r"(\d+)[-–]for[-–](\d+)\s+split\s+of",
            r"(\d+)\s*for\s*(\d+)\s+split\s+of",
            r"(\d+)[-–]for[-–](\d+)\s+stock\s+split\s+effected",
            r"(\d+)\s*for\s*(\d+)\s+stock\s+split\s+effected",
            # Hybrid patterns for "X-for-one" format
            r"(\d+)[-–]for[-–](\w+)\s+(?:stock|share)?\s*split",
            r"(\d+)\s*for\s*(\w+)\s+(?:stock|share)?\s*split",
            r"(\d+)[-–]for[-–](\w+)",
            r"(\d+)\s*for\s*(\w+)",
            r"(\d+)\s*for\s*(\w+)\s+stock\s+split",
            r"(\d+)\s*for\s*(\w+)\s+share\s+split",
            r"(\d+)\s*for\s*(\w+)\s+split",
            r"executed\s+a\s+(\d+)[-–]for[-–](\w+)\s+stock\s+split",
            r"executed\s+a\s+(\d+)\s*for\s*(\w+)\s+stock\s+split",
            r"(\d+)[-–]for[-–](\w+)\s+stock\s+split.*effected",
            r"(\d+)\s*for\s*(\w+)\s+stock\s+split.*effected",
            r"(\d+)\s*for\s*one\s+stock\s+split",
            r"(\d+)\s*for\s*one\s+share\s+split",
            r"(\d+)\s*for\s*one\s+split",
            r"(\d+)[-–]for[-–]one\s+stock\s+split",
            r"(\d+)[-–]for[-–]one\s+share\s+split",
            r"(\d+)[-–]for[-–]one\s+split",
        ]

        for pattern in ratio_patterns:
            for match in re.finditer(pattern, filing_text, re.IGNORECASE):
                groups = match.groups()
                num_str, denom_str = groups

                # Handle first group
                try:
                    num = int(num_str)
                except ValueError:
                    if num_str.lower() == "one":
                        num = 1
                    else:
                        continue

                # Handle second group
                try:
                    denom = int(denom_str)
                except ValueError:
                    denom = self.number_words.get(denom_str.lower())
                    if not denom:
                        continue

                # Validate ratio
                if denom != 0 and 0 < num < 100 and 0 < denom < 100:
                    start, end = match.span()
                    context = filing_text[
                        max(0, start - context_window) : min(
                            len(filing_text), end + context_window
                        )
                    ]

                    if re.search(r"split", context, re.IGNORECASE):
                        if self._is_valid_split_context(context):
                            return num / denom

        return None

    def _extract_word_ratio(
        self, filing_text: str, context_window: int
    ) -> Optional[float]:
        """Extract stock split ratio using word-based patterns."""
        word_patterns = [
            r"(\w+)[-–]for[-–](\w+)\s+(?:stock|share)?\s*split",
            r"(\w+)\s*for\s*(\w+)\s+(?:stock|share)?\s*split",
            r"(\w+)[-–]for[-–](\w+)",
            r"(\w+)\s*for\s*(\w+)",
            r"(\w+)\s*for\s*(\w+)\s+stock\s+split",
            r"(\w+)\s*for\s*(\w+)\s+share\s+split",
            r"(\w+)\s*for\s*(\w+)\s+split",
            r"executed\s+a\s+(\w+)[-–]for[-–](\w+)\s+stock\s+split",
            r"executed\s+a\s+(\w+)\s*for\s*(\w+)\s+stock\s+split",
            r"(\w+)[-–]for[-–](\w+)\s+stock\s+split.*effected",
            r"(\w+)\s*for\s*(\w+)\s+stock\s+split.*effected",
            r"twenty\s*for\s*one\s+stock\s+split",
            r"twenty\s*for\s*one\s+share\s+split",
            r"twenty\s*for\s*one\s+split",
            r"twenty[-–]for[-–]one\s+stock\s+split",
            r"twenty[-–]for[-–]one\s+share\s+split",
            r"twenty[-–]for[-–]one\s+split",
            r"(\w+)\s*for\s*one\s+stock\s+split",
            r"(\w+)\s*for\s*one\s+share\s+split",
            r"(\w+)\s*for\s*one\s+split",
            r"(\w+)[-–]for[-–]one\s+stock\s+split",
            r"(\w+)[-–]for[-–]one\s+share\s+split",
            r"(\w+)[-–]for[-–]one\s+split",
        ]

        for pattern in word_patterns:
            for match in re.finditer(pattern, filing_text, re.IGNORECASE):
                groups = match.groups()
                num_word, denom_word = groups

                num = self.number_words.get(num_word.lower())
                denom = self.number_words.get(denom_word.lower())

                if num and denom and denom != 0 and 0 < num < 100 and 0 < denom < 100:
                    start, end = match.span()
                    context = filing_text[
                        max(0, start - context_window) : min(
                            len(filing_text), end + context_window
                        )
                    ]

                    if re.search(r"split", context, re.IGNORECASE):
                        if self._is_valid_split_context(context):
                            return num / denom

        return None

    def _extract_split_date(self, filing_text: str, filing_date: str) -> str:
        """Extract the split announcement date from filing text."""
        # Try Month DD, YYYY format
        date_match = re.search(
            r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
            filing_text,
        )
        if date_match:
            try:
                return datetime.strptime(date_match.group(0), "%B %d, %Y").strftime(
                    "%Y-%m-%d"
                )
            except Exception:
                pass

        # Try MM/DD/YYYY format
        alt_date_match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4})", filing_text)
        if alt_date_match:
            try:
                month, day, year = alt_date_match.groups()
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            except Exception:
                pass

        # Fall back to filing date
        return filing_date

    def _is_valid_split_context(self, context: str) -> bool:
        """Validate if the context indicates a valid stock split announcement."""
        context_lower = context.lower()

        # Check for execution/completion language
        execution_keywords = [
            "executed",
            "effected",
            "implemented",
            "completed",
            "carried out",
        ]
        if any(word in context_lower for word in execution_keywords):
            return True

        # Check for dividend-related language
        dividend_keywords = ["dividend", "special dividend", "stock dividend"]
        if any(word in context_lower for word in dividend_keywords):
            return True

        # Check for date-specific language
        date_patterns = [
            r"on\s+\w+\s+\d{1,2},\s+\d{4}",
            r"effective\s+\w+\s+\d{1,2},\s+\d{4}",
            r"record\s+date\s+of\s+\w+\s+\d{1,2},\s+\d{4}",
        ]
        for pattern in date_patterns:
            if re.search(pattern, context_lower):
                return True

        return False

    def _select_best_split(self, splits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best split from multiple candidates based on date and form priority."""
        # Group splits by ratio and select best for each
        best_split_for_ratio = {}
        for split in splits:
            ratio = split["ratio"]
            key = ratio
            pri = self.form_priority.get(split["filing"], 4)

            if key not in best_split_for_ratio:
                best_split_for_ratio[key] = (
                    split["date"],
                    pri,
                    split["filing_date"],
                    split,
                )
            else:
                old_date, old_pri, old_filing_date, _ = best_split_for_ratio[key]
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

        # Get the best split (earliest announcement date, then best form)
        filtered_splits = [v[3] for v in best_split_for_ratio.values()]
        filtered_splits.sort(
            key=lambda s: (
                s["date"],  # Earliest announcement date first
                self.form_priority.get(s["filing"], 4),  # Lower priority number first
                s["filing_date"],  # Earliest filing date first
            ),
            reverse=False,
        )

        return filtered_splits[0]

    def _sort_all_splits(self, splits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort all splits by date and form priority."""
        # Group splits by ratio and select best for each
        best_split_for_ratio = {}
        for split in splits:
            ratio = split["ratio"]
            key = ratio
            pri = self.form_priority.get(split["filing"], 4)

            if key not in best_split_for_ratio:
                best_split_for_ratio[key] = (
                    split["date"],
                    pri,
                    split["filing_date"],
                    split,
                )
            else:
                old_date, old_pri, old_filing_date, _ = best_split_for_ratio[key]
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

        # Sort all splits (earliest announcement date, then best form)
        filtered_splits = [v[3] for v in best_split_for_ratio.values()]
        filtered_splits.sort(
            key=lambda s: (
                s["date"],  # Earliest announcement date first
                self.form_priority.get(s["filing"], 4),  # Lower priority number first
                s["filing_date"],  # Earliest filing date first
            ),
            reverse=False,
        )

        return filtered_splits

    def adjust_financials_for_split(
        self, data: List[Dict[str, Any]], split: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Adjust financial data for a stock split.

        Args:
            data: List of financial data dicts
            split: Stock split information dict

        Returns:
            Adjusted financial data
        """
        if not data or not split:
            return data

        ratio = split["ratio"]
        split_filing_date = split["filing_date"]

        logger.info(
            f"[DEBUG] Adjusting financial data for split: ratio={ratio}, filing_date={split_filing_date}"
        )

        try:
            split_filing_date_obj = datetime.strptime(split_filing_date, "%Y-%m-%d")
        except Exception as e:
            logger.error(
                f"[DEBUG] Error parsing split filing date '{split_filing_date}': {e}"
            )
            return data

        adjusted_count = 0
        for i, item in enumerate(data):
            item_date = (
                item.get("date") or item.get("fiscalDateEnding") or item.get("endDate")
            )

            if item_date:
                try:
                    item_date_obj = datetime.strptime(item_date, "%Y-%m-%d")
                    should_adjust = item_date_obj < split_filing_date_obj

                    if should_adjust:
                        logger.info(
                            f"[DEBUG] Adjusting item {i}: date={item_date} (split filing date={split_filing_date})"
                        )

                        # Adjust per-share metrics
                        for field in [
                            "eps",
                            "epsDiluted",
                            "earningsPerShareBasic",
                            "earningsPerShareDiluted",
                        ]:
                            if field in item and item[field] is not None:
                                old_value = item[field]
                                item[field] /= ratio
                                logger.info(
                                    f"[DEBUG] Adjusted {field}: {old_value:.2f} -> {item[field]:.2f}"
                                )

                        # Adjust share count metrics
                        for field in [
                            "weightedAverageShsOut",
                            "weightedAverageShsOutDil",
                            "weightedAverageNumberOfSharesOutstandingBasic",
                            "weightedAverageNumberOfDilutedSharesOutstanding",
                            "commonStockSharesOutstanding",
                        ]:
                            if field in item and item[field] is not None:
                                old_value = item[field]
                                item[field] *= ratio
                                logger.info(
                                    f"[DEBUG] Adjusted {field}: {old_value:,.0f} -> {item[field]:,.0f}"
                                )

                        # Add metadata
                        item["_split_adjusted"] = True
                        item["_split_ratio"] = ratio
                        item["_split_date"] = split["date"]
                        item["_split_filing_date"] = split_filing_date
                        adjusted_count += 1

                except Exception as e:
                    logger.error(f"[DEBUG] Error processing item {i}: {e}")

        logger.info(
            f"[DEBUG] Stock split adjustment complete: {adjusted_count}/{len(data)} items adjusted"
        )
        return data

    def adjust_financials_for_latest_split(
        self, data: List[Dict[str, Any]], split: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Adjusts financial data for a stock split. Only data prior to the split filing date is adjusted.

        Args:
            data: List of financial data dicts (e.g., from get_income_statement)
            split: Split info dict as returned by find_stock_splits

        Returns:
            Adjusted financial data
        """
        logger.info(
            f"[DEBUG] adjust_financials_for_latest_split called with data length: {len(data) if data else 0}"
        )
        logger.info(f"[DEBUG] split provided: {split is not None}")

        if not data:
            logger.info("[DEBUG] No data provided, returning empty data")
            return data

        if not split:
            logger.info("[DEBUG] No split provided, returning unadjusted data")
            return data

        ratio = split["ratio"]
        split_date = split["date"]
        split_filing_date = split["filing_date"]  # Use filing date for adjustment logic
        logger.info(
            f"[DEBUG] Using split: ratio={ratio}, announcement_date={split_date}, filing_date={split_filing_date}"
        )

        # Ensure split_filing_date is in proper format for comparison
        try:
            from datetime import datetime

            split_filing_date_obj = datetime.strptime(split_filing_date, "%Y-%m-%d")
        except Exception as e:
            logger.error(
                f"[DEBUG] Error parsing split filing date '{split_filing_date}': {e}"
            )
            return data

        adjusted_count = 0
        for i, item in enumerate(data):
            item_date = (
                item.get("date") or item.get("fiscalDateEnding") or item.get("endDate")
            )

            if item_date:
                # Parse item date for proper comparison
                try:
                    item_date_obj = datetime.strptime(item_date, "%Y-%m-%d")

                    # Use datetime objects for comparison
                    # IMPORTANT: We compare against the FILING DATE, not the announcement date
                    # because the filing date represents when the split information became publicly available
                    # Financial data should be adjusted if it's from BEFORE the filing date
                    should_adjust = item_date_obj < split_filing_date_obj

                    if should_adjust:
                        logger.info(
                            f"[DEBUG] Adjusting item {i}: date={item_date} (split filing date={split_filing_date})"
                        )

                        # Adjust per-share metrics
                        for field in [
                            "eps",
                            "epsDiluted",
                            "earningsPerShareBasic",
                            "earningsPerShareDiluted",
                        ]:
                            if field in item and item[field] is not None:
                                old_value = item[field]
                                item[field] /= ratio
                                logger.info(
                                    f"[DEBUG] Adjusted {field}: {old_value:.2f} -> {item[field]:.2f}"
                                )
                            else:
                                logger.info(
                                    f"[DEBUG] Field {field} not present in item {i}"
                                )

                        # Adjust share count metrics
                        for field in [
                            "weightedAverageShsOut",
                            "weightedAverageShsOutDil",
                            "weightedAverageNumberOfSharesOutstandingBasic",
                            "weightedAverageNumberOfDilutedSharesOutstanding",
                            "commonStockSharesOutstanding",
                        ]:
                            if field in item and item[field] is not None:
                                old_value = item[field]
                                item[field] *= ratio
                                logger.info(
                                    f"[DEBUG] Adjusted {field}: {old_value:,.0f} -> {item[field]:,.0f}"
                                )
                            else:
                                logger.info(
                                    f"[DEBUG] Field {field} not present in item {i}"
                                )

                        # Add metadata
                        item["_split_adjusted"] = True
                        item["_split_ratio"] = ratio
                        item["_split_date"] = split_date
                        item["_split_filing_date"] = split_filing_date
                        adjusted_count += 1
                    else:
                        logger.info(
                            f"[DEBUG] Item {i} not adjusted: date={item_date} >= split filing date={split_filing_date}"
                        )

                except Exception as e:
                    logger.error(f"[DEBUG] Error processing item {i}: {e}")
            else:
                logger.info(f"[DEBUG] Item {i} has no valid date field")

        logger.info(
            f"[DEBUG] Stock split adjustment complete: {adjusted_count}/{len(data)} items adjusted"
        )
        return data

    def clear_cache(self):
        """Clear all cached data to free memory."""
        self._cik_cache.clear()
        self._split_cache.clear()
        logger.info("[DEBUG] Stock split cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics for monitoring."""
        return {
            "cik_cache_size": len(self._cik_cache),
            "split_cache_size": len(self._split_cache),
        }

    def set_cache_size(self, cik_cache_size: int = 1000, split_cache_size: int = 500):
        """Set cache sizes for memory management."""
        # For now, we'll just log the request
        # In a more sophisticated implementation, you could implement LRU eviction
        logger.info(
            f"[DEBUG] Cache size set to: CIK={cik_cache_size}, Split={split_cache_size}"
        )

    def warm_cache(self, symbols: List[str]):
        """
        Pre-populate cache with common symbols for faster subsequent lookups.

        Args:
            symbols: List of symbols to pre-load into cache
        """
        logger.info(f"[DEBUG] Warming cache with {len(symbols)} symbols")

        # Process symbols in parallel
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(self.find_stock_splits, symbol) for symbol in symbols
            ]

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.warning(f"Error warming cache: {e}")

        logger.info(f"[DEBUG] Cache warming complete. Stats: {self.get_cache_stats()}")
