#!/usr/bin/env python3
"""
Shared Patterns Module

This module contains common regex patterns and constants used across
the SEC financial data modules to eliminate code duplication.
"""

import re
from typing import List, Dict

# Number words dictionary for parsing word-based ratios
NUMBER_WORDS = {
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
# Only include filing types that actually announce or report EXECUTED stock splits
FORM_PRIORITY = {
    "8-K": 1,  # Highest priority - immediate announcements of executed splits
    "10-Q": 2,  # High priority - quarterly reports that may mention recent splits
    "10-K": 3,  # Medium priority - annual reports that may mention recent splits
    # Removed DEF 14A, DEF 14C, 424B3, 424B4, 424B5 as they are unreliable for stock splits
    # DEF 14A/C are proxy statements that may mention historical or proposed splits, not executed ones
    # 424B3-5 are prospectus filings that are not reliable for split announcements
}


# Stock split regex patterns
def get_numeric_ratio_patterns() -> List[str]:
    """Get numeric ratio patterns for stock split detection."""
    return [
        # Decimal ratio patterns (e.g., "1.37 for 1") - MUST come before integer patterns
        r"(\d+\.\d+)\s*for\s*(\d+)\s+(?:stock|share)?\s*split",
        r"(\d+\.\d+)[-–]for[-–](\d+)\s+(?:stock|share)?\s*split",
        r"(\d+\.\d+)\s*for\s*(\d+)",
        r"(\d+\.\d+)[-–]for[-–](\d+)",
        # Integer ratio patterns (existing) - MUST come after decimal patterns and be more restrictive
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


def get_word_ratio_patterns() -> List[str]:
    """Get word-based ratio patterns for stock split detection."""
    return [
        r"(\w+)[-–]for[-–](\w+)\s+(?:stock|share)?\s*split",
        r"(\w+)\s*for\s*(\w+)\s+(?:stock|share)?\s*split",
        r"(\w+)[-–]for[-–](\w+)",
        r"(\w+)\s*for\s*(\w+)",
        # Additional patterns for better coverage
        r"(\w+)\s*for\s*(\w+)\s+stock\s+split",
        r"(\w+)\s*for\s*(\w+)\s+share\s+split",
        r"(\w+)\s*for\s*(\w+)\s+split",
        # Pattern for "executed a X-for-Y stock split" format
        r"executed\s+a\s+(\w+)[-–]for[-–](\w+)\s+stock\s+split",
        r"executed\s+a\s+(\w+)\s*for\s*(\w+)\s+stock\s+split",
        # Pattern for "effected in the form of" format
        r"(\w+)[-–]for[-–](\w+)\s+stock\s+split.*effected",
        r"(\w+)\s*for\s*(\w+)\s+stock\s+split.*effected",
        # Additional word-based patterns for common formats
        r"twenty\s*for\s*one\s+stock\s+split",
        r"twenty\s*for\s*one\s+share\s+split",
        r"twenty\s*for\s*one\s+split",
        r"twenty[-–]for[-–]one\s+stock\s+split",
        r"twenty[-–]for[-–]one\s+share\s+split",
        r"twenty[-–]for[-–]one\s+split",
        # Patterns for other common ratios
        r"(\w+)\s*for\s*one\s+stock\s+split",
        r"(\w+)\s*for\s*one\s+share\s+split",
        r"(\w+)\s*for\s*one\s+split",
        r"(\w+)[-–]for[-–]one\s+stock\s+split",
        r"(\w+)[-–]for[-–]one\s+share\s+split",
        r"(\w+)[-–]for[-–]one\s+split",
    ]


def get_high_priority_patterns() -> List[re.Pattern]:
    """Get pre-compiled high-priority regex patterns for better performance."""
    return [
        # Decimal ratio patterns (e.g., "1.37 for 1") - MUST come before integer patterns
        re.compile(r"(\d+\.\d+)\s*for\s*one\s+stock\s+split", re.IGNORECASE),
        re.compile(r"(\d+\.\d+)\s*for\s*one\s+split", re.IGNORECASE),
        re.compile(r"(\d+\.\d+)[-–]for[-–]one\s+stock\s+split", re.IGNORECASE),
        # Integer ratio patterns (existing) - MUST come after decimal patterns
        re.compile(
            r"\b(\d+)\s*for\s*one\s+stock\s+split\b", re.IGNORECASE
        ),  # Most common: "20-for-one"
        re.compile(r"\b(\d+)\s*for\s*one\s+split\b", re.IGNORECASE),
        re.compile(r"\b(\d+)[-–]for[-–]one\s+stock\s+split\b", re.IGNORECASE),
    ]


def get_medium_priority_patterns() -> List[re.Pattern]:
    """Get pre-compiled medium-priority regex patterns."""
    return [
        # Decimal ratio patterns (e.g., "1.37 for 1") - MUST come before integer patterns
        re.compile(r"(\d+\.\d+)\s*for\s*(\d+)\s+stock\s+split", re.IGNORECASE),
        re.compile(r"(\d+\.\d+)\s*for\s*(\d+)\s+split", re.IGNORECASE),
        re.compile(r"(\d+\.\d+)[-–]for[-–](\d+)\s+stock\s+split", re.IGNORECASE),
        # Integer ratio patterns (existing) - MUST come after decimal patterns and be more restrictive
        re.compile(r"\b(\d+)\s*for\s*(\d+)\s+stock\s+split\b", re.IGNORECASE),
        re.compile(r"\b(\d+)\s*for\s*(\d+)\s+split\b", re.IGNORECASE),
        re.compile(r"\b(\d+)[-–]for[-–](\d+)\s+stock\s+split\b", re.IGNORECASE),
    ]


def get_low_priority_patterns() -> List[re.Pattern]:
    """Get pre-compiled low-priority regex patterns."""
    return [
        # Decimal ratio patterns (e.g., "1.37 for 1") - MUST come before integer patterns
        re.compile(r"(\d+\.\d+)\s*for\s*(\w+)\s+stock\s+split", re.IGNORECASE),
        re.compile(r"(\d+\.\d+)\s*for\s*(\w+)\s+split", re.IGNORECASE),
        # Integer ratio patterns (existing) - MUST come after decimal patterns and be more restrictive
        re.compile(r"\b(\d+)\s*for\s*(\w+)\s+stock\s+split\b", re.IGNORECASE),
        re.compile(r"\b(\d+)\s*for\s*(\w+)\s+split\b", re.IGNORECASE),
        re.compile(r"\b(\w+)\s*for\s*(\w+)\s+stock\s+split\b", re.IGNORECASE),
    ]


def get_context_patterns() -> List[re.Pattern]:
    """Get pre-compiled context patterns for validation."""
    return [
        re.compile(r"executed\s+a\s+", re.IGNORECASE),
        re.compile(r"effected\s+", re.IGNORECASE),
        re.compile(r"implemented\s+", re.IGNORECASE),
    ]


def get_section_patterns() -> List[str]:
    """Get patterns for extracting relevant filing sections."""
    return [
        r"stock\s+split.*?(?=\n\n|\n[A-Z]|$)",
        r"split.*?(?=\n\n|\n[A-Z]|$)",
        r"capital\s+stock.*?(?=\n\n|\n[A-Z]|$)",
        r"authorized\s+shares.*?(?=\n\n|\n[A-Z]|$)",
    ]


def get_date_patterns() -> Dict[str, str]:
    """Get date extraction patterns."""
    return {
        "month_day_year": r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
        "mm_dd_yyyy": r"(\d{1,2})/(\d{1,2})/(\d{4})",
    }


def get_execution_keywords() -> List[str]:
    """Get keywords that indicate a stock split has been executed."""
    return [
        "executed",
        "effected",
        "implemented",
        "completed",
        "carried out",
        "performed",
        "realized",
        "actualized",
        "put into effect",
        "made effective",
        "took effect",
        "became effective",
        "was effective",
        "effective date",
        "record date",
        "ex-dividend date",
        "distribution date",
    ]


def get_announcement_keywords() -> List[str]:
    """Get keywords that indicate a stock split announcement."""
    return [
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
        "proposed",
        "recommended",
        "suggested",
        "considered",
        "discussed",
        "planned",
        "intended",
        "expected",
    ]


def get_historical_indicators() -> List[str]:
    """Get keywords that indicate historical or proposed information (NOT executed)."""
    return [
        "previously",
        "historically",
        "in the past",
        "was filed",
        "had been",
        "used to be",
        "former",
        "prior",
        "earlier",
        "before",
        "proposed",
        "recommended",
        "suggested",
        "considered",
        "discussed",
        "planned",
        "intended",
        "expected",
        "may",
        "might",
        "could",
        "would",
        "should",
    ]
