"""Per-share-class EPS facts read from a filing's XBRL instance.

Filers with several share classes (V, BRK-B, HSY, STZ, TSN, KKR) often tag
EPS only with a StatementClassOfStockAxis member, and partnerships (MPLX)
tag per-unit income by partner type. SEC companyfacts drops dimensioned
facts, so their EPS is recoverable only from the filing itself.
"""

import json
import re
import xml.etree.ElementTree as ET
from collections import OrderedDict
from io import BytesIO

_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
_LINKBASE_SUFFIXES = ("_cal.xml", "_def.xml", "_lab.xml", "_pre.xml")
_CLASS_AXIS_SUFFIXES = (
    ":StatementClassOfStockAxis",
    ":PartnerTypeOfPartnersCapitalAccountAxis",
)
_XBRLI = "{http://www.xbrl.org/2003/instance}"
_XBRLDI = "{http://xbrl.org/2006/xbrldi}"
_US_GAAP_NAMESPACE_MARKER = "fasb.org/us-gaap/"
_TICKER_CLASS_SUFFIX = re.compile(r"[.\-/]([A-Z])$")

# Filings are immutable, so successful parses are cached by accession.
_CACHE_SIZE = 64
_cache = OrderedDict()


def _instance_url(cik, accession, fetch):
    """URL of the filing's XBRL instance, or None when the index has no single one."""
    base = _ARCHIVES_URL.format(cik=int(cik), accession=accession.replace("-", ""))
    index = json.loads(fetch(base + "index.json"))
    names = [
        item.get("name", "") for item in index.get("directory", {}).get("item", [])
    ]
    inline = [name for name in names if name.endswith("_htm.xml")]
    if inline:
        return base + inline[0]
    plain = [
        name
        for name in names
        if name.endswith(".xml")
        and name != "FilingSummary.xml"
        and not name.endswith(_LINKBASE_SUFFIXES)
    ]
    return base + plain[0] if len(plain) == 1 else None


def _class_context(context):
    """(start, end, member) of a context dimensioned only by share class or partner type, else None."""
    if context.find(".//" + _XBRLDI + "typedMember") is not None:
        return None
    members = list(context.iter(_XBRLDI + "explicitMember"))
    if len(members) != 1 or not (members[0].get("dimension") or "").endswith(
        _CLASS_AXIS_SUFFIXES
    ):
        return None
    period = context.find(_XBRLI + "period")
    if period is None:
        return None
    start = period.findtext(_XBRLI + "startDate")
    end = period.findtext(_XBRLI + "endDate")
    member = (members[0].text or "").strip().rsplit(":", 1)[-1]
    if not (start and end and member):
        return None
    return start.strip(), end.strip(), member


def parse_class_eps_facts(instance_bytes, concepts):
    """Share-class facts of the us-gaap `concepts` as dicts with concept/start/end/member/value."""
    contexts = {}
    raw_facts = []
    root = None
    depth = 0
    for event, element in ET.iterparse(
        BytesIO(instance_bytes), events=("start", "end")
    ):
        if event == "start":
            if root is None:
                root = element
            depth += 1
            continue
        depth -= 1
        if depth != 1:
            continue
        if element.tag == _XBRLI + "context":
            context = _class_context(element)
            if context is not None:
                contexts[element.get("id")] = context
        else:
            namespace, _, concept = element.tag.rpartition("}")
            if concept in concepts and _US_GAAP_NAMESPACE_MARKER in namespace:
                raw_facts.append((concept, element.get("contextRef"), element.text))
        # Top-level elements are fully handled once closed.
        root.clear()

    facts = []
    for concept, context_ref, text in raw_facts:
        context = contexts.get(context_ref)
        if context is None or not text:
            continue
        try:
            value = float(text.strip())
        except ValueError:
            continue
        start, end, member = context
        facts.append(
            {
                "concept": concept,
                "start": start,
                "end": end,
                "member": member,
                "value": value,
            }
        )
    return facts


def fetch_class_eps_facts(cik, accession, fetch, concepts):
    """Share-class facts of `concepts` in one filing; `fetch(url)` returns the response bytes."""
    concepts = frozenset(concepts)
    key = (str(cik), accession, concepts)
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    url = _instance_url(cik, accession, fetch)
    facts = parse_class_eps_facts(fetch(url), concepts) if url else []
    _cache[key] = facts
    if len(_cache) > _CACHE_SIZE:
        _cache.popitem(last=False)
    return facts


def preferred_class_member(members, symbol):
    """Member of the symbol's class: ticker suffix (BRK-B), else common stock, Class A, LP common units, or the only one."""
    members = sorted(set(members))
    suffix = _TICKER_CLASS_SUFFIX.search((symbol or "").upper())
    if suffix:
        share_class = "Class" + suffix.group(1)
        for name in members:
            if name.endswith(share_class + "Member"):
                return name
        for name in members:
            if share_class in name:
                return name
        # Another class's EPS would be off by the conversion ratio (BRK A/B).
        return None
    if "CommonStockMember" in members:
        return "CommonStockMember"
    common = [name for name in members if name.endswith("CommonStockMember")]
    if len(common) == 1:
        # Filer-specific common stock member (HersheyCommonStockMember).
        return common[0]
    for name in members:
        if name.endswith("ClassAMember"):
            return name
    # Partnerships: limited partners' common units, not the general partner.
    limited = [
        name for name in members if "LimitedPartner" in name or "CommonUnit" in name
    ]
    if len(limited) == 1:
        return limited[0]
    if len(members) == 1 and "GeneralPartner" not in members[0]:
        return members[0]
    return None
