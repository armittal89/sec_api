"""EPS for filers that tag it per share class or only for continuing operations."""

import json

import pytest

from sec_financial_data import share_class_eps
from sec_financial_data import sec_financial_data as sfd
from sec_financial_data.sec_financial_data import (
    _EPS_TAGS,
    _get_eps_value,
    _get_financial_statement_data,
)
from sec_financial_data.share_class_eps import (
    parse_class_eps_facts,
    preferred_class_member,
)

CIK = "0000000001"
ACCN = "0000000001-26-000001"
INDEX_URL = "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/index.json"
INSTANCE_URL = (
    "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/x-20251231_htm.xml"
)


def _context(context_id, start, end, member=None, extra_member=None):
    members = "".join(
        f'<xbrldi:explicitMember dimension="{axis}">{name}</xbrldi:explicitMember>'
        for axis, name in (
            ("us-gaap:StatementClassOfStockAxis", member),
            ("srt:ProductOrServiceAxis", extra_member),
        )
        if name
    )
    segment = f"<entity><segment>{members}</segment></entity>" if members else ""
    return (
        f'<context id="{context_id}">{segment}<period>'
        f"<startDate>{start}</startDate><endDate>{end}</endDate></period></context>"
    )


def _instance(contexts, facts):
    body = "".join(contexts) + "".join(
        f'<us-gaap:{concept} contextRef="{ref}" unitRef="u">{value}</us-gaap:{concept}>'
        for concept, ref, value in facts
    )
    return (
        '<xbrl xmlns="http://www.xbrl.org/2003/instance" '
        'xmlns:xbrldi="http://xbrl.org/2006/xbrldi" '
        'xmlns:us-gaap="http://fasb.org/us-gaap/2025">'
        f"{body}</xbrl>"
    ).encode()


BRK_INSTANCE = _instance(
    [
        _context("a", "2025-01-01", "2025-12-31", "us-gaap:EquivalentClassAMember"),
        _context("b", "2025-01-01", "2025-12-31", "us-gaap:EquivalentClassBMember"),
        _context("q", "2025-10-01", "2025-12-31", "us-gaap:EquivalentClassBMember"),
        _context(
            "seg",
            "2025-01-01",
            "2025-12-31",
            "us-gaap:EquivalentClassBMember",
            "x:OtherMember",
        ),
    ],
    [
        ("EarningsPerShareBasic", "a", "46563"),
        ("EarningsPerShareBasic", "b", "31.04"),
        ("EarningsPerShareBasic", "q", "7.10"),
        ("EarningsPerShareBasic", "seg", "999"),
    ],
)


@pytest.fixture(autouse=True)
def _clear_cache():
    share_class_eps._cache.clear()
    yield
    share_class_eps._cache.clear()


@pytest.fixture
def fake_sec(monkeypatch):
    """Serves `fake_sec.instance` as the filing's XBRL instance and records URLs."""

    class FakeSec:
        instance = BRK_INSTANCE
        urls = []

        def fetch(self, url, headers):
            self.urls.append(url)
            if url == INDEX_URL:
                names = [
                    "x-20251231_htm.xml",
                    "x-20251231_cal.xml",
                    "FilingSummary.xml",
                ]
                return json.dumps(
                    {"directory": {"item": [{"name": n} for n in names]}}
                ).encode()
            assert url == INSTANCE_URL
            return self.instance

    fake = FakeSec()
    monkeypatch.setattr(sfd, "_fetch_sec_bytes", fake.fetch)
    return fake


def _income_statement(
    symbol, eps_facts, report_type="10-K", end="2025-12-31", start="2025-01-01"
):
    def fact(val):
        return {
            "val": val,
            "start": start,
            "end": end,
            "form": report_type,
            "filed": "2026-02-27",
            "fy": 2025,
            "fp": "FY" if report_type == "10-K" else "Q4",
            "accn": ACCN,
        }

    tags = {
        "NetIncomeLoss": {"USD": [fact(66_968_000_000)]},
        "Revenues": {"USD": [fact(371_000_000_000)]},
    }
    for tag, val in eps_facts.items():
        tags[tag] = {"USD/shares": [fact(val)]}
    company_facts = {"facts": {"us-gaap": {t: {"units": u} for t, u in tags.items()}}}
    return _get_financial_statement_data(
        symbol,
        "income_statement",
        10,
        report_type,
        {},
        lambda _: CIK,
        lambda _: company_facts,
    )


def test_preferred_member_follows_ticker_class_suffix():
    members = ["EquivalentClassAMember", "EquivalentClassBMember"]
    assert preferred_class_member(members, "BRK-B") == "EquivalentClassBMember"
    assert preferred_class_member(members, "BRK.A") == "EquivalentClassAMember"


def test_preferred_member_never_substitutes_another_class_for_suffix():
    assert preferred_class_member(["CommonClassAMember"], "BRK-B") is None


def test_preferred_member_without_suffix():
    # V: Class A over B-1/B-2/C; HSY: common stock (custom member) over Class B.
    visa = ["CommonClassAMember", "CommonClassB1Member", "CommonClassCMember"]
    assert preferred_class_member(visa, "V") == "CommonClassAMember"
    assert (
        preferred_class_member(["CommonStockMember", "CommonClassBMember"], "HSY")
        == "CommonStockMember"
    )
    assert (
        preferred_class_member(
            ["HersheyCommonStockMember", "CommonClassBMember"], "HSY"
        )
        == "HersheyCommonStockMember"
    )
    assert preferred_class_member(["SeriesXMember"], "X") == "SeriesXMember"
    assert preferred_class_member(["SeriesXMember", "SeriesYMember"], "X") is None


def test_parse_keeps_only_single_class_dimension_facts():
    facts = parse_class_eps_facts(BRK_INSTANCE, _EPS_TAGS)
    assert {(f["member"], f["start"], f["value"]) for f in facts} == {
        ("EquivalentClassAMember", "2025-01-01", 46563.0),
        ("EquivalentClassBMember", "2025-01-01", 31.04),
        ("EquivalentClassBMember", "2025-10-01", 7.10),
    }


def test_continuing_operations_eps_used_without_total_eps():
    # ABNB/MNST tag only continuing-operations EPS.
    data = {
        "IncomeLossFromContinuingOperationsPerBasicShare": 7.52,
        "IncomeLossFromDiscontinuedOperationsNetOfTaxPerBasicShare": -0.5,
        "IncomeLossFromContinuingOperationsPerDilutedShare": 7.24,
    }
    assert _get_eps_value(data, "eps") == pytest.approx(7.02)
    assert _get_eps_value(data, "epsDiluted") == 7.24
    assert _get_eps_value({"EarningsPerShareBasic": 1.5, **data}, "eps") == 1.5
    assert _get_eps_value({}, "eps") == 0


def test_class_eps_read_from_filing_when_companyfacts_has_none(fake_sec):
    [row] = _income_statement("BRK-B", {})
    assert row["eps"] == 31.04
    # No diluted EPS tagged at all: no dilutive securities, diluted = basic.
    assert row["epsDiluted"] == 31.04
    assert fake_sec.urls == [INDEX_URL, INSTANCE_URL]


def test_class_eps_ten_q_uses_discrete_quarter(fake_sec):
    [row] = _income_statement("BRK-B", {}, report_type="10-Q", start="2025-10-01")
    assert row["eps"] == 7.10


def test_companyfacts_eps_does_not_fetch_filing(fake_sec):
    [row] = _income_statement(
        "BRK-B", {"EarningsPerShareBasic": 2.0, "EarningsPerShareDiluted": 1.9}
    )
    assert (row["eps"], row["epsDiluted"]) == (2.0, 1.9)
    assert fake_sec.urls == []


def test_only_missing_diluted_taken_from_filing(fake_sec):
    # TSN: undimensioned diluted EPS, basic only per class.
    fake_sec.instance = _instance(
        [_context("a", "2025-01-01", "2025-12-31", "us-gaap:CommonClassAMember")],
        [("EarningsPerShareBasic", "a", "1.31")],
    )
    [row] = _income_statement("TSN", {"EarningsPerShareDiluted": 1.28})
    assert (row["eps"], row["epsDiluted"]) == (1.31, 1.28)


def test_fetch_failure_leaves_eps_zero(monkeypatch):
    def fail(url, headers):
        raise sfd.requests.exceptions.ConnectionError("offline")

    monkeypatch.setattr(sfd, "_fetch_sec_bytes", fail)
    [row] = _income_statement("BRK-B", {})
    assert (row["eps"], row["epsDiluted"]) == (0, 0)


def test_preferred_member_partnership_units():
    # MPLX tags per-unit income by partner type; never the general partner.
    members = ["GeneralPartnerMember", "LimitedPartnersCommonUnitsMember"]
    assert preferred_class_member(members, "MPLX") == "LimitedPartnersCommonUnitsMember"
    assert preferred_class_member(["GeneralPartnerMember"], "MPLX") is None


def test_parse_reads_partner_type_axis():
    instance = _instance(
        [
            '<context id="lp"><entity><segment><xbrldi:explicitMember '
            'dimension="us-gaap:PartnerTypeOfPartnersCapitalAccountAxis">'
            "x:LimitedPartnersCommonUnitsMember</xbrldi:explicitMember></segment>"
            "</entity><period><startDate>2019-01-01</startDate>"
            "<endDate>2019-12-31</endDate></period></context>"
        ],
        [
            (
                "NetIncomeLossPerOutstandingLimitedPartnershipUnitBasicNetOfTax",
                "lp",
                "1.00",
            )
        ],
    )
    [fact] = parse_class_eps_facts(instance, _EPS_TAGS)
    assert (fact["member"], fact["value"]) == ("LimitedPartnersCommonUnitsMember", 1.0)


def test_new_taxonomy_discontinued_eps_is_added():
    # AGL tags discontinued operations EPS with the 2022+ element name.
    data = {
        "IncomeLossFromContinuingOperationsPerBasicShare": -0.98,
        "DiscontinuedOperationIncomeLossFromDiscontinuedOperationNetOfTaxPerBasicShare": 0.03,
    }
    assert _get_eps_value(data, "eps") == pytest.approx(-0.95)


def test_fetch_retries_throttling(monkeypatch):
    class Response:
        def __init__(self, status_code):
            self.status_code = status_code
            self.content = b"ok"

        def raise_for_status(self):
            if self.status_code >= 400:
                raise sfd.requests.exceptions.HTTPError(str(self.status_code))

    statuses = iter([503, 200])
    monkeypatch.setattr(sfd.requests, "get", lambda *a, **k: Response(next(statuses)))
    monkeypatch.setattr(sfd.time, "sleep", lambda _: None)
    assert sfd._fetch_sec_bytes("https://www.sec.gov/x", {}) == b"ok"

    statuses = iter([503, 503, 503])
    with pytest.raises(sfd.requests.exceptions.HTTPError):
        sfd._fetch_sec_bytes("https://www.sec.gov/x", {})
