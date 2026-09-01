"""API surface tests.

The most important assertion in this file is the last one: there is no route
anywhere in the app that could place an order.
"""

import json

import pytest
from fastapi.testclient import TestClient

from alpha.api import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["places_orders"] is False
    assert body["disclaimer"]


def test_brief_returns_every_section(client):
    body = client.get("/api/brief").json()
    for key in ("options", "equities", "sectors", "expiries", "disclaimer", "as_of"):
        assert key in body, f"missing {key}"
    assert body["errors"] == []
    assert len(body["options"]) == 2


def test_brief_is_json_serialisable(client):
    json.dumps(client.get("/api/brief").json())


def test_option_signals_endpoint(client):
    body = client.get("/api/signals/options?symbols=NIFTY,BANKNIFTY").json()
    assert len(body["signals"]) == 2
    for sig in body["signals"]:
        assert sig["direction"] in ("long", "short", "no_trade")
        assert sig["scorecard"]["factors"]
        assert sig["summary"]


def test_unknown_symbol_is_reported_not_crashed(client):
    body = client.get("/api/signals/options?symbols=NOTAREALINDEX").json()
    assert body["signals"] == []
    assert body["errors"] and "NOTAREALINDEX" in body["errors"][0]["symbol"]


def test_equity_endpoint_respects_top(client):
    body = client.get("/api/signals/equity?top=3").json()
    assert len(body["signals"]) <= 3


def test_equity_include_rejected_returns_more(client):
    few = client.get("/api/signals/equity?top=5").json()["signals"]
    many = client.get("/api/signals/equity?top=5&include_rejected=true").json()["signals"]
    assert len(many) >= len(few)


def test_sectors_endpoint_is_ranked(client):
    rows = client.get("/api/sectors").json()["sectors"]
    assert rows and [r["rank"] for r in rows] == list(range(1, len(rows) + 1))


def test_expiries_reflect_the_sebi_weekly_rule(client):
    rows = {e["symbol"]: e for e in client.get("/api/expiries").json()["expiries"]}
    assert rows["NIFTY"]["has_weekly"] is True
    assert rows["BANKNIFTY"]["has_weekly"] is False       # monthly-only since Nov 2024
    assert rows["FINNIFTY"]["has_weekly"] is False


def test_universe_endpoint(client):
    body = client.get("/api/universe").json()
    assert body["count"] == len(body["constituents"]) > 0
    assert body["benchmark"] == "NIFTY"


def test_bad_as_of_is_a_400_not_a_500(client):
    assert client.get("/api/brief?as_of=not-a-date").status_code == 400


def test_dashboard_and_assets_are_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/static/app.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_data_quality_is_surfaced_when_simulated(client):
    body = client.get("/api/brief").json()
    assert body["live_data"] is False          # tests run in fixture mode
    assert body["degraded"]


def test_no_route_can_place_an_order(client):
    """The product boundary, asserted. If a route ever appears that could send
    an order, this fails and the reviewer has to justify it explicitly."""
    paths = {r.path for r in app.routes}
    forbidden = ("order", "trade", "buy", "sell", "execute", "position", "broker")
    offenders = [p for p in paths if any(w in p.lower() for w in forbidden)]
    assert not offenders, f"order-capable routes must not exist: {offenders}"

    methods = {m for r in app.routes for m in (getattr(r, "methods", None) or set())}
    assert methods <= {"GET", "HEAD"}, f"the API must be read-only, found: {methods}"
