import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from corep import calc as C
from corep import data as D
from corep import export as X
from corep import templates as T
from corep import validation as V
from corep import variation as VA


@pytest.fixture(scope="module")
def setup():
    rds = D.generate(11)
    reps = {d: T.build(rd) for d, rd in rds.items()}
    return rds, reps


def test_all_rules_pass_on_clean_data(setup):
    _, reps = setup
    for r in reps.values():
        v = V.run(r)
        assert (v["status"] != "fail").all(), v[v["status"] == "fail"][["rule", "description"]]
        assert (V.data_quality(r.exposures)["status"] == "pass").all()


def test_irb_reference_rw():
    assert C.irb_rw(0.01, 0.45, 2.5, "Corporates - other general", 1000, False) == pytest.approx(0.9244, abs=0.005)
    # residential mortgage, PD 1 %, LGD 15 %, R = 15 %: K = 0.15 × (N(−1.225) − 0.01) ≈ 1.50 % → RW ≈ 18.8 %
    assert C.irb_rw(0.01, 0.15, 1, "Retail - secured by residential immovable property", 0, False) == pytest.approx(0.188, abs=0.002)


def test_sa_risk_weights():
    base = dict(counterparty_id="X", counterparty="X", country="FR", currency="EUR", grade="G05", rated=True, sme=False, ltv=0.4,
                property_type="Residential", original_exposure=100.0, provisions=0.0)
    assert C.sa_risk_weight({**base, "sa_class": "Secured by mortgages on immovable property"}) == pytest.approx(0.20)
    assert C.sa_risk_weight({**base, "sa_class": "Secured by mortgages on immovable property", "ltv": 0.8}) == pytest.approx(0.2 * 0.55 / 0.8 + 0.75 * 0.25 / 0.8)
    assert C.sa_risk_weight({**base, "sa_class": "Corporates", "rated": False, "sme": True}) == 0.85
    assert C.sa_risk_weight({**base, "sa_class": "Exposures in default", "provisions": 30.0}) == 1.0
    assert C.sa_risk_weight({**base, "sa_class": "Exposures in default", "provisions": 5.0, "property_type": ""}) == 1.5


def test_ratios_and_identities(setup):
    _, reps = setup
    r = reps[D.REF_DATES[1]]
    assert r.get("C 03.00", "0010", "0010") == pytest.approx(r.get("C 01.00", "0020", "0010") / r.get("C 02.00", "0010", "0010"))
    assert r.get("C 02.00", "0010", "0010") >= r.get("C 02.00", "0034", "0010")
    assert 0.08 < r.get("C 03.00", "0010", "0010") < 0.25


def test_output_floor_binding_when_forced(setup):
    rds, _ = setup
    rd = rds[D.REF_DATES[1]]
    old = T.OUTPUT_FLOOR.copy()
    try:
        T.OUTPUT_FLOOR[2026] = 0.95
        r = T.build(rd)
        assert r.get("C 02.00", "0035", "0010") > 0
        assert r.get("C 02.00", "0010", "0010") <= 1.25 * r.get("C 02.00", "0034", "0010") + 1
        assert (V.run(r)["status"] != "fail").all()
    finally:
        T.OUTPUT_FLOOR.clear()
        T.OUTPUT_FLOOR.update(old)


def test_topside_adjustment_breaks_rules(setup):
    _, reps = setup
    r = reps[D.REF_DATES[1]]
    adj = pd.DataFrame([{"template": "C 02.00", "sheet": "", "row": "0050", "col": "0010", "amount": 1e8, "comment": ""}])
    v = V.run(T.apply_adjustments(r, adj))
    failed = set(v.loc[v["status"] == "fail", "rule"])
    assert {"vC02-001", "vC02-005"} <= failed


def test_defects_are_caught(setup):
    rds, _ = setup
    rd, log = D.inject_defects(rds[D.REF_DATES[1]])
    dq = V.data_quality(rd.exposures)
    assert set(dq.loc[dq["status"] == "fail", "check"]) >= {"DQ-01", "DQ-02", "DQ-03"}
    v = V.run(T.build(rd))
    assert "vC09-101" in set(v.loc[v["status"] == "fail", "rule"])


def test_lineage_reconciles(setup):
    _, reps = setup
    r = reps[D.REF_DATES[1]]
    for t, s, row, col in [("C 07.00", "Corporates", "0010", "0220"), ("C 08.01", "Corporates - SME", "0010", "0260"), ("C 09.01", "FR", "0030", "0090")]:
        c = r.contrib[(r.contrib["template"] == t) & (r.contrib["sheet"] == s) & (r.contrib["row"] == row) & (r.contrib["col"] == col)]
        assert c["value"].sum() == pytest.approx(r.get(t, row, col, s), rel=1e-9)


def test_variation_bridges_add_up(setup):
    rds, reps = setup
    r0, r1 = reps[D.REF_DATES[0]], reps[D.REF_DATES[1]]
    s, det = VA.rwea_bridge(r0, r1, rds[D.REF_DATES[0]].fx, rds[D.REF_DATES[1]].fx)
    assert s.sum() == pytest.approx(s.attrs["end"] - s.attrs["start"], rel=1e-9, abs=1)
    assert s["FX effect"] != 0 and s["rating migrations"] > 0
    tb = VA.trea_bridge(r0, r1, s)
    assert tb.sum() == pytest.approx(tb.attrs["end"] - tb.attrs["start"], abs=1)
    assert abs(tb["other / top-side adjustments"]) < 1
    cb = VA.cet1_ratio_bridge(r0, r1, tb)
    assert cb.sum() == pytest.approx(cb.attrs["end"] - cb.attrs["start"], abs=1e-12)


def test_exports(setup):
    _, reps = setup
    r = reps[D.REF_DATES[1]]
    assert len(X.to_excel(r)) > 10_000
    z = zipfile.ZipFile(io.BytesIO(X.to_xbrl_csv(r)))
    names = z.namelist()
    assert any(n.endswith("META-INF/reportPackage.json") for n in names)
    assert any(n.endswith("reports/FilingIndicators.csv") for n in names)
    assert any(n.endswith("c_01.00.csv") for n in names)


def test_import_validation(setup):
    rds, _ = setup
    e = rds[D.REF_DATES[1]].exposures.drop(columns=["amount_ccy"])
    out = D.validate_exposures(pd.read_csv(io.StringIO(e.to_csv(index=False))))
    assert len(out) == len(e)
    with pytest.raises(ValueError):
        D.validate_exposures(e.drop(columns=["lgd"]))
