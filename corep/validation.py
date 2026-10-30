"""
Validation rules in the spirit of the EBA validation rules (intra-template, inter-template, sign and range checks)
and data-quality pre-checks on the source exposures. Rule identifiers are internal (vC…), not the official EBA IDs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import IRB_CLASSES, SA_CLASSES
from .templates import RW_ROWS, Report

TOL_ABS = 1.0           # € (rounding tolerance)
TOL_REL = 1e-6


def _rule(rid, kind, templates, desc, lhs, rhs, op="=", severity="error"):
    if op == "=":
        ok = abs(lhs - rhs) <= max(TOL_ABS, TOL_REL * max(abs(lhs), abs(rhs)))
    elif op == "<=":
        ok = lhs <= rhs + TOL_ABS * 1e-9 if abs(rhs) < 10 else lhs <= rhs + TOL_ABS
    else:
        ok = lhs >= rhs - (1e-9 if abs(rhs) < 10 else TOL_ABS)
    return {"rule": rid, "type": kind, "templates": templates, "description": desc, "lhs": lhs, "op": op, "rhs": rhs, "difference": lhs - rhs,
            "severity": severity, "status": "pass" if ok else ("fail" if severity == "error" else "warning")}


def run(rep: Report) -> pd.DataFrame:
    g, t = rep.get, rep.total
    out = []
    # ---- C 01.00
    out.append(_rule("vC01-001", "intra", "C 01.00", "Own funds = Tier 1 + Tier 2 {r0010 = r0015 + r0750}", g("C 01.00", "0010", "0010"),
                     g("C 01.00", "0015", "0010") + g("C 01.00", "0750", "0010")))
    out.append(_rule("vC01-002", "intra", "C 01.00", "Tier 1 = CET1 + AT1 {r0015 = r0020 + r0530}", g("C 01.00", "0015", "0010"),
                     g("C 01.00", "0020", "0010") + g("C 01.00", "0530", "0010")))
    comp = ["0030", "0130", "0180", "0200", "0230", "0250", "0290", "0300", "0340", "0370", "0380"]
    out.append(_rule("vC01-003", "intra", "C 01.00", "CET1 = sum of its components {r0020 = r0030 + r0130 + … + r0380}", g("C 01.00", "0020", "0010"),
                     sum(g("C 01.00", r, "0010") for r in comp)))
    for r in ["0290", "0300", "0340", "0370", "0380"]:
        out.append(_rule(f"vC01-1{r[1:]}", "sign", "C 01.00", f"Deduction r{r} reported with a negative sign", g("C 01.00", r, "0010"), 0.0, "<="))
    # ---- C 02.00
    out.append(_rule("vC02-001", "intra", "C 02.00", "Credit risk = SA + IRB {r0040 = r0050 + r0240}", g("C 02.00", "0040", "0010"),
                     g("C 02.00", "0050", "0010") + g("C 02.00", "0240", "0010")))
    out.append(_rule("vC02-002", "intra", "C 02.00", "U-TREA = credit + settlement + market + operational + CVA {r0034}", g("C 02.00", "0034", "0010"),
                     sum(g("C 02.00", r, "0010") for r in ["0040", "0490", "0520", "0590", "0640"])))
    out.append(_rule("vC02-003", "intra", "C 02.00", "TREA = U-TREA + output floor adjustment {r0010 = r0034 + r0035}", g("C 02.00", "0010", "0010"),
                     g("C 02.00", "0034", "0010") + g("C 02.00", "0035", "0010")))
    out.append(_rule("vC02-004", "intra", "C 02.00", "Floored TREA ≥ un-floored TREA", g("C 02.00", "0010", "0010"), g("C 02.00", "0034", "0010"), ">="))
    out.append(_rule("vC02-005", "inter", "C 02.00 / C 07.00", "SA credit RWEA = Σ C 07.00 {r0010; c0220} over exposure classes",
                     g("C 02.00", "0050", "0010"), sum(g("C 07.00", "0010", "0220", cls) for cls in SA_CLASSES)))
    out.append(_rule("vC02-006", "inter", "C 02.00 / C 08.01", "IRB credit RWEA = Σ C 08.01 {r0010; c0260} over exposure classes",
                     g("C 02.00", "0240", "0010"), sum(g("C 08.01", "0010", "0260", cls) for cls in IRB_CLASSES)))
    out.append(_rule("vC02-007", "inter", "C 02.00 / C 18-23", "Market risk RWEA = C 18-23 total", g("C 02.00", "0520", "0010"), g("C 18-23", "0990", "0020")))
    out.append(_rule("vC02-008", "inter", "C 02.00 / C 16.01", "Operational risk RWEA = 12.5 × BIC", g("C 02.00", "0590", "0010"), 12.5 * g("C 16.01", "0050", "0010")))
    out.append(_rule("vC02-009", "inter", "C 02.00 / C 25.01", "CVA RWEA = C 25.01 {r0010; c0060}", g("C 02.00", "0640", "0010"), g("C 25.01", "0010", "0060")))
    # ---- C 03.00
    trea = g("C 02.00", "0010", "0010")
    out.append(_rule("vC03-001", "inter", "C 03.00 / C 01.00 / C 02.00", "CET1 ratio = CET1 / TREA", g("C 03.00", "0010", "0010"),
                     g("C 01.00", "0020", "0010") / trea if trea else 0.0))
    out.append(_rule("vC03-002", "inter", "C 03.00 / C 01.00 / C 02.00", "T1 ratio = T1 / TREA", g("C 03.00", "0030", "0010"),
                     g("C 01.00", "0015", "0010") / trea if trea else 0.0))
    out.append(_rule("vC03-003", "inter", "C 03.00 / C 01.00 / C 02.00", "Total capital ratio = own funds / TREA", g("C 03.00", "0050", "0010"),
                     g("C 01.00", "0010", "0010") / trea if trea else 0.0))
    out.append(_rule("vC03-101", "range", "C 03.00", "CET1 ratio ≥ 4.5 % (Pillar 1 minimum)", g("C 03.00", "0010", "0010"), 0.045, ">=", "warning"))
    # ---- C 07.00 (each class)
    for cls in SA_CLASSES:
        s = cls
        out.append(_rule(f"vC07-001", "intra", "C 07.00", f"[{s}] Net exposure {{c0040 = c0010 + c0030}}", g("C 07.00", "0010", "0040", s),
                         g("C 07.00", "0010", "0010", s) + g("C 07.00", "0010", "0030", s)))
        out.append(_rule(f"vC07-002", "intra", "C 07.00", f"[{s}] After substitution {{c0110 = c0040 + c0050 + c0100}}", g("C 07.00", "0010", "0110", s),
                         g("C 07.00", "0010", "0040", s) + g("C 07.00", "0010", "0050", s) + g("C 07.00", "0010", "0100", s)))
        out.append(_rule(f"vC07-003", "intra", "C 07.00", f"[{s}] Exposure value: on + off-balance + derivatives {{r0010 = r0070 + r0080 + r0090}}",
                         g("C 07.00", "0010", "0200", s), sum(g("C 07.00", r, "0200", s) for r in ["0070", "0080", "0090"])))
        out.append(_rule(f"vC07-004", "intra", "C 07.00", f"[{s}] Breakdown by risk weight sums to total exposure value {{c0200}}",
                         g("C 07.00", "0010", "0200", s), sum(g("C 07.00", r, "0200", s) for r in list(RW_ROWS.values()) + ["0280"])))
        out.append(_rule(f"vC07-005", "intra", "C 07.00", f"[{s}] RWEA after SME supporting factor ≤ RWEA before {{c0220 ≤ c0215}}",
                         g("C 07.00", "0010", "0220", s), g("C 07.00", "0010", "0215", s), "<="))
        out.append(_rule(f"vC07-101", "sign", "C 07.00", f"[{s}] Value adjustments and provisions reported negative {{c0030 ≤ 0}}",
                         g("C 07.00", "0010", "0030", s), 0.0, "<="))
    out.append(_rule("vC07-006", "inter", "C 07.00", "CRM substitution: Σ outflows {c0050} = − Σ inflows {c0100} across classes",
                     -t("C 07.00", "0010", "0050"), t("C 07.00", "0010", "0100")))
    # ---- C 08.01 / C 08.02
    for cls in IRB_CLASSES:
        if not rep.cells[(rep.cells["template"] == "C 08.01") & (rep.cells["sheet"] == cls)].shape[0]:
            continue
        c2 = rep.cells[(rep.cells["template"] == "C 08.02") & (rep.cells["sheet"] == cls)]
        out.append(_rule("vC08-001", "inter", "C 08.01 / C 08.02", f"[{cls}] Exposure value = Σ obligor grades {{c0110}}",
                         g("C 08.01", "0010", "0110", cls), float(c2.loc[c2["col"] == "0110", "value"].sum())))
        out.append(_rule("vC08-002", "inter", "C 08.01 / C 08.02", f"[{cls}] RWEA = Σ obligor grades {{c0260}}",
                         g("C 08.01", "0010", "0260", cls), float(c2.loc[c2["col"] == "0260", "value"].sum())))
        out.append(_rule("vC08-101", "range", "C 08.01", f"[{cls}] Exposure-weighted LGD within [0, 100 %] {{c0230}}", g("C 08.01", "0010", "0230", cls), 1.0, "<="))
        pdv = c2.loc[c2["col"] == "0010", "value"]
        out.append(_rule("vC08-102", "range", "C 08.02", f"[{cls}] PD of every obligor grade ≥ 0.03 % {{c0010}}",
                         float(pdv.min()) if len(pdv) else 1.0, 0.0003, ">="))
    # ---- C 09
    out.append(_rule("vC09-001", "inter", "C 09.01 / C 07.00", "Σ countries C 09.01 {c0090} = Σ classes C 07.00 {r0010; c0220}",
                     sum(t("C 09.01", f"{i + 1:03d}0", "0090") for i in range(len(SA_CLASSES))),
                     sum(g("C 07.00", "0010", "0220", cls) for cls in SA_CLASSES)))
    out.append(_rule("vC09-002", "inter", "C 09.02 / C 08.01", "Σ countries C 09.02 {c0105} = Σ classes C 08.01 {r0010; c0260}",
                     sum(t("C 09.02", f"{i + 1:03d}0", "0105") for i in range(len(IRB_CLASSES))), sum(g("C 08.01", "0010", "0260", cls) for cls in IRB_CLASSES)))
    na = rep.cells[(rep.cells["template"].isin(["C 09.01", "C 09.02"])) & (rep.cells["sheet"] == "Not allocated")]
    out.append(_rule("vC09-101", "completeness", "C 09.01 / C 09.02", "No exposure without country of residence (sheet 'Not allocated' empty)",
                     float(na["value"].abs().sum()), 0.0))
    # ---- C 47.00, C 28.00
    out.append(_rule("vC47-001", "inter", "C 47.00 / C 01.00", "Tier 1 in the leverage ratio = C 01.00 {r0015}", g("C 47.00", "0310", "0010"), g("C 01.00", "0015", "0010")))
    out.append(_rule("vC47-002", "intra", "C 47.00", "Leverage ratio = Tier 1 / total exposure measure", g("C 47.00", "0340", "0010"),
                     g("C 47.00", "0310", "0010") / max(g("C 47.00", "0290", "0010"), 1.0)))
    out.append(_rule("vC47-101", "range", "C 47.00", "Leverage ratio ≥ 3 %", g("C 47.00", "0340", "0010"), 0.03, ">=", "warning"))
    c28 = rep.cells[(rep.cells["template"] == "C 28.00") & (rep.cells["col"] == "0350")]
    out.append(_rule("vC28-101", "limit", "C 28.00 / C 26.00", "No exposure after CRM above 25 % of Tier 1", float(c28["value"].max()) if len(c28) else 0.0, 0.25,
                     "<=", "warning"))
    df = pd.DataFrame(out)
    return df


def data_quality(e: pd.DataFrame) -> pd.DataFrame:
    """Pre-checks on the source exposures (before the templates are built)."""
    checks = [
        ("DQ-01", "country of residence filled", e["country"].fillna("").eq("")),
        ("DQ-02", "provisions are not negative", e["provisions"] < 0),
        ("DQ-03", "IRB LGD within [0, 100 %]", (e["approach"] == "IRB") & ((e["lgd"] < 0) | (e["lgd"] > 1))),
        ("DQ-04", "IRB exposures have a PD", (e["approach"] == "IRB") & ~e["defaulted"] & (e["pd"] <= 0)),
        ("DQ-05", "off-balance items have a CCF bucket", (e["bs_type"] == "off") & e["ccf_bucket"].fillna("").eq("")),
        ("DQ-06", "real-estate exposures have an LTV", (e["sa_class"] == "Secured by mortgages on immovable property") & (e["approach"] == "SA") & (e["ltv"] <= 0)),
        ("DQ-07", "defaulted flag consistent with grade D", e["defaulted"] != e["grade"].eq("D") & e["grade"].ne("")),
        ("DQ-08", "unique exposure identifier", e["exposure_id"].duplicated(keep=False)),
    ]
    rows = []
    for cid, desc, m in checks:
        m = m.fillna(False).astype(bool)
        rows.append({"check": cid, "description": desc, "failing records": int(m.sum()), "status": "pass" if m.sum() == 0 else "fail",
                     "examples": ", ".join(e.loc[m, "exposure_id"].head(5))})
    return pd.DataFrame(rows)
