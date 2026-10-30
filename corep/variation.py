"""
Quarter-on-quarter variation analysis.

* Cell-level variation of any template between two reference dates.
* RWEA bridge at exposure level (credit risk), matching exposures by identifier:
    new business                      RWEA of exposures that exist only at the later date
    maturities and repayments in full − RWEA of exposures that exist only at the earlier date
    FX effect                         (USD exposures) RWEA_0 × (FX_0 / FX_1 − 1)
    volume (existing)                 (EV_1 − EV_0 − FX part) × RW_0
    then the density change EV_1 × (RW_1 − RW_0) is attributed, per exposure, to the first matching cause:
    new defaults → rating migrations (grade change) → IRB model / parameter changes (LGD) → other (CRM, LTV, CCF)
* Other risk types (market, CVA, operational, output floor) taken from C 02.00.
* CET1 ratio bridge: CET1 movements by C 01.00 component and TREA movements by risk type.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .templates import Report


def cell_variation(r0: Report, r1: Report, template: str) -> pd.DataFrame:
    a = r0.cells[r0.cells["template"] == template].set_index(["sheet", "row", "col"])
    b = r1.cells[r1.cells["template"] == template].set_index(["sheet", "row", "col"])
    idx = a.index.union(b.index)
    out = pd.DataFrame(index=idx)
    out["item"] = b["row_label"].reindex(idx).fillna(a["row_label"].reindex(idx))
    out["column"] = b["col_label"].reindex(idx).fillna(a["col_label"].reindex(idx))
    out["previous"] = a["value"].reindex(idx).fillna(0.0)
    out["current"] = b["value"].reindex(idx).fillna(0.0)
    out["change"] = out["current"] - out["previous"]
    out["change %"] = np.where(out["previous"].abs() > 0, out["change"] / out["previous"].abs(), np.nan)
    return out.reset_index()


def rwea_bridge(r0: Report, r1: Report, fx0: float, fx1: float) -> tuple[pd.Series, pd.DataFrame]:
    e0 = r0.exposures.set_index("exposure_id")
    e1 = r1.exposures.set_index("exposure_id")
    both = e0.index.intersection(e1.index)
    new = e1.index.difference(e0.index)
    gone = e0.index.difference(e1.index)
    a, b = e0.loc[both], e1.loc[both]
    rw0 = np.where(a["exposure_value"] > 0, a["rwea"] / a["exposure_value"], 0.0)
    rw1 = np.where(b["exposure_value"] > 0, b["rwea"] / b["exposure_value"], 0.0)
    usd = (a["currency"] == "USD").to_numpy()
    fx_ev = np.where(usd, a["exposure_value"] * (fx0 / fx1 - 1), 0.0)
    fx = fx_ev * rw0
    vol = (b["exposure_value"].to_numpy() - a["exposure_value"].to_numpy() - fx_ev) * rw0
    dens = b["exposure_value"].to_numpy() * (rw1 - rw0)
    cause = np.select([(~a["defaulted"] & b["defaulted"]).to_numpy(), (a["grade"] != b["grade"]).to_numpy(),
                       ((a["lgd"] - b["lgd"]).abs() > 1e-9).to_numpy()],
                      ["new defaults", "rating migrations", "IRB model / parameter changes"], "other (CRM, LTV, CCF, provisions)")
    detail = pd.DataFrame({"exposure_id": both, "portfolio": b["irb_class"].where(b["approach"] == "IRB", b["sa_class"]).to_numpy(),
                           "approach": b["approach"].to_numpy(), "FX effect": fx, "volume (existing)": vol, "density change": dens, "cause": cause})
    s = pd.Series({"new business": float(e1.loc[new, "rwea"].sum()), "maturities and full repayments": -float(e0.loc[gone, "rwea"].sum()),
                   "volume (existing exposures)": float(vol.sum()), "FX effect": float(fx.sum())})
    for c in ["new defaults", "rating migrations", "IRB model / parameter changes", "other (CRM, LTV, CCF, provisions)"]:
        s[c] = float(dens[cause == c].sum())
    s.attrs["start"] = float(r0.exposures["rwea"].sum())
    s.attrs["end"] = float(r1.exposures["rwea"].sum())
    new_d = e1.loc[new].assign(cause="new business", delta=e1.loc[new, "rwea"])
    gone_d = e0.loc[gone].assign(cause="maturities and full repayments", delta=-e0.loc[gone, "rwea"])
    detail = pd.concat([detail.assign(delta=detail["density change"]),
                        detail.assign(cause="volume (existing exposures)", delta=detail["volume (existing)"]),
                        detail.assign(cause="FX effect", delta=detail["FX effect"]),
                        pd.DataFrame({"exposure_id": new_d.index, "portfolio": new_d["irb_class"].where(new_d["approach"] == "IRB", new_d["sa_class"]),
                                      "approach": new_d["approach"], "cause": "new business", "delta": new_d["delta"]}),
                        pd.DataFrame({"exposure_id": gone_d.index, "portfolio": gone_d["irb_class"].where(gone_d["approach"] == "IRB", gone_d["sa_class"]),
                                      "approach": gone_d["approach"], "cause": "maturities and full repayments", "delta": gone_d["delta"]})],
                       ignore_index=True)
    return s, detail


def trea_bridge(r0: Report, r1: Report, credit: pd.Series) -> pd.Series:
    g0, g1 = r0.get, r1.get
    s = credit.copy()
    for lab, r in [("market risk", "0520"), ("operational risk", "0590"), ("CVA", "0640"), ("output floor adjustment", "0035")]:
        s[lab] = g1("C 02.00", r, "0010") - g0("C 02.00", r, "0010")
    s.attrs["start"] = g0("C 02.00", "0010", "0010")
    s.attrs["end"] = g1("C 02.00", "0010", "0010")
    s["other / top-side adjustments"] = (s.attrs["end"] - s.attrs["start"]) - s.sum()
    return s


def cet1_ratio_bridge(r0: Report, r1: Report, trea: pd.Series) -> pd.Series:
    """Change of the CET1 ratio: CET1 components at constant TREA, then the TREA effect."""
    t0 = r0.get("C 02.00", "0010", "0010")
    t1 = r1.get("C 02.00", "0010", "0010")
    c0, c1 = r0.get("C 01.00", "0020", "0010"), r1.get("C 01.00", "0020", "0010")
    comps = {"interim profit": "0150", "OCI": "0180", "prudent valuation": "0290", "intangibles": "0340", "IRB shortfall": "0380",
             "goodwill": "0300", "deferred tax assets": "0370", "other reserves": "0200"}
    s = pd.Series({k: (r1.get("C 01.00", r, "0010") - r0.get("C 01.00", r, "0010")) / t0 for k, r in comps.items()})
    s["other CET1 movements"] = (c1 - c0) / t0 - s.sum()
    s["TREA increase"] = c1 / t1 - c1 / t0
    s.attrs.update(start=c0 / t0, end=c1 / t1)
    return s
