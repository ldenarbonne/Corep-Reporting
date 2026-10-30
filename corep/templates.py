"""
COREP templates (CRR3 / EBA reporting framework 4.x), built as a single long table of cells
(template, sheet, row, column, value) plus a long table of exposure-level contributions for data → cell lineage.

Row and column codes follow the numbering of the EBA templates for the main cells; where the CRR3 layouts add many
detailed rows (risk-weight buckets, new exposure sub-classes) the app keeps a reduced set and the codes are indicative.
They are not a substitute for the official annotated templates and DPM.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import calc as C
from .data import IRB_CLASSES, SA_CLASSES, ReportingData

OUTPUT_FLOOR = {2025: 0.50, 2026: 0.55, 2027: 0.60, 2028: 0.65, 2029: 0.70}
OF_CAP = 1.25                      # art. 465(2): floored TREA ≤ 125 % of un-floored TREA during the transition

TITLES = {
    "C 01.00": "Own funds (CA1)", "C 02.00": "Own funds requirements (CA2)", "C 03.00": "Capital ratios and capital levels (CA3)",
    "C 07.00": "Credit and counterparty credit risks – Standardised approach (CR SA)",
    "C 08.01": "Credit and counterparty credit risks – IRB approach (CR IRB 1)",
    "C 08.02": "Credit and counterparty credit risks – IRB approach, breakdown by obligor grade (CR IRB 2)",
    "C 09.01": "Geographical breakdown of exposures – SA (CR GB 1)", "C 09.02": "Geographical breakdown of exposures – IRB (CR GB 2)",
    "C 16.01": "Operational risk – business indicator and own funds requirements (OPR)",
    "C 18-23": "Market risk – standardised approaches (MKR SA TDI / EQU / FX / COM, summary)",
    "C 25.01": "Credit valuation adjustment risk – reduced basic approach (CVA)",
    "C 26.00": "Large exposures limits (LE Limits)", "C 28.00": "Exposures in the non-trading and trading book (LE 2)",
    "C 29.00": "Detail of the exposures to individual clients within groups of connected clients (LE 3)",
    "C 47.00": "Leverage ratio calculation (LRCalc)",
}

C07_COLS = {"0010": "Original exposure pre conversion factors", "0030": "(-) Value adjustments and provisions",
            "0040": "Exposure net of value adjustments and provisions", "0050": "(-) Unfunded credit protection: guarantees (outflows)",
            "0090": "(-) Funded credit protection: financial collateral, adjusted value", "0100": "Substitution of the exposure due to CRM: inflows (+)",
            "0110": "Net exposure after CRM substitution effects pre conversion factors", "0150": "Fully adjusted exposure value (E*)",
            "0200": "Exposure value", "0215": "Risk weighted exposure amount pre SME supporting factor",
            "0220": "Risk weighted exposure amount after supporting factors"}
C07_ROWS = {"0010": "TOTAL EXPOSURES", "0015": "of which: defaulted exposures", "0020": "of which: SME", "0030": "of which: exposures subject to the SME supporting factor",
            "0070": "On-balance sheet exposures", "0080": "Off-balance sheet exposures", "0090": "Derivatives (counterparty credit risk)"}
RW_ROWS = {0.0: "0140", 0.20: "0180", 0.30: "0185", 0.40: "0195", 0.50: "0200", 0.70: "0210", 0.75: "0220", 0.85: "0225", 0.90: "0226",
           1.0: "0230", 1.10: "0232", 1.30: "0235", 1.50: "0240"}
RW_LABEL = {v: f"Risk weight {int(round(k * 100))} %" for k, v in RW_ROWS.items()}
RW_LABEL["0280"] = "Other risk weights"

C08_COLS = {"0010": "PD assigned to the obligor grade or pool (%)", "0020": "Original exposure pre conversion factors",
            "0040": "(-) Unfunded credit protection: guarantees", "0110": "Exposure value", "0230": "Exposure-weighted average LGD (%)",
            "0250": "Exposure-weighted average maturity (days)", "0255": "RWEA pre SME supporting factor", "0260": "Risk weighted exposure amount",
            "0280": "Expected loss amount", "0290": "(-) Value adjustments and provisions", "0300": "Number of obligors"}
C08_ROWS = {"0010": "TOTAL EXPOSURES", "0015": "of which: defaulted", "0020": "of which: SME", "0030": "On-balance sheet items",
            "0040": "Off-balance sheet items", "0050": "Derivatives"}


@dataclass
class Report:
    ref_date: pd.Timestamp
    cells: pd.DataFrame                      # template, sheet, row, col, value, row_label, col_label
    contrib: pd.DataFrame                    # template, sheet, row, col, exposure_id, value
    formulas: dict                           # (template, row, col) → formula text
    exposures: pd.DataFrame
    aux: dict = field(default_factory=dict)
    adjustments: pd.DataFrame | None = None

    def get(self, template, row, col, sheet="") -> float:
        m = (self.cells["template"] == template) & (self.cells["row"] == row) & (self.cells["col"] == col) & (self.cells["sheet"] == sheet)
        v = self.cells.loc[m, "value"]
        return float(v.iloc[0]) if len(v) else 0.0

    def total(self, template, row, col) -> float:
        m = (self.cells["template"] == template) & (self.cells["row"] == row) & (self.cells["col"] == col)
        return float(self.cells.loc[m, "value"].sum())

    def table(self, template, sheet="") -> pd.DataFrame:
        c = self.cells[(self.cells["template"] == template) & (self.cells["sheet"] == sheet)]
        if c.empty:
            return pd.DataFrame()
        rows = c.drop_duplicates("row")[["row", "row_label"]]
        cols = c.drop_duplicates("col")[["col", "col_label"]]
        t = c.pivot_table(index="row", columns="col", values="value", aggfunc="sum", sort=False)
        t = t.reindex(index=rows["row"], columns=cols["col"])
        t.index = pd.MultiIndex.from_arrays([rows["row"], rows["row_label"]], names=["row", "item"])
        t.columns = pd.MultiIndex.from_arrays([cols["col"], cols["col_label"]], names=["col", "column"])
        return t

    def sheets(self, template) -> list[str]:
        return list(dict.fromkeys(self.cells.loc[self.cells["template"] == template, "sheet"]))


def _cells(template, sheet, data: dict, row_labels: dict, col_labels: dict) -> list[dict]:
    out = []
    for (r, c), v in data.items():
        out.append({"template": template, "sheet": sheet, "row": r, "col": c, "value": float(v), "row_label": row_labels.get(r, r),
                    "col_label": col_labels.get(c, c)})
    return out


def _rw_row(rw: float) -> str:
    for k, code in RW_ROWS.items():
        if abs(rw - k) < 1e-6:
            return code
    return "0280"


def _contrib_long(df: pd.DataFrame, template: str, sheet_col: str, rows: list[tuple[str, pd.Series]], cols: dict[str, str]) -> pd.DataFrame:
    """rows: (row code, boolean mask); cols: col code → measure column of df."""
    frames = []
    for rcode, mask in rows:
        sub = df[mask]
        if sub.empty:
            continue
        for ccode, meas in cols.items():
            frames.append(pd.DataFrame({"template": template, "sheet": sub[sheet_col].to_numpy() if sheet_col else "", "row": rcode, "col": ccode,
                                        "exposure_id": sub["exposure_id"].to_numpy(), "value": sub[meas].to_numpy()}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["template", "sheet", "row", "col", "exposure_id", "value"])


def build(rd: ReportingData) -> Report:
    e = C.compute(rd.exposures)
    lines = C.sa_lines(e)
    contrib = []
    cells = []
    formulas = {}
    # ------------------------------------------------ C 07.00
    L = lines.copy()
    ob = L["part"] == "obligor"
    L["m0010"] = np.where(ob, L["gross_for_c07"], 0.0)
    L["m0030"] = np.where(ob & (L["bs_type"] != "deriv"), -L["provisions"], 0.0)
    L["m0040"] = L["m0010"] + L["m0030"]
    L["m0050"] = np.where(ob, -L["guar_cover"], 0.0)
    L["m0090"] = np.where(ob, -L["coll_adj"], 0.0)
    L["m0100"] = np.where(~ob, L["guar_cover"], 0.0)
    L["m0110"] = L["m0040"] + L["m0050"] + L["m0100"]
    L["m0150"] = np.where(ob, L["e_star"], L["guar_cover"])
    L["m0200"] = L["ev"]
    L["m0215"] = L["rwea_pre_sf"]
    L["m0220"] = L["rwea_line"]
    L["rw_row"] = L["rw"].map(_rw_row)
    colmap = {c: f"m{c}" for c in C07_COLS}
    rows = [("0010", pd.Series(True, index=L.index)), ("0015", L["defaulted"] & ob), ("0020", L["sme"] & ob), ("0030", ob & (L["sme_sf"] < 1)),
            ("0070", L["bs_type"] == "on"), ("0080", L["bs_type"] == "off"), ("0090", L["bs_type"] == "deriv")]
    rows += [(code, L["rw_row"] == code) for code in list(RW_ROWS.values()) + ["0280"]]
    c07 = _contrib_long(L, "C 07.00", "line_class", rows, colmap)
    contrib.append(c07)
    agg = c07.groupby(["sheet", "row", "col"])["value"].sum()
    rl = {**C07_ROWS, **RW_LABEL}
    for cls in SA_CLASSES:
        d = {(r, c): agg.get((cls, r, c), 0.0) for r in list(C07_ROWS) for c in C07_COLS}
        d.update({(r, c): agg.get((cls, r, c), 0.0) for r in list(RW_ROWS.values()) + ["0280"] for c in C07_COLS if (cls, r, c) in agg.index})
        cells += _cells("C 07.00", cls, d, rl, C07_COLS)
    # ------------------------------------------------ C 08.01 / C 08.02
    I = e[e["approach"] == "IRB"].copy()
    I["m0010"] = I["pd_irb"] * I["ev_irb"]
    I["m0020"] = np.where(I["bs_type"] == "deriv", I["net_exposure"], I["original_exposure"])
    I["m0040"] = -I["guar_cover"]
    I["m0110"] = I["ev_irb"]
    I["m0230"] = I["lgd_irb"] * I["ev_irb"]
    I["m0250"] = I["maturity"] * 365 * I["ev_irb"]
    I["m0255"] = I["rwea_irb"]
    I["m0260"] = I["rwea_irb"]
    I["m0280"] = I["el_irb"]
    I["m0290"] = -I["provisions"].clip(lower=0)
    rows = [("0010", pd.Series(True, index=I.index)), ("0015", I["defaulted"]), ("0020", I["sme"]), ("0030", I["bs_type"] == "on"),
            ("0040", I["bs_type"] == "off"), ("0050", I["bs_type"] == "deriv")]
    cm = {c: f"m{c}" for c in C08_COLS if c != "0300"}
    c08 = _contrib_long(I, "C 08.01", "irb_class", rows, cm)
    contrib.append(c08)
    g = c08.groupby(["sheet", "row", "col"])["value"].sum()
    W = ["0010", "0230", "0250"]                                  # exposure-weighted averages
    for cls in IRB_CLASSES:
        d = {}
        for r in C08_ROWS:
            ev = g.get((cls, r, "0110"), 0.0)
            for c in C08_COLS:
                if c == "0300":
                    m = {"0010": True, "0015": I["defaulted"], "0020": I["sme"], "0030": I["bs_type"] == "on", "0040": I["bs_type"] == "off",
                         "0050": I["bs_type"] == "deriv"}[r]
                    sub = I[(I["irb_class"] == cls) & (m if not isinstance(m, bool) else True)]
                    d[(r, c)] = sub["counterparty_id"].nunique()
                elif c in W:
                    d[(r, c)] = g.get((cls, r, c), 0.0) / ev if ev else 0.0
                else:
                    d[(r, c)] = g.get((cls, r, c), 0.0)
        cells += _cells("C 08.01", cls, d, C08_ROWS, C08_COLS)
    # C 08.02 rows = obligor grades
    I["grade_row"] = I["grade"].replace("", "NA")
    rows = [(gr, I["grade_row"] == gr) for gr in sorted(I["grade_row"].unique())]
    c082 = _contrib_long(I, "C 08.02", "irb_class", rows, {c: f"m{c}" for c in ["0010", "0020", "0110", "0230", "0260", "0280", "0290"]})
    contrib.append(c082)
    g2 = c082.groupby(["sheet", "row", "col"])["value"].sum()
    c2cols = {k: C08_COLS[k] for k in ["0010", "0020", "0110", "0230", "0260", "0280", "0290", "0300"]}
    for cls in IRB_CLASSES:
        d = {}
        for gr, _ in rows:
            sub = I[(I["irb_class"] == cls) & (I["grade_row"] == gr)]
            if sub.empty:
                continue
            ev = g2.get((cls, gr, "0110"), 0.0)
            for c in c2cols:
                if c == "0300":
                    d[(gr, c)] = sub["counterparty_id"].nunique()
                elif c in ("0010", "0230"):
                    d[(gr, c)] = g2.get((cls, gr, c), 0.0) / ev if ev else 0.0
                else:
                    d[(gr, c)] = g2.get((cls, gr, c), 0.0)
        cells += _cells("C 08.02", cls, d, {gr: f"Obligor grade {gr}" for gr, _ in rows}, c2cols)
    # ------------------------------------------------ C 09.01 / C 09.02 (by country of residence of the obligor)
    S = e[e["approach"] == "SA"].copy()
    S["country_sheet"] = S["country"].replace("", "Not allocated")
    S["m0010"] = S["gross_for_c07"]
    S["m0020"] = np.where(S["defaulted"], S["gross_for_c07"], 0.0)
    S["m0060"] = -S["provisions"].clip(lower=0)
    S["m0075"] = S["ev_sa"] + S["ev_guar"]
    S["m0080"] = S["rwea_sa_pre_sf"]
    S["m0090"] = S["rwea_sa"]
    gcols = {"0010": "Original exposure", "0020": "Defaulted exposures", "0060": "Specific credit risk adjustments", "0075": "Exposure value",
             "0080": "RWEA pre supporting factors", "0090": "Risk weighted exposure amount"}
    rows = [(f"{i + 1:03d}0", S["sa_class"] == cls) for i, cls in enumerate(SA_CLASSES)]
    c091 = _contrib_long(S, "C 09.01", "country_sheet", rows, {c: f"m{c}" for c in gcols})
    contrib.append(c091)
    g3 = c091.groupby(["sheet", "row", "col"])["value"].sum()
    rlab = {f"{i + 1:03d}0": cls for i, cls in enumerate(SA_CLASSES)}
    for ctry in sorted(S["country_sheet"].unique()):
        d = {(r, c): g3.get((ctry, r, c), 0.0) for r in rlab for c in gcols if (ctry, r, c) in g3.index}
        cells += _cells("C 09.01", ctry, d, rlab, gcols)
    I["country_sheet"] = I["country"].replace("", "Not allocated")
    I["m0030"] = np.where(I["defaulted"], I["m0020"], 0.0)
    gcols2 = {"0010": "Original exposure", "0030": "Defaulted exposures", "0080": "Exposure value", "0090": "Exposure-weighted average PD (%)",
              "0100": "Exposure-weighted average LGD (%)", "0105": "Risk weighted exposure amount", "0110": "Expected loss amount"}
    I["m0080"], I["m0090"], I["m0100"], I["m0105"], I["m0110"] = I["ev_irb"], I["pd_irb"] * I["ev_irb"], I["lgd_irb"] * I["ev_irb"], I["rwea_irb"], I["el_irb"]
    rows = [(f"{i + 1:03d}0", I["irb_class"] == cls) for i, cls in enumerate(IRB_CLASSES)]
    c092 = _contrib_long(I, "C 09.02", "country_sheet", rows, {c: f"m{c}" for c in gcols2})
    contrib.append(c092)
    g4 = c092.groupby(["sheet", "row", "col"])["value"].sum()
    rlab2 = {f"{i + 1:03d}0": cls for i, cls in enumerate(IRB_CLASSES)}
    for ctry in sorted(I["country_sheet"].unique()):
        d = {}
        for r in rlab2:
            ev = g4.get((ctry, r, "0080"), 0.0)
            if not ev:
                continue
            for c in gcols2:
                d[(r, c)] = g4.get((ctry, r, c), 0.0) / ev if c in ("0090", "0100") else g4.get((ctry, r, c), 0.0)
        cells += _cells("C 09.02", ctry, d, rlab2, gcols2)
    # ------------------------------------------------ market risk, CVA, operational risk
    of_items = rd.own_funds
    prelim_of = of_items.sum()
    mkt = C.market_risk(rd.positions, prelim_of)
    for i, r in mkt.iterrows():
        cells += _cells("C 18-23", "", {(f"{i + 1:03d}0", "0010"): r["own funds requirement"], (f"{i + 1:03d}0", "0020"): r["RWEA"]},
                        {f"{i + 1:03d}0": f"{r['template']} – {r['component']}"}, {"0010": "Own funds requirements", "0020": "Total risk exposure amount"})
    cells += _cells("C 18-23", "", {("0990", "0010"): mkt["own funds requirement"].sum(), ("0990", "0020"): mkt["RWEA"].sum()},
                    {"0990": "TOTAL market risk (standardised)"}, {"0010": "Own funds requirements", "0020": "Total risk exposure amount"})
    cv = C.cva(e)
    cells += _cells("C 25.01", "", {("0010", "0010"): len(cv), ("0010", "0020"): cv["EAD"].sum(), ("0010", "0030"): cv["SCVA"].sum(),
                                    ("0010", "0040"): cv.attrs["K_reduced"], ("0010", "0050"): cv.attrs["own_funds"], ("0010", "0060"): cv.attrs["RWEA"]},
                    {"0010": "Reduced basic approach (BA-CVA)"}, {"0010": "Number of counterparties", "0020": "Exposure value (SA-CCR)",
                                                                   "0030": "Σ SCVA", "0040": "K reduced", "0050": "Own funds requirement (DS × K)", "0060": "RWEA"})
    opr = C.operational_risk(rd.pnl_bi)
    oprow = {"0010": "Interest, leases and dividend component (ILDC)", "0020": "Services component (SC)", "0030": "Financial component (FC)",
             "0040": "Business indicator (BI)", "0050": "Business indicator component (BIC)", "0060": "Own funds requirement (ILM = 1)", "0070": "RWEA"}
    cells += _cells("C 16.01", "", {("0010", "0010"): opr["ILDC"], ("0020", "0010"): opr["SC"], ("0030", "0010"): opr["FC"], ("0040", "0010"): opr["BI"],
                                    ("0050", "0010"): opr["BIC"], ("0060", "0010"): opr["own funds"], ("0070", "0010"): opr["RWEA"]},
                    oprow, {"0010": "Amount (3-year average)"})
    # ------------------------------------------------ C 02.00 (risk exposure amounts, output floor)
    sa_rwea = float(L["rwea_line"].sum())
    irb_rwea = float(I["rwea_irb"].sum())
    credit = sa_rwea + irb_rwea
    u_trea = credit + mkt["RWEA"].sum() + opr["RWEA"] + cv.attrs["RWEA"]
    s_credit = float(e["rwea_sa"].sum())
    s_trea = s_credit + mkt["RWEA"].sum() + opr["RWEA"] + cv.attrs["RWEA"]
    x = OUTPUT_FLOOR.get(rd.ref_date.year, 0.725)
    floored = max(u_trea, min(x * s_trea, OF_CAP * u_trea))
    of_adj = floored - u_trea
    c02 = {("0010", "0010"): floored, ("0034", "0010"): u_trea, ("0035", "0010"): of_adj, ("0040", "0010"): credit,
           ("0050", "0010"): sa_rwea, ("0240", "0010"): irb_rwea, ("0490", "0010"): 0.0, ("0520", "0010"): mkt["RWEA"].sum(),
           ("0590", "0010"): opr["RWEA"], ("0640", "0010"): cv.attrs["RWEA"],
           ("0010", "0020"): s_trea, ("0040", "0020"): s_credit, ("0050", "0020"): sa_rwea, ("0240", "0020"): float(I["rwea_sa"].sum()),
           ("0520", "0020"): mkt["RWEA"].sum(), ("0590", "0020"): opr["RWEA"], ("0640", "0020"): cv.attrs["RWEA"]}
    c02rows = {"0010": "TOTAL RISK EXPOSURE AMOUNT", "0034": "Un-floored total risk exposure amount (U-TREA)", "0035": "Output floor adjustment",
               "0040": "Credit, counterparty credit and dilution risks and free deliveries", "0050": "Standardised approach (SA)",
               "0240": "Internal ratings based approach (IRB)", "0490": "Settlement / delivery", "0520": "Position, foreign exchange and commodities risks",
               "0590": "Operational risk", "0640": "Credit valuation adjustment"}
    cells += _cells("C 02.00", "", c02, c02rows, {"0010": "Risk exposure amount", "0020": "Standardised total exposure amount (output floor)"})
    for k, v in {("C 02.00", "0050", "0010"): "Σ C 07.00 {r0010; c0220} over exposure classes", ("C 02.00", "0240", "0010"): "Σ C 08.01 {r0010; c0260} over exposure classes",
                 ("C 02.00", "0520", "0010"): "C 18-23 {r0990; c0020}", ("C 02.00", "0590", "0010"): "C 16.01 {r0070; c0010}",
                 ("C 02.00", "0640", "0010"): "C 25.01 {r0010; c0060}", ("C 02.00", "0034", "0010"): "r0040 + r0490 + r0520 + r0590 + r0640",
                 ("C 02.00", "0010", "0010"): f"max(U-TREA, min({x:.0%} × S-TREA, {OF_CAP:.0%} × U-TREA))",
                 ("C 02.00", "0040", "0010"): "r0050 + r0240", ("C 02.00", "0240", "0020"): "Σ SA-equivalent RWEA of IRB exposures"}.items():
        formulas[k] = v
    # ------------------------------------------------ C 01.00 own funds
    el, prov = float(I.loc[~I["defaulted"], "el_irb"].sum() + I.loc[I["defaulted"], "el_irb"].sum()), float(I["provisions"].clip(lower=0).sum())
    shortfall = max(el - prov, 0.0)
    excess = min(max(prov - el, 0.0), 0.006 * irb_rwea)
    fv = float(e.loc[e["fair_value"], "net_exposure"].abs().sum() + rd.positions["net_position"].abs().sum())
    ava = 0.001 * fv
    o = of_items
    cet1_items = {"0040": o["Paid up capital instruments"], "0060": o["Share premium"], "0140": o["Retained earnings"],
                  "0150": o["Eligible interim profit"], "0180": o["Accumulated other comprehensive income"], "0200": o["Other reserves"],
                  "0230": o["Minority interests"], "0250": o["Cash flow hedge reserve (filter)"], "0290": -ava, "0300": o["Goodwill"],
                  "0340": o["Other intangible assets"], "0370": o["Deferred tax assets relying on future profitability"], "0380": -shortfall}
    cet1 = sum(cet1_items.values())
    at1 = o["AT1 instruments"]
    t2_items = {"0760": o["T2 instruments"] + o["T2 amortisation (final 5 years)"], "0890": excess}
    t2 = sum(t2_items.values())
    c01 = {("0010", "0010"): cet1 + at1 + t2, ("0015", "0010"): cet1 + at1, ("0020", "0010"): cet1,
           ("0030", "0010"): cet1_items["0040"] + cet1_items["0060"], ("0130", "0010"): cet1_items["0140"] + cet1_items["0150"],
           ("0530", "0010"): at1, ("0750", "0010"): t2}
    c01.update({(k, "0010"): v for k, v in cet1_items.items()})
    c01.update({(k, "0010"): v for k, v in t2_items.items()})
    c01rows = {"0010": "OWN FUNDS", "0015": "TIER 1 CAPITAL", "0020": "COMMON EQUITY TIER 1 CAPITAL", "0030": "Capital instruments eligible as CET1 capital",
               "0040": "Fully paid up capital instruments", "0060": "Share premium", "0130": "Retained earnings", "0140": "Previous years retained earnings",
               "0150": "Profit or loss eligible", "0180": "Accumulated other comprehensive income", "0200": "Other reserves", "0230": "Minority interests given recognition in CET1",
               "0250": "Adjustments to CET1 due to prudential filters", "0290": "(-) Value adjustments due to the requirements for prudent valuation",
               "0300": "(-) Goodwill", "0340": "(-) Other intangible assets", "0370": "(-) Deferred tax assets that rely on future profitability",
               "0380": "(-) IRB shortfall of credit risk adjustments to expected losses", "0530": "ADDITIONAL TIER 1 CAPITAL", "0750": "TIER 2 CAPITAL",
               "0760": "Capital instruments eligible as T2 capital (after amortisation)", "0890": "IRB excess of provisions over expected losses eligible"}
    order = ["0010", "0015", "0020", "0030", "0040", "0060", "0130", "0140", "0150", "0180", "0200", "0230", "0250", "0290", "0300", "0340", "0370", "0380",
             "0530", "0750", "0760", "0890"]
    cells += _cells("C 01.00", "", {(r, "0010"): c01[(r, "0010")] for r in order}, c01rows, {"0010": "Amount"})
    formulas.update({("C 01.00", "0010", "0010"): "r0015 + r0750", ("C 01.00", "0015", "0010"): "r0020 + r0530",
                     ("C 01.00", "0020", "0010"): "r0030 + r0130 + r0180 + r0200 + r0230 + r0250 + r0290 + r0300 + r0340 + r0370 + r0380",
                     ("C 01.00", "0290", "0010"): "0.1 % × Σ |fair-valued assets and positions| (simplified approach)",
                     ("C 01.00", "0380", "0010"): "−max(0, Σ C 08.01 {r0010; c0280} + Σ C 08.01 {r0010; c0290})",
                     ("C 01.00", "0890", "0010"): "min(max(0, provisions − EL), 0.6 % × IRB RWEA)"})
    # ------------------------------------------------ C 03.00 ratios
    c03 = {("0010", "0010"): cet1 / floored, ("0030", "0010"): (cet1 + at1) / floored, ("0050", "0010"): (cet1 + at1 + t2) / floored,
           ("0013", "0010"): cet1 / u_trea, ("0033", "0010"): (cet1 + at1) / u_trea, ("0053", "0010"): (cet1 + at1 + t2) / u_trea}
    c03rows = {"0010": "CET1 capital ratio", "0013": "CET1 capital ratio considering un-floored TREA", "0030": "T1 capital ratio",
               "0033": "T1 capital ratio considering un-floored TREA", "0050": "Total capital ratio", "0053": "Total capital ratio considering un-floored TREA"}
    cells += _cells("C 03.00", "", {(r, "0010"): c03[(r, "0010")] for r in ["0010", "0013", "0030", "0033", "0050", "0053"]}, c03rows, {"0010": "Ratio"})
    formulas.update({("C 03.00", "0010", "0010"): "C 01.00 {r0020} / C 02.00 {r0010}", ("C 03.00", "0013", "0010"): "C 01.00 {r0020} / C 02.00 {r0034}",
                     ("C 03.00", "0030", "0010"): "C 01.00 {r0015} / C 02.00 {r0010}", ("C 03.00", "0050", "0010"): "C 01.00 {r0010} / C 02.00 {r0010}"})
    # ------------------------------------------------ C 47.00 leverage
    lv = C.leverage_items(e).iloc[0]
    ded = -(o["Goodwill"] + o["Other intangible assets"] + o["Deferred tax assets relying on future profitability"]) + ava
    lvr = {("0010", "0010"): lv["derivatives – replacement cost × 1.4"], ("0020", "0010"): lv["derivatives – potential future exposure × 1.4"],
           ("0190", "0010"): lv["off-balance sheet items (leverage CCFs)"], ("0210", "0010"): lv["on-balance sheet items (carrying amount)"],
           ("0260", "0010"): -ded}
    tot = sum(lvr.values())
    lvr.update({("0290", "0010"): tot, ("0310", "0010"): cet1 + at1, ("0340", "0010"): (cet1 + at1) / tot})
    cells += _cells("C 47.00", "", lvr, {"0010": "Derivatives: replacement cost (SA-CCR, α = 1.4)", "0020": "Derivatives: potential future exposure (SA-CCR, α = 1.4)",
                                         "0190": "Off-balance sheet items (leverage CCFs, 10 % floor)", "0210": "Other assets (on-balance sheet)",
                                         "0260": "(-) Asset amounts deducted from Tier 1", "0290": "Total leverage ratio exposure",
                                         "0310": "Tier 1 capital", "0340": "Leverage ratio"}, {"0010": "Amount"})
    formulas.update({("C 47.00", "0310", "0010"): "C 01.00 {r0015}", ("C 47.00", "0340", "0010"): "r0310 / r0290"})
    # ------------------------------------------------ large exposures
    t1 = cet1 + at1
    le = C.large_exposures(e, t1)
    cells += _cells("C 26.00", "", {("0010", "0010"): 0.25 * t1, ("0020", "0010"): 0.25}, {"0010": "Non-institutions: limit (25 % of Tier 1)",
                                                                                         "0020": "Limit in % of Tier 1"}, {"0010": "Amount"})
    big = le[le["large exposure"]]
    d28, l28 = {}, {}
    for i, (gid, r) in enumerate(big.iterrows()):
        rc = f"{i + 1:03d}"
        l28[rc] = gid
        d28.update({(rc, "0080"): r["exposure"], (rc, "0210"): 0.0, (rc, "0310"): r["exposure"], (rc, "0320"): r["% of Tier 1 before CRM"],
                    (rc, "0330"): -r["crm"], (rc, "0340"): r["after CRM"], (rc, "0350"): r["% of Tier 1"]})
    cells += _cells("C 28.00", "", d28, l28, {"0080": "Original exposure", "0210": "(-) Exposures exempted", "0310": "Exposure value before CRM",
                                              "0320": "% of Tier 1 before CRM", "0330": "(-) Substitution effect of CRM", "0340": "Exposure value after CRM and exemptions",
                                              "0350": "% of Tier 1 after CRM"})
    x29 = e[e["group_id"].isin(big.index)].copy()
    x29["le_value"] = np.where(x29["bs_type"] == "deriv", 1.4 * (x29["rc"] + x29["pfe"]),
                               np.where(x29["bs_type"] == "off", x29["original_exposure"] * x29["ccf_bucket"].map(C.LEV_CCF).fillna(1.0).clip(lower=0.10), x29["original_exposure"]))
    contrib.append(pd.DataFrame({"template": "C 28.00", "sheet": "", "row": x29["group_id"].map({v: k for k, v in l28.items()}).to_numpy(), "col": "0310",
                                 "exposure_id": x29["exposure_id"].to_numpy(), "value": x29["le_value"].to_numpy()}))
    d29, l29 = {}, {}
    cli = x29.groupby(["group_id", "counterparty_id", "counterparty"])["le_value"].sum().reset_index()
    for i, r in cli.iterrows():
        rc = f"{i + 1:03d}"
        l29[rc] = f"{r['group_id']} / {r['counterparty']}"
        d29.update({(rc, "0080"): r["le_value"], (rc, "0320"): r["le_value"] / t1})
    cells += _cells("C 29.00", "", d29, l29, {"0080": "Original exposure", "0320": "% of Tier 1"})
    rep = Report(rd.ref_date, pd.DataFrame(cells), pd.concat(contrib, ignore_index=True), formulas, e,
                 aux={"market": mkt, "cva": cv, "opr": opr, "large_exposures": le, "floor_x": x, "s_trea": s_trea, "u_trea": u_trea,
                      "ava": ava, "shortfall": shortfall, "excess": excess, "lev_items": lv, "lines": L})
    return rep


def apply_adjustments(rep: Report, adj: pd.DataFrame) -> Report:
    """Top-side (manual) adjustments to reported cells: template, sheet, row, col, amount, comment. Dependent cells are NOT recomputed."""
    if adj is None or adj.empty:
        return rep
    cells = rep.cells.copy()
    for a in adj.itertuples():
        m = (cells["template"] == a.template) & (cells["sheet"] == (a.sheet or "")) & (cells["row"] == str(a.row)) & (cells["col"] == str(a.col))
        if m.any():
            cells.loc[m, "value"] += float(a.amount)
    return Report(rep.ref_date, cells, rep.contrib, rep.formulas, rep.exposures, rep.aux, adj)
