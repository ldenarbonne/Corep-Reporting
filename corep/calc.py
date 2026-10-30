"""
Own-funds requirement calculations at exposure level (CRR3, simplified where noted).

Standardised approach (Part Three, Title II, Chapter 2)
* CCF (art. 111, CRR3 buckets): 10 % unconditionally cancellable, 20 %, 40 % other commitments, 50 % medium risk,
  100 % full risk. Derivatives: SA-CCR exposure value 1.4 × (RC + PFE).
* Funded protection: financial collateral comprehensive method, E* = max(0, E − C·(1 − H_c)) (no currency mismatch).
* Unfunded protection: substitution — the guaranteed part moves to the guarantor's exposure class and risk weight
  (outflow / inflow columns of C 07.00).
* Risk weights: central governments by CQS (domestic EU sovereigns in own currency 0 %); institutions ECRA by CQS,
  unrated SCRA grade A 40 %; corporates by CQS, unrated 100 % (investment-grade option 65 % not used), unrated SME 85 %;
  retail 75 %; residential real estate loan-splitting (20 % up to 55 % LTV, counterparty RW above); income-producing
  commercial real estate 70 / 90 / 110 % by LTV; defaulted 100 % if specific provisions ≥ 20 % of the unsecured part,
  else 150 % (RRE 100 %); equity: 2026 transitional weight (art. 495a); other items 100 %, cash 0 %.
* SME supporting factor (art. 501): 0.7619 up to €2.5m per obligor, 0.85 above.

IRB approach (Part Three, Title II, Chapter 3)
* Corporate / institution risk-weight function with maturity adjustment, SME correlation adjustment (turnover €5-50m),
  1.25 correlation multiplier for large financial sector entities; retail: RRE R = 15 %, QRRE R = 4 %, other retail
  R = 3-16 %.
* CRR3 input floors: PD ≥ 0.05 % (QRRE revolvers 0.10 %); A-IRB LGD floors 25 % unsecured corporate, 5 % RRE,
  50 % QRRE, 30 % other retail. F-IRB (institutions, large corporates): supervisory LGD 45 % / 40 %.
* Defaulted exposures: RW = 12.5 × max(0, LGD − ELBE) with ELBE = LGD − 8 %.
* Expected loss EL = PD × LGD × EAD (defaulted: ELBE × EAD); shortfall / excess against provisions.
* SA-equivalent risk weight computed for every IRB exposure (output floor, art. 92(3)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from .data import CQS_OF_GRADE

CCF = {"": 1.0, "10% (unconditionally cancellable)": 0.10, "20% (trade-related)": 0.20, "40% (other commitments)": 0.40,
       "50% (medium risk)": 0.50, "100% (full risk)": 1.0}
CCF_LEV_FLOOR = 0.10
EQUITY_RW_2026 = 1.30            # art. 495a transitional schedule (illustrative)
SOV_RW = {1: 0.0, 2: 0.20, 3: 0.50, 4: 1.0, 5: 1.0, 6: 1.5}
INST_RW = {1: 0.20, 2: 0.30, 3: 0.50, 4: 1.0, 5: 1.0, 6: 1.5}
CORP_RW = {1: 0.20, 2: 0.50, 3: 0.75, 4: 1.0, 5: 1.5, 6: 1.5}
GUARANTOR_RW = {"Central governments or central banks": 0.0, "Institutions": 0.20}
LGD_FLOOR = {"Corporates - SME": 0.25, "Corporates - other general": 0.25, "Retail - secured by residential immovable property": 0.05,
             "Retail - qualifying revolving": 0.50, "Retail - other": 0.30}
FIRB_LGD = {"Institutions": 0.45, "Corporates - large corporates": 0.40}


def cqs(grade: str) -> int:
    if not grade or grade == "D":
        return 0
    return CQS_OF_GRADE.get(int(grade[1:]), 6)


def sa_risk_weight(r) -> float:
    c = r["sa_class"]
    q = cqs(r["grade"])
    if c == "Central governments or central banks":
        if r["counterparty_id"] == "ECB" or (r["country"] in ("FR", "DE", "NL", "ES", "IT", "BE", "LU") and r["currency"] == "EUR"):
            return 0.0
        return SOV_RW.get(q, 1.0)
    if c == "Institutions":
        return INST_RW.get(q, 1.0) if r["rated"] else 0.40
    if c == "Corporates":
        if r["rated"]:
            return CORP_RW.get(q, 1.0)
        return 0.85 if r["sme"] else 1.0
    if c == "Retail":
        return 0.75
    if c == "Secured by mortgages on immovable property":
        ltv = max(r["ltv"], 1e-6)
        if r["property_type"] == "Residential":
            return 0.20 * min(ltv, 0.55) / ltv + 0.75 * max(ltv - 0.55, 0) / ltv
        return 0.70 if ltv <= 0.60 else (0.90 if ltv <= 0.80 else 1.10)
    if c == "Exposures in default":
        unsecured = max(r["original_exposure"], 1.0)
        return 1.0 if (r["provisions"] / unsecured >= 0.20 or r["property_type"] == "Residential") else 1.5
    if c == "Equity exposures":
        return EQUITY_RW_2026
    if c == "Other items":
        return 0.0 if "Cash" in r["counterparty"] else 1.0
    return 1.0


def _sa_class_equivalent(r) -> str:
    """SA exposure class an IRB exposure would fall in (for the output floor)."""
    if r["defaulted"]:
        return "Exposures in default"
    m = {"Institutions": "Institutions", "Corporates - SME": "Corporates", "Corporates - large corporates": "Corporates",
         "Corporates - other general": "Corporates", "Retail - secured by residential immovable property": "Secured by mortgages on immovable property",
         "Retail - qualifying revolving": "Retail", "Retail - other": "Retail"}
    return m.get(r["irb_class"], r["sa_class"])


def irb_rw(pd_, lgd, M, irb_class, turnover, defaulted):
    if defaulted:
        return 12.5 * 0.08
    if irb_class.startswith("Retail"):
        if irb_class == "Retail - secured by residential immovable property":
            R = 0.15
        elif irb_class == "Retail - qualifying revolving":
            R = 0.04
        else:
            w = (1 - np.exp(-35 * pd_)) / (1 - np.exp(-35))
            R = 0.03 * w + 0.16 * (1 - w)
        k = lgd * norm.cdf((norm.ppf(pd_) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R)) - pd_ * lgd
        return 12.5 * k
    w = (1 - np.exp(-50 * pd_)) / (1 - np.exp(-50))
    R = 0.12 * w + 0.24 * (1 - w)
    if irb_class == "Corporates - SME" or (5 <= turnover <= 50):
        S = min(max(turnover, 5.0), 50.0)
        R -= 0.04 * (1 - (S - 5) / 45)
    if irb_class == "Institutions":
        R *= 1.25
    b = (0.11852 - 0.05478 * np.log(pd_)) ** 2
    ma = (1 + (min(max(M, 1.0), 5.0) - 2.5) * b) / (1 - 1.5 * b)
    k = (lgd * norm.cdf((norm.ppf(pd_) + np.sqrt(R) * norm.ppf(0.999)) / np.sqrt(1 - R)) - pd_ * lgd) * ma
    return 12.5 * k


def compute(exposures: pd.DataFrame) -> pd.DataFrame:
    """Adds all regulatory measures to the exposure table (one row per exposure)."""
    e = exposures.copy()
    e["ccf"] = np.where(e["bs_type"] == "off", e["ccf_bucket"].map(CCF).fillna(1.0), 1.0)
    e["net_exposure"] = np.where(e["bs_type"] == "deriv", 1.4 * (e["rc"] + e["pfe"]), e["original_exposure"] - e["provisions"].clip(lower=0))
    e.loc[e["bs_type"] == "off", "net_exposure"] = e.loc[e["bs_type"] == "off", "original_exposure"] - e.loc[e["bs_type"] == "off", "provisions"].clip(lower=0)
    e["gross_for_c07"] = np.where(e["bs_type"] == "deriv", e["net_exposure"], e["original_exposure"])
    # ---------------- SA view (actual SA exposures, and SA-equivalent of IRB exposures)
    e["sa_class_eff"] = np.where(e["approach"] == "IRB", e.apply(_sa_class_equivalent, axis=1), e["sa_class"])
    tmp = e.assign(sa_class=e["sa_class_eff"])
    e["rw_sa"] = tmp.apply(sa_risk_weight, axis=1)
    e["guar_cover"] = np.minimum(e["guarantee"], e["net_exposure"]).clip(lower=0)
    e["coll_adj"] = (e["fin_collateral"] * (1 - e["collateral_haircut"])).clip(lower=0)
    e["e_star"] = (e["net_exposure"] - e["guar_cover"] - e["coll_adj"]).clip(lower=0)
    e["ev_sa"] = e["e_star"] * e["ccf"]
    e["ev_guar"] = e["guar_cover"] * e["ccf"]
    e["rw_guar"] = e["guarantor_type"].map(GUARANTOR_RW).fillna(1.0)
    e["sme_sf"] = 1.0
    sme = e["sme"] & (e["sa_class_eff"] == "Corporates")
    tot = e.loc[sme].groupby("counterparty_id")["ev_sa"].transform("sum")
    e.loc[sme, "sme_sf"] = np.where(tot <= 2.5e6, 0.7619, (0.7619 * 2.5e6 + 0.85 * (tot - 2.5e6)) / tot.clip(lower=1))
    e["rwea_sa_pre_sf"] = e["ev_sa"] * e["rw_sa"] + e["ev_guar"] * e["rw_guar"]
    e["rwea_sa"] = e["ev_sa"] * e["rw_sa"] * e["sme_sf"] + e["ev_guar"] * e["rw_guar"]
    # ---------------- IRB view
    irb = e["approach"] == "IRB"
    pd_floor = np.where(e["irb_class"] == "Retail - qualifying revolving", 0.001, 0.0005)
    e["pd_irb"] = np.where(e["defaulted"], 1.0, np.maximum(e["pd"], pd_floor))
    lgd_in = e["lgd"].where(e["lgd"] > 0, 0.45)
    lgd_f = e["irb_class"].map(LGD_FLOOR).fillna(0.0)
    firb = e["irb_class"].map(FIRB_LGD)
    e["lgd_irb"] = np.where(firb.notna(), firb, np.maximum(lgd_in, lgd_f))
    e["ev_irb"] = np.where(irb, e["net_exposure"] * e["ccf"] + e["provisions"].clip(lower=0) * (e["bs_type"] == "on"), 0.0)   # IRB EAD gross of provisions
    rw = [irb_rw(p, l, m, c, t, d) if a else 0.0 for p, l, m, c, t, d, a in
          zip(e["pd_irb"], e["lgd_irb"], e["maturity"], e["irb_class"], e["turnover_m"], e["defaulted"], irb)]
    e["rw_irb"] = rw
    e["rwea_irb"] = e["ev_irb"] * e["rw_irb"]
    e["el_irb"] = np.where(irb, np.where(e["defaulted"], (e["lgd_irb"] - 0.08).clip(lower=0), e["pd_irb"] * e["lgd_irb"]) * e["ev_irb"], 0.0)
    # ---------------- reported view
    e["rwea"] = np.where(irb, e["rwea_irb"], e["rwea_sa"])
    e["exposure_value"] = np.where(irb, e["ev_irb"], e["ev_sa"] + e["ev_guar"])
    e["rwea_sa_equivalent"] = e["rwea_sa"]          # SA-equivalent for all (= actual for SA exposures)
    return e


def sa_lines(e: pd.DataFrame) -> pd.DataFrame:
    """C 07.00 lines: the non-guaranteed part in the obligor's class, the guaranteed part substituted into the guarantor's class."""
    sa = e[e["approach"] == "SA"]
    base = sa.assign(line_class=sa["sa_class"], part="obligor", ev=sa["ev_sa"], rw=sa["rw_sa"], rwea_line=sa["ev_sa"] * sa["rw_sa"] * sa["sme_sf"],
                     rwea_pre_sf=sa["ev_sa"] * sa["rw_sa"])
    g = sa[sa["guar_cover"] > 0]
    sub = g.assign(line_class=g["guarantor_type"], part="guarantor (inflow)", ev=g["ev_guar"], rw=g["rw_guar"], rwea_line=g["ev_guar"] * g["rw_guar"],
                   rwea_pre_sf=g["ev_guar"] * g["rw_guar"])
    return pd.concat([base, sub], ignore_index=True)


# ---------------------------------------------------------------- market risk (CRR2 standardised, still applicable in 2026)
GENERAL_IR_WEIGHT = {"0-1y": 0.0020, "1-3y": 0.0080, "3-5y": 0.0175, "5-10y": 0.0325, "10y+": 0.0525}
SPECIFIC_IR = {"Government (0 %)": 0.0, "Qualifying (1.6 %)": 0.016, "Other (8 %)": 0.08, "No specific risk": 0.0}


def market_risk(pos: pd.DataFrame, own_funds_total: float) -> pd.DataFrame:
    ir = pos[pos["risk_type"] == "Interest rate"]
    gen = abs((ir["net_position"] * ir["bucket"].map(GENERAL_IR_WEIGHT)).sum()) + 0.1 * (ir["net_position"].abs() * ir["bucket"].map(GENERAL_IR_WEIGHT)).sum()
    spec = (ir["net_position"].abs() * ir["category"].map(SPECIFIC_IR)).sum()
    eq = pos[pos["risk_type"] == "Equity"]
    eq_spec = 0.08 * eq["net_position"].abs().sum()
    eq_gen = 0.08 * abs(eq["net_position"].sum())
    fx = pos[pos["risk_type"] == "FX"]
    long_, short_ = fx.loc[fx["net_position"] > 0, "net_position"].sum(), -fx.loc[fx["net_position"] < 0, "net_position"].sum()
    net_fx = max(long_, short_)
    fx_req = 0.08 * net_fx if net_fx > 0.02 * own_funds_total else 0.0
    co = pos[pos["risk_type"] == "Commodity"]
    co_req = 0.15 * co["net_position"].abs().sum() + 0.03 * co["net_position"].abs().sum()
    rows = [("C 18.00", "Traded debt instruments – general risk", gen), ("C 18.00", "Traded debt instruments – specific risk", spec),
            ("C 21.00", "Equity – specific risk", eq_spec), ("C 21.00", "Equity – general risk", eq_gen),
            ("C 22.00", "Foreign exchange", fx_req), ("C 23.00", "Commodities", co_req)]
    out = pd.DataFrame(rows, columns=["template", "component", "own funds requirement"])
    out["RWEA"] = 12.5 * out["own funds requirement"]
    out.attrs["fx_net"] = net_fx
    return out


# ---------------------------------------------------------------- CVA (CRR3 reduced basic approach)
BA_CVA_RW = {"Financials": (0.05, 0.12), "Basic materials, energy, industrials": (0.03, 0.07), "Consumer goods and services": (0.03, 0.085),
             "Technology, telecommunications": (0.02, 0.055), "Health care, utilities": (0.015, 0.05), "Sovereigns": (0.005, 0.03), "Other": (0.05, 0.12)}


def cva(e: pd.DataFrame, rho: float = 0.5, ds: float = 0.65) -> pd.DataFrame:
    d = e[e["bs_type"] == "deriv"].copy()
    d["investment grade"] = d["grade"].map(lambda g: bool(g) and g != "D" and int(g[1:]) <= 6)
    d["RW"] = [BA_CVA_RW.get(s or "Other", BA_CVA_RW["Other"])[0 if ig else 1] for s, ig in zip(d["cva_sector"], d["investment grade"])]
    M = d["maturity"].clip(lower=1.0)
    d["DF"] = (1 - np.exp(-0.05 * M)) / (0.05 * M)
    d["EAD"] = 1.4 * (d["rc"] + d["pfe"])
    g = d.groupby(["counterparty_id", "counterparty", "cva_sector", "investment grade"], as_index=False).apply(
        lambda x: pd.Series({"EAD": x["EAD"].sum(), "M": (x["maturity"] * x["EAD"]).sum() / x["EAD"].sum(), "RW": x["RW"].iloc[0],
                             "SCVA": x["RW"].iloc[0] * (x["maturity"].clip(lower=1) * x["EAD"] * x["DF"]).sum() / 1.4}), include_groups=False)
    K = np.sqrt((rho * g["SCVA"].sum()) ** 2 + (1 - rho ** 2) * (g["SCVA"] ** 2).sum())
    g.attrs.update(K_reduced=K, own_funds=ds * K, RWEA=12.5 * ds * K)
    return g


# ---------------------------------------------------------------- operational risk (CRR3 business indicator, art. 314)
def operational_risk(pnl: pd.DataFrame) -> dict:
    ildc = np.minimum((pnl["interest income"] - pnl["interest expense"]).abs(), 0.0225 * pnl["interest earning assets"]) + pnl["dividend income"]
    sc = np.maximum(pnl["other operating income"], pnl["other operating expense"]) + np.maximum(pnl["fee and commission income"], pnl["fee and commission expense"])
    fc = pnl["net P&L trading book"].abs() + pnl["net P&L banking book"].abs()
    bi_y = ildc + sc + fc
    BI = float(bi_y.mean())
    bic = 0.12 * min(BI, 1e9) + 0.15 * min(max(BI - 1e9, 0), 29e9) + 0.18 * max(BI - 30e9, 0)
    return {"ILDC": float(ildc.mean()), "SC": float(sc.mean()), "FC": float(fc.mean()), "BI": BI, "BI by year": bi_y,
            "BIC": bic, "own funds": bic, "RWEA": 12.5 * bic}


# ---------------------------------------------------------------- leverage (C 47.00)
LEV_CCF = {"10% (unconditionally cancellable)": 0.10, "20% (trade-related)": 0.20, "40% (other commitments)": 0.40,
           "50% (medium risk)": 0.50, "100% (full risk)": 1.0}


def leverage_items(e: pd.DataFrame) -> pd.DataFrame:
    on = e[e["bs_type"] == "on"]
    off = e[e["bs_type"] == "off"]
    der = e[e["bs_type"] == "deriv"]
    return pd.DataFrame({
        "on-balance sheet items (carrying amount)": [(on["original_exposure"] - on["provisions"].clip(lower=0)).sum()],
        "derivatives – replacement cost × 1.4": [1.4 * der["rc"].sum()],
        "derivatives – potential future exposure × 1.4": [1.4 * der["pfe"].sum()],
        "off-balance sheet items (leverage CCFs)": [(off["original_exposure"] * off["ccf_bucket"].map(LEV_CCF).fillna(1.0).clip(lower=CCF_LEV_FLOOR)).sum()],
    })


# ---------------------------------------------------------------- large exposures (C 28 / C 29)
def large_exposures(e: pd.DataFrame, tier1: float) -> pd.DataFrame:
    x = e[~e["segment"].isin(["Other", "Retail"]) & (e["sa_class"] != "Central governments or central banks")].copy()
    x["le_value"] = np.where(x["bs_type"] == "deriv", 1.4 * (x["rc"] + x["pfe"]),
                             np.where(x["bs_type"] == "off", x["original_exposure"] * x["ccf_bucket"].map(LEV_CCF).fillna(1.0).clip(lower=0.10),
                                      x["original_exposure"]))
    x["crm"] = x["guar_cover"] + x["coll_adj"]
    g = x.groupby("group_id").agg(clients=("counterparty_id", "nunique"), exposure=("le_value", "sum"), crm=("crm", "sum"))
    g["after CRM"] = (g["exposure"] - g["crm"]).clip(lower=0)
    g["% of Tier 1"] = g["after CRM"] / tier1
    g["% of Tier 1 before CRM"] = g["exposure"] / tier1
    g["large exposure"] = g["% of Tier 1 before CRM"] >= 0.10
    g["limit breach"] = g["% of Tier 1"] > 0.25
    return g.sort_values("exposure", ascending=False)
