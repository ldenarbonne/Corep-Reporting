"""
Synthetic reporting data of a mid-sized EU bank at two reference dates (30 June and 30 September 2026).

Tables (one set per reference date)
* exposures : line-level credit exposures (loans, mortgages, revolving facilities, guarantees given, undrawn commitments,
              bonds, derivatives, equities, other assets) with the attributes needed by the SA and IRB calculations.
* positions : trading-book positions for market risk (interest rate, equity, FX, commodities).
* own_funds : accounting and regulatory own-funds items.
* pnl_bi    : three years of P&L items for the operational-risk business indicator.

Q3 is derived from Q2 with the movements a reporting team has to explain: new business, repayments and maturities,
rating migrations (including a downgrade cluster in commercial real estate), new defaults, a USD/EUR move, a profit
for the quarter, and an IRB model change on one portfolio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REF_DATES = [pd.Timestamp("2026-06-30"), pd.Timestamp("2026-09-30")]
COUNTRIES = ["FR", "DE", "IT", "ES", "NL", "BE", "LU", "GB", "US"]
FX = {pd.Timestamp("2026-06-30"): 1.08, pd.Timestamp("2026-09-30"): 1.04}       # USD per EUR

SA_CLASSES = ["Central governments or central banks", "Institutions", "Corporates", "Retail",
              "Secured by mortgages on immovable property", "Exposures in default", "Equity exposures", "Other items"]
IRB_CLASSES = ["Institutions", "Corporates - SME", "Corporates - large corporates", "Corporates - other general",
               "Retail - secured by residential immovable property", "Retail - qualifying revolving", "Retail - other"]

# master scale of the bank: grade → PD
GRADES = pd.DataFrame({"grade": [f"G{i:02d}" for i in range(1, 14)] + ["D"],
                       "pd": [0.0003, 0.0005, 0.0008, 0.0012, 0.0020, 0.0035, 0.0060, 0.0100, 0.0170, 0.0300, 0.0550, 0.1000, 0.2000, 1.0]})
CQS_OF_GRADE = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5, 11: 6, 12: 6, 13: 6}

EXPOSURE_COLS = ["exposure_id", "ref_date", "counterparty_id", "counterparty", "group_id", "country", "currency", "segment",
                 "approach", "sa_class", "irb_class", "product", "bs_type", "ccf_bucket", "original_exposure", "drawn",
                 "provisions", "fin_collateral", "collateral_haircut", "guarantee", "guarantor_type", "grade", "pd", "lgd",
                 "maturity", "defaulted", "ltv", "property_type", "sme", "turnover_m", "rated", "rc", "pfe", "cva_sector",
                 "stage", "fair_value"]


@dataclass
class ReportingData:
    ref_date: pd.Timestamp
    exposures: pd.DataFrame
    positions: pd.DataFrame
    own_funds: pd.Series
    pnl_bi: pd.DataFrame
    fx: float


def _grade_idx(rng, mean, n):
    g = np.clip(np.round(rng.normal(mean, 1.6, n)), 1, 13).astype(int)
    return g


def _q2(seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    k = 0

    def add(n, **kw):
        nonlocal k
        for i in range(n):
            r = {c: kw.get(c) for c in EXPOSURE_COLS}
            for c, v in kw.items():
                r[c] = v(i) if callable(v) else v
            r["exposure_id"] = f"E{k:06d}"
            k += 1
            rows.append(r)

    # --- counterparties
    n_corp, n_sme, n_inst = 220, 420, 45
    corp_grade = _grade_idx(rng, 6.5, n_corp)
    sme_grade = _grade_idx(rng, 8.0, n_sme)
    inst_grade = _grade_idx(rng, 4.0, n_inst)
    groups = [f"GRP{g:03d}" for g in rng.integers(0, 150, n_corp)]
    # a few big groups to create large exposures
    for j in range(6):
        groups[j * 3:(j * 3) + 3] = [f"GRP9{j:02d}"] * 3
    # sovereigns (SA, mostly 0 %) and central bank
    for c, amt, cqs in [("FR", 9.0e9, 1), ("DE", 3.5e9, 1), ("IT", 2.8e9, 3), ("ES", 2.0e9, 2), ("NL", 1.0e9, 1), ("US", 1.2e9, 1)]:
        add(1, counterparty_id=f"SOV_{c}", counterparty=f"Republic {c}", group_id=f"SOV_{c}", country=c, currency="USD" if c == "US" else "EUR",
            segment="Sovereign", approach="SA", sa_class="Central governments or central banks", irb_class="", product="Bond", bs_type="on",
            ccf_bucket="", original_exposure=amt, drawn=amt, provisions=amt * 0.00005, grade=f"G{cqs * 2 - 1:02d}", rated=True,
            maturity=float(rng.uniform(2, 9)), defaulted=False, sme=False, stage=1, fair_value=True)
    add(1, counterparty_id="ECB", counterparty="Banque de France (reserves)", group_id="ECB", country="FR", currency="EUR", segment="Sovereign",
        approach="SA", sa_class="Central governments or central banks", irb_class="", product="Other asset", bs_type="on", ccf_bucket="",
        original_exposure=7.5e9, drawn=7.5e9, provisions=0.0, grade="G01", rated=True, maturity=0.1, defaulted=False, sme=False, stage=1, fair_value=False)
    # institutions: IRB (F-IRB) placements and derivatives
    for i in range(n_inst):
        g = int(inst_grade[i])
        cid = f"INS{i:03d}"
        ctry = rng.choice(COUNTRIES, p=[0.22, 0.18, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.1])
        amt = float(rng.lognormal(np.log(60e6), 0.7))
        add(1, counterparty_id=cid, counterparty=f"Bank {i:03d}", group_id=f"BKG{i:03d}", country=ctry, currency="USD" if ctry == "US" else "EUR",
            segment="Institution", approach="IRB", sa_class="Institutions", irb_class="Institutions", product="Loan", bs_type="on", ccf_bucket="",
            original_exposure=amt, drawn=amt, provisions=amt * 0.0002, grade=f"G{g:02d}", lgd=0.45, maturity=float(rng.uniform(0.3, 3)),
            defaulted=False, sme=False, rated=True, stage=1, fair_value=False)
        if rng.random() < 0.7:
            rc = float(rng.lognormal(np.log(15e6), 0.9))
            add(1, counterparty_id=cid, counterparty=f"Bank {i:03d}", group_id=f"BKG{i:03d}", country=ctry, currency="EUR", segment="Institution",
                approach="IRB", sa_class="Institutions", irb_class="Institutions", product="Derivative", bs_type="deriv", ccf_bucket="",
                original_exposure=0.0, drawn=0.0, provisions=0.0, rc=rc, pfe=rc * float(rng.uniform(0.4, 1.2)), grade=f"G{g:02d}", lgd=0.45,
                maturity=float(rng.uniform(1, 5)), defaulted=False, sme=False, rated=True, cva_sector="Financials", stage=1, fair_value=True)
    # large and mid corporates (IRB) — loans, revolvers, guarantees, derivatives
    for i in range(n_corp):
        g = int(corp_grade[i])
        cid = f"COR{i:04d}"
        ctry = rng.choice(COUNTRIES[:8], p=[0.4, 0.15, 0.12, 0.1, 0.07, 0.06, 0.04, 0.06])
        large = rng.random() < 0.45
        turnover = float(rng.uniform(550, 25000) if large else rng.uniform(55, 500))
        size = rng.lognormal(np.log(70e6 if large else 25e6), 0.8)
        cls = "Corporates - large corporates" if large else "Corporates - other general"
        sec = rng.choice(["Basic materials, energy, industrials", "Consumer goods and services", "Technology, telecommunications", "Health care, utilities"])
        dflt = g == 13 and rng.random() < 0.5
        common = dict(counterparty_id=cid, counterparty=f"Corporate {i:04d}", group_id=groups[i], country=ctry,
                      currency="USD" if (large and rng.random() < 0.2) else "EUR",
                      segment="Corporate", approach="IRB", sa_class="Corporates", irb_class=cls, grade="D" if dflt else f"G{g:02d}",
                      defaulted=dflt, sme=False, turnover_m=turnover, rated=bool(large and rng.random() < 0.6), stage=3 if dflt else (2 if g >= 10 else 1),
                      fair_value=False, cva_sector=sec)
        lgd_ = 0.40 if large else float(np.clip(rng.normal(0.33, 0.07), 0.25, 0.5))
        amt = float(size)
        add(1, **common, product="Loan", bs_type="on", ccf_bucket="", original_exposure=amt, drawn=amt,
            provisions=amt * (0.45 if dflt else (0.02 if g >= 10 else 0.002)), lgd=lgd_, maturity=float(rng.uniform(1, 7)),
            fin_collateral=amt * 0.2 if rng.random() < 0.15 else 0.0, collateral_haircut=0.10)
        if rng.random() < 0.6:
            lim = float(size * rng.uniform(0.5, 1.2))
            add(1, **common, product="Revolving facility", bs_type="off", ccf_bucket="40% (other commitments)", original_exposure=lim, drawn=0.0,
                provisions=lim * 0.001, lgd=lgd_, maturity=float(rng.uniform(1, 5)))
        if rng.random() < 0.25:
            ga = float(size * rng.uniform(0.1, 0.4))
            add(1, **common, product="Guarantee given", bs_type="off", ccf_bucket="50% (medium risk)", original_exposure=ga, drawn=0.0,
                provisions=0.0, lgd=lgd_, maturity=float(rng.uniform(1, 4)))
        if large and rng.random() < 0.35:
            rc = float(rng.lognormal(np.log(4e6), 0.8))
            add(1, **common, product="Derivative", bs_type="deriv", ccf_bucket="", original_exposure=0.0, drawn=0.0, provisions=0.0, rc=rc,
                pfe=rc * float(rng.uniform(0.5, 1.5)), lgd=lgd_, maturity=float(rng.uniform(1, 6)))
    # SMEs: SA for half of the book (permanent partial use), IRB for the rest
    for i in range(n_sme):
        g = int(sme_grade[i])
        cid = f"SME{i:04d}"
        ctry = rng.choice(["FR", "BE", "LU", "DE"], p=[0.7, 0.12, 0.08, 0.1])
        irb = i % 2 == 0
        dflt = g == 13 and rng.random() < 0.6
        amt = float(rng.lognormal(np.log(1.6e6), 0.8))
        guar = amt * 0.5 if (not irb and rng.random() < 0.12) else 0.0
        common = dict(counterparty_id=cid, counterparty=f"SME {i:04d}", group_id=f"SMG{i:04d}", country=ctry, currency="EUR", segment="SME",
                      approach="IRB" if irb else "SA", sa_class="Exposures in default" if (dflt and not irb) else "Corporates",
                      irb_class="Corporates - SME", grade="D" if dflt else f"G{g:02d}", defaulted=dflt, sme=True,
                      turnover_m=float(rng.uniform(2, 50)), rated=False, stage=3 if dflt else (2 if g >= 10 else 1), fair_value=False)
        add(1, **common, product="Loan", bs_type="on", ccf_bucket="", original_exposure=amt, drawn=amt,
            provisions=amt * (0.40 if dflt else (0.025 if g >= 10 else 0.004)), lgd=float(np.clip(rng.normal(0.30, 0.08), 0.25, 0.55)),
            maturity=float(rng.uniform(1, 6)), guarantee=guar, guarantor_type="Central governments or central banks" if guar else "",
            fin_collateral=0.0, collateral_haircut=0.0)
        if rng.random() < 0.3:
            lim = amt * float(rng.uniform(0.2, 0.6))
            add(1, **common | {"sa_class": "Corporates" if not dflt else common["sa_class"]}, product="Undrawn commitment", bs_type="off",
                ccf_bucket="10% (unconditionally cancellable)", original_exposure=lim, drawn=0.0, provisions=0.0, lgd=0.35, maturity=1.0)
    # commercial real estate (SA, secured by immovable property)
    for i in range(140):
        g = int(_grade_idx(rng, 8.5, 1)[0])
        dflt = g == 13 and rng.random() < 0.5
        amt = float(rng.lognormal(np.log(12e6), 0.7))
        ctry = rng.choice(["FR", "DE", "NL", "ES", "IT"], p=[0.5, 0.2, 0.1, 0.1, 0.1])
        add(1, counterparty_id=f"CRE{i:04d}", counterparty=f"Property company {i:04d}", group_id=f"PRG{i // 3:03d}", country=ctry, currency="EUR",
            segment="Real estate", approach="SA", sa_class="Exposures in default" if dflt else "Secured by mortgages on immovable property",
            irb_class="", product="Mortgage", bs_type="on", ccf_bucket="", original_exposure=amt, drawn=amt,
            provisions=amt * (0.30 if dflt else (0.015 if g >= 10 else 0.003)), grade="D" if dflt else f"G{g:02d}", defaulted=dflt, sme=False,
            ltv=float(np.clip(rng.normal(0.62, 0.15), 0.25, 1.1)), property_type="Commercial – income producing", maturity=float(rng.uniform(3, 10)),
            rated=False, stage=3 if dflt else (2 if g >= 10 else 1), fair_value=False)
    # retail: residential mortgages (IRB), credit cards (QRRE), consumer loans (IRB other retail); pooled by bucket
    for i in range(900):
        g = int(_grade_idx(rng, 5.5, 1)[0])
        dflt = g == 13 and rng.random() < 0.6
        amt = float(rng.lognormal(np.log(30e6), 0.35))          # pool of homogeneous loans
        ctry = rng.choice(["FR", "BE", "LU"], p=[0.85, 0.1, 0.05])
        add(1, counterparty_id=f"RRE{i:04d}", counterparty=f"Mortgage pool {i:04d}", group_id=f"RTL{i:04d}", country=ctry, currency="EUR",
            segment="Retail", approach="IRB", sa_class="Secured by mortgages on immovable property", irb_class="Retail - secured by residential immovable property",
            product="Mortgage", bs_type="on", ccf_bucket="", original_exposure=amt, drawn=amt, provisions=amt * (0.15 if dflt else 0.0008),
            grade="D" if dflt else f"G{g:02d}", lgd=float(np.clip(rng.normal(0.14, 0.04), 0.06, 0.3)), maturity=20.0, defaulted=dflt, sme=False,
            ltv=float(np.clip(rng.normal(0.68, 0.14), 0.2, 1.05)), property_type="Residential", rated=False, stage=3 if dflt else (2 if g >= 10 else 1),
            fair_value=False)
    for i in range(160):
        g = int(_grade_idx(rng, 7.5, 1)[0])
        dflt = g == 13 and rng.random() < 0.6
        lim = float(rng.lognormal(np.log(14e6), 0.3))
        dr = lim * float(rng.uniform(0.25, 0.45))
        add(1, counterparty_id=f"CCD{i:04d}", counterparty=f"Card pool {i:04d}", group_id=f"RTC{i:04d}", country="FR", currency="EUR", segment="Retail",
            approach="IRB", sa_class="Retail", irb_class="Retail - qualifying revolving", product="Revolving facility", bs_type="on", ccf_bucket="",
            original_exposure=dr, drawn=dr, provisions=dr * (0.6 if dflt else 0.02), grade="D" if dflt else f"G{g:02d}", lgd=0.75, maturity=1.0,
            defaulted=dflt, sme=False, rated=False, stage=3 if dflt else (2 if g >= 10 else 1), fair_value=False)
        add(1, counterparty_id=f"CCD{i:04d}", counterparty=f"Card pool {i:04d}", group_id=f"RTC{i:04d}", country="FR", currency="EUR", segment="Retail",
            approach="IRB", sa_class="Retail", irb_class="Retail - qualifying revolving", product="Undrawn commitment", bs_type="off",
            ccf_bucket="10% (unconditionally cancellable)", original_exposure=lim - dr, drawn=0.0, provisions=0.0, grade="D" if dflt else f"G{g:02d}",
            lgd=0.75, maturity=1.0, defaulted=dflt, sme=False, rated=False, stage=1, fair_value=False)
    for i in range(300):
        g = int(_grade_idx(rng, 8.0, 1)[0])
        dflt = g == 13 and rng.random() < 0.6
        amt = float(rng.lognormal(np.log(9e6), 0.35))
        add(1, counterparty_id=f"CON{i:04d}", counterparty=f"Consumer pool {i:04d}", group_id=f"RTO{i:04d}", country=rng.choice(["FR", "BE"], p=[0.8, 0.2]),
            currency="EUR", segment="Retail", approach="IRB", sa_class="Retail", irb_class="Retail - other", product="Loan", bs_type="on",
            ccf_bucket="", original_exposure=amt, drawn=amt, provisions=amt * (0.55 if dflt else 0.01), grade="D" if dflt else f"G{g:02d}",
            lgd=float(np.clip(rng.normal(0.42, 0.08), 0.3, 0.7)), maturity=3.0, defaulted=dflt, sme=False, rated=False,
            stage=3 if dflt else (2 if g >= 10 else 1), fair_value=False)
    # SA retail (subsidiary not on IRB)
    for i in range(120):
        dflt = rng.random() < 0.03
        amt = float(rng.lognormal(np.log(6e6), 0.4))
        add(1, counterparty_id=f"SRT{i:04d}", counterparty=f"Retail pool (SA) {i:04d}", group_id=f"SRT{i:04d}", country="LU", currency="EUR",
            segment="Retail", approach="SA", sa_class="Exposures in default" if dflt else "Retail", irb_class="", product="Loan", bs_type="on", ccf_bucket="",
            original_exposure=amt, drawn=amt, provisions=amt * (0.35 if dflt else 0.008), grade="D" if dflt else "G08", defaulted=dflt, sme=False,
            rated=False, maturity=3.0, stage=3 if dflt else 1, fair_value=False)
    # equities and other items
    for i in range(12):
        v = float(rng.lognormal(np.log(25e6), 0.6))
        add(1, counterparty_id=f"EQ{i:03d}", counterparty=f"Equity holding {i:03d}", group_id=f"EQ{i:03d}", country="FR", currency="EUR",
            segment="Equity", approach="SA", sa_class="Equity exposures", irb_class="", product="Equity", bs_type="on", ccf_bucket="",
            original_exposure=v, drawn=v, provisions=0.0, grade="", defaulted=False, sme=False, rated=False, maturity=1.0, stage=1, fair_value=True)
    for nm, v in [("Tangible assets", 0.9e9), ("Other assets", 1.4e9), ("Cash in hand", 0.3e9)]:
        add(1, counterparty_id=nm.upper().replace(" ", "_"), counterparty=nm, group_id="OTHER", country="FR", currency="EUR", segment="Other",
            approach="SA", sa_class="Other items", irb_class="", product="Other asset", bs_type="on", ccf_bucket="", original_exposure=v, drawn=v,
            provisions=0.0, grade="", defaulted=False, sme=False, rated=False, maturity=1.0, stage=1, fair_value=False)
    df = pd.DataFrame(rows)
    num = ["original_exposure", "drawn", "provisions", "fin_collateral", "collateral_haircut", "guarantee", "lgd", "maturity", "ltv", "turnover_m", "rc", "pfe"]
    for c in num:
        df[c] = pd.to_numeric(df[c]).fillna(0.0)
    for c in ["guarantor_type", "property_type", "cva_sector", "irb_class", "ccf_bucket", "grade"]:
        df[c] = df[c].fillna("")
    df["defaulted"] = df["defaulted"].fillna(False).astype(bool)
    df["sme"] = df["sme"].fillna(False).astype(bool)
    df["rated"] = df["rated"].fillna(False).astype(bool)
    df["fair_value"] = df["fair_value"].fillna(False).astype(bool)
    df["stage"] = df["stage"].fillna(1).astype(int)
    pdm = GRADES.set_index("grade")["pd"]
    df["pd"] = df["grade"].map(pdm).fillna(0.0)
    df.loc[df["approach"] == "SA", "lgd"] = df.loc[df["approach"] == "SA", "lgd"].where(df["lgd"] > 0, 0.0)
    df["ref_date"] = REF_DATES[0]
    # USD exposures are expressed in EUR at the reference-date rate; keep the USD amount for FX analysis
    df["amount_ccy"] = np.where(df["currency"] == "USD", df["original_exposure"] * FX[REF_DATES[0]], df["original_exposure"])
    return df


def _roll_forward(q2: pd.DataFrame, seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    q3 = q2.copy()
    q3["ref_date"] = REF_DATES[1]
    n = len(q3)
    # repayments: amortising loans −3 % on average; maturities: 4 % of the lines disappear
    amort = q3["product"].isin(["Loan", "Mortgage"]) & ~q3["counterparty_id"].isin(["ECB"]) & ~q3["segment"].eq("Sovereign")
    q3.loc[amort, ["original_exposure", "drawn"]] = q3.loc[amort, ["original_exposure", "drawn"]].mul(rng.uniform(0.93, 0.995, amort.sum()), axis=0)
    mature = rng.random(n) < 0.04
    mature &= ~q3["segment"].isin(["Sovereign", "Other", "Equity"])
    q3 = q3[~mature].copy()
    # rating migrations: 8 % of graded non-defaulted lines move one or two notches, mostly downwards; CRE cluster downgraded
    pdm = GRADES.set_index("grade")["pd"]
    graded = q3["grade"].str.match(r"G\d\d") & ~q3["defaulted"]
    mig = graded & (rng.random(len(q3)) < 0.08)
    gnum = q3.loc[mig, "grade"].str[1:].astype(int)
    step = rng.choice([-1, 1, 1, 2], mig.sum())
    q3.loc[mig, "grade"] = [f"G{int(np.clip(a + b, 1, 13)):02d}" for a, b in zip(gnum, step)]
    cre = (q3["segment"] == "Real estate") & graded
    cre_down = cre & (rng.random(len(q3)) < 0.35)
    q3.loc[cre_down, "grade"] = [f"G{min(int(a[1:]) + 2, 13):02d}" for a in q3.loc[cre_down, "grade"]]
    q3.loc[cre, "ltv"] = q3.loc[cre, "ltv"] * 1.06                      # property values −6 %
    # new defaults: grade-13 and some grade-12 lines default
    newdef = graded & (q3["grade"].isin(["G13"]) & (rng.random(len(q3)) < 0.5) | (q3["grade"].isin(["G12"]) & (rng.random(len(q3)) < 0.08)))
    newdef &= q3["bs_type"] != "deriv"
    q3.loc[newdef, "grade"] = "D"
    q3.loc[newdef, "defaulted"] = True
    q3.loc[newdef, "stage"] = 3
    q3.loc[newdef, "provisions"] = q3.loc[newdef, "original_exposure"] * 0.35
    q3.loc[newdef & (q3["approach"] == "SA") & (q3["bs_type"] == "on"), "sa_class"] = "Exposures in default"
    q3["pd"] = q3["grade"].map(pdm).fillna(0.0)
    q3.loc[q3["grade"].isin(["G10", "G11", "G12"]) & (q3["stage"] == 1), "stage"] = 2
    # IRB model change: consumer loans re-calibrated LGD (−4 pp)
    con = q3["irb_class"] == "Retail - other"
    q3.loc[con, "lgd"] = (q3.loc[con, "lgd"] - 0.04).clip(lower=0.30)
    # FX: USD amounts constant in USD, EUR value moves with the rate
    usd = q3["currency"] == "USD"
    q3.loc[usd, "original_exposure"] = q3.loc[usd, "amount_ccy"] / FX[REF_DATES[1]]
    q3.loc[usd, "drawn"] = np.where(q3.loc[usd, "bs_type"] == "on", q3.loc[usd, "original_exposure"], q3.loc[usd, "drawn"])
    # new business: 150 new lines cloned from existing ones (new ids)
    pool = q3[q3["segment"].isin(["Corporate", "SME", "Retail", "Real estate"]) & ~q3["defaulted"]]
    new = pool.sample(150, random_state=seed).copy()
    new["exposure_id"] = [f"N{i:06d}" for i in range(len(new))]
    new[["original_exposure", "drawn"]] = new[["original_exposure", "drawn"]].mul(rng.uniform(0.6, 1.4, len(new)), axis=0)
    new["provisions"] = new["original_exposure"] * 0.002
    q3 = pd.concat([q3, new], ignore_index=True)
    q3["amount_ccy"] = np.where(q3["currency"] == "USD", q3["original_exposure"] * FX[REF_DATES[1]], q3["original_exposure"])
    return q3


def _positions(date_idx: int, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed + date_idx)
    rows = []
    buckets = [(0.5, "0-1y"), (2, "1-3y"), (4, "3-5y"), (7, "5-10y"), (15, "10y+")]
    for issuer, cat in [("Sovereign FR", "Government (0 %)"), ("Sovereign IT", "Qualifying (1.6 %)"), ("Corporate IG basket", "Qualifying (1.6 %)"),
                        ("Corporate HY basket", "Other (8 %)"), ("Swap book", "No specific risk")]:
        for m, b in buckets:
            rows.append(("Interest rate", issuer, cat, b, m, float(rng.normal(0, 60e6) * (1 + 0.1 * date_idx)), "EUR"))
    for nm in ["Euro Stoxx 50 basket", "CAC 40 basket", "Single stocks"]:
        rows.append(("Equity", nm, "Equity", "", 0, float(rng.normal(40e6, 25e6)), "EUR"))
        rows.append(("Equity", nm + " (short)", "Equity", "", 0, -float(abs(rng.normal(20e6, 10e6))), "EUR"))
    for ccy, v in [("USD", 95e6), ("GBP", -35e6), ("CHF", 20e6), ("JPY", -15e6), ("Gold", 8e6)]:
        rows.append(("FX", ccy, "Currency", "", 0, v * (1 + 0.15 * date_idx * (ccy == "USD")), ccy))
    for c, v in [("Energy", 18e6), ("Metals", -9e6), ("Agricultural", 5e6)]:
        rows.append(("Commodity", c, "Commodity", "", 0, v, "EUR"))
    return pd.DataFrame(rows, columns=["risk_type", "instrument", "category", "bucket", "maturity", "net_position", "currency"])


def _own_funds(date_idx: int) -> pd.Series:
    base = {"Paid up capital instruments": 1.00e9, "Share premium": 0.60e9, "Retained earnings": 2.95e9,
            "Accumulated other comprehensive income": -0.12e9, "Other reserves": 0.20e9, "Eligible interim profit": 0.0,
            "Minority interests": 0.05e9, "Goodwill": -0.20e9, "Other intangible assets": -0.15e9,
            "Deferred tax assets relying on future profitability": -0.08e9, "Cash flow hedge reserve (filter)": 0.02e9,
            "AT1 instruments": 0.60e9, "T2 instruments": 0.90e9, "T2 amortisation (final 5 years)": -0.08e9}
    if date_idx == 1:
        base["Eligible interim profit"] = 0.11e9           # Q3 profit net of foreseeable dividends, with supervisory permission
        base["Accumulated other comprehensive income"] = -0.15e9
        base["Other intangible assets"] = -0.16e9
        base["T2 amortisation (final 5 years)"] = -0.11e9
    return pd.Series(base)


def _pnl_bi() -> pd.DataFrame:
    return pd.DataFrame({
        "interest income": [2.40e9, 2.70e9, 2.80e9], "interest expense": [1.00e9, 1.30e9, 1.35e9],
        "interest earning assets": [78e9, 80e9, 82e9], "dividend income": [0.03e9, 0.03e9, 0.04e9],
        "fee and commission income": [0.78e9, 0.80e9, 0.84e9], "fee and commission expense": [0.20e9, 0.21e9, 0.22e9],
        "other operating income": [0.08e9, 0.07e9, 0.09e9], "other operating expense": [0.10e9, 0.12e9, 0.11e9],
        "net P&L trading book": [0.18e9, 0.22e9, 0.20e9], "net P&L banking book": [0.04e9, -0.03e9, 0.05e9]},
        index=["Y-3", "Y-2", "Y-1"])


def generate(seed: int = 11) -> dict[pd.Timestamp, ReportingData]:
    q2 = _q2(seed)
    q3 = _roll_forward(q2, seed + 1)
    out = {}
    for i, (d, df) in enumerate(zip(REF_DATES, [q2, q3])):
        out[d] = ReportingData(d, df.reset_index(drop=True), _positions(i), _own_funds(i), _pnl_bi(), FX[d])
    return out


def inject_defects(rd: ReportingData, seed: int = 5) -> tuple[ReportingData, pd.DataFrame]:
    """Typical source-data defects that the data-quality pre-checks must catch."""
    rng = np.random.default_rng(seed)
    e = rd.exposures.copy()
    idx = rng.choice(e.index[e["segment"].isin(["Corporate", "SME"])], 3, replace=False)
    log = []
    e.loc[idx[0], "country"] = ""
    log.append((e.loc[idx[0], "exposure_id"], "country", "missing country of residence"))
    e.loc[idx[1], "provisions"] = -abs(e.loc[idx[1], "provisions"]) - 1e5
    log.append((e.loc[idx[1], "exposure_id"], "provisions", "negative credit risk adjustment"))
    irb = e.index[(e["approach"] == "IRB") & ~e["defaulted"]]
    j = rng.choice(irb, 1)[0]
    e.loc[j, "lgd"] = 1.35
    log.append((e.loc[j, "exposure_id"], "lgd", "LGD above 100 %"))
    return ReportingData(rd.ref_date, e, rd.positions, rd.own_funds, rd.pnl_bi, rd.fx), pd.DataFrame(log, columns=["exposure_id", "field", "defect"])


def validate_exposures(df: pd.DataFrame) -> pd.DataFrame:
    need = ["exposure_id", "counterparty_id", "group_id", "country", "approach", "sa_class", "irb_class", "product", "bs_type",
            "original_exposure", "provisions", "pd", "lgd", "maturity", "defaulted"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"exposures: missing columns {', '.join(miss)}")
    df = df.copy()
    for c in EXPOSURE_COLS:
        if c not in df.columns:
            df[c] = "" if c in ("counterparty", "currency", "segment", "ccf_bucket", "guarantor_type", "grade", "property_type", "cva_sector") else 0.0
    for c in ["original_exposure", "drawn", "provisions", "fin_collateral", "collateral_haircut", "guarantee", "pd", "lgd", "maturity", "ltv",
              "turnover_m", "rc", "pfe"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["defaulted", "sme", "rated", "fair_value"]:
        df[c] = df[c].astype(str).str.lower().isin(["true", "1", "yes"])
    if not set(df["approach"]) <= {"SA", "IRB"}:
        raise ValueError("approach must be SA or IRB")
    if not set(df["sa_class"]) <= set(SA_CLASSES):
        raise ValueError(f"sa_class must be one of {SA_CLASSES}")
    df["currency"] = df["currency"].replace("", "EUR")
    df["amount_ccy"] = df["original_exposure"]
    df["stage"] = pd.to_numeric(df.get("stage", 1), errors="coerce").fillna(1).astype(int)
    return df[EXPOSURE_COLS + ["amount_ccy"]]
