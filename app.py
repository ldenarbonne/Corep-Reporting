"""COREP Reporting – CRR3 own funds, credit / market / CVA / operational risk, leverage and large exposures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from corep import data as D
from corep import export as X
from corep import templates as T
from corep import validation as V
from corep import variation as VA

st.set_page_config(page_title="COREP Reporting", page_icon="📑", layout="wide")
TEAL, ORANGE, RED, GREEN, GREY, NAVY, SAND = "#264653", "#e76f51", "#c1121f", "#2a9d8f", "#9aa5b1", "#1d3557", "#e9c46a"
LEG = dict(orientation="h", y=-0.2)
MN = 1e6
PCT_CELLS = {("C 03.00", None), ("C 47.00", "0340"), ("C 26.00", "0020")}
PCT_COLS = {("C 08.01", "0010"), ("C 08.01", "0230"), ("C 08.02", "0010"), ("C 08.02", "0230"), ("C 09.02", "0090"), ("C 09.02", "0100"),
            ("C 28.00", "0320"), ("C 28.00", "0350"), ("C 29.00", "0320")}
COUNT_COLS = {("C 08.01", "0300"), ("C 08.02", "0300"), ("C 25.01", "0010")}


def h(df):
    return min(35 * (len(df) + 1) + 3, 900)


def fmt_cell(template, row, col, v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if template == "C 03.00" or (template, row) in {("C 47.00", "0340"), ("C 26.00", "0020")} or (template, col) in PCT_COLS:
        return f"{v:.2%}"
    if (template, col) in COUNT_COLS:
        return f"{v:,.0f}"
    if template == "C 08.01" and col == "0250":
        return f"{v:,.0f}"
    return f"{v / 1e3:,.0f}"          # € thousands, as reported


def show_template(rep, template, sheet):
    tb = rep.table(template, sheet)
    if tb.empty:
        st.info("No data for this sheet.")
        return tb
    rows = [r for r, _ in tb.index]
    cols = [c for c, _ in tb.columns]
    vals = tb.to_numpy()
    out = pd.DataFrame([[fmt_cell(template, r, c, vals[i, j]) for j, c in enumerate(cols)] for i, r in enumerate(rows)])
    out.columns = [f"c{c} · {lab}" for c, lab in tb.columns]
    out.index = [f"r{r} · {lab}" for r, lab in tb.index]
    st.dataframe(out, width="stretch", height=min(h(out), 640))
    return tb


@st.cache_data(show_spinner="Building the COREP templates…")
def build_reports(seed, defects, import_key, _imported):
    rds = D.generate(seed)
    dq_log = pd.DataFrame()
    if _imported is not None:
        d1 = D.REF_DATES[1]
        rds[d1] = D.ReportingData(d1, _imported.assign(ref_date=d1), rds[d1].positions, rds[d1].own_funds, rds[d1].pnl_bi, rds[d1].fx)
    if defects:
        d1 = D.REF_DATES[1]
        rds[d1], dq_log = D.inject_defects(rds[d1])
    reps = {d: T.build(rd) for d, rd in rds.items()}
    return rds, reps, dq_log


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Reporting")
    ref = st.radio("Reference date", D.REF_DATES, index=1, format_func=lambda d: d.strftime("%d %B %Y"))
    seed = st.number_input("Seed (synthetic bank)", 0, 9999, 11)
    defects = st.checkbox("Inject source-data defects (Sep 2026)", value=False,
                          help="Missing country, a negative provision and an LGD above 100 % — caught by the data-quality pre-checks and validation rules.")
    st.caption("Entity: DEMO BANK SA (synthetic) · consolidated · EUR · amounts reported in € thousands.")

imported = st.session_state.get("imported")
rds, reps_raw, dq_log = build_reports(seed, defects, st.session_state.get("imported_name", ""), imported)
prev_date = [d for d in D.REF_DATES if d != ref][0] if ref == D.REF_DATES[1] else None

st.title("COREP Reporting")
st.caption("Supervisory reporting of own funds and own-funds requirements under CRR3 in the format of the EBA COREP templates — "
           "from line-level source data to C 01.00 – C 47.00, with validation rules, data-to-cell lineage, quarter-on-quarter "
           "variation analysis and Excel / xBRL-CSV exports.")

tabs = st.tabs(["Dashboard", "Templates", "Validation", "Data lineage", "Variation analysis", "Credit risk", "Market, CVA & OpRisk",
                "Leverage & large exposures", "Export", "Data & import", "Methodology"])

# top-side adjustments (rendered first: they change every template)
with tabs[2]:
    st.subheader("Top-side adjustments")
    st.caption("Manual adjustments posted on reported cells after the calculation (the dependent cells are not recomputed) — "
               "the classic source of inter-template inconsistencies. Activate a line to see which validation rules break.")
    adj0 = pd.DataFrame([
        {"active": False, "template": "C 02.00", "sheet": "", "row": "0050", "col": "0010", "amount": 150e6, "comment": "late booking of a SA corporate loan"},
        {"active": False, "template": "C 01.00", "sheet": "", "row": "0020", "col": "0010", "amount": -40e6, "comment": "additional deduction (insufficient coverage NPE)"},
        {"active": False, "template": "C 07.00", "sheet": "Corporates", "row": "0010", "col": "0200", "amount": 25e6, "comment": "manual reclassification"}])
    adj = st.data_editor(adj0, num_rows="dynamic", key="adj", width="stretch",
                         column_config={"amount": st.column_config.NumberColumn("amount (€)", format="%.0f")})
reps = {d: (T.apply_adjustments(r, adj[adj["active"]].drop(columns="active")) if d == ref else r) for d, r in reps_raw.items()}
rep = reps[ref]
val = V.run(rep)
dq = V.data_quality(rep.exposures)

# ---------------------------------------------------------------- dashboard
with tabs[0]:
    g = rep.get
    k = st.columns(6)
    k[0].metric("CET1 ratio", f"{g('C 03.00', '0010', '0010'):.2%}", None if prev_date is None else
                f"{(g('C 03.00', '0010', '0010') - reps[prev_date].get('C 03.00', '0010', '0010')) * 1e4:+.0f} bp vs Q2")
    k[1].metric("Tier 1 ratio", f"{g('C 03.00', '0030', '0010'):.2%}")
    k[2].metric("Total capital ratio", f"{g('C 03.00', '0050', '0010'):.2%}")
    k[3].metric("Leverage ratio", f"{g('C 47.00', '0340', '0010'):.2%}")
    k[4].metric("TREA", f"€{g('C 02.00', '0010', '0010') / 1e9:,.2f}bn", None if prev_date is None else
                f"{(g('C 02.00', '0010', '0010') / reps[prev_date].get('C 02.00', '0010', '0010') - 1):+.1%}")
    head = 1 - rep.aux["floor_x"] * rep.aux["s_trea"] / rep.aux["u_trea"]
    k[5].metric(f"Headroom to the output floor ({rep.aux['floor_x']:.0%})", f"{head:.1%}", help="1 − x × S-TREA / U-TREA; negative = floor binding.")
    nfail, nwarn = (val["status"] == "fail").sum(), (val["status"] == "warning").sum()
    ndq = (dq["status"] == "fail").sum()
    if nfail or ndq:
        st.error(f"{nfail} validation rule(s) failed, {nwarn} warning(s), {ndq} data-quality pre-check(s) failed — see the Validation tab.")
    else:
        st.success(f"All {len(val)} validation rules and {len(dq)} data-quality pre-checks pass ({nwarn} warning(s)).")
    c1, c2 = st.columns(2)
    with c1:
        comp = pd.Series({"Credit risk – SA": g("C 02.00", "0050", "0010"), "Credit risk – IRB": g("C 02.00", "0240", "0010"),
                          "Market risk": g("C 02.00", "0520", "0010"), "Operational risk": g("C 02.00", "0590", "0010"),
                          "CVA": g("C 02.00", "0640", "0010"), "Output floor adjustment": g("C 02.00", "0035", "0010")})
        fig = go.Figure(go.Bar(x=comp.index, y=comp.values / 1e9, marker_color=[TEAL, NAVY, ORANGE, SAND, GREEN, RED],
                               text=[f"{v / comp.sum():.0%}" for v in comp.values], textposition="outside"))
        fig.update_layout(height=380, margin=dict(t=30, b=10), yaxis_title="€bn", title="Total risk exposure amount by risk type (C 02.00)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        trea = g("C 02.00", "0010", "0010")
        cet1, at1, t2 = g("C 01.00", "0020", "0010"), g("C 01.00", "0530", "0010"), g("C 01.00", "0750", "0010")
        fig = go.Figure()
        for lab, v, col in (("CET1", cet1, TEAL), ("AT1", at1, GREEN), ("Tier 2", t2, SAND)):
            fig.add_bar(x=["Own funds"], y=[v / trea], name=lab, marker_color=col)
        req = [("Pillar 1 CET1 4.5 %", 0.045), ("Pillar 1 T1 6 %", 0.06), ("Pillar 1 total 8 %", 0.08)]
        for lab, v, col in (("CET1 4.5 %", 0.045, TEAL), ("+ AT1 1.5 %", 0.015, GREEN), ("+ T2 2 %", 0.02, SAND), ("Capital conservation buffer 2.5 %", 0.025, GREY)):
            fig.add_bar(x=["Pillar 1 + CCB"], y=[v], name=lab, marker_color=col, opacity=0.5, showlegend=False)
        fig.update_layout(barmode="stack", height=380, margin=dict(t=30, b=10), yaxis_tickformat=".0%", legend=LEG, title="Capital stack vs minimum requirements")
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Templates filed**")
    idx = pd.DataFrame([{"template": t_, "title": T.TITLES.get(t_, ""), "sheets (z-axis)": len(rep.sheets(t_)),
                         "cells": int((rep.cells["template"] == t_).sum()),
                         "rules failed": int(val[val["templates"].str.contains(t_, regex=False) & (val["status"] == "fail")].shape[0])}
                        for t_ in rep.cells["template"].unique()])
    st.dataframe(idx, width="stretch", hide_index=True, height=h(idx))

# ---------------------------------------------------------------- templates
with tabs[1]:
    c1, c2 = st.columns([1, 2])
    tsel = c1.selectbox("Template", list(dict.fromkeys(rep.cells["template"])), format_func=lambda t_: f"{t_} – {T.TITLES.get(t_, '')}", key="tpl")
    sheets = rep.sheets(tsel)
    ssel = c2.selectbox("Sheet (z-axis: exposure class / country)", sheets, key="sheet") if len(sheets) > 1 or sheets[0] else ""
    st.markdown(f"#### {tsel} – {T.TITLES.get(tsel, '')}" + (f" · {ssel}" if ssel else ""))
    tb = show_template(rep, tsel, ssel)
    st.caption("Amounts in € thousands; ratios and averages in %. Row and column codes follow the EBA templates for the main cells; "
               "reduced layouts use indicative codes.")
    fs = {k_: v for k_, v in rep.formulas.items() if k_[0] == tsel}
    if fs:
        st.markdown("**Derivation of computed cells**")
        st.dataframe(pd.DataFrame([{"cell": f"{k_[0]} {{r{k_[1]}; c{k_[2]}}}", "derivation": v} for k_, v in fs.items()]), width="stretch", hide_index=True)

# ---------------------------------------------------------------- validation
with tabs[2]:
    st.subheader("Data-quality pre-checks on the source exposures")
    st.dataframe(dq.style.map(lambda v: f"color: {RED}; font-weight: 600" if v == "fail" else "", subset=["status"]), width="stretch", hide_index=True)
    if defects and not dq_log.empty:
        st.caption("Injected defects: " + "; ".join(f"{r.exposure_id} – {r.defect}" for r in dq_log.itertuples()))
    st.subheader("Validation rules")
    k = st.columns(4)
    k[0].metric("Rules executed", len(val))
    k[1].metric("Passed", int((val["status"] == "pass").sum()))
    k[2].metric("Failed", int((val["status"] == "fail").sum()))
    k[3].metric("Warnings", int((val["status"] == "warning").sum()))
    flt = st.multiselect("Show", ["fail", "warning", "pass"], default=["fail", "warning"] if (val["status"] != "pass").any() else ["pass"])
    vv = val[val["status"].isin(flt)].copy()
    small = vv["lhs"].abs().lt(5) & vv["rhs"].abs().lt(5) & vv["description"].str.contains("ratio|LGD|PD|%")
    for c in ["lhs", "rhs", "difference"]:
        vv[c] = [f"{a:.4%}" if s else f"{a / 1e3:,.0f}" for a, s in zip(vv[c], small)]
    st.dataframe(vv.style.map(lambda v: f"color: {RED}; font-weight: 600" if v == "fail" else (f"color: {ORANGE}" if v == "warning" else f"color: {GREEN}"),
                              subset=["status"]), width="stretch", hide_index=True, height=min(h(vv), 600))
    st.caption("Amounts in € thousands. Rule types: intra-template, inter-template, sign, range, completeness and limit checks, "
               "modelled on the EBA validation rules (internal identifiers).")

# ---------------------------------------------------------------- lineage
with tabs[3]:
    st.markdown("Pick any cell fed by line-level data to see the exposures that make it up — the audit trail required by "
                "BCBS 239 and by the supervisor's on-site inspections.")
    lin_t = [t_ for t_ in ["C 07.00", "C 08.01", "C 08.02", "C 09.01", "C 09.02", "C 28.00"] if t_ in set(rep.contrib["template"])]
    c1, c2, c3, c4 = st.columns(4)
    lt = c1.selectbox("Template", lin_t, key="lt")
    cc = rep.contrib[rep.contrib["template"] == lt]
    ls = c2.selectbox("Sheet", sorted(cc["sheet"].unique()), key="ls")
    cc = cc[cc["sheet"] == ls]
    lab_r = rep.cells[(rep.cells["template"] == lt) & (rep.cells["sheet"] == ls)].drop_duplicates("row").set_index("row")["row_label"]
    lab_c = rep.cells[(rep.cells["template"] == lt) & (rep.cells["sheet"] == ls)].drop_duplicates("col").set_index("col")["col_label"]
    lr = c3.selectbox("Row", sorted(cc["row"].unique()), format_func=lambda r: f"r{r} · {lab_r.get(r, '')}", key="lr")
    lc = c4.selectbox("Column", sorted(cc[cc["row"] == lr]["col"].unique()), format_func=lambda c: f"c{c} · {lab_c.get(c, '')}", key="lc")
    cs = cc[(cc["row"] == lr) & (cc["col"] == lc) & (cc["value"].abs() > 1e-9)]
    det = cs.merge(rep.exposures, on="exposure_id", how="left")
    reported = rep.get(lt, lr, lc, ls)
    s_contrib = cs["value"].sum()
    weighted = (lt, lc) in PCT_COLS or (lt == "C 08.01" and lc == "0250")
    k = st.columns(4)
    k[0].metric("Reported value", fmt_cell(lt, lr, lc, reported))
    k[1].metric("Contributing exposures", f"{len(cs):,}")
    k[2].metric("Σ contributions" + (" (numerator of the weighted average)" if weighted else ""), f"{s_contrib / 1e3:,.0f}")
    adjd = (rep.adjustments is not None and not rep.adjustments.empty and
            ((rep.adjustments["template"] == lt) & (rep.adjustments["row"] == lr) & (rep.adjustments["col"] == lc) & (rep.adjustments["sheet"] == ls)).any())
    k[3].metric("Reconciles to the source", "n/a (weighted average)" if weighted else ("yes" if abs(reported - s_contrib) < 1 else "NO – top-side adjustment" if adjd else "NO"))
    st.markdown(f"**Lineage path:** source exposures (`{'Q3' if ref == D.REF_DATES[1] else 'Q2'} exposures`) → regulatory calculation "
                f"(`corep.calc.compute`: CCF, CRM, risk weights / IRB formula) → aggregation by exposure class, row filter r{lr}, measure c{lc} → "
                f"**{lt} {{{ls}; r{lr}; c{lc}}}**")
    cols = ["exposure_id", "value", "counterparty", "group_id", "country", "approach", "product", "bs_type", "original_exposure", "provisions",
            "grade", "pd", "lgd", "exposure_value", "rwea"]
    dd = det[cols].sort_values("value", key=abs, ascending=False).head(300)
    st.dataframe(dd.style.format({"value": "{:,.0f}", "original_exposure": "{:,.0f}", "provisions": "{:,.0f}", "pd": "{:.2%}", "lgd": "{:.0%}",
                                  "exposure_value": "{:,.0f}", "rwea": "{:,.0f}"}), width="stretch", hide_index=True, height=400)
    st.download_button("Download the contributing exposures (CSV)", det.to_csv(index=False).encode(), f"lineage_{lt}_{ls}_r{lr}_c{lc}.csv".replace(" ", "_"), "text/csv")

# ---------------------------------------------------------------- variation
with tabs[4]:
    r0, r1 = reps[D.REF_DATES[0]], reps[D.REF_DATES[1]]
    cred, detail = VA.rwea_bridge(r0, r1, rds[D.REF_DATES[0]].fx, rds[D.REF_DATES[1]].fx)
    tb_ = VA.trea_bridge(r0, r1, cred)
    cb = VA.cet1_ratio_bridge(r0, r1, tb_)
    st.caption(f"30 June 2026 → 30 September 2026. USD/EUR {rds[D.REF_DATES[0]].fx:.2f} → {rds[D.REF_DATES[1]].fx:.2f}.")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Waterfall(x=["30 Jun"] + list(tb_.index) + ["30 Sep"], y=[tb_.attrs["start"] / 1e9] + list(tb_.values / 1e9) + [0],
                                     measure=["absolute"] + ["relative"] * len(tb_) + ["total"],
                                     text=[f"{tb_.attrs['start'] / 1e9:.2f}"] + [f"{v / 1e6:+,.0f}m" for v in tb_.values] + [f"{tb_.attrs['end'] / 1e9:.2f}"],
                                     increasing_marker_color=RED, decreasing_marker_color=GREEN, totals_marker_color=TEAL))
        fig.update_layout(height=440, margin=dict(t=30, b=10), yaxis_title="€bn", title="TREA bridge Q2 → Q3",
                          yaxis_range=[min(tb_.attrs["start"], tb_.attrs["end"]) / 1e9 * 0.94, max(tb_.attrs["start"], tb_.attrs["end"]) / 1e9 * 1.04])
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = go.Figure(go.Waterfall(x=["30 Jun"] + list(cb.index) + ["30 Sep"], y=[cb.attrs["start"] * 1e4] + list(cb.values * 1e4) + [0],
                                     measure=["absolute"] + ["relative"] * len(cb) + ["total"],
                                     text=[f"{cb.attrs['start']:.2%}"] + [f"{v * 1e4:+.0f}" for v in cb.values] + [f"{cb.attrs['end']:.2%}"],
                                     increasing_marker_color=GREEN, decreasing_marker_color=RED, totals_marker_color=TEAL))
        fig.update_layout(height=440, margin=dict(t=30, b=10), yaxis_title="bp", title="CET1 ratio bridge Q2 → Q3 (bp)",
                          yaxis_range=[min(cb.attrs["start"], cb.attrs["end"]) * 1e4 - 80, max(cb.attrs["start"], cb.attrs["end"]) * 1e4 + 60])
        st.plotly_chart(fig, width="stretch")
    st.subheader("Credit RWEA change by portfolio and driver")
    pv = detail.pivot_table(index="portfolio", columns="cause", values="delta", aggfunc="sum", fill_value=0.0)
    fig = px.imshow(pv / MN, text_auto=".0f", color_continuous_scale="RdBu_r", color_continuous_midpoint=0, aspect="auto")
    fig.update_layout(height=420, margin=dict(t=30, b=10), coloraxis_colorbar=dict(title="€m"))
    st.plotly_chart(fig, width="stretch")
    st.caption("Existing exposures: volume and FX effects at the old risk weight, then the density change attributed to the first matching "
               "cause (new default → grade change → LGD / model change → other).")
    st.subheader("Cell-level variation")
    c1, c2, c3 = st.columns(3)
    vt = c1.selectbox("Template", list(dict.fromkeys(r1.cells["template"])), key="vt")
    cv_ = VA.cell_variation(r0, r1, vt)
    thr = c2.number_input("Materiality threshold (€m)", 0.0, 1000.0, 10.0, 5.0) * MN
    pct_thr = c3.number_input("… or relative change above", 0.0, 1.0, 0.10, 0.05)
    is_pct = cv_.apply(lambda r: (vt == "C 03.00") or (vt, r["col"]) in PCT_COLS or (vt, r["row"]) in {("C 47.00", "0340")}, axis=1)
    mat = cv_[((cv_["change"].abs() >= thr) & ~is_pct) | (cv_["change %"].abs() >= pct_thr)].copy()
    mat = mat.reindex(mat["change"].abs().sort_values(ascending=False).index)
    mat["explanation required"] = np.where(mat["change"].abs() >= thr * 5, "yes – material", "review")
    show = mat.copy()
    for c in ["previous", "current", "change"]:
        show[c] = [fmt_cell(vt, r, cc_, v) for r, cc_, v in zip(mat["row"], mat["col"], mat[c])]
    show["change %"] = mat["change %"].map(lambda v: "" if pd.isna(v) else f"{v:+.1%}")
    st.dataframe(show, width="stretch", hide_index=True, height=min(h(show), 500))
    st.caption(f"{len(mat)} cell(s) above the thresholds out of {len(cv_)}. Amounts in € thousands.")

# ---------------------------------------------------------------- credit risk
with tabs[5]:
    e = rep.exposures
    c1, c2 = st.columns(2)
    with c1:
        s7 = pd.DataFrame({cls: {"exposure value": rep.get("C 07.00", "0010", "0200", cls), "RWEA": rep.get("C 07.00", "0010", "0220", cls)} for cls in D.SA_CLASSES}).T
        s7["RW density"] = s7["RWEA"] / s7["exposure value"].replace(0, np.nan)
        fig = go.Figure()
        fig.add_bar(y=s7.index, x=s7["exposure value"] / 1e9, orientation="h", name="exposure value", marker_color=GREY)
        fig.add_bar(y=s7.index, x=s7["RWEA"] / 1e9, orientation="h", name="RWEA", marker_color=TEAL)
        fig.update_layout(barmode="group", height=420, margin=dict(t=30, b=10), xaxis_title="€bn", legend=LEG, title="Standardised approach by exposure class (C 07.00)")
        st.plotly_chart(fig, width="stretch")
    with c2:
        s8 = pd.DataFrame({cls: {"IRB RWEA": rep.get("C 08.01", "0010", "0260", cls),
                                 "SA-equivalent RWEA": float(e.loc[(e["approach"] == "IRB") & (e["irb_class"] == cls), "rwea_sa"].sum()),
                                 "EV": rep.get("C 08.01", "0010", "0110", cls)} for cls in D.IRB_CLASSES}).T
        fig = go.Figure()
        fig.add_bar(y=s8.index, x=s8["IRB RWEA"] / 1e9, orientation="h", name="IRB RWEA", marker_color=NAVY)
        fig.add_bar(y=s8.index, x=s8["SA-equivalent RWEA"] / 1e9, orientation="h", name="SA-equivalent (output floor)", marker_color=SAND)
        fig.update_layout(barmode="group", height=420, margin=dict(t=30, b=10), xaxis_title="€bn", legend=LEG, title="IRB vs SA-equivalent by exposure class")
        st.plotly_chart(fig, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        ic = st.selectbox("IRB exposure class (C 08.02)", D.IRB_CLASSES, index=1)
        g2 = rep.table("C 08.02", ic)
        if not g2.empty:
            g2 = g2.droplevel(1, axis=1)
            fig = go.Figure()
            fig.add_bar(x=[r for r, _ in g2.index], y=g2["0110"] / MN, name="exposure value (€m)", marker_color=GREY)
            fig.add_scatter(x=[r for r, _ in g2.index], y=g2["0260"] / g2["0110"].replace(0, np.nan), name="RW density", yaxis="y2", line_color=RED, mode="lines+markers")
            fig.update_layout(height=380, margin=dict(t=30, b=10), legend=LEG, title=f"{ic}: exposure and RW density by obligor grade",
                              yaxis=dict(title="€m"), yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False))
            st.plotly_chart(fig, width="stretch")
    with c2:
        ipr = e[e["approach"] == "IRB"]
        el, prov = float(ipr["el_irb"].sum()), float(ipr["provisions"].clip(lower=0).sum())
        st.markdown("**IRB expected loss vs provisions**")
        st.dataframe(pd.DataFrame({"€m": [el / MN, prov / MN, rep.aux["shortfall"] / MN, rep.aux["excess"] / MN]},
                                  index=["expected loss", "credit risk adjustments", "shortfall (CET1 deduction)", "excess (T2, capped at 0.6 % IRB RWEA)"]).style.format("{:,.1f}"),
                     width="stretch")
        st.markdown("**SME supporting factor and CRM effects (SA)**")
        L = rep.aux["lines"]
        st.dataframe(pd.DataFrame({"€m": [float((L["rwea_pre_sf"] - L["rwea_line"]).sum()) / MN, float(e.loc[e["approach"] == "SA", "guar_cover"].sum()) / MN,
                                          float(e.loc[e["approach"] == "SA", "coll_adj"].sum()) / MN]},
                                  index=["RWEA relief from the SME supporting factor", "guaranteed exposure substituted", "financial collateral (adjusted)"]).style.format("{:,.1f}"),
                     width="stretch")

# ---------------------------------------------------------------- market, CVA, op risk
with tabs[6]:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Market risk (C 18 – C 23, standardised)**")
        mk = rep.aux["market"]
        st.dataframe(mk.style.format({"own funds requirement": "{:,.0f}", "RWEA": "{:,.0f}"}), width="stretch", hide_index=True)
        st.caption("Under the EU delegated act of June 2026, the FRTB own-funds requirements apply from 1 January 2027 (subject to "
                   "scrutiny); for 2026 reference dates the CRR2 standardised approaches still determine the requirement.")
    with c2:
        st.markdown("**CVA risk (C 25.01, reduced BA-CVA)**")
        cv = rep.aux["cva"]
        st.dataframe(cv.style.format({"EAD": "{:,.0f}", "M": "{:.1f}", "RW": "{:.1%}", "SCVA": "{:,.0f}"}), width="stretch", hide_index=True, height=300)
        st.caption(f"K = √[(ρ ΣSCVA)² + (1 − ρ²) ΣSCVA²], ρ = 50 %; own funds = 0.65 × K = €{cv.attrs['own_funds'] / MN:,.1f}m.")
    with c3:
        st.markdown("**Operational risk (C 16.01, business indicator)**")
        op = rep.aux["opr"]
        st.dataframe(pd.DataFrame({"€m": [op["ILDC"] / MN, op["SC"] / MN, op["FC"] / MN, op["BI"] / MN, op["BIC"] / MN, op["RWEA"] / MN]},
                                  index=["ILDC", "SC", "FC", "BI (3-year average)", "BIC (12 % / 15 % / 18 %)", "RWEA (ILM = 1)"]).style.format("{:,.1f}"), width="stretch")
        st.dataframe((op["BI by year"] / MN).rename("BI by year (€m)").to_frame().style.format("{:,.1f}"), width="stretch")

# ---------------------------------------------------------------- leverage & large exposures
with tabs[7]:
    c1, c2 = st.columns(2)
    with c1:
        lt_ = rep.table("C 47.00")
        lv = lt_.droplevel(1, axis=1)["0010"]
        parts = lv.loc[["0210", "0010", "0020", "0190", "0260"]].droplevel(0) if isinstance(lv.index, pd.MultiIndex) else lv
        fig = go.Figure(go.Waterfall(x=list(parts.index) + ["Total exposure measure"], y=list(parts.values / 1e9) + [0],
                                     measure=["relative"] * len(parts) + ["total"], increasing_marker_color=TEAL, decreasing_marker_color=RED, totals_marker_color=NAVY))
        fig.update_layout(height=420, margin=dict(t=30, b=10), yaxis_title="€bn", title=f"Leverage exposure measure (C 47.00) — ratio {rep.get('C 47.00', '0340', '0010'):.2%}")
        st.plotly_chart(fig, width="stretch")
    with c2:
        le = rep.aux["large_exposures"].head(15)
        t1 = rep.get("C 01.00", "0015", "0010")
        fig = go.Figure(go.Bar(y=le.index, x=le["% of Tier 1"], orientation="h", marker_color=[RED if v > 0.25 else (ORANGE if v >= 0.10 else GREY) for v in le["% of Tier 1 before CRM"]]))
        fig.add_vline(x=0.10, line_dash="dot", line_color=ORANGE, annotation_text="10 % reporting")
        fig.add_vline(x=0.25, line_dash="dash", line_color=RED, annotation_text="25 % limit")
        fig.update_layout(height=420, margin=dict(t=30, b=10), xaxis_tickformat=".0%", yaxis=dict(autorange="reversed"),
                          title=f"Groups of connected clients, % of Tier 1 (€{t1 / 1e9:.2f}bn)")
        st.plotly_chart(fig, width="stretch")
    st.markdown("**C 28.00 – large exposures (≥ 10 % of Tier 1)**")
    show_template(rep, "C 28.00", "")
    st.markdown("**C 29.00 – individual clients within the reported groups**")
    show_template(rep, "C 29.00", "")
    st.caption("Exposure values before CRM: on-balance amounts, off-balance items with conversion factors (10 % floor) and SA-CCR for derivatives; "
               "exposures to EU sovereigns exempted (art. 400). Retail pools are not groups of connected clients.")

# ---------------------------------------------------------------- export
with tabs[8]:
    st.markdown(f"Reporting package for **{ref:%d %B %Y}**" + (" — including the active top-side adjustments" if rep.adjustments is not None and not rep.adjustments.empty else ""))
    c1, c2, c3 = st.columns(3)
    c1.download_button("COREP workbook (Excel)", X.to_excel(rep), f"COREP_{ref:%Y%m%d}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    c2.download_button("xBRL-CSV-style report package (zip)", X.to_xbrl_csv(rep), f"COREP_{ref:%Y%m%d}_xbrl_csv.zip", "application/zip")
    c3.download_button("Validation report (CSV)", val.to_csv(index=False).encode(), f"COREP_{ref:%Y%m%d}_validation.csv", "text/csv")
    st.markdown("""
**xBRL-CSV package structure** (EBA filing rules since the March 2026 reference date, framework 4.2):
```
<LEI>.CON_FR_COREP040200_COREPOF_<date>_<timestamp>.zip
├── META-INF/reportPackage.json
└── reports/
    ├── report.json          ← "extends": taxonomy entry point (module)
    ├── parameters.csv       ← entityID, refPeriod, baseCurrency, decimals
    ├── FilingIndicators.csv ← templates reported
    └── c_01.00.csv, c_02.00.csv, …  ← datapoint, factValue
```
Here datapoints are named by template / sheet / row / column codes, not by the official DPM identifiers: the package shows
the structure but is not submittable.
""")

# ---------------------------------------------------------------- data & import
with tabs[9]:
    st.markdown("**Exposures file** — one row per exposure, same columns as the template. Uploading a file replaces the "
                "30 September 2026 exposures (own funds, trading positions and P&L stay synthetic).")
    st.download_button("Exposures template (CSV)", rds[D.REF_DATES[1]].exposures.drop(columns=["amount_ccy"], errors="ignore").head(200).to_csv(index=False).encode(),
                       "exposures_template.csv", "text/csv")
    up = st.file_uploader("Exposures CSV", type="csv")
    if up is not None and st.button("Load"):
        try:
            st.session_state["imported"] = D.validate_exposures(pd.read_csv(up))
            st.session_state["imported_name"] = up.name
            st.rerun()
        except Exception as err:  # noqa: BLE001
            st.error(f"Import failed: {err}")
    if "imported" in st.session_state and st.button("Back to synthetic data"):
        del st.session_state["imported"]
        st.session_state.pop("imported_name", None)
        st.rerun()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Own funds items**")
        st.dataframe((rds[ref].own_funds / MN).rename("€m").to_frame().style.format("{:,.1f}"), width="stretch")
    with c2:
        st.markdown("**Business indicator P&L (3 years)**")
        st.dataframe((rds[ref].pnl_bi.T / MN).style.format("{:,.0f}"), width="stretch")
    st.markdown("**Exposures**")
    st.dataframe(rep.exposures[D.EXPOSURE_COLS].head(500), width="stretch", hide_index=True, height=360)
    st.markdown("**Trading positions**")
    st.dataframe(rds[ref].positions.style.format({"net_position": "{:,.0f}"}), width="stretch", hide_index=True, height=300)

# ---------------------------------------------------------------- methodology
with tabs[10]:
    st.markdown(r"""
### Framework
* **CRR3** (Regulation (EU) 2024/1623) applies from 1 January 2025; the first COREP reference date under CRR3 was 31 March 2025
  (EBA reporting framework 4.0). Framework 4.2 moved all reports to DPM 2.0, and from the **March 2026 reference date**
  submissions must be in **xBRL-CSV** (xBRL-XML no longer accepted). Release 4.4 (October 2026) covers IFRS 18 FINREP and
  Pillar 3 changes; the reporting simplification package (phase 2, release 4.4.1) is expected in early 2027.
* Market risk: the FRTB own-funds requirements are postponed in the EU (delegated act adopted on 4 June 2026, application from
  1 January 2027 subject to scrutiny); for 2026 reference dates the CRR2 standardised approaches apply (C 18 – C 24).

### Calculation chain
1. **Source data** — line-level exposures, trading positions, own-funds items, P&L for the business indicator.
2. **Data-quality pre-checks** — completeness and plausibility of the attributes the calculation uses.
3. **Regulatory calculation** — CCF, CRM (financial collateral comprehensive method, substitution for guarantees), SA risk weights
   (CRR3 buckets incl. real-estate loan splitting and the 2026 equity transitional weight), SME supporting factor, IRB formula with
   CRR3 input floors and F-IRB supervisory LGDs, expected loss vs provisions, SA-equivalent RWEA for the output floor
   (55 % in 2026, 125 % cap of art. 465(2)), market risk SA, reduced BA-CVA, business-indicator component, leverage exposure,
   large exposures.
4. **Templates** — aggregation by exposure class / country / obligor grade into the COREP cells, with exposure-level contributions
   kept for lineage.
5. **Validation** — intra- and inter-template rules, signs, ranges, completeness, limits.
6. **Variation analysis** — cell movements and analytical bridges for the quarter.
7. **Export** — Excel workbook and xBRL-CSV-style package.

### Simplifications
* Reduced layouts for the templates with many detailed rows (risk-weight buckets, CRR3 sub-classes); indicative codes for those rows.
* No securitisation, CIU look-through, NPE insufficient coverage, significant investments in financial sector entities, or
  threshold deductions; IFRS 9 transitional arrangements not used.
* Prudent valuation by the simplified approach (0.1 % of fair-valued positions).
* Exposure values for large exposures and leverage simplified (no SFTs).
""")
