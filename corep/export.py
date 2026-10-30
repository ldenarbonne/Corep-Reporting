"""
Exports: Excel workbook (one sheet per template and z-axis value) and an xBRL-CSV-style report package.

The xBRL-CSV package follows the structure of the EBA filing rules (report package zip with META-INF/reportPackage.json,
reports/report.json pointing to the taxonomy entry point, parameters.csv, FilingIndicators.csv and one CSV per table),
but the datapoints are identified by template/sheet/row/column codes instead of the official DPM datapoint IDs, so the
package is a structural illustration and not submittable.
"""
from __future__ import annotations

import io
import json
import re
import zipfile

import pandas as pd

from .templates import TITLES, Report


def _sheet_name(t: str, s: str) -> str:
    n = f"{t} {s}".strip()
    n = re.sub(r"[\[\]\*\?/\\:]", "-", n)
    return n[:31]


def to_excel(rep: Report, entity: str = "DEMO BANK SA") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        idx = pd.DataFrame([{"template": t, "title": TITLES.get(t, ""), "sheets": len(rep.sheets(t))} for t in rep.cells["template"].unique()])
        pd.DataFrame({"entity": [entity], "reference date": [rep.ref_date.date()], "framework": ["EBA reporting framework 4.x (CRR3) – illustrative"],
                      "unit": ["EUR"]}).to_excel(xw, sheet_name="Cover", index=False)
        idx.to_excel(xw, sheet_name="Index", index=False)
        used = set()
        for t in rep.cells["template"].unique():
            for s in rep.sheets(t):
                tb = rep.table(t, s)
                if tb.empty:
                    continue
                nm = _sheet_name(t, s)
                k = 1
                while nm in used:
                    nm = _sheet_name(t, s)[:28] + f"~{k}"
                    k += 1
                used.add(nm)
                flat = tb.copy()
                flat.columns = [f"c{c} {lab}" for c, lab in flat.columns]
                flat.index = [f"r{r} {lab}" for r, lab in flat.index]
                flat.to_excel(xw, sheet_name=nm)
        if rep.adjustments is not None and not rep.adjustments.empty:
            rep.adjustments.to_excel(xw, sheet_name="Top-side adjustments", index=False)
    return buf.getvalue()


def to_xbrl_csv(rep: Report, lei: str = "529900DEMOBANK000001", entry_point: str = "http://www.eba.europa.eu/eu/fr/xbrl/crr/fws/corep/4.2/mod/corep_of.json") -> bytes:
    buf = io.BytesIO()
    name = f"{lei}.CON_FR_COREP040200_COREPOF_{rep.ref_date:%Y-%m-%d}_20261030000000000"
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{name}/META-INF/reportPackage.json", json.dumps({"documentInfo": {"documentType": "https://xbrl.org/report-package/2023"}}, indent=2))
        tables = {}
        for t in rep.cells["template"].unique():
            code = t.lower().replace(" ", "_")
            tables[code] = {"template": code, "url": f"{code}.csv"}
        z.writestr(f"{name}/reports/report.json", json.dumps({"documentInfo": {"documentType": "https://xbrl.org/2021/xbrl-csv",
                                                                               "extends": [entry_point]}, "tables": tables}, indent=2))
        z.writestr(f"{name}/reports/parameters.csv", f"name,value\nentityID,lei:{lei}\nrefPeriod,{rep.ref_date:%Y-%m-%d}\nbaseCurrency,iso4217:EUR\ndecimalsMonetary,-3\ndecimalsPercentage,4\n")
        fi = pd.DataFrame({"templateID": [t.replace(" ", "_") for t in rep.cells["template"].unique()], "reported": "true"})
        z.writestr(f"{name}/reports/FilingIndicators.csv", fi.to_csv(index=False))
        for t in rep.cells["template"].unique():
            c = rep.cells[rep.cells["template"] == t]
            dp = c.assign(datapoint=lambda d: d["template"].str.replace(" ", "_") + ("_" + d["sheet"].str.replace(" ", "_")).where(d["sheet"] != "", "")
                          + "_r" + d["row"] + "_c" + d["col"], factValue=lambda d: d["value"].round(4))
            z.writestr(f"{name}/reports/{t.lower().replace(' ', '_')}.csv", dp[["datapoint", "factValue"]].to_csv(index=False))
    return buf.getvalue()
