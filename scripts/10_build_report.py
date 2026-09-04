"""Assemble the standalone HTML findings report.

Numbers are read from the generated CSVs in `reports/` and figures are inlined
as data URIs, so the page is regenerable and always agrees with the analysis
that produced it. Run every other script first.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
OUT = REPORTS / "durable_dominance_report.html"

FIGURE_KEYS = {
    "FIG_REPRO": "02_nba_paper_cb_win.png",
    "FIG_IDENT": "06_identifiability.png",
    "FIG_ROBUST": "07_robustness.png",
    "FIG_FIT": "05_nba_ranked_wins.png",
    "FIG_DIAG": "05_nba_diagnostics.png",
    "FIG_ML": "09_machine_learning.png",
    "FIG_INDIV": "03_individual_sports.png",
}


def data_uri(name: str) -> str:
    raw = (FIGURES / name).read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def load() -> dict:
    r = {}
    for n in ["audit_nfl_alignment", "audit_detrending", "audit_league_size",
              "simulation_diagnostics", "ml_season_cv", "ml_team_persistence",
              "ml_feature_importance", "database_benchmark", "individual_sports",
              "fit_nba", "fit_nfl", "identifiability_clip"]:
        r[n] = pd.read_csv(REPORTS / f"{n}.csv")
    return r


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.1e}".replace("e-0", "e−").replace("e-", "e−")
    return f"{p:.3f}"


def pill(text: str, kind: str) -> str:
    return f'<span class="pill pill--{kind}">{text}</span>'


def build_size_chart(size: pd.DataFrame) -> str:
    """Inline SVG: NBA p-values before and after controlling for team count."""
    nba = size[(size["league"] == "nba") & (size["window"] == "Full")].reset_index(drop=True)
    W, H = 720, 300
    pad_l, pad_r, pad_t, pad_b = 168, 24, 26, 46
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    row_h = plot_h / len(nba)

    def x(p):
        return pad_l + plot_w * min(max(p, 0.0), 1.0)

    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="NBA p-values before and after controlling for league size" '
        f'class="chart">'
    ]
    # gridlines + axis labels
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        gx = x(tick)
        parts.append(f'<line x1="{gx:.1f}" y1="{pad_t}" x2="{gx:.1f}" '
                     f'y2="{pad_t+plot_h}" class="grid"/>')
        parts.append(f'<text x="{gx:.1f}" y="{pad_t+plot_h+22}" '
                     f'class="axis" text-anchor="middle">{tick:g}</text>')
    # p = .05 threshold
    tx = x(0.05)
    parts.append(f'<line x1="{tx:.1f}" y1="{pad_t-6}" x2="{tx:.1f}" '
                 f'y2="{pad_t+plot_h}" class="thresh"/>')
    parts.append(f'<text x="{tx+6:.1f}" y="{pad_t-10}" class="axis thresh-t">'
                 f'p = .05</text>')

    for i, row in nba.iterrows():
        cy = pad_t + i * row_h
        bar_h = row_h * 0.30
        y1 = cy + row_h * 0.14
        y2 = cy + row_h * 0.52
        parts.append(
            f'<text x="{pad_l-12}" y="{cy+row_h/2+4:.1f}" class="ylab" '
            f'text-anchor="end">{row["metric"]}</text>')
        parts.append(f'<rect x="{pad_l}" y="{y1:.1f}" '
                     f'width="{max(x(row["p_raw"])-pad_l,1.5):.1f}" '
                     f'height="{bar_h:.1f}" class="bar bar--raw"/>')
        parts.append(f'<rect x="{pad_l}" y="{y2:.1f}" '
                     f'width="{max(x(row["p_size"])-pad_l,1.5):.1f}" '
                     f'height="{bar_h:.1f}" class="bar bar--ctl"/>')
        parts.append(f'<text x="{x(row["p_size"])+7:.1f}" '
                     f'y="{y2+bar_h-1:.1f}" class="vlab">'
                     f'{row["p_size"]:.2f}</text>')
    parts.append(f'<text x="{pad_l+plot_w/2:.1f}" y="{H-6}" class="axis" '
                 f'text-anchor="middle">p-value of the balance coefficient</text>')
    parts.append("</svg>")
    return "".join(parts)


def main() -> None:
    d = load()
    tpl = (ROOT / "scripts" / "report_template.html").read_text()

    # ---- verdict numbers -------------------------------------------------
    nfl_a = d["audit_nfl_alignment"]
    orig, fixed = nfl_a.iloc[0], nfl_a.iloc[1]
    detr = d["audit_detrending"].set_index(["league", "balance_metric"])
    ncaaf = detr.loc[("ncaaf", "rating_sd")]
    size = d["audit_league_size"]
    nba_size = size[(size["league"] == "nba")]
    sim = d["simulation_diagnostics"].set_index(["league", "relationship"])
    persist = d["ml_team_persistence"]
    fit_nba, fit_nfl = d["fit_nba"], d["fit_nfl"]

    repl = {
        "SIZE_CHART": build_size_chart(size),
        "N_SIG_RAW": str(int((size["p_raw"] < .05).sum())),
        "N_SIG_CTL": str(int((size["p_size"] < .05).sum())),
        "N_TESTS": str(len(size)),
        "NBA_NULL_MIN_P": f"{nba_size['p_size'].min():.2f}",
        "NFL_ORIG_SLOPE": f"{orig['slope']:+.1f}",
        "NFL_ORIG_P": fmt_p(orig["p_value"]),
        "NFL_FIX_SLOPE": f"{fixed['slope']:+.1f}",
        "NFL_FIX_P": fmt_p(fixed["p_value"]),
        "NCAAF_RAW_R2": f"{ncaaf['raw_r2']:.3f}",
        "NCAAF_DET_R2": f"{ncaaf['detrended_r2']:.3f}",
        "NCAAF_DET_P": fmt_p(ncaaf["detrended_p"]),
        "NCAAF_YEAR_R2": f"{ncaaf['dominance_vs_year_r2']:.2f}",
        "HHI_SIZE_R2": f"{size.loc[(size.league=='nba')&(size.window=='Full')&(size.metric=='HHI of wins'),'size_r2'].iloc[0]:.2f}",
        "DECK_SIZE_R2": f"{size.loc[(size.league=='nba')&(size.window=='Full')&(size.metric==chr(100)+'eck’s win% spread'),'size_r2'].iloc[0]:.2f}" if False else f"{size.loc[(size.league=='nba')&(size.window=='Full')&(size.metric.str.startswith('deck')),'size_r2'].iloc[0]:.2f}",
        "NBA_S_R2": f"{sim.loc[('NBA','s ~ win_pct_sd'),'r_squared']:.3f}",
        "NFL_S_R2": f"{sim.loc[('NFL','s ~ win_pct_sd'),'r_squared']:.3f}",
        "NBA_SD_P": fmt_p(sim.loc[("NBA", "s ~ dominance"), "p_value"]),
        "NFL_SD_P": fmt_p(sim.loc[("NFL", "s ~ dominance"), "p_value"]),
        "NBA_SD_R2": f"{sim.loc[('NBA','s ~ dominance'),'r_squared']:.3f}",
        "NFL_SD_R2": f"{sim.loc[('NFL','s ~ dominance'),'r_squared']:.3f}",
        "FIT_C": f"{fit_nba['c'].iloc[0]:.3f}",
        "NBA_S_LO": f"{fit_nba['s'].min():.2f}", "NBA_S_HI": f"{fit_nba['s'].max():.2f}",
        "NFL_S_LO": f"{fit_nfl['s'].min():.2f}", "NFL_S_HI": f"{fit_nfl['s'].max():.2f}",
        "AUC_NBA": f"{persist[(persist.league=='nba')&(persist.model=='Logistic regression')]['test_auc'].iloc[0]:.3f}",
        "AUC_NFL": f"{persist[(persist.league=='nfl')&(persist.model=='Logistic regression')]['test_auc'].iloc[0]:.3f}",
        "AUC_NCAAF": f"{persist[(persist.league=='ncaaf')&(persist.model=='Logistic regression')]['test_auc'].iloc[0]:.3f}",
    }

    # ---- generated table bodies -----------------------------------------
    repl["ROWS_SIZE"] = "".join(
        f'<tr><td class="tl">{r.metric}</td>'
        f'<td>{"yes" if r.metric in SCALE_FREE else "<b>no</b>"}</td>'
        f'<td class="num">{r.size_r2:.2f}</td>'
        f'<td class="num">{fmt_p(r.p_raw)}</td>'
        f'<td class="num">{fmt_p(r.p_size)}</td>'
        f'<td>{pill("null","fail") if r.p_size>=.05 else pill("holds","hold")}</td></tr>'
        for r in size[(size.league == "nba") & (size.window == "Full")].itertuples()
    )
    repl["ROWS_SIM"] = "".join(
        f'<tr><td class="tl">{lg}</td><td class="tl">{rel.replace("s ~ ","fitted s vs. ").replace("win_pct_sd","real win% spread")}</td>'
        f'<td class="num">{row.r_squared:.3f}</td><td class="num">{fmt_p(row.p_value)}</td>'
        f'<td>{pill("calibrated","hold") if row.p_value<.01 else pill("no relation","fail")}</td></tr>'
        for (lg, rel), row in sim.iterrows()
    )
    cv = d["ml_season_cv"].pivot(index="league", columns="model", values="cv_r2")
    repl["ROWS_CV"] = "".join(
        f'<tr><td class="tl">{lg.upper()}</td>'
        + "".join(f'<td class="num">{cv.loc[lg, m]:.2f}</td>'
                  for m in ["Baseline (train mean)", "Lasso", "ElasticNet", "Random Forest"])
        + "</tr>" for lg in cv.index
    )
    pv = persist.pivot(index="league", columns="model", values="test_auc")
    base = persist.groupby("league")["base_rate"].first()
    repl["ROWS_AUC"] = "".join(
        f'<tr><td class="tl">{lg.upper()}</td><td class="num">{base[lg]*100:.1f}%</td>'
        f'<td class="num strong">{pv.loc[lg,"Logistic regression"]:.3f}</td>'
        f'<td class="num">{pv.loc[lg,"Gradient boosting"]:.3f}</td></tr>'
        for lg in pv.index
    )
    imp = d["ml_feature_importance"]
    imp = imp[imp.league == "nba"].sort_values("importance", ascending=False)
    mx = imp["importance"].max()
    repl["ROWS_IMP"] = "".join(
        f'<tr><td class="tl">{FEATURE_NAMES.get(r.feature, r.feature)}</td>'
        f'<td class="num">{r.importance:+.3f}</td>'
        f'<td class="barcell"><span class="minibar" style="width:'
        f'{max(r.importance,0)/mx*100:.0f}%"></span></td></tr>'
        for r in imp.itertuples()
    )
    bm = d["database_benchmark"]
    repl["ROWS_BM"] = "".join(
        f'<tr><td class="tl">{r.source}</td><td class="tl">{r.operation}</td>'
        f'<td class="num">{r.seconds:.3f}</td>'
        f'<td class="num">{r.relative:.1f}&times;</td></tr>' for r in bm.itertuples()
    )
    ind = d["individual_sports"]
    repl["ROWS_IND"] = "".join(
        f'<tr><td class="tl">{r.sport}</td><td class="num">{r.n_years}</td>'
        f'<td class="num">{r.r_squared:.3f}</td><td class="num">{fmt_p(r.p_value)}</td>'
        f'<td>{pill("signal","hold") if r.p_value<.05 else pill("none","fail")}</td></tr>'
        for r in ind.itertuples()
    )
    clip = d["identifiability_clip"]
    c2 = clip[clip.multiplier == 2.0]
    repl["ROWS_CLIP"] = "".join(
        f'<tr><td class="num">{int(r.s)}</td>'
        f'<td class="num">{r.max_abs_diff_wins:.3f}</td>'
        f'<td>{pill("identical","hold") if r.max_abs_diff_wins<0.5 else pill("clip binds","warn")}</td></tr>'
        for r in c2.itertuples()
    )

    for key, fname in FIGURE_KEYS.items():
        repl[key] = data_uri(fname)

    html = tpl
    for k, v in repl.items():
        html = html.replace("{{" + k + "}}", str(v))

    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024/1024:.2f} MB)")
    leftover = [t for t in ("{{",) if t in html]
    if leftover:
        import re
        print("UNRESOLVED:", set(re.findall(r"\{\{(\w+)\}\}", html)))


SCALE_FREE = {"SD of win%", "Noll-Scully", "RSD of wins", "mean |win% - .5|"}
FEATURE_NAMES = {
    "mov": "margin of victory", "srs": "SRS", "win_pct": "win %",
    "elite": "elite this season", "prior_elite": "prior elite finishes",
    "prior_seasons": "seasons of history", "elite_rate": "historical elite rate",
}

if __name__ == "__main__":
    main()
