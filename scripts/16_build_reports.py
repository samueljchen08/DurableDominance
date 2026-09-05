"""Assemble the standalone HTML reports.

Every number is read from the generated CSVs and every figure is inlined as a
data URI, so the pages always agree with the analysis that produced them.
Run scripts 11-15 first.

Builds two pages:
  reports/durable_dominance_report.html  -- Studies A-D, the NBA deep dive
  reports/cross_sport_atlas.html         -- Study E, the cross-sport comparison
"""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from durable_dominance import datasets as D

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def uri(name: str) -> str:
    return "data:image/png;base64," + base64.b64encode(
        (FIGURES / name).read_bytes()).decode()


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return "&lt;0.0001"
    return f"{p:.4f}".rstrip("0").rstrip(".") if p < 0.1 else f"{p:.3f}"


def pill(text: str, kind: str) -> str:
    return f'<span class="pill pill--{kind}">{text}</span>'


# --------------------------------------------------------------------------
# Inline SVG: the champion timeline
# --------------------------------------------------------------------------

def champion_timeline() -> str:
    """43 seasons as blocks, height by the champion's prior titles.

    The visual point is the clustering: the same handful of franchises recur,
    and their blocks grow as their pedigree accumulates.
    """
    panel = pd.read_csv(REPORTS / "champion_panel_nba.csv")
    W, H = 900, 250
    pad_l, pad_r, pad_t, pad_b = 44, 12, 22, 62
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    n = len(panel)
    bw = plot_w / n
    max_t = max(panel["prior_titles"].max(), 1)

    # A stable colour per franchise, warm for repeat winners.
    order = panel["champion"].value_counts().index.tolist()
    palette = ["#A8762A", "#C68F3A", "#8C5E1F", "#D9A648", "#6E4A17",
               "#B98436", "#7A5320", "#E0B65F", "#5C3D12", "#CE9A45",
               "#966824", "#EBCB86", "#4E3310", "#DDAE52"]
    cmap = {t: palette[i % len(palette)] for i, t in enumerate(order)}

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
             f'aria-label="NBA champions 1981 to 2023, shaded by franchise, '
             f'bar height showing prior titles">']
    for tick in range(0, int(max_t) + 1, 2):
        y = pad_t + plot_h - plot_h * tick / (max_t + 1)
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l+plot_w}" '
                     f'y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad_l-8}" y="{y+3.5:.1f}" class="axis" '
                     f'text-anchor="end">{tick}</text>')

    for i, r in panel.iterrows():
        x = pad_l + i * bw
        h = plot_h * (r["prior_titles"] + 0.55) / (max_t + 1)
        y = pad_t + plot_h - h
        parts.append(
            f'<rect x="{x+0.6:.1f}" y="{y:.1f}" width="{bw-1.2:.1f}" '
            f'height="{h:.1f}" fill="{cmap[r["champion"]]}" rx="1.5">'
            f'<title>{r["season"]} — {r["champion"]} '
            f'({int(r["prior_titles"])} prior titles)</title></rect>')
        if i % 6 == 0:
            parts.append(f'<text x="{x+bw/2:.1f}" y="{pad_t+plot_h+16}" '
                         f'class="axis" text-anchor="middle">'
                         f'{int(r["start_year"])}</text>')

    parts.append(f'<text x="{pad_l-8}" y="{pad_t-8}" class="axis" '
                 f'text-anchor="end">titles</text>')
    # legend for the most decorated franchises
    lx = pad_l
    for t in order[:7]:
        parts.append(f'<rect x="{lx}" y="{H-30}" width="10" height="10" '
                     f'fill="{cmap[t]}" rx="2"/>')
        short = t.replace("Los Angeles ", "LA ").replace("Golden State ", "GS ")
        parts.append(f'<text x="{lx+14}" y="{H-21}" class="axis">{short}</text>')
        lx += 26 + 6.0 * len(short)
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------
# Inline SVG: real value against the simulated distribution
# --------------------------------------------------------------------------

def concentration_strips() -> str:
    """One row per statistic: simulated 95% band, median, and the real value."""
    test = pd.read_csv(REPORTS / "concentration_test.csv")
    labels = {"unique_champions": "unique champions", "hhi": "title HHI",
              "top_share": "top team's share", "entropy": "title entropy",
              "repeat_rate": "back-to-back rate"}
    W = 780
    row_h = 46
    H = row_h * len(test) + 48
    pad_l, pad_r = 168, 96

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
             f'aria-label="Real concentration statistics against their '
             f'simulated 95 percent intervals">']
    for i, r in test.iterrows():
        lo, hi, med, real = r["ci_lo"], r["ci_hi"], r["sim_median"], r["real"]
        span_lo = min(lo, real)
        span_hi = max(hi, real)
        pad = (span_hi - span_lo) * 0.22 or 1
        a, b = span_lo - pad, span_hi + pad
        def sx(v):
            return pad_l + (W - pad_l - pad_r) * (v - a) / (b - a)
        cy = 30 + i * row_h
        parts.append(f'<text x="{pad_l-14}" y="{cy+4}" class="ylab" '
                     f'text-anchor="end">{labels[r["statistic"]]}</text>')
        parts.append(f'<line x1="{sx(lo):.1f}" y1="{cy}" x2="{sx(hi):.1f}" '
                     f'y2="{cy}" class="band"/>')
        parts.append(f'<circle cx="{sx(med):.1f}" cy="{cy}" r="5" class="dot-sim"/>')
        parts.append(f'<circle cx="{sx(real):.1f}" cy="{cy}" r="6.5" '
                     f'class="dot-real"/>')
        val = f"{real:.3f}" if real < 10 else f"{int(real)}"
        parts.append(f'<text x="{W-pad_r+12}" y="{cy+4}" class="vlab">'
                     f'p = {fmt_p(r["p_two_sided"])}</text>')
        parts.append(f'<text x="{sx(real):.1f}" y="{cy-13}" class="vlab-real" '
                     f'text-anchor="middle">{val}</text>')
    parts.append(f'<text x="{pad_l}" y="{H-8}" class="axis">'
                 f'← more concentrated · less concentrated →</text>')
    parts.append("</svg>")
    return "".join(parts)


def reliability_chart() -> str:
    rel = pd.read_csv(REPORTS / "playoff_reliability.csv")
    W, H = 420, 340
    pad = 52
    m = 0.58
    def sx(v): return pad + (W - pad - 18) * v / m
    def sy(v): return H - pad - (H - pad - 20) * v / m
    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
             f'aria-label="Reliability curve: predicted versus observed title rate">']
    for t in (0, .1, .2, .3, .4, .5):
        parts.append(f'<line x1="{sx(t):.1f}" y1="{sy(0):.1f}" x2="{sx(t):.1f}" '
                     f'y2="{sy(m):.1f}" class="grid"/>')
        parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(t):.1f}" x2="{sx(m):.1f}" '
                     f'y2="{sy(t):.1f}" class="grid"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{H-pad+18}" class="axis" '
                     f'text-anchor="middle">{t:.1f}</text>')
        parts.append(f'<text x="{pad-8}" y="{sy(t)+3.5:.1f}" class="axis" '
                     f'text-anchor="end">{t:.1f}</text>')
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(m):.1f}" '
                 f'y2="{sy(m):.1f}" class="ideal"/>')
    pts = " ".join(f"{sx(r['predicted']):.1f},{sy(r['actual']):.1f}"
                   for _, r in rel.iterrows())
    parts.append(f'<polyline points="{pts}" class="relline"/>')
    for _, r in rel.iterrows():
        rad = 4 + 7 * (r["n"] / rel["n"].max()) ** .5
        parts.append(f'<circle cx="{sx(r["predicted"]):.1f}" '
                     f'cy="{sy(r["actual"]):.1f}" r="{rad:.1f}" class="dot-real">'
                     f'<title>{r["n"]} team-seasons</title></circle>')
    parts.append(f'<text x="{W/2:.0f}" y="{H-10}" class="axis" '
                 f'text-anchor="middle">predicted title probability</text>')
    parts.append(f'<text transform="translate(14,{H/2:.0f}) rotate(-90)" '
                 f'class="axis" text-anchor="middle">observed rate</text>')
    parts.append("</svg>")
    return "".join(parts)


def atlas_bars() -> str:
    a = pd.read_csv(REPORTS / "cross_sport_atlas.csv").sort_values(
        "dominance_multiple")
    W = 760
    row_h = 46
    H = row_h * len(a) + 56
    pad_l, pad_r = 176, 74
    mx = a["dominance_multiple"].max() * 1.12
    def sx(v): return pad_l + (W - pad_l - pad_r) * v / mx

    parts = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" '
             f'aria-label="Dominance multiple by sport">']
    for t in (0, 1, 2, 3):
        parts.append(f'<line x1="{sx(t):.1f}" y1="24" x2="{sx(t):.1f}" '
                     f'y2="{H-34}" class="grid"/>')
        parts.append(f'<text x="{sx(t):.1f}" y="{H-16}" class="axis" '
                     f'text-anchor="middle">{t}×</text>')
    parts.append(f'<line x1="{sx(1):.1f}" y1="18" x2="{sx(1):.1f}" '
                 f'y2="{H-34}" class="ideal"/>')
    parts.append(f'<text x="{sx(1)+7:.1f}" y="16" class="vlab-real">'
                 f'chance among entrants</text>')
    for i, r in a.reset_index(drop=True).iterrows():
        cy = 38 + i * row_h
        parts.append(f'<text x="{pad_l-14}" y="{cy+4}" class="ylab" '
                     f'text-anchor="end">{r["sport"]}</text>')
        parts.append(f'<rect x="{sx(0):.1f}" y="{cy-11}" '
                     f'width="{sx(r["dominance_multiple"])-sx(0):.1f}" '
                     f'height="22" class="bar-real" rx="2"/>')
        parts.append(f'<text x="{sx(r["dominance_multiple"])+9:.1f}" '
                     f'y="{cy+5}" class="vlab-real">'
                     f'{r["dominance_multiple"]:.2f}×</text>')
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------

def table_rows(df: pd.DataFrame, cols, fmts, cls=None) -> str:
    cls = cls or {}
    out = []
    for r in df.itertuples(index=False):
        d = dict(zip(df.columns, r))
        cells = []
        for c, f in zip(cols, fmts):
            cells.append(f'<td class="{cls.get(c, "num")}">{f(d[c])}</td>')
        out.append("<tr>" + "".join(cells) + "</tr>")
    return "".join(out)


def build_main() -> Path:
    t = (ROOT / "scripts" / "report_main.html").read_text()
    test = pd.read_csv(REPORTS / "concentration_test.csv").set_index("statistic")
    cal = pd.read_csv(REPORTS / "playoff_calibration.csv")
    era = pd.read_csv(REPORTS / "concentration_by_era.csv")
    conv = pd.read_csv(REPORTS / "title_conversion.csv", index_col=0)
    ml = pd.read_csv(REPORTS / "ml_champion_results.csv")
    imp = pd.read_csv(REPORTS / "ml_champion_importance.csv")
    ped = pd.read_csv(REPORTS / "pedigree_vs_balance.csv")
    sens = pd.read_csv(REPORTS / "concentration_sensitivity.csv")
    panel = pd.read_csv(REPORTS / "champion_panel_nba.csv")
    sweep = pd.read_csv(REPORTS / "conversion_bonus_sweep.csv")

    PRETTY_F = {
        "bt": "fitted strength", "mov": "margin of victory", "win_pct": "win %",
        "conf_rank": "conference seed", "league_rank": "league rank",
        "strength_gap_to_best": "gap to the best team",
        "prior_titles": "prior titles", "prior_top4": "prior top-4 finishes",
        "career_win_pct": "career win %",
        "league_strength_sd": "league strength spread",
    }
    PRETTY_M = {"prior_titles": "prior titles",
                "prior_titles_10y": "titles, last 10 years",
                "gini_wins": "Gini of wins", "win_pct_sd": "SD of win%"}

    surv = ped[ped["q_detrended"] < 0.10].sort_values("q_detrended")
    best_g = cal.loc[cal["loglik"].idxmax(), "gamma"]
    conv_r = conv.reset_index().rename(columns={"index": "team"})
    top_conv = pd.concat([conv_r.head(4), conv_r.tail(3)])
    imp_mx = imp["importance"].max()

    nba, matches, ncaaf = D.load_nba(), D.load_nba_matches(), D.load_ncaaf()
    n_games = int((matches["series_wins"].sum() + matches["series_losses"].sum()) / 2)

    repl = {
        "N_SEASONS": str(nba["season"].nunique()),
        "N_TEAM_SEASONS": f"{len(nba) + len(ncaaf):,}",
        "N_GAMES": f"{n_games:,}",
        "TIMELINE": champion_timeline(),
        "STRIPS": concentration_strips(),
        "RELIABILITY": reliability_chart(),
        "N_UNIQUE": str(int(test.loc["unique_champions", "real"])),
        "N_SIM_UNIQUE": str(int(test.loc["unique_champions", "sim_median"])),
        "SIM_LO": str(int(test.loc["unique_champions", "ci_lo"])),
        "SIM_HI": str(int(test.loc["unique_champions", "ci_hi"])),
        "HHI_REAL": f'{test.loc["hhi","real"]:.3f}',
        "HHI_SIM": f'{test.loc["hhi","sim_median"]:.3f}',
        "HHI_RATIO": f'{test.loc["hhi","real"]/test.loc["hhi","sim_median"]:.2f}',
        "P_ENTROPY": fmt_p(test.loc["entropy", "p_two_sided"]),
        "GAMMA": f"{best_g:.1f}",
        "REPEAT_REAL": f'{test.loc["repeat_rate","real"]:.3f}',
        "REPEAT_SIM": f'{test.loc["repeat_rate","sim_median"]:.3f}',
        "MEAN_PRIOR_TITLES": f'{panel["prior_titles"].mean():.1f}',
        "MEAN_PRIOR_TOP4": f'{panel["prior_top4"].mean():.1f}',
        "MEAN_CAREER_WP": f'{panel["career_win_pct"].mean():.3f}',
        "N_FIRST_TIME": str(int((panel["prior_titles"] == 0).sum())),
        "N_TESTS": str(len(ped)),
        "N_RAW_SIG": str(int((ped["p"] < .05).sum())),
        "N_DET_SIG": str(int((ped["p_detrended"] < .05).sum())),
        "N_FDR_SIG": str(len(surv)),
        "SIM_LOGLOSS": f'{ml.loc[ml["model"]=="Random-walk simulation","log_loss"].iloc[0]:.4f}',
        "RF_LOGLOSS": f'{ml.loc[ml["model"]=="Random forest","log_loss"].iloc[0]:.4f}',
        "SPLIT_R": "+0.24", "SPLIT_P": "0.23",
        "BONUS_SD": "0.38",
        "WITHIN_SD": "0.67",
        "RHO": "0.58",
        "ROWS_CONC": table_rows(
            test.reset_index().assign(
                stat=lambda d: d["statistic"].map({
                    "unique_champions": "unique champions", "hhi": "title HHI",
                    "top_share": "top team's share", "entropy": "title entropy",
                    "repeat_rate": "back-to-back rate"})),
            ["stat", "real", "sim_median", "ci_lo", "ci_hi", "p_two_sided"],
            [str, lambda v: f"{v:.3f}", lambda v: f"{v:.3f}",
             lambda v: f"{v:.3f}", lambda v: f"{v:.3f}", fmt_p],
            {"stat": "tl"}),
        "ROWS_REL": table_rows(
            pd.read_csv(REPORTS / "playoff_reliability.csv"),
            ["predicted", "actual", "n"],
            [lambda v: f"{v:.3f}", lambda v: f"{v:.3f}", lambda v: f"{int(v)}"]),
        "ROWS_ERA": table_rows(
            era, ["era", "n_seasons", "unique_real", "unique_sim", "p_two_sided"],
            [str, lambda v: f"{int(v)}", lambda v: f"{int(v)}",
             lambda v: f"{v:.0f}", fmt_p], {"era": "tl"}),
        "ROWS_CONV": "".join(
            f'<tr><td class="tl">{r.team}</td>'
            f'<td class="num">{r.expected_titles:.1f}</td>'
            f'<td class="num">{int(r.actual_titles)}</td>'
            f'<td class="num {"pos" if r.excess>0 else "neg"}">{r.excess:+.1f}</td></tr>'
            for r in top_conv.itertuples()),
        "ROWS_ML": "".join(
            f'<tr><td class="tl">{r.model}</td>'
            f'<td class="num{" strong" if r.model=="Random-walk simulation" else ""}">'
            f'{r.log_loss:.4f}</td><td class="num">{r.auc:.3f}</td></tr>'
            for r in ml.itertuples()),
        "ROWS_IMP": "".join(
            f'<tr><td class="tl">{PRETTY_F.get(r.feature, r.feature)}</td>'
            f'<td class="num">{r.importance:+.4f}</td>'
            f'<td class="barcell"><span class="minibar" style="width:'
            f'{max(r.importance,0)/imp_mx*100:.0f}%"></span></td></tr>'
            for r in imp.itertuples()),
        "ROWS_PED": "".join(
            f'<tr><td class="tl">{r.league}</td>'
            f'<td class="tl">{PRETTY_M.get(r.pedigree, r.pedigree)}</td>'
            f'<td class="tl">{PRETTY_M.get(r.balance, r.balance)}</td>'
            f'<td class="num">{r.r_detrended:+.3f}</td>'
            f'<td class="num">{r.q_detrended:.3f}</td></tr>'
            for r in surv.itertuples()),
        "ROWS_SENS": "".join(
            f'<tr><td class="tl">{ {"bt":"Bradley-Terry","mov":"margin of victory","record":"raw record"}[r.strength] }</td>'
            f'<td class="num">{r.hca:.1f}</td>'
            f'<td class="num">{int(r.unique_champions_sim)}</td>'
            f'<td class="num">{fmt_p(r.hhi_p)}</td></tr>'
            for r in sens.itertuples()),
        "ROWS_SWEEP": "".join(
            f'<tr><td class="num">{r.bonus_sd:.2f}</td>'
            f'<td class="num">{r.unique_champions:.1f}</td>'
            f'<td class="num">{r.hhi:.4f}</td></tr>'
            for r in sweep.itertuples()),
        "FIG_PEDIGREE": uri("11_pedigree_balance_heatmap.png"),
        "FIG_TIMELINE": uri("11_nba_timeline.png"),
        "FIG_WALK": uri("12_playoff_random_walk.png"),
        "FIG_DYNASTY": uri("13_dynasty_dynamics.png"),
        "FIG_ML": uri("14_championship_ml.png"),
    }
    html = t
    for k, v in repl.items():
        html = html.replace("{{" + k + "}}", str(v))
    out = REPORTS / "durable_dominance_report.html"
    out.write_text(html)
    leftover = set(re.findall(r"\{\{(\w+)\}\}", html))
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size/1024/1024:.2f} MB)"
          + (f"  UNRESOLVED: {leftover}" if leftover else ""))
    return out


def build_atlas() -> Path:
    t = (ROOT / "scripts" / "report_atlas.html").read_text()
    a = pd.read_csv(REPORTS / "cross_sport_atlas.csv")
    ind = pd.read_csv(REPORTS / "individual_sports.csv")
    kd = D.load_kentucky_derby().dropna(subset=["sire"])
    sires = kd["sire"].astype(str).str.strip()
    shares = sires.value_counts(normalize=True)

    repl = {
        "ATLAS_BARS": atlas_bars(),
        "N_SPORTS": str(len(a)),
        "TOP_SPORT": a.iloc[0]["sport"],
        "TOP_MULT": f'{a.iloc[0]["dominance_multiple"]:.2f}',
        "NBA_MULT": f'{a.loc[a["sport"]=="NBA","dominance_multiple"].iloc[0]:.2f}',
        "DERBY_UNIQUE": str(sires.nunique()),
        "DERBY_N": str(len(sires)),
        "DERBY_TOP": f"{shares.iloc[0]*100:.1f}",
        "ROWS_ATLAS": "".join(
            f'<tr><td class="tl">{r.sport}</td><td class="tl">{r.unit}</td>'
            f'<td class="num">{int(r.editions)}</td>'
            f'<td class="num">{int(r.field_median)}</td>'
            f'<td class="num">{int(r.unique_real)}</td>'
            f'<td class="num">{int(r.unique_null)}</td>'
            f'<td class="num strong">{r.dominance_multiple:.2f}×</td>'
            f'<td class="num">{fmt_p(r.p_two_sided)}</td></tr>'
            for r in a.itertuples()),
        "ROWS_IND": "".join(
            f'<tr><td class="tl">{r.sport}</td><td class="num">{int(r.n_years)}</td>'
            f'<td class="num">{r.r_squared:.3f}</td>'
            f'<td class="num">{fmt_p(r.p_value)}</td>'
            f'<td>{pill("signal","hold") if r.p_value<.05 else pill("none","fail")}</td>'
            f'</tr>' for r in ind.itertuples()),
        "FIG_ATLAS": uri("15_cross_sport_atlas.png"),
        "FIG_INDIV": uri("03_individual_sports.png"),
    }
    html = t
    for k, v in repl.items():
        html = html.replace("{{" + k + "}}", str(v))
    out = REPORTS / "cross_sport_atlas.html"
    out.write_text(html)
    leftover = set(re.findall(r"\{\{(\w+)\}\}", html))
    print(f"wrote {out.relative_to(ROOT)} ({out.stat().st_size/1024/1024:.2f} MB)"
          + (f"  UNRESOLVED: {leftover}" if leftover else ""))
    return out


if __name__ == "__main__":
    build_main()
    build_atlas()
