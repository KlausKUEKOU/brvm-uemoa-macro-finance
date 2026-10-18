"""CFA franc zones and the BRVM: inflation, growth and stock market behaviour.

Part 1  Does membership of a CFA monetary union (UEMOA / BCEAO, CEMAC / BEAC) deliver lower and
        more stable inflation than neighbouring countries with their own currency, and at what
        cost in growth?
Part 2  How does the BRVM Composite behave: returns, risk, drawdowns, and is it weak-form
        efficient?

Usage: python run_analysis.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

ROOT = Path(__file__).parent
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

UEMOA = {"BEN": "Benin", "BFA": "Burkina Faso", "CIV": "Cote d'Ivoire", "GNB": "Guinea-Bissau",
         "MLI": "Mali", "NER": "Niger", "SEN": "Senegal", "TGO": "Togo"}
CEMAC = {"CMR": "Cameroon", "CAF": "Central African Republic", "TCD": "Chad", "COG": "Congo",
         "GNQ": "Equatorial Guinea", "GAB": "Gabon"}
# West and Central African neighbours with their own floating or managed currency
OTHERS = {"NGA": "Nigeria", "GHA": "Ghana", "GIN": "Guinea", "SLE": "Sierra Leone", "GMB": "Gambia",
          "LBR": "Liberia", "MRT": "Mauritania", "COD": "Democratic Republic of Congo"}
GROUP = {**{k: "UEMOA" for k in UEMOA}, **{k: "CEMAC" for k in CEMAC}, **{k: "Non-CFA" for k in OTHERS}}
NAMES = {**UEMOA, **CEMAC, **OTHERS}
COLORS = {"UEMOA": "#1f77b4", "CEMAC": "#2ca02c", "Non-CFA": "#d62728"}
START, END = 1995, 2023            # after the January 1994 CFA devaluation

# ======================================================================= PART 1: macro
cpi = pd.read_csv(ROOT / "data/cpi_inflation_world_bank.csv")
cpi = cpi[cpi["Country Code"].isin(GROUP) & cpi["Year"].between(START, END)].copy()
cpi["group"] = cpi["Country Code"].map(GROUP)
cpi["cfa"] = (cpi["group"] != "Non-CFA").astype(int)
cpi = cpi.rename(columns={"Inflation": "infl", "Country Code": "iso"})

summary = cpi.groupby("group")["infl"].agg(
    mean="mean", median="median", std="std",
    share_below_3=lambda s: (s <= 3).mean(),
    share_above_10=lambda s: (s > 10).mean(),
    country_years="count")
vol = cpi.groupby(["group", "iso"])["infl"].std().groupby("group").median()
summary["median_country_volatility"] = vol
summary = summary.loc[["UEMOA", "CEMAC", "Non-CFA"]]
summary.to_csv(ROOT / "results/inflation_summary.csv", float_format="%.3f")

# Panel regression with year fixed effects; errors clustered by country.
# Inflation is winsorised at 1%/99% so that DRC hyperinflation does not drive the result.
lo, hi = cpi["infl"].quantile([0.01, 0.99])
cpi["infl_w"] = cpi["infl"].clip(lo, hi)
cpi["uemoa"] = (cpi["group"] == "UEMOA").astype(int)
cpi["cemac"] = (cpi["group"] == "CEMAC").astype(int)
reg_rows = []
for label, formula in [("CFA dummy", "infl_w ~ cfa + C(Year)"),
                       ("By union", "infl_w ~ uemoa + cemac + C(Year)")]:
    m = smf.ols(formula, data=cpi).fit(cov_type="cluster", cov_kwds={"groups": cpi["iso"]})
    for v in ["cfa", "uemoa", "cemac"]:
        if v in m.params:
            reg_rows.append({"Specification": label, "Variable": v, "Coefficient (pp)": m.params[v],
                             "Clustered SE": m.bse[v], "p-value": m.pvalues[v], "N": int(m.nobs),
                             "R2": m.rsquared})
# Median (quantile) regression: robust to outliers without winsorising
mq = smf.quantreg("infl ~ cfa + C(Year)", data=cpi).fit(q=0.5)
reg_rows.append({"Specification": "Median regression", "Variable": "cfa",
                 "Coefficient (pp)": mq.params["cfa"], "Clustered SE": mq.bse["cfa"],
                 "p-value": mq.pvalues["cfa"], "N": int(mq.nobs), "R2": np.nan})
pd.DataFrame(reg_rows).to_csv(ROOT / "results/inflation_regressions.csv", index=False, float_format="%.4f")

# 2022 imported-inflation shock
shock = cpi[cpi["Year"] == 2022].groupby("group")["infl"].median().reindex(["UEMOA", "CEMAC", "Non-CFA"])
shock.to_csv(ROOT / "results/inflation_2022_median.csv", header=["median_inflation_2022"], float_format="%.2f")

fig, ax = plt.subplots(figsize=(9, 4.5))
for g in ["UEMOA", "CEMAC", "Non-CFA"]:
    s = cpi[cpi["group"] == g].groupby("Year")["infl"]
    med, q1, q3 = s.median(), s.quantile(0.25), s.quantile(0.75)
    ax.plot(med.index, med, color=COLORS[g], lw=2, label=f"{g} (median)")
    ax.fill_between(med.index, q1, q3, color=COLORS[g], alpha=0.12)
ax.axhline(3, color="black", ls="--", lw=1, label="Convergence criterion (3%)")
ax.set_ylim(-5, 30)
ax.set_ylabel("CPI inflation (%)")
ax.set_title("Inflation since the 1994 devaluation: CFA unions vs neighbours (median, interquartile band)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/inflation_by_group.png")
plt.close(fig)

# ---------------------------------------------------------------------- growth (Maddison)
mad = pd.read_csv(ROOT / "data/maddison_2020.csv")
inv = {v: k for k, v in NAMES.items()}
mad = mad[mad["Entity"].isin(inv)].copy()
mad["iso"] = mad["Entity"].map(inv)
mad["group"] = mad["iso"].map(GROUP)
mad = mad.sort_values(["iso", "Year"])
mad["growth"] = 100 * mad.groupby("iso")["GDP per capita"].transform(lambda s: np.log(s).diff())
gr = mad[mad["Year"].between(START, 2018)].dropna(subset=["growth"])
gsum = gr.groupby("group")["growth"].agg(mean="mean", median="median", std="std", country_years="count")
gsum["median_country_volatility"] = gr.groupby(["group", "iso"])["growth"].std().groupby("group").median()
cum = (mad[mad["Year"].isin([START, 2018])].pivot(index="iso", columns="Year", values="GDP per capita"))
cum["growth_1995_2018_pct"] = 100 * (cum[2018] / cum[START] - 1)
cum["group"] = cum.index.map(GROUP)
gsum["median_cumulative_growth_1995_2018"] = cum.groupby("group")["growth_1995_2018_pct"].median()
gsum = gsum.loc[["UEMOA", "CEMAC", "Non-CFA"]]
gsum.to_csv(ROOT / "results/growth_summary.csv", float_format="%.3f")
cum.drop(columns="group").assign(group=cum["group"], country=cum.index.map(NAMES)).to_csv(
    ROOT / "results/gdp_per_capita_by_country.csv", float_format="%.1f")

gr["cfa"] = (gr["group"] != "Non-CFA").astype(int)
lo, hi = gr["growth"].quantile([0.01, 0.99])
gr["growth_w"] = gr["growth"].clip(lo, hi)
mg = smf.ols("growth_w ~ cfa + C(Year)", data=gr).fit(cov_type="cluster", cov_kwds={"groups": gr["iso"]})
pd.DataFrame([{"Variable": "cfa", "Coefficient (pp)": mg.params["cfa"], "Clustered SE": mg.bse["cfa"],
               "p-value": mg.pvalues["cfa"], "N": int(mg.nobs)}]).to_csv(
    ROOT / "results/growth_regression.csv", index=False, float_format="%.4f")

idx = mad[mad["Year"].between(START, 2018)].copy()
idx["index"] = idx.groupby("iso")["GDP per capita"].transform(lambda s: 100 * s / s.iloc[0])
fig, ax = plt.subplots(figsize=(9, 4.5))
for g in ["UEMOA", "CEMAC", "Non-CFA"]:
    s = idx[idx["group"] == g].groupby("Year")["index"].median()
    ax.plot(s.index, s, color=COLORS[g], lw=2, label=g)
ax.set_ylabel("Real GDP per capita, 1995 = 100 (median country)")
ax.set_title("Real GDP per capita since 1995 (Maddison Project 2020, 2011 PPP)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/gdp_per_capita_by_group.png")
plt.close(fig)

# ======================================================================= PART 2: BRVM
b = pd.read_csv(ROOT / "data/brvm_composite_daily.csv", parse_dates=["Date"], index_col="Date")["Close"]
b = b.loc[:"2025-12-31"]                             # full calendar years only
r = np.log(b).diff().dropna()


def stats_block(x, per_year):
    ann_r = x.mean() * per_year
    ann_v = x.std() * np.sqrt(per_year)
    return ann_r, ann_v


periods = {"1998-2025 (full)": ("1998", "2025"), "1998-2005": ("1998", "2005"),
           "2006-2015 (boom)": ("2006", "2015"), "2016-2020 (bear market)": ("2016", "2020"),
           "2021-2025 (recovery)": ("2021", "2025")}
rows = []
for name, (a, z) in periods.items():
    x = r.loc[a:z]
    wealth = np.exp(x.cumsum())
    rows.append({"Period": name, "Trading days": len(x),
                 "Annualised log return": x.mean() * 252, "Annualised volatility": x.std() * np.sqrt(252),
                 "Skewness": stats.skew(x), "Excess kurtosis": stats.kurtosis(x),
                 "Max drawdown": (wealth / wealth.cummax() - 1).min(),
                 "Share of zero-return days": (x == 0).mean()})
brvm_stats = pd.DataFrame(rows).set_index("Period")
brvm_stats.to_csv(ROOT / "results/brvm_statistics.csv", float_format="%.4f")

yearly = b.resample("YE").last()
yearly_ret = (yearly.pct_change().dropna() * 100).rename("Annual return (%)")
yearly_ret.index = yearly_ret.index.year
yearly_ret.to_csv(ROOT / "results/brvm_annual_returns.csv", float_format="%.2f")


def variance_ratio(x, q):
    """Lo and MacKinlay (1988) variance ratio with heteroskedasticity-robust z-statistic."""
    x = np.asarray(x)
    T = len(x)
    mu = x.mean()
    s1 = np.sum((x - mu) ** 2) / (T - 1)
    xq = np.convolve(x, np.ones(q), "valid")
    m = q * (T - q + 1) * (1 - q / T)
    sq = np.sum((xq - q * mu) ** 2) / m
    vr = sq / s1
    dev2 = (x - mu) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = np.sum(dev2[j:] * dev2[:-j]) / np.sum(dev2) ** 2 * T
        theta += (2 * (q - j) / q) ** 2 * delta
    z = (vr - 1) / np.sqrt(theta / T)
    return vr, z, 2 * (1 - stats.norm.cdf(abs(z)))


eff_rows = []
for name, (a, z) in [("1998-2010", ("1998", "2010")), ("2011-2025", ("2011", "2025"))]:
    x = r.loc[a:z]
    ac1 = x.autocorr(1)
    lb = acorr_ljungbox(x, lags=[10])["lb_pvalue"].iloc[0]
    row = {"Sample": name, "AR(1) autocorrelation": ac1, "Ljung-Box(10) p-value": lb}
    for q in (2, 5, 10, 20):
        vr, zz, p = variance_ratio(x.values, q)
        row[f"VR({q})"] = vr
        row[f"VR({q}) p-value"] = p
    eff_rows.append(row)
eff = pd.DataFrame(eff_rows).set_index("Sample")
eff.to_csv(ROOT / "results/brvm_efficiency_tests.csv", float_format="%.4f")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
axes[0].plot(b, color="#1f4e79", lw=1)
axes[0].set_ylabel("BRVM Composite")
axes[0].set_title("BRVM Composite index, 1998 to 2025")
dd = b / b.cummax() - 1
axes[1].fill_between(dd.index, dd, 0, color="#c0392b", alpha=0.5)
axes[1].set_ylabel("Drawdown")
fig.tight_layout()
fig.savefig(ROOT / "figures/brvm_composite.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.bar(yearly_ret.index, yearly_ret, color=np.where(yearly_ret >= 0, "#2ca02c", "#c0392b"))
ax.set_ylabel("%")
ax.set_title("BRVM Composite annual returns (price index, dividends excluded)")
fig.tight_layout()
fig.savefig(ROOT / "figures/brvm_annual_returns.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
lags = range(1, 21)
for name, (a, z), c in [("1998-2010", ("1998", "2010"), "#d62728"), ("2011-2025", ("2011", "2025"), "#1f77b4")]:
    x = r.loc[a:z]
    ax.bar(np.array(lags) + (-0.2 if c == "#d62728" else 0.2), [x.autocorr(k) for k in lags], width=0.4,
           color=c, label=name)
    ax.axhline(1.96 / np.sqrt(len(x)), color=c, ls=":", lw=1)
    ax.axhline(-1.96 / np.sqrt(len(x)), color=c, ls=":", lw=1)
ax.set_xticks(range(1, 21))
ax.set_xlabel("Lag (trading days)")
ax.set_ylabel("Autocorrelation of daily returns")
ax.set_title("BRVM returns are predictable from their own past")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/brvm_autocorrelation.png")
plt.close(fig)

pd.set_option("display.width", 200)
for t in (summary, pd.DataFrame(reg_rows), shock, gsum, brvm_stats, yearly_ret.tail(8), eff):
    print(t.round(3) if hasattr(t, "round") else t, "\n")
print("growth reg cfa:", round(mg.params["cfa"], 3), "p", round(mg.pvalues["cfa"], 3))
