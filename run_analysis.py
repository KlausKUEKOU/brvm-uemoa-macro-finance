# Zones franc CFA et BRVM : inflation, croissance et comportement de la bourse.
#
# Partie 1 : les pays d'une union monétaire CFA (UEMOA / BCEAO, CEMAC / BEAC) ont-ils une
#            inflation plus basse et plus stable que leurs voisins à monnaie propre ?
#            Est-ce que cela leur coûte de la croissance ?
# Partie 2 : rendement, risque et efficience du BRVM Composite.
#
# Lancement : python run_analysis.py

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf

from src.basic_stats import (excess_kurtosis, lag_correlation, ljung_box_pvalue, mean, median, quantile,
                             skewness, std, variance_ratio)

ROOT = Path(__file__).parent
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3})

UEMOA = {"BEN": "Benin", "BFA": "Burkina Faso", "CIV": "Cote d'Ivoire", "GNB": "Guinea-Bissau",
         "MLI": "Mali", "NER": "Niger", "SEN": "Senegal", "TGO": "Togo"}
CEMAC = {"CMR": "Cameroon", "CAF": "Central African Republic", "TCD": "Chad", "COG": "Congo",
         "GNQ": "Equatorial Guinea", "GAB": "Gabon"}
OTHERS = {"NGA": "Nigeria", "GHA": "Ghana", "GIN": "Guinea", "SLE": "Sierra Leone", "GMB": "Gambia",
          "LBR": "Liberia", "MRT": "Mauritania", "COD": "Democratic Republic of Congo"}
GROUPS = ["UEMOA", "CEMAC", "Non-CFA"]
COLORS = {"UEMOA": "#1f77b4", "CEMAC": "#2ca02c", "Non-CFA": "#d62728"}
START, END = 1995, 2023                 # après la dévaluation du franc CFA de janvier 1994

GROUP_OF = {}                           # code pays -> groupe
NAME_OF = {}                            # code pays -> nom
for code in UEMOA:
    GROUP_OF[code], NAME_OF[code] = "UEMOA", UEMOA[code]
for code in CEMAC:
    GROUP_OF[code], NAME_OF[code] = "CEMAC", CEMAC[code]
for code in OTHERS:
    GROUP_OF[code], NAME_OF[code] = "Non-CFA", OTHERS[code]


def write_table(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.3f")


# ======================================================================= PARTIE 1 : MACRO
# ---------------- 1.a Lecture de l'inflation : une ligne = (pays, année, inflation)
raw = pd.read_csv(ROOT / "data/cpi_inflation_world_bank.csv")
panel = []
for k in range(len(raw)):
    code = raw["Country Code"].iloc[k]
    year = int(raw["Year"].iloc[k])
    if code in GROUP_OF and START <= year <= END:
        panel.append({"iso": code, "Year": year, "infl": float(raw["Inflation"].iloc[k]),
                      "group": GROUP_OF[code]})

# ---------------- 1.b Résumé par groupe
# POUR chaque groupe : moyenne, médiane, écart-type, part des années <= 3 % (critère de
# convergence), part des années > 10 %, et volatilité médiane (écart-type pays par pays).
summary = []
for g in GROUPS:
    values = []
    by_country = {}
    for obs in panel:
        if obs["group"] == g:
            values.append(obs["infl"])
            by_country.setdefault(obs["iso"], []).append(obs["infl"])
    below_3, above_10 = 0, 0
    for v in values:
        if v <= 3:
            below_3 = below_3 + 1
        if v > 10:
            above_10 = above_10 + 1
    country_vol = []
    for iso in by_country:
        country_vol.append(std(by_country[iso]))
    summary.append({"group": g, "mean": mean(values), "median": median(values), "std": std(values),
                    "share_below_3": below_3 / len(values), "share_above_10": above_10 / len(values),
                    "country_years": len(values), "median_country_volatility": median(country_vol)})
write_table(summary, ROOT / "results/inflation_summary.csv")

# ---------------- 1.c Régression en panel
# inflation(pays, année) = a + b x CFA(pays) + effet de l'année + erreur
#   - l'effet de l'année absorbe les chocs communs (pétrole, prix alimentaires, 2022...)
#   - erreurs "clusterisées" par pays : les années d'un même pays ne sont pas indépendantes
#   - inflation écrêtée aux centiles 1 % et 99 % pour que l'hyperinflation de la RDC ne domine pas
all_infl = []
for obs in panel:
    all_infl.append(obs["infl"])
low, high = quantile(all_infl, 0.01), quantile(all_infl, 0.99)
for obs in panel:
    obs["infl_w"] = min(max(obs["infl"], low), high)
    obs["cfa"] = 1 if obs["group"] != "Non-CFA" else 0
    obs["uemoa"] = 1 if obs["group"] == "UEMOA" else 0
    obs["cemac"] = 1 if obs["group"] == "CEMAC" else 0
df = pd.DataFrame(panel)

regressions = []
for label, formula in [("CFA dummy", "infl_w ~ cfa + C(Year)"),
                       ("By union", "infl_w ~ uemoa + cemac + C(Year)")]:
    model = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["iso"]})
    for v in ["cfa", "uemoa", "cemac"]:
        if v in model.params:
            regressions.append({"Specification": label, "Variable": v, "Coefficient (pp)": model.params[v],
                                "Clustered SE": model.bse[v], "p-value": model.pvalues[v],
                                "N": int(model.nobs), "R2": model.rsquared})
# régression médiane : insensible aux valeurs extrêmes sans avoir besoin d'écrêter
model = smf.quantreg("infl ~ cfa + C(Year)", data=df).fit(q=0.5)
regressions.append({"Specification": "Median regression", "Variable": "cfa",
                    "Coefficient (pp)": model.params["cfa"], "Clustered SE": model.bse["cfa"],
                    "p-value": model.pvalues["cfa"], "N": int(model.nobs), "R2": float("nan")})
pd.DataFrame(regressions).to_csv(ROOT / "results/inflation_regressions.csv", index=False, float_format="%.4f")

# ---------------- 1.d Le choc de 2022
shock = []
for g in GROUPS:
    values = []
    for obs in panel:
        if obs["group"] == g and obs["Year"] == 2022:
            values.append(obs["infl"])
    shock.append({"group": g, "median_inflation_2022": median(values)})
pd.DataFrame(shock).to_csv(ROOT / "results/inflation_2022_median.csv", index=False, float_format="%.2f")

# graphique : médiane par année et bande entre les quartiles
fig, ax = plt.subplots(figsize=(9, 4.5))
for g in GROUPS:
    years, med, q1, q3 = [], [], [], []
    for year in range(START, END + 1):
        values = []
        for obs in panel:
            if obs["group"] == g and obs["Year"] == year:
                values.append(obs["infl"])
        if len(values) > 0:
            years.append(year)
            med.append(median(values))
            q1.append(quantile(values, 0.25))
            q3.append(quantile(values, 0.75))
    ax.plot(years, med, color=COLORS[g], lw=2, label=g + " (median)")
    ax.fill_between(years, q1, q3, color=COLORS[g], alpha=0.12)
ax.axhline(3, color="black", ls="--", lw=1, label="Convergence criterion (3%)")
ax.set_ylim(-5, 30)
ax.set_ylabel("CPI inflation (%)")
ax.set_title("Inflation since the 1994 devaluation: CFA unions vs neighbours (median, interquartile band)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/inflation_by_group.png")
plt.close(fig)

# ---------------- 1.e Croissance du PIB par habitant (Maddison)
# croissance(t) = 100 x ln(PIB par hab.(t) / PIB par hab.(t-1))
mad = pd.read_csv(ROOT / "data/maddison_2020.csv")
code_of_name = {}
for code in NAME_OF:
    code_of_name[NAME_OF[code]] = code

gdp = {}                                            # gdp[code][année] = PIB par habitant
for k in range(len(mad)):
    name = mad["Entity"].iloc[k]
    value = mad["GDP per capita"].iloc[k]
    if name in code_of_name and not pd.isna(value):
        gdp.setdefault(code_of_name[name], {})[int(mad["Year"].iloc[k])] = float(value)

growth_panel = []
for code in gdp:
    for year in range(START, 2019):
        if year in gdp[code] and (year - 1) in gdp[code]:
            growth = 100 * math.log(gdp[code][year] / gdp[code][year - 1])
            growth_panel.append({"iso": code, "Year": year, "growth": growth, "group": GROUP_OF[code]})

growth_summary = []
cumulative_rows = []
for g in GROUPS:
    values = []
    by_country = {}
    for obs in growth_panel:
        if obs["group"] == g:
            values.append(obs["growth"])
            by_country.setdefault(obs["iso"], []).append(obs["growth"])
    country_vol = []
    for iso in by_country:
        country_vol.append(std(by_country[iso]))
    cumulative = []
    for code in gdp:
        if GROUP_OF[code] == g:
            pct = 100 * (gdp[code][2018] / gdp[code][START] - 1)
            cumulative.append(pct)
            cumulative_rows.append({"iso": code, "country": NAME_OF[code], "group": g,
                                    "gdp_pc_1995": gdp[code][START], "gdp_pc_2018": gdp[code][2018],
                                    "growth_1995_2018_pct": pct})
    growth_summary.append({"group": g, "mean": mean(values), "median": median(values), "std": std(values),
                           "country_years": len(values), "median_country_volatility": median(country_vol),
                           "median_cumulative_growth_1995_2018": median(cumulative)})
write_table(growth_summary, ROOT / "results/growth_summary.csv")
pd.DataFrame(cumulative_rows).to_csv(ROOT / "results/gdp_per_capita_by_country.csv", index=False,
                                     float_format="%.1f")

all_growth = []
for obs in growth_panel:
    all_growth.append(obs["growth"])
low, high = quantile(all_growth, 0.01), quantile(all_growth, 0.99)
for obs in growth_panel:
    obs["growth_w"] = min(max(obs["growth"], low), high)
    obs["cfa"] = 1 if obs["group"] != "Non-CFA" else 0
gdf = pd.DataFrame(growth_panel)
model = smf.ols("growth_w ~ cfa + C(Year)", data=gdf).fit(cov_type="cluster", cov_kwds={"groups": gdf["iso"]})
pd.DataFrame([{"Variable": "cfa", "Coefficient (pp)": model.params["cfa"], "Clustered SE": model.bse["cfa"],
               "p-value": model.pvalues["cfa"], "N": int(model.nobs)}]).to_csv(
    ROOT / "results/growth_regression.csv", index=False, float_format="%.4f")

fig, ax = plt.subplots(figsize=(9, 4.5))
for g in GROUPS:
    years, med = [], []
    for year in range(START, 2019):
        index_values = []
        for code in gdp:
            if GROUP_OF[code] == g and year in gdp[code]:
                index_values.append(100 * gdp[code][year] / gdp[code][START])
        years.append(year)
        med.append(median(index_values))
    ax.plot(years, med, color=COLORS[g], lw=2, label=g)
ax.set_ylabel("Real GDP per capita, 1995 = 100 (median country)")
ax.set_title("Real GDP per capita since 1995 (Maddison Project 2020, 2011 PPP)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/gdp_per_capita_by_group.png")
plt.close(fig)

# ======================================================================= PARTIE 2 : BRVM
b = pd.read_csv(ROOT / "data/brvm_composite_daily.csv", parse_dates=["Date"], index_col="Date")["Close"]
b = b.loc[:"2025-12-31"]                            # années civiles complètes seulement
dates = list(b.index[1:])
returns = []                                        # rendement log du jour
for t in range(1, len(b)):
    returns.append(math.log(b.iloc[t] / b.iloc[t - 1]))


def between(first_year, last_year):
    """Rendements des années first_year à last_year incluses."""
    out = []
    for t in range(len(returns)):
        if first_year <= dates[t].year <= last_year:
            out.append(returns[t])
    return out


periods = [("1998-2025 (full)", 1998, 2025), ("1998-2005", 1998, 2005), ("2006-2015 (boom)", 2006, 2015),
           ("2016-2020 (bear market)", 2016, 2020), ("2021-2025 (recovery)", 2021, 2025)]
brvm_rows = []
for label, y0, y1 in periods:
    x = between(y0, y1)
    level, peak, worst, zeros = 1.0, 1.0, 0.0, 0
    for v in x:
        level = level * math.exp(v)
        if level > peak:
            peak = level
        if level / peak - 1 < worst:
            worst = level / peak - 1
        if v == 0:
            zeros = zeros + 1
    brvm_rows.append({"Period": label, "Trading days": len(x),
                      "Annualised log return": mean(x) * 252, "Annualised volatility": std(x) * math.sqrt(252),
                      "Skewness": skewness(x), "Excess kurtosis": excess_kurtosis(x),
                      "Max drawdown": worst, "Share of zero-return days": zeros / len(x)})
pd.DataFrame(brvm_rows).to_csv(ROOT / "results/brvm_statistics.csv", index=False, float_format="%.4f")

# rendement annuel = dernier cours de l'année / dernier cours de l'année précédente - 1
year_end = {}
for t in range(len(b)):
    year_end[b.index[t].year] = b.iloc[t]           # le dernier jour de l'année écrase les autres
annual = []
for year in sorted(year_end):
    if (year - 1) in year_end:
        annual.append({"Date": year, "Annual return (%)": 100 * (year_end[year] / year_end[year - 1] - 1)})
pd.DataFrame(annual).to_csv(ROOT / "results/brvm_annual_returns.csv", index=False, float_format="%.2f")

# tests d'efficience au sens faible : les rendements passés prédisent-ils les futurs ?
efficiency = []
for label, y0, y1 in [("1998-2010", 1998, 2010), ("2011-2025", 2011, 2025)]:
    x = between(y0, y1)
    row = {"Sample": label, "AR(1) autocorrelation": lag_correlation(x, 1),
           "Ljung-Box(10) p-value": ljung_box_pvalue(x, 10)}
    for q in [2, 5, 10, 20]:
        vr, z, p = variance_ratio(x, q)
        row["VR(%d)" % q] = vr
        row["VR(%d) p-value" % q] = p
    efficiency.append(row)
pd.DataFrame(efficiency).to_csv(ROOT / "results/brvm_efficiency_tests.csv", index=False, float_format="%.4f")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
axes[0].plot(b, color="#1f4e79", lw=1)
axes[0].set_ylabel("BRVM Composite")
axes[0].set_title("BRVM Composite index, 1998 to 2025")
drawdown = []
peak = b.iloc[0]
for value in b.values:
    peak = max(peak, value)
    drawdown.append(value / peak - 1)
axes[1].fill_between(b.index, drawdown, 0, color="#c0392b", alpha=0.5)
axes[1].set_ylabel("Drawdown")
fig.tight_layout()
fig.savefig(ROOT / "figures/brvm_composite.png")
plt.close(fig)

years, values, bar_colors = [], [], []
for row in annual:
    years.append(row["Date"])
    values.append(row["Annual return (%)"])
    bar_colors.append("#2ca02c" if row["Annual return (%)"] >= 0 else "#c0392b")
fig, ax = plt.subplots(figsize=(9, 3.8))
ax.bar(years, values, color=bar_colors)
ax.set_ylabel("%")
ax.set_title("BRVM Composite annual returns (price index, dividends excluded)")
fig.tight_layout()
fig.savefig(ROOT / "figures/brvm_annual_returns.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
for label, y0, y1, color, shift in [("1998-2010", 1998, 2010, "#d62728", -0.2),
                                    ("2011-2025", 2011, 2025, "#1f77b4", 0.2)]:
    x = between(y0, y1)
    lags, correlations = [], []
    for k in range(1, 21):
        lags.append(k + shift)
        correlations.append(lag_correlation(x, k))
    ax.bar(lags, correlations, width=0.4, color=color, label=label)
    band = 1.96 / math.sqrt(len(x))                 # zone "non significatif" à 5 %
    ax.axhline(band, color=color, ls=":", lw=1)
    ax.axhline(-band, color=color, ls=":", lw=1)
ax.set_xticks(range(1, 21))
ax.set_xlabel("Lag (trading days)")
ax.set_ylabel("Autocorrelation of daily returns")
ax.set_title("BRVM returns are predictable from their own past")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "figures/brvm_autocorrelation.png")
plt.close(fig)

for table in (summary, regressions, growth_summary, brvm_rows, efficiency):
    print(pd.DataFrame(table).round(3).to_string())
