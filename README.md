# CFA Franc Zones and the BRVM

This project has two parts:

1. Inflation and growth in the two CFA franc monetary unions (UEMOA, central bank BCEAO, and
   CEMAC, central bank BEAC) compared with 8 neighbouring countries that have their own currency,
   from the 1994 devaluation to 2023.
2. Returns, risk and efficiency of the BRVM Composite, the index of the regional stock exchange
   of West Africa, from 1998 to 2025.

## Countries

- UEMOA: Benin, Burkina Faso, Côte d'Ivoire, Guinea-Bissau, Mali, Niger, Senegal, Togo
- CEMAC: Cameroon, Central African Republic, Chad, Congo, Equatorial Guinea, Gabon
- Non-CFA: Nigeria, Ghana, Guinea, Sierra Leone, Gambia, Liberia, Mauritania, DR Congo

## Part 1. Inflation and growth

### Method

```
FOR each group (UEMOA, CEMAC, Non-CFA)
    mean, median and standard deviation of yearly inflation
    share of country-years with inflation <= 3% (convergence criterion of both unions)
    share of country-years with inflation > 10%
END FOR

Panel regression:
    inflation(country, year) = a + b x CFA(country) + year effect + error
    year effect: removes shocks common to all countries (oil, food prices, 2022)
    errors clustered by country
    inflation cut at the 1st and 99th percentiles (DR Congo hyperinflation)
```

### Inflation, 1995 to 2023 (World Bank CPI)

| | UEMOA | CEMAC | Non-CFA |
|---|---|---|---|
| Median inflation | 2.1% | 2.9% | 9.8% |
| Mean inflation | 3.3% | 3.4% | 23.8% |
| Years within the 3% criterion | 65% | 51% | 8% |
| Years above 10% | 5% | 5% | 47% |
| Volatility of inflation (median country, points) | 3.5 | 3.3 | 9.4 |
| Median inflation in 2022 | 8.7% | 5.2% | 11.5% |

| Regression | Effect on inflation | Clustered SE | p-value |
|---|---|---|---|
| CFA member | -11.0 points | 2.9 | < 0.001 |
| UEMOA member | -11.1 points | 3.0 | < 0.001 |
| CEMAC member | -10.9 points | 2.9 | < 0.001 |
| Median regression, CFA member | -7.4 points | 0.3 | < 0.001 |

![Inflation](figures/inflation_by_group.png)

CFA countries have inflation 7 to 11 points lower than their neighbours, with about one third of
the volatility. The two unions give almost the same result. In 2022, inflation rose everywhere but
stayed much lower in CFA countries than in Ghana (31%) or Nigeria (19%).

### Growth of real GDP per capita, 1995 to 2018 (Maddison Project)

| | UEMOA | CEMAC | Non-CFA |
|---|---|---|---|
| Mean yearly growth | 1.3% | 3.2% | 2.5% |
| Median total growth 1995 to 2018 | +35% | +77% | +91% |
| Volatility of growth (median country, points) | 2.7 | 6.3 | 4.3 |

With year effects, the growth difference between CFA and non-CFA countries is -0.6 points per year
and is not significant (p = 0.28). CEMAC growth is high but very unstable because several members
depend on oil (Equatorial Guinea, Congo, Gabon, Chad). UEMOA growth is low but regular. Its low
average comes in part from crises in some countries (Côte d'Ivoire 1999 to 2011, Guinea-Bissau, Mali).

![GDP per capita](figures/gdp_per_capita_by_group.png)

These are comparisons, not causal effects: countries did not join the CFA zone at random.

## Part 2. The BRVM Composite

### Returns and risk

| Period | Annual return | Volatility | Max drawdown | Days with no change |
|---|---|---|---|---|
| 1998 to 2025 | 4.8% | 12.4% | -62% | 2.8% |
| 1998 to 2005 | 1.9% | 10.1% | -42% | 8.1% |
| 2006 to 2015 | 10.1% | 14.7% | -49% | 2.1% |
| 2016 to 2020 | -15.0% | 12.6% | -62% | 0.2% |
| 2021 to 2025 | 17.5% | 9.1% | -14% | 0.3% |

Log returns of the price index, dividends not included. The index rose by 77% in 2007 and 39% in
2013, then fell five years in a row from 2016 to 2020 (-62% from the top). It then rose again:
+39% in 2021, +29% in 2024 and +25% in 2025. Daily returns have an excess kurtosis of 17.

![BRVM](figures/brvm_composite.png)

### Weak-form efficiency

If prices follow a random walk, past returns do not predict future returns. Two tests:

- autocorrelation of daily returns
- variance ratio of Lo and MacKinlay (1988)

```
VR(q) = variance of returns over q days / (q x variance of daily returns)
VR(q) = 1  : random walk
VR(q) > 1  : returns follow each other (prices adjust slowly)
```

| Sample | AR(1) | VR(5) | VR(10) | VR(20) |
|---|---|---|---|---|
| 1998 to 2010 | 0.11 | 1.53 (p < 0.001) | 1.93 (p < 0.001) | 2.06 (p < 0.001) |
| 2011 to 2025 | -0.03 | 1.04 (p = 0.45) | 1.13 (p = 0.08) | 1.24 (p = 0.02) |

Before 2011 the ratios are far above 1 and returns are positively autocorrelated: past returns
predicted future returns. This is typical of a market with few trades, where prices are often not
updated (the index did not move on 8% of days between 1998 and 2005). Since 2011 the market is much
closer to a random walk, with a small remaining effect at 20 days.

![Autocorrelation](figures/brvm_autocorrelation.png)

## Data

| File | Source |
|---|---|
| `data/cpi_inflation_world_bank.csv` | World Bank, consumer price inflation (annual %), from the [datasets/inflation](https://github.com/datasets/inflation) package |
| `data/maddison_2020.csv` | Maddison Project Database 2020 (Bolt and van Zanden), from [Our World in Data](https://github.com/owid/owid-datasets) |
| `data/brvm_composite_daily.csv` | BRVM Composite daily closes, from the community dataset [brvm-data](https://github.com/balletyanick/brvm-data) |

Note: in the `datasets/inflation` package the two files have swapped names (the file called
"consumer" contains the GDP deflator). This was found by comparing with published CPI values, for
example Gabon 2022: 4.2% for the CPI and 14.8% for the deflator. The correct CPI series is used here.

## Run it

```bash
pip install -r requirements.txt
python run_analysis.py      # a few seconds
```

```
src/basic_stats.py     mean, median, quantiles, autocorrelation, Ljung-Box, variance ratio
run_analysis.py        part 1 and part 2, tables and charts
results/, figures/     outputs
```

Author: Klaus KUEKOU
