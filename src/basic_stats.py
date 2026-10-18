# Statistiques de base écrites à la main, avec des boucles.

import math

from scipy import stats


def mean(x):
    total = 0.0
    for v in x:
        total = total + v
    return total / len(x)


def std(x):
    """Écart-type avec diviseur n - 1."""
    m = mean(x)
    total = 0.0
    for v in x:
        total = total + (v - m) ** 2
    return math.sqrt(total / (len(x) - 1))


def quantile(x, q):
    """Quantile : on trie puis on interpole entre les deux valeurs voisines."""
    values = sorted(x)
    position = (len(values) - 1) * q
    below = int(math.floor(position))
    above = min(below + 1, len(values) - 1)
    return values[below] + (position - below) * (values[above] - values[below])


def median(x):
    return quantile(x, 0.5)


def skewness(x):
    m = mean(x)
    m2, m3 = 0.0, 0.0
    for v in x:
        m2 = m2 + (v - m) ** 2
        m3 = m3 + (v - m) ** 3
    return (m3 / len(x)) / (m2 / len(x)) ** 1.5


def excess_kurtosis(x):
    m = mean(x)
    m2, m4 = 0.0, 0.0
    for v in x:
        m2 = m2 + (v - m) ** 2
        m4 = m4 + (v - m) ** 4
    return (m4 / len(x)) / (m2 / len(x)) ** 2 - 3


def lag_correlation(x, k):
    """Corrélation de Pearson entre x(t) et x(t-k)."""
    a = x[k:]
    b = x[:-k]
    ma, mb = mean(a), mean(b)
    num, va, vb = 0.0, 0.0, 0.0
    for t in range(len(a)):
        num = num + (a[t] - ma) * (b[t] - mb)
        va = va + (a[t] - ma) ** 2
        vb = vb + (b[t] - mb) ** 2
    return num / math.sqrt(va * vb)


def acf(x, k):
    """Autocorrélation "classique" (moyenne et variance de toute la série)."""
    m = mean(x)
    num, den = 0.0, 0.0
    for t in range(len(x)):
        den = den + (x[t] - m) ** 2
        if t >= k:
            num = num + (x[t] - m) * (x[t - k] - m)
    return num / den


def ljung_box_pvalue(x, h=10):
    """H0 : aucune autocorrélation jusqu'au retard h. Q = n(n+2) x somme rho(k)² / (n-k)."""
    n = len(x)
    q = 0.0
    for k in range(1, h + 1):
        q = q + acf(x, k) ** 2 / (n - k)
    q = n * (n + 2) * q
    return stats.chi2.sf(q, h)


# ------------------------------------------------------------------------------
# ALGORITHME RatioDeVariance (Lo et MacKinlay, 1988)
# Si les prix suivent une marche aléatoire, la variance des rendements sur q jours
# vaut q fois la variance sur 1 jour, donc VR(q) = 1.
#   VR(q) > 1 : les rendements se suivent (tendance, prix "en retard")
#   VR(q) < 1 : les rendements se compensent (retour à la moyenne)
# DEBUT
#   s1 <- variance des rendements journaliers
#   POUR chaque t : somme_q(t) <- r(t) + ... + r(t+q-1)
#   sq <- variance de somme_q (corrigée des chevauchements) / q
#   VR <- sq / s1
#   z <- (VR - 1) / écart-type robuste à l'hétéroscédasticité ; p-value par la loi normale
# FIN
# ------------------------------------------------------------------------------
def variance_ratio(x, q):
    T = len(x)
    mu = mean(x)
    s1 = 0.0
    for v in x:
        s1 = s1 + (v - mu) ** 2
    s1 = s1 / (T - 1)

    sq = 0.0
    for t in range(T - q + 1):
        block = 0.0
        for j in range(q):
            block = block + x[t + j]
        sq = sq + (block - q * mu) ** 2
    m = q * (T - q + 1) * (1 - q / T)
    sq = sq / m
    vr = sq / s1

    dev2 = []
    total_dev2 = 0.0
    for v in x:
        dev2.append((v - mu) ** 2)
        total_dev2 = total_dev2 + (v - mu) ** 2
    theta = 0.0
    for j in range(1, q):
        cross = 0.0
        for t in range(j, T):
            cross = cross + dev2[t] * dev2[t - j]
        delta = cross / total_dev2 ** 2 * T
        theta = theta + (2 * (q - j) / q) ** 2 * delta
    z = (vr - 1) / math.sqrt(theta / T)
    return vr, z, 2 * (1 - stats.norm.cdf(abs(z)))
