"""
====================================================================
 ANALYSE DES HYPOTHESES — Memoire ESLSCA
 "Analyse des differences de perception du risque entre investisseurs
 et non-investisseurs a l'ere de l'intelligence artificielle"
====================================================================

USAGE
-----
1. Depuis le tableau de bord de l'enquete (code chercheur), cliquer sur
   "Export JSON (donnees brutes)" une fois la collecte terminee
   (100 reponses minimum, 150-200 visees).
2. Placer le fichier telecharge a cote de ce script et le renommer
   'reponses.json' (ou passer son chemin en argument).
3. Installer les dependances si besoin :
       pip install pandas numpy scipy statsmodels matplotlib --break-system-packages
4. Lancer :
       python3 analyse_hypotheses.py reponses.json

SORTIE
------
- Affichage console de chaque etape (H1, H2, H3) avec formulation APA
- resultats_chapitre4.md   : texte pret a coller / adapter dans le chapitre 4
- figures/*.png            : graphiques (matrice de correlation, moyennes par segment)

CE QUE CE SCRIPT NE FAIT PAS
-----------------------------
- Il ne remplace pas la lecture critique attendue en 4.4 (Discussion) :
  les p-values et coefficients sont fournis, mais leur interpretation
  au regard de la litterature reste a rediger.
- Le PLS-SEM "complet" (mesure de la validite convergente/discriminante,
  HTMT, bootstrap) n'est pas implemente ici : ce script utilise une
  regression lineaire multiple (OLS) et une ANOVA a deux facteurs, une
  approche plus simple mais methodologiquement defendable pour un
  echantillon de 100-200 repondants avec des construits mesures par
  des scores composites plutot que par un modele a variables latentes
  complet. A mentionner en 3.5 (Methodes d'analyse) et en 4.4 (limites).
"""

import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

ALPHA = 0.05  # seuil de significativite conventionnel

# --------------------------------------------------------------------
# Grille des items par dimension (doit rester synchronisee avec le
# tableau Q de enquete_perception_risque.html)
# --------------------------------------------------------------------
ITEMS = {
    "ob": ["ob1", "ob2", "ob3", "ob4", "ob5", "ob6"],
    "la": ["la1", "la2", "la3", "la4", "la5", "la6"],
    "ab": ["ab1", "ab2", "ab3", "ab4", "ab5"],
    "rp": ["rp1", "rp2", "rp3", "rp4", "rp5", "rp6"],
    "ai": ["ai1", "ai2", "ai3", "ai4", "ai5", "ai6", "ai7"],
}

INVESTOR_LABELS = {0: "Investisseur actif", 1: "Investisseur occasionnel", 2: "Non-investisseur"}
AI_LABELS = {0: "Usage regulier", 1: "Usage occasionnel", 2: "Jamais"}


# ======================================================================
# 1. CHARGEMENT ET MISE EN FORME
# ======================================================================
def load_data(path: str) -> pd.DataFrame:
    """
    Accepts two possible sources, auto-detected by file extension:

    - .json : the "Export JSON (donnees brutes)" file from the in-app
              researcher dashboard (window.storage-based).
    - .csv  : the "Download as CSV" export from Netlify's Forms panel
              (Site > Forms > reponses). Each row's 'data' column holds
              the same JSON payload as above, so both paths converge.
    """
    if path.lower().endswith(".csv"):
        csv_df = pd.read_csv(path)
        raw = []
        for _, csv_row in csv_df.iterrows():
            try:
                raw.append(json.loads(csv_row["data"]))
            except (KeyError, TypeError, json.JSONDecodeError):
                # Fallback if the 'data' column is missing/corrupted: rebuild
                # a minimal record from the individual Netlify form fields.
                raw.append({
                    "tier": csv_row.get("tier"), "age": csv_row.get("age"),
                    "gender": csv_row.get("gender"), "edu": csv_row.get("edu"),
                    "status": csv_row.get("status"), "investor": csv_row.get("investor"),
                    "ai_use": csv_row.get("ai_use"), "fl": csv_row.get("fl"),
                    "ob": csv_row.get("ob"), "la": csv_row.get("la"),
                    "ab": csv_row.get("ab"), "rp": csv_row.get("rp"), "ai": csv_row.get("ai"),
                    "raw": {},
                })
    else:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

    rows = []
    for rec in raw:
        row = {
            "tier": rec.get("tier"),
            "age": rec.get("age"),
            "gender": rec.get("gender"),
            "edu": rec.get("edu"),
            "status": rec.get("status"),
            "investor": rec.get("investor"),
            "ai_use": rec.get("ai_use"),
            "fl": rec.get("fl"),
            "ob": rec.get("ob"),
            "la": rec.get("la"),
            "ab": rec.get("ab"),
            "rp": rec.get("rp"),
            "ai": rec.get("ai"),
        }
        item_answers = rec.get("raw", {}) or {}
        for item_list in ITEMS.values():
            for item_id in item_list:
                row[item_id] = item_answers.get(item_id, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)

    # Variables derivees utiles pour H3
    df["investor_bin"] = df["investor"].apply(lambda x: "Non-investisseur" if x == 2 else "Investisseur")
    df["ai_bin"] = df["ai_use"].apply(lambda x: "Jamais" if x == 2 else "Utilisateur d'IA")
    df["ai_intensity"] = df["ai_use"].apply(lambda x: np.nan if pd.isna(x) else (2 - x))  # 0=jamais .. 2=regulier

    return df


# ======================================================================
# 2. STATISTIQUES DESCRIPTIVES (4.1)
# ======================================================================
def descriptive_stats(df: pd.DataFrame) -> str:
    lines = ["## 4.1 Presentation des resultats\n"]
    n = len(df)
    lines.append(f"L'echantillon final comprend **n = {n}** repondants.\n")

    inv_counts = df["investor"].map(INVESTOR_LABELS).value_counts()
    ai_counts = df["ai_use"].map(AI_LABELS).value_counts()

    lines.append("**Repartition par statut d'investisseur :**\n")
    for label, count in inv_counts.items():
        lines.append(f"- {label} : {count} ({count/n*100:.1f} %)")
    lines.append("")

    lines.append("**Repartition par usage des outils d'IA :**\n")
    for label, count in ai_counts.items():
        lines.append(f"- {label} : {count} ({count/n*100:.1f} %)")
    lines.append("")

    lines.append("**Statistiques descriptives par dimension (echelle 0-100) :**\n")
    lines.append("| Dimension | Moyenne | Ecart-type | Min | Max |")
    lines.append("|---|---|---|---|---|")
    for dim, label in [("ob", "Exces de confiance (OB)"), ("la", "Aversion aux pertes (LA)"),
                        ("ab", "Biais d'ancrage (AB)"), ("rp", "Perception du risque (RP)"),
                        ("ai", "Rapport a l'IA")]:
        s = df[dim].dropna()
        lines.append(f"| {label} | {s.mean():.1f} | {s.std():.1f} | {s.min():.0f} | {s.max():.0f} |")
    lines.append("")
    return "\n".join(lines)


# ======================================================================
# 3. FIABILITE DES ECHELLES — Alpha de Cronbach
#    (calcule sur le sous-echantillon "deep" = seul groupe ayant
#    repondu a l'integralite des items de chaque dimension)
# ======================================================================
def cronbach_alpha(item_df: pd.DataFrame) -> float:
    item_df = item_df.dropna()
    if item_df.shape[0] < 3 or item_df.shape[1] < 2:
        return np.nan
    item_vars = item_df.var(axis=0, ddof=1)
    total_var = item_df.sum(axis=1).var(ddof=1)
    k = item_df.shape[1]
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)


def reliability_report(df: pd.DataFrame) -> str:
    lines = ["## 3.5 (complement) Fiabilite des echelles de mesure\n"]
    deep = df[df["tier"] == "deep"]
    lines.append(
        f"Le parcours adaptatif de l'enquete ne soumet l'integralite des items d'une dimension "
        f"qu'aux repondants ayant atteint le niveau 'complet' (n = {len(deep)}). "
        f"L'alpha de Cronbach est donc calcule sur ce sous-echantillon uniquement, "
        f"les autres profils n'ayant repondu qu'a un noyau reduit d'items communs.\n"
    )
    lines.append("| Dimension | Nb items | n (deep) | Alpha de Cronbach |")
    lines.append("|---|---|---|---|")
    for dim, items in ITEMS.items():
        alpha = cronbach_alpha(deep[items])
        lines.append(f"| {dim.upper()} | {len(items)} | {deep[items].dropna().shape[0]} | "
                      f"{alpha:.3f}" if not np.isnan(alpha) else f"| {dim.upper()} | {len(items)} | 0 | n/a |")
    lines.append(
        "\nSeuil generalement admis : alpha >= 0.70 (Nunnally, 1978). "
        "Un alpha inferieur invite a retirer l'item le moins correle au score total "
        "avant de reconduire le test.\n"
    )
    return "\n".join(lines)


# ======================================================================
# 4. MATRICE DE CORRELATION
# ======================================================================
def correlation_report(df: pd.DataFrame, fig_dir: Path) -> str:
    cols = ["ob", "la", "ab", "ai", "rp"]
    sub = df[cols].dropna()
    corr = sub.corr(method="pearson")

    lines = ["## 4.2 Analyse des resultats — matrice de correlation\n"]
    lines.append("| | OB | LA | AB | AI | RP |")
    lines.append("|---|---|---|---|---|---|")
    for row_name in cols:
        row = corr.loc[row_name]
        lines.append(f"| {row_name.upper()} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
    lines.append("")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdYlBu_r")
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.upper() for c in cols])
        ax.set_yticks(range(len(cols))); ax.set_yticklabels([c.upper() for c in cols])
        for i in range(len(cols)):
            for j in range(len(cols)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im, fraction=0.046, pad=0.04)
        ax.set_title("Matrice de correlation (Pearson)")
        fig.tight_layout()
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / "correlation_matrix.png", dpi=200)
        plt.close(fig)
        lines.append("Figure enregistree : `figures/correlation_matrix.png`\n")
    except ImportError:
        lines.append("(matplotlib non installe — figure non generee)\n")

    return "\n".join(lines)


# ======================================================================
# 5. H1 — Regression multiple : RP ~ OB + LA + AB
# ======================================================================
def test_h1(df: pd.DataFrame) -> str:
    sub = df[["rp", "ob", "la", "ab", "fl"]].dropna()
    model = smf.ols("rp ~ ob + la + ab + fl", data=sub).fit()

    lines = ["## 4.3 Test des hypotheses — H1\n"]
    lines.append(
        "**H1** : Les biais cognitifs (exces de confiance, aversion aux pertes, biais d'ancrage) "
        "influencent significativement la perception du risque financier.\n"
    )
    lines.append(f"Modele estime sur n = {len(sub)} repondants : "
                  f"RP = f(OB, LA, AB, litteratie financiere).\n")
    lines.append("| Variable | β | Erreur std. | t | p |")
    lines.append("|---|---|---|---|---|")
    for var in ["Intercept", "ob", "la", "ab", "fl"]:
        b = model.params[var]; se = model.bse[var]; t = model.tvalues[var]; p = model.pvalues[var]
        lines.append(f"| {var} | {b:.3f} | {se:.3f} | {t:.2f} | {p:.4f}{' *' if p < ALPHA else ''} |")
    lines.append("")
    lines.append(f"R² = {model.rsquared:.3f} ; R² ajuste = {model.rsquared_adj:.3f} ; "
                  f"F({int(model.df_model)}, {int(model.df_resid)}) = {model.fvalue:.2f}, "
                  f"p {'< .001' if model.f_pvalue < .001 else f'= {model.f_pvalue:.3f}'}\n")

    sig_vars = [v for v in ["ob", "la", "ab"] if model.pvalues[v] < ALPHA]
    verdict = "H1 est validee" if sig_vars else "H1 n'est pas validee"
    lines.append(f"**Verdict (seuil p < .05) : {verdict}.** "
                  f"Predicteurs significatifs : {', '.join(v.upper() for v in sig_vars) or 'aucun'}.\n")
    return "\n".join(lines)


# ======================================================================
# 6. H2 — Regression hierarchique : ajout de l'usage de l'IA
# ======================================================================
def test_h2(df: pd.DataFrame) -> str:
    sub = df[["rp", "ob", "la", "ab", "fl", "ai", "ai_intensity"]].dropna()

    base = smf.ols("rp ~ ob + la + ab + fl", data=sub).fit()
    full = smf.ols("rp ~ ob + la + ab + fl + ai + ai_intensity", data=sub).fit()
    delta_r2 = full.rsquared - base.rsquared

    lines = ["\n## H2\n"]
    lines.append(
        "**H2** : L'utilisation des outils d'intelligence artificielle influence significativement "
        "la perception du risque financier.\n"
    )
    lines.append(f"Modele hierarchique sur n = {len(sub)} repondants : "
                  f"Modele 1 (biais + litteratie) vs Modele 2 (+ attitude et intensite d'usage de l'IA).\n")
    lines.append(f"ΔR² apporte par les variables IA = {delta_r2:.3f}\n")
    lines.append("| Variable (Modele 2) | β | t | p |")
    lines.append("|---|---|---|---|")
    for var in ["ai", "ai_intensity"]:
        b = full.params[var]; t = full.tvalues[var]; p = full.pvalues[var]
        lines.append(f"| {var} | {b:.3f} | {t:.2f} | {p:.4f}{' *' if p < ALPHA else ''} |")
    lines.append("")

    sig = full.pvalues["ai"] < ALPHA or full.pvalues["ai_intensity"] < ALPHA
    verdict = "H2 est validee" if sig else "H2 n'est pas validee"
    lines.append(f"**Verdict (seuil p < .05) : {verdict}.**\n")
    return "\n".join(lines)


# ======================================================================
# 7. H3 — ANOVA a deux facteurs : statut investisseur x usage IA
# ======================================================================
def test_h3(df: pd.DataFrame, fig_dir: Path) -> str:
    sub = df[["rp", "investor_bin", "ai_bin"]].dropna()

    model = smf.ols("rp ~ C(investor_bin) * C(ai_bin)", data=sub).fit()
    from statsmodels.stats.anova import anova_lm
    aov = anova_lm(model, typ=2)

    lines = ["\n## H3\n"]
    lines.append(
        "**H3** : Les investisseurs utilisant des outils d'intelligence artificielle presentent "
        "une perception du risque distincte de celle des non-utilisateurs et des non-investisseurs.\n"
    )
    lines.append(f"ANOVA a deux facteurs (statut investisseur x usage de l'IA) sur n = {len(sub)}.\n")
    lines.append("| Source | Somme des carres | ddl | F | p |")
    lines.append("|---|---|---|---|---|")
    for idx_name, row in aov.iterrows():
        if idx_name == "Residual":
            continue
        lines.append(f"| {idx_name} | {row['sum_sq']:.2f} | {row['df']:.0f} | "
                      f"{row['F']:.2f} | {row['PR(>F)']:.4f}{' *' if row['PR(>F)'] < ALPHA else ''} |")
    lines.append("")

    means = sub.groupby(["investor_bin", "ai_bin"])["rp"].agg(["mean", "std", "count"]).round(1)
    lines.append("**Moyennes de RP par segment :**\n")
    lines.append(means.to_markdown())
    lines.append("")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        means_reset = means.reset_index()
        labels = means_reset["investor_bin"] + " · " + means_reset["ai_bin"]
        ax.bar(labels, means_reset["mean"], yerr=means_reset["std"], capsize=4)
        ax.set_ylabel("Perception du risque (0-100)")
        ax.set_title("RP moyen par segment (H3)")
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig_dir.mkdir(exist_ok=True)
        fig.savefig(fig_dir / "h3_segments.png", dpi=200)
        plt.close(fig)
        lines.append("Figure enregistree : `figures/h3_segments.png`\n")
    except ImportError:
        pass

    interaction_p = aov.loc["C(investor_bin):C(ai_bin)", "PR(>F)"]
    verdict = "H3 est validee" if interaction_p < ALPHA else "H3 n'est pas validee (effet d'interaction non significatif)"
    lines.append(f"**Verdict (seuil p < .05, effet d'interaction) : {verdict}.**\n")
    return "\n".join(lines)


# ======================================================================
# 8. SYNTHESE
# ======================================================================
def summary_table(h1_valid, h2_valid, h3_valid) -> str:
    lines = ["\n## Synthese — validation des hypotheses\n"]
    lines.append("| Hypothese | Enonce (resume) | Statut |")
    lines.append("|---|---|---|")
    lines.append(f"| H1 | Biais cognitifs -> perception du risque | {'Validee' if h1_valid else 'Non validee'} |")
    lines.append(f"| H2 | Usage de l'IA -> perception du risque | {'Validee' if h2_valid else 'Non validee'} |")
    lines.append(f"| H3 | Interaction investisseur x IA sur RP | {'Validee' if h3_valid else 'Non validee'} |")
    lines.append(
        "\nCette synthese alimente directement la conclusion generale "
        "(reponse a la problematique, validation des hypotheses).\n"
    )
    return "\n".join(lines)


# ======================================================================
# MAIN
# ======================================================================
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "reponses.json"
    if not Path(path).exists():
        print(f"Fichier introuvable : {path}")
        print("Deux sources possibles :")
        print("  1) Netlify : Site > Forms > reponses > 'Download as CSV'")
        print("  2) Tableau de bord in-app : 'Export JSON (donnees brutes)'")
        print("Place le fichier a cote de ce script et relance avec son chemin en argument.")
        sys.exit(1)

    df = load_data(path)
    fig_dir = Path("figures")

    report_parts = [
        "# Resultats — Chapitre 4\n",
        f"_Genere automatiquement a partir de {len(df)} reponses._\n",
        descriptive_stats(df),
        reliability_report(df),
        correlation_report(df, fig_dir),
        test_h1(df),
        test_h2(df),
        test_h3(df, fig_dir),
    ]

    sub1 = df[["rp", "ob", "la", "ab", "fl"]].dropna()
    m1 = smf.ols("rp ~ ob + la + ab + fl", data=sub1).fit()
    h1_valid = any(m1.pvalues[v] < ALPHA for v in ["ob", "la", "ab"])

    sub2 = df[["rp", "ob", "la", "ab", "fl", "ai", "ai_intensity"]].dropna()
    m2 = smf.ols("rp ~ ob + la + ab + fl + ai + ai_intensity", data=sub2).fit()
    h2_valid = (m2.pvalues["ai"] < ALPHA) or (m2.pvalues["ai_intensity"] < ALPHA)

    sub3 = df[["rp", "investor_bin", "ai_bin"]].dropna()
    m3 = smf.ols("rp ~ C(investor_bin) * C(ai_bin)", data=sub3).fit()
    from statsmodels.stats.anova import anova_lm
    aov3 = anova_lm(m3, typ=2)
    h3_valid = aov3.loc["C(investor_bin):C(ai_bin)", "PR(>F)"] < ALPHA

    report_parts.append(summary_table(h1_valid, h2_valid, h3_valid))

    report = "\n".join(report_parts)
    Path("resultats_chapitre4.md").write_text(report, encoding="utf-8")

    print(report)
    print("\n" + "=" * 70)
    print("Rapport ecrit dans resultats_chapitre4.md")
    if fig_dir.exists():
        print(f"Figures enregistrees dans {fig_dir}/")


if __name__ == "__main__":
    main()
