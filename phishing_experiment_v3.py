"""
V3 - addresses review: registered-domain grouping, class-balanced grouped folds,
stronger static baselines, full metrics (incl. ROC-AUC, PR-AUC, TPR@1%FPR),
all-combination ablation, corrected resampled t-tests, dataset audit, better figures.

Colab usage:
  !pip install tldextract xgboost scipy
  upload dataset_phishing.csv (= dataset_B_05_2020.csv renamed)
  run this file. Runtime ~10-25 min.
Outputs: audit.txt, results_v3_raw.csv, results_v3_summary.csv, results_v3_tests.csv,
         results_v3_main.png, results_v3_ablation.png, prevalence_precision.csv
"""
import re
import warnings
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
try:
    from xgboost import XGBClassifier
except ImportError:
    raise SystemExit("pip install xgboost first")

try:
    import tldextract
    _ext = tldextract.TLDExtract(suffix_list_urls=())  # use bundled snapshot, no network
    def reg_domain(host):
        e = _ext(host)
        d = getattr(e, "top_domain_under_public_suffix", None) or getattr(e, "registered_domain", "")
        return d or host
except Exception:
    print("tldextract not available: falling back to last-two-labels heuristic")
    def reg_domain(host):
        host = host.split(":")[0]
        parts = host.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else host

CSV = "dataset_phishing.csv"
N_SPLITS, REPEATS = 5, 3

df = pd.read_csv(CSV)
df["label"] = (df["status"].str.lower() == "phishing").astype(int)
df["host"] = df["url"].apply(lambda u: urlparse(u if "://" in u else "http://" + u).netloc.lower().split(":")[0])
df["domain"] = df["host"].apply(reg_domain)
y = df["label"].values

# ---------------- feature groups ----------------
all_cols = [c for c in df.columns if c not in ("url", "status", "label", "host", "domain")]
EXT = [c for c in ["whois_registered_domain", "domain_registration_length", "domain_age",
                   "web_traffic", "dns_record", "google_index", "page_rank"] if c in all_cols]
URL = all_cols[: all_cols.index("nb_hyperlinks")]
CONT = [c for c in all_cols if c not in URL and c not in EXT]
print(f"URL {len(URL)} | content {len(CONT)} | external {len(EXT)}")
FS = {
    "All 87": all_cols,
    "No external (URL+content)": URL + CONT,
    "URL-only": URL,
}
FS_EXTRA = {
    "Content-only": CONT,
    "External-only": EXT,
    "URL+external": URL + EXT,
    "Content+external": CONT + EXT,
}

# ---------------- dataset audit ----------------
with open("audit.txt", "w") as f:
    def w(s=""):
        print(s); f.write(s + "\n")
    w(f"Rows: {len(df)}; exact duplicate URLs: {df['url'].duplicated().sum()}")
    w(f"Unique hostnames: {df['host'].nunique()}; unique registered domains: {df['domain'].nunique()}")
    gs = df.groupby("domain").size()
    w("URLs per registered domain: " + gs.describe().round(2).to_dict().__str__())
    for lab, name in [(0, "legitimate"), (1, "phishing")]:
        sub = df[df.label == lab]
        w(f"{name}: {len(sub)} URLs, {sub['domain'].nunique()} domains, "
          f"max URLs/domain {sub.groupby('domain').size().max()}, "
          f"mean {sub.groupby('domain').size().mean():.2f}")
    both = df.groupby("domain")["label"].nunique()
    w(f"Domains with both labels: {(both > 1).sum()}")
    w(f"Hostnames starting with www.: {df['host'].str.startswith('www.').mean():.3f} "
      f"(legit {df[df.label==0]['host'].str.startswith('www.').mean():.3f}, "
      f"phish {df[df.label==1]['host'].str.startswith('www.').mean():.3f})")
    w("Top 10 registered domains by URL count:\n" + gs.sort_values(ascending=False).head(10).to_string())
    w("External features: share equal to -1 / 0 by class")
    for c in EXT:
        w(f"  {c}: -1 -> legit {(df[df.label==0][c]==-1).mean():.3f} phish {(df[df.label==1][c]==-1).mean():.3f}; "
          f" 0 -> legit {(df[df.label==0][c]==0).mean():.3f} phish {(df[df.label==1][c]==0).mean():.3f}")

# ---------------- static baselines ----------------
def rule_based(url):
    host = urlparse(url if "://" in url else "http://" + url).netloc
    s = bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}(:\d+)?", host))
    s += "@" in url
    s += len(url) > 75
    s += not url.lower().startswith("https")
    s += host.count(".") > 3
    s += "-" in host
    return int(s >= 2)

RULE_PRED = df["url"].apply(rule_based).values


def metrics(yt, yp, score=None):
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    m = {"accuracy": accuracy_score(yt, yp),
         "precision": precision_score(yt, yp, zero_division=0) if (tp + fp) else np.nan,
         "recall": recall_score(yt, yp, zero_division=0),
         "f1": f1_score(yt, yp, zero_division=0),
         "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
         "roc_auc": np.nan, "pr_auc": np.nan, "tpr_at_1pct_fpr": np.nan}
    if score is not None:
        m["roc_auc"] = roc_auc_score(yt, score)
        m["pr_auc"] = average_precision_score(yt, score)
        fpr, tpr, _ = roc_curve(yt, score)
        m["tpr_at_1pct_fpr"] = float(np.interp(0.01, fpr, tpr))
    return m


def ml_models(seed):
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                 eval_metric="logloss", random_state=seed),
    }


rows, fold_info = [], []
X_all = df[all_cols]
for rep in range(REPEATS):
    seed = 100 + rep
    splitters = {
        "A: Random (stratified)": StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(X_all, y),
        "B: Domain-grouped (stratified)": StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed).split(
            X_all, y, groups=df["domain"]),
    }
    for setting, gen in splitters.items():
        for fold, (tr, te) in enumerate(gen):
            ytr, yte = y[tr], y[te]
            base = {"repeat": rep, "fold": fold, "setting": setting}
            fold_info.append({**base, "n_train": len(tr), "n_test": len(te),
                              "test_phish_share": yte.mean(),
                              "test_domains": df["domain"].iloc[te].nunique(),
                              "shared_domains_train_test": len(set(df["domain"].iloc[tr]) & set(df["domain"].iloc[te]))})
            # static: rule heuristic
            rows.append({**base, "features": "n/a", "model": "Rule heuristic", **metrics(yte, RULE_PRED[te])})
            # static: domain blacklist
            bl = set(df["domain"].iloc[tr][ytr == 1])
            pred = df["domain"].iloc[te].isin(bl).astype(int).values
            rows.append({**base, "features": "n/a", "model": "Domain blacklist", **metrics(yte, pred)})
            # static-like learned baselines: stump and depth-3 tree
            for nm, cols in [("Decision stump", all_cols), ("Depth-3 tree (all)", all_cols),
                             ("Depth-3 tree (URL-only)", URL)]:
                depth = 1 if "stump" in nm else 3
                t = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(df[cols].iloc[tr], ytr)
                rows.append({**base, "features": "n/a", "model": nm,
                             **metrics(yte, t.predict(df[cols].iloc[te]), t.predict_proba(df[cols].iloc[te])[:, 1])})
            # ML models
            for fname, cols in FS.items():
                for mname, model in ml_models(seed).items():
                    model.fit(df[cols].iloc[tr], ytr)
                    sc = model.predict_proba(df[cols].iloc[te])[:, 1]
                    rows.append({**base, "features": fname, "model": mname,
                                 **metrics(yte, (sc >= 0.5).astype(int), sc)})
            for fname, cols in FS_EXTRA.items():
                model = ml_models(seed)["XGBoost"].fit(df[cols].iloc[tr], ytr)
                sc = model.predict_proba(df[cols].iloc[te])[:, 1]
                rows.append({**base, "features": fname, "model": "XGBoost",
                             **metrics(yte, (sc >= 0.5).astype(int), sc)})
    print("finished repeat", rep)

raw = pd.DataFrame(rows)
raw.to_csv("results_v3_raw.csv", index=False)
fi = pd.DataFrame(fold_info)
fi.groupby("setting")[["test_phish_share", "test_domains", "shared_domains_train_test"]].agg(["mean", "std", "min", "max"]) \
    .round(3).to_csv("fold_balance.csv")
print(fi.groupby("setting")[["n_train", "n_test", "test_phish_share", "shared_domains_train_test"]].agg(["mean", "std"]).round(3))

# ---------------- summary ----------------
mets = ["accuracy", "precision", "recall", "f1", "fpr", "roc_auc", "pr_auc", "tpr_at_1pct_fpr"]
g = raw.groupby(["setting", "features", "model"])[mets].agg(["mean", "std"])
out = pd.DataFrame(index=g.index)
for m in mets:
    out[m] = (g[m]["mean"] * 100).round(2).astype(str) + " ± " + (g[m]["std"] * 100).round(2).astype(str)
out = out.reset_index()
out.to_csv("results_v3_summary.csv", index=False)
print(out[["setting", "features", "model", "accuracy", "recall", "fpr", "roc_auc", "pr_auc"]].to_string(index=False))

# ---------------- corrected resampled t-tests (Nadeau & Bengio) on accuracy ----------------
def fold_series(setting, features, model):
    d = raw[(raw.setting == setting) & (raw.features == features) & (raw.model == model)]
    return d.sort_values(["repeat", "fold"])["accuracy"].values


def nb_test(a, b, ratio=1 / (N_SPLITS - 1)):
    d = a - b
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return d.mean(), np.nan, np.nan
    t = d.mean() / np.sqrt((1 / k + ratio) * var)
    return d.mean(), t, 2 * stats.t.sf(abs(t), k - 1)


tests = []
for s in raw.setting.unique():
    pairs = [
        ("XGB vs RF (All 87)", fold_series(s, "All 87", "XGBoost"), fold_series(s, "All 87", "Random Forest")),
        ("XGB vs LR (All 87)", fold_series(s, "All 87", "XGBoost"), fold_series(s, "All 87", "Logistic Regression")),
        ("XGB All 87 vs No external", fold_series(s, "All 87", "XGBoost"), fold_series(s, "No external (URL+content)", "XGBoost")),
        ("XGB No external vs URL-only", fold_series(s, "No external (URL+content)", "XGBoost"), fold_series(s, "URL-only", "XGBoost")),
        ("XGB URL-only vs Depth-3 tree (URL-only)", fold_series(s, "URL-only", "XGBoost"), fold_series(s, "n/a", "Depth-3 tree (URL-only)")),
        ("XGB URL-only vs Rule heuristic", fold_series(s, "URL-only", "XGBoost"), fold_series(s, "n/a", "Rule heuristic")),
        ("XGB All 87 vs Decision stump", fold_series(s, "All 87", "XGBoost"), fold_series(s, "n/a", "Decision stump")),
    ]
    for name, a, b in pairs:
        md, t, p = nb_test(a, b)
        tests.append({"setting": s, "comparison": name, "mean_diff_pts": round(md * 100, 2),
                      "t": round(t, 2) if t == t else None, "p_value": round(p, 4) if p == p else None})
tests = pd.DataFrame(tests)
tests.to_csv("results_v3_tests.csv", index=False)
print(tests.to_string(index=False))

# ---------------- prevalence-adjusted precision ----------------
pp = []
for s in raw.setting.unique():
    for feat in ["All 87", "URL-only"]:
        d = raw[(raw.setting == s) & (raw.features == feat) & (raw.model == "XGBoost")]
        rec, fpr = d["recall"].mean(), d["fpr"].mean()
        for prev in [0.5, 0.05, 0.01, 0.001]:
            prec = rec * prev / (rec * prev + fpr * (1 - prev))
            pp.append({"setting": s, "features": feat, "prevalence": prev,
                       "recall": round(rec, 4), "fpr": round(fpr, 4), "implied_precision": round(prec, 4)})
pd.DataFrame(pp).to_csv("prevalence_precision.csv", index=False)
print(pd.DataFrame(pp).to_string(index=False))

# ---------------- figures (zero baseline, error bars) ----------------
def acc_stats(setting, features, model):
    v = raw[(raw.setting == setting) & (raw.features == features) & (raw.model == model)]["accuracy"] * 100
    return v.mean(), v.std()


bars = [("Rule heuristic", "n/a", "Rule heuristic"), ("Decision stump", "n/a", "Decision stump"),
        ("Depth-3 tree (URL-only)", "n/a", "Depth-3 tree (URL-only)"), ("Depth-3 tree (all)", "n/a", "Depth-3 tree (all)"),
        ("Domain blacklist", "n/a", "Domain blacklist"),
        ("Logistic Reg. (URL-only)", "URL-only", "Logistic Regression"), ("Random Forest (URL-only)", "URL-only", "Random Forest"),
        ("XGBoost (URL-only)", "URL-only", "XGBoost"),
        ("Logistic Reg. (all 87)", "All 87", "Logistic Regression"), ("Random Forest (all 87)", "All 87", "Random Forest"),
        ("XGBoost (all 87)", "All 87", "XGBoost")]
SA, SB = "A: Random (stratified)", "B: Domain-grouped (stratified)"
fig, ax = plt.subplots(figsize=(7.5, 5.2))
h = 0.38
for i, (lab, feat, mod) in enumerate(bars):
    ma, sa = acc_stats(SA, feat, mod)
    ax.barh(i + h / 2, ma, h, xerr=sa, color="#1f77b4", label="A: random split" if i == 0 else None, capsize=2)
    if mod == "Domain blacklist":
        ax.text(2, i - h / 2, "B: zero recall by construction (not plotted)", va="center", fontsize=7)
    else:
        mb, sb = acc_stats(SB, feat, mod)
        ax.barh(i - h / 2, mb, h, xerr=sb, color="#ff7f0e", label="B: domain-grouped split" if i == 0 else None, capsize=2)
ax.set_yticks(range(len(bars)))
ax.set_yticklabels([b[0] for b in bars], fontsize=8)
ax.set_xlim(0, 100)
ax.set_xlabel("Accuracy (%), mean ± SD over 15 folds")
ax.set_ylabel("Method")
ax.legend(title="Evaluation", fontsize=8, loc="lower right")
ax.set_title("Static baselines vs ML models", fontsize=10)
plt.tight_layout()
plt.savefig("results_v3_main.png", dpi=220)

abl = ["URL-only", "Content-only", "External-only", "URL+external", "Content+external",
       "No external (URL+content)", "All 87"]
fig, ax = plt.subplots(figsize=(7.5, 4))
for j, (s, c, lab) in enumerate([(SA, "#1f77b4", "A: random split"), (SB, "#ff7f0e", "B: domain-grouped split")]):
    m = [acc_stats(s, a, "XGBoost")[0] for a in abl]
    sd = [acc_stats(s, a, "XGBoost")[1] for a in abl]
    ax.bar(np.arange(len(abl)) + (j - 0.5) * 0.38, m, 0.38, yerr=sd, color=c, label=lab, capsize=2)
ax.set_xticks(range(len(abl)))
ax.set_xticklabels(abl, rotation=25, ha="right", fontsize=8)
ax.set_ylim(50, 100)
ax.set_ylabel("XGBoost accuracy (%)")
ax.set_xlabel("Feature set")
ax.legend(title="Evaluation", fontsize=8)
ax.set_title("Feature-group ablation (note: y-axis starts at 50%)", fontsize=10)
plt.tight_layout()
plt.savefig("results_v3_ablation.png", dpi=220)
print("done")
