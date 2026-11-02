"""Week 8: Word2Vec training, inspection, parameter experiments and a mini-project.

Run with:  PYTHONHASHSEED=0 python word2vec_experiments.py
(PYTHONHASHSEED and workers=1 + seed make the results reproducible.)
"""
import csv
import itertools
import json
import os
import time

import numpy as np
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")

# ------------------------------------------------------------------ corpus
with open(os.path.join(HERE, "corpus.txt"), encoding="utf-8") as fh:
    raw = [ln.strip() for ln in fh if ln.strip()]
sents = [simple_preprocess(s) for s in raw]          # tokenise + lowercase

BASE = dict(vector_size=50, window=5, min_count=2, sg=1, epochs=100, seed=42, workers=1)


def train(data, **over):
    cfg = {**BASE, **over}
    t0 = time.time()
    m = Word2Vec(sentences=data, **cfg)
    m.train_seconds = time.time() - t0
    return m


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def neighbours(m, w, n=3):
    return [(t, round(float(s), 2)) for t, s in m.wv.most_similar(w, topn=n)] if w in m.wv else "OOV"


# ------------------------------------------------ 1. corpus + representations
hdr("1. CORPUS DESCRIPTION AND PREPROCESSING")
tokens = [t for s in sents for t in s]
print(f"sentences: {len(sents)}   tokens: {len(tokens)}   distinct words: {len(set(tokens))}")
print("raw     :", raw[0])
print("tokens  :", sents[0])

hdr("1b. ONE-HOT vs LEARNED EMBEDDINGS")
vocab4 = ["teacher", "student", "school", "book"]
onehot = {w: np.eye(4)[i] for i, w in enumerate(vocab4)}
cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
print("one-hot cosine(teacher, student) =", cos(onehot["teacher"], onehot["student"]))
print("one-hot cosine(teacher, book)    =", cos(onehot["teacher"], onehot["book"]))

# --------------------------------------------------------- 2. Task 1: train
hdr("2. TASK 1 - TRAIN AND INSPECT (Skip-gram, size 50, window 5, min_count 2, 100 epochs, seed 42)")
model = train(sents)
print(f"vocabulary size: {len(model.wv)}   training time: {model.train_seconds:.2f}s")
for w in ["teacher", "student", "computer", "model", "exam"]:
    v = model.wv[w]
    print(f"vector[{w}] (first 6 of {len(v)} dims): {np.round(v[:6], 3)}   |v|={np.linalg.norm(v):.2f}")

print("\nThree most similar words for three selected terms:")
for w in ["teacher", "computer", "model"]:
    print(f"  {w:<9}", neighbours(model, w, 3))

hdr("2b. NEAREST NEIGHBOURS OF TEN SELECTED WORDS (top 5)")
ten = ["teacher", "student", "lecturer", "exam", "library", "computer", "program", "data", "network", "model"]
for w in ten:
    print(f"  {w:<9}", neighbours(model, w, 5))

hdr("2c. COSINE SIMILARITY CHECK (manual formula vs gensim)")
a, b = model.wv["teacher"], model.wv["lecturer"]
print("manual  cosine(teacher, lecturer):", round(cos(a, b), 4))
print("gensim  similarity(teacher, lecturer):", round(float(model.wv.similarity("teacher", "lecturer")), 4))
print("similarity(teacher, server)  :", round(float(model.wv.similarity("teacher", "server")), 4))

hdr("2d. OUT-OF-VOCABULARY WORDS")
for w in ["blockchain", "professor"]:
    try:
        model.wv[w]
    except KeyError:
        print(f"'{w}' -> KeyError: not in the learned vocabulary (OOV). Word2Vec cannot embed unseen words.")

# ------------------------------------------------------ 3. experiments A-E
RELATED = [("teacher", "lecturer"), ("student", "students"), ("program", "code"), ("computer", "laptop"),
           ("exam", "assignment"), ("school", "university"), ("server", "network"), ("data", "model"),
           ("library", "books"), ("software", "application")]
UNRELATED = [("teacher", "server"), ("exam", "network"), ("library", "algorithm"), ("student", "database"),
             ("lesson", "internet"), ("classroom", "model"), ("book", "bug"), ("lecturer", "software"),
             ("school", "data"), ("desks", "program")]


def separation(m):
    rel = [m.wv.similarity(a, b) for a, b in RELATED if a in m.wv and b in m.wv]
    unr = [m.wv.similarity(a, b) for a, b in UNRELATED if a in m.wv and b in m.wv]
    return float(np.mean(rel)), float(np.mean(unr)), len(rel), len(unr)


EXPERIMENTS = [
    ("Base", "size 50, window 5, min_count 2, skip-gram, 100 sentences", {}, sents),
    ("A1", "vector size 25", dict(vector_size=25), sents),
    ("A2", "vector size 100", dict(vector_size=100), sents),
    ("B1", "window 2", dict(window=2), sents),
    ("B2", "window 5 (= base)", dict(window=5), sents),
    ("C1", "min_count 1", dict(min_count=1), sents),
    ("C2", "min_count 2 (= base)", dict(min_count=2), sents),
    ("D1", "CBOW (sg=0)", dict(sg=0), sents),
    ("D2", "Skip-gram (sg=1, = base)", dict(sg=1), sents),
    ("E1", "small corpus: first 30 sentences, min_count 1", dict(min_count=1), sents[:30]),
    ("E2", "full corpus: 100 sentences, min_count 1", dict(min_count=1), sents),
]
hdr("3. CONTROLLED PARAMETER EXPERIMENTS (one setting changed at a time; seed 42)")
print(f"{'ID':<5}{'setting':<54}{'vocab':>6}{'time(s)':>9}{'rel':>7}{'unrel':>7}{'gap':>7}  pairs(r/u)")
exp_rows, models = [], {}
for eid, desc, over, data in EXPERIMENTS:
    m = train(data, **over)
    models[eid] = m
    r, u, nr, nu = separation(m)
    exp_rows.append(dict(id=eid, setting=desc, vocab=len(m.wv), seconds=round(m.train_seconds, 2),
                         rel=round(r, 3), unrel=round(u, 3), gap=round(r - u, 3), pairs=f"{nr}/{nu}",
                         teacher=neighbours(m, "teacher", 3), computer=neighbours(m, "computer", 3)))
    print(f"{eid:<5}{desc:<54}{len(m.wv):>6}{m.train_seconds:>9.2f}{r:>7.2f}{u:>7.2f}{r-u:>7.2f}  {nr}/{nu}")
print("\n'rel' = mean cosine of 10 hand-picked related pairs, 'unrel' = mean of 10 unrelated pairs,")
print("'gap' = rel - unrel (higher = embeddings separate related from unrelated words better).")
print("Pairs containing a word missing from a model's vocabulary are skipped (see pairs r/u).")

print("\nNeighbours of 'teacher' (top 3) per configuration:")
for row in exp_rows:
    print(f"  {row['id']:<5}", row["teacher"])
print("\nNeighbours of 'computer' (top 3) per configuration:")
for row in exp_rows:
    print(f"  {row['id']:<5}", row["computer"])

c1, c2 = models["C1"], models["C2"]
excluded = sorted(set(c1.wv.index_to_key) - set(c2.wv.index_to_key))
print(f"\nMin-frequency effect: min_count=1 keeps {len(c1.wv)} words, min_count=2 keeps {len(c2.wv)}; "
      f"{len(excluded)} words removed, e.g. {excluded[:10]}")

# --------------------------------------------------------------- 4. stability
hdr("4. STABILITY ACROSS RANDOM SEEDS (5 seeds; mean Jaccard overlap of top-5 neighbours, 10 words)")
stab_words = ["teacher", "student", "lecturer", "exam", "library", "computer", "program", "data", "network", "model"]


def stability(data, **over):
    ms = [train(data, seed=s, **over) for s in range(5)]
    scores = []
    for w in stab_words:
        if not all(w in m.wv for m in ms):
            continue
        sets = [set(t for t, _ in m.wv.most_similar(w, topn=5)) for m in ms]
        for x, y in itertools.combinations(sets, 2):
            scores.append(len(x & y) / len(x | y))
    return float(np.mean(scores)), len(scores)


stab = {}
for name, data, over in [("Skip-gram, 100 sentences", sents, {}),
                         ("CBOW, 100 sentences", sents, dict(sg=0)),
                         ("Skip-gram, 30 sentences (min_count 1)", sents[:30], dict(min_count=1))]:
    s, n = stability(data, **over)
    stab[name] = round(s, 3)
    print(f"  {name:<42} mean neighbour overlap = {s:.3f}   (from {n} seed-pair comparisons)")
print("1.0 = identical neighbours for every seed; values near 0 = neighbours are essentially random.")

# ---------------------------------------------------------------- 5. PCA plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

edu = ["teacher", "student", "students", "lecturer", "exam", "lesson", "classroom", "library", "school", "university", "assignment"]
tech = ["computer", "program", "code", "software", "server", "network", "data", "model", "database", "internet", "application"]
words = [w for w in edu + tech if w in model.wv]
xy = PCA(n_components=2, random_state=0).fit_transform(np.array([model.wv[w] for w in words]))
fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)
for (x, y), w in zip(xy, words):
    colour = "#1f6feb" if w in edu else "#d4572a"
    ax.scatter(x, y, c=colour, s=28)
    ax.annotate(w, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
ax.set_title("Word2Vec vectors projected to 2-D with PCA\nblue = education words, orange = technology words", fontsize=10)
ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "w8_pca.png"))
print("\nSaved figure: figures/w8_pca.png")

# ------------------------------------------------- 6. applied mini-project
hdr("6. APPLIED MINI-PROJECT - classify student feedback (teaching / facilities / technology)")
texts, labels = [], []
with open(os.path.join(HERE, "feedback.csv"), encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        texts.append(row["text"]); labels.append(row["label"])
print(f"feedback sentences: {len(texts)}   classes: { {l: labels.count(l) for l in sorted(set(labels))} }")
print("majority-class baseline accuracy:", round(max(labels.count(l) for l in set(labels)) / len(labels), 3))


def doc_vec(m, text):
    vs = [m.wv[w] for w in simple_preprocess(text) if w in m.wv]
    return np.mean(vs, axis=0) if vs else np.zeros(m.vector_size)


y = np.array(labels)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
results = {}
for name, feats in [("Word2Vec skip-gram (avg vector)", np.array([doc_vec(models["D2"], t) for t in texts])),
                    ("Word2Vec CBOW (avg vector)", np.array([doc_vec(models["D1"], t) for t in texts]))]:
    sc = cross_val_score(LogisticRegression(max_iter=2000), feats, y, cv=cv)
    results[name] = (sc.mean(), sc.std())
sc = cross_val_score(make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=2000)), texts, y, cv=cv)
results["TF-IDF bag of words (baseline)"] = (sc.mean(), sc.std())
print("\n5-fold cross-validation, repeated 5 times (25 scores):")
for k, (mu, sd) in results.items():
    print(f"  {k:<36} accuracy = {mu:.3f} +/- {sd:.3f}")

feats = np.array([doc_vec(models["D2"], t) for t in texts])
pred = cross_val_predict(LogisticRegression(max_iter=2000), feats, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
print("\nSkip-gram embedding classifier - per-class report (single 5-fold run):")
print(classification_report(y, pred, digits=2))
cls = sorted(set(labels))
print("confusion matrix (rows = true, cols = predicted):", cls)
print(confusion_matrix(y, pred, labels=cls))
oov = sorted({w for t in texts for w in simple_preprocess(t) if w not in models["D2"].wv})
print(f"\nfeedback words missing from the embedding vocabulary ({len(oov)}): {oov[:20]}")

with open(os.path.join(HERE, "w8_results.json"), "w", encoding="utf-8") as fh:
    json.dump(dict(experiments=exp_rows, stability=stab,
                   classification={k: [round(a, 3), round(b, 3)] for k, (a, b) in results.items()}), fh, indent=2)
