#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "results.csv").open()))
rows.sort(key=lambda r: float(r["TI-HOTA"]))
names = [r["profile"].replace("_", " ") for r in rows]
scores = [float(r["TI-HOTA"]) for r in rows]
colors = ["#c23b32" if n == "extreme oblique" else "#e18d48" if n.startswith("hard ") else "#327f8b" for n in names]

fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
bars = ax.barh(names, scores, color=colors, height=0.68)
ax.axvline(float(next(r["TI-HOTA"] for r in rows if r["profile"] == "base")),
           color="#333333", linestyle="--", linewidth=1.2, label="base view")
ax.set_xlim(0.70, 1.005)
ax.set_xlabel("TI-HOTA")
ax.set_title("Current pipeline under 2D virtual-camera stress profiles\nIMG_0108 · 600 frames")
ax.grid(axis="x", alpha=0.22)
ax.legend(loc="lower right", frameon=False)
for bar, value in zip(bars, scores):
    ax.text(value + 0.003, bar.get_y() + bar.get_height() / 2, f"{value:.4f}",
            va="center", ha="left", fontsize=9)
fig.savefig(HERE / "results.png", dpi=180)
