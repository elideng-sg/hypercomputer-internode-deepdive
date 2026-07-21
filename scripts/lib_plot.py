#!/usr/bin/env python3
import argparse, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True); ap.add_argument("--x", required=True)
    ap.add_argument("--y", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--logx", action="store_true"); ap.add_argument("--kind", default="line")
    ap.add_argument("--title", default=""); ap.add_argument("--xlabel", default="")
    ap.add_argument("--ylabel", default="")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    fig, ax = plt.subplots(figsize=(8,5))
    for col in a.y.split(","):
        if a.kind == "bar": ax.bar(df[a.x].astype(str), df[col], label=col)
        else: ax.plot(df[a.x], df[col], marker="o", label=col)
    if a.logx: ax.set_xscale("log", base=2)
    ax.set_title(a.title); ax.set_xlabel(a.xlabel or a.x); ax.set_ylabel(a.ylabel)
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(a.out, dpi=120)
    print(f"wrote {a.out}")

if __name__ == "__main__": main()
