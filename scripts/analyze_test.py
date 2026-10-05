"""Análises sobre a avaliação no teste do modelo final, sem nova inferência (E17).

Lê <run>/eval_test/predictions.csv e calcula:

  1. intervalos de confiança de 95% por bootstrap sobre radiografias;
  2. análise de sensibilidade post hoc sem predições degeneradas (largura ou
     altura < 2 px), rotulada como tal: não altera o resultado principal;
  3. calibração do escore: fração de verdadeiros positivos por faixa de escore
     e erro de calibração esperado (ECE);
  4. figuras da curva FROC e da calibração.

  uv run --group analysis python scripts/analyze_test.py --run runs/s3_final \
      --metadata data/metadata.csv --splits splits/splits.csv --fig-dir ../Tcc/assets/experimentos
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from node21det.data.metadata import boxes_by_image, load_metadata
from node21det.data.splits import select
from node21det.evaluation import FP_RATES, Prediction, evaluate, froc

SERIES = "#2a78d6"  # slot 1 da paleta de referência (validada)
SERIES2 = "#eb6834"  # slot 2 (par validado: CVD ΔE 24,7)
MIN_CAL_SCORE = 0.1  # abaixo disso estão ~71 mil predições de fundo, que dominariam a calibração
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = ["mean_sens", "sens@0.125", "sens@0.25", "sens@0.5", "ap@0.2", "auc", "rank", "auc_outside_nodules"]


def predictions_by_image(pred: pd.DataFrame, names: list[str]) -> list[Prediction]:
    groups = {n: g for n, g in pred.groupby("img_name")}
    out = []
    for n in names:
        g = groups.get(n)
        if g is None:
            out.append(Prediction(np.zeros((0, 4)), np.zeros(0)))
        else:
            out.append(Prediction(g[["x1", "y1", "x2", "y2"]].to_numpy(), g["score"].to_numpy()))
    return out


def bootstrap(preds, gts, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(preds)
    rows = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        rows.append(evaluate([preds[i] for i in idx], [gts[i] for i in idx]))
    df = pd.DataFrame(rows)
    return {m: (float(df[m].quantile(0.025)), float(df[m].quantile(0.975))) for m in METRICS}


def calibration(pred: pd.DataFrame, bins: int = 9, min_score: float = MIN_CAL_SCORE):
    """Faixas de igual largura entre min_score e 1. ECE ponderado pelo número de predições."""
    pred = pred[pred["score"] >= min_score]
    edges = np.linspace(min_score, 1, bins + 1)
    b = np.clip(np.digitize(pred["score"], edges) - 1, 0, bins - 1)
    table = pred.assign(bin=b).groupby("bin").agg(n=("score", "size"), mean_score=("score", "mean"),
                                                 frac_tp=("is_tp", "mean")).reset_index()
    ece = float((table["n"] * (table["mean_score"] - table["frac_tp"]).abs()).sum() / table["n"].sum())
    return table, ece


def br(x: float, nd: int = 3) -> str:
    """Número com vírgula decimal, como no texto do trabalho."""
    return f"{x:.{nd}f}".replace(".", ",")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 1)))
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def froc_figure(preds, gts, reps, seed, path):
    curve = froc(preds, gts)
    grid = np.geomspace(1 / 32, 4, 60)
    main = np.array([curve.sensitivity_at(r) for r in grid])
    rng = np.random.default_rng(seed)
    band = []
    for _ in range(reps):
        idx = rng.integers(0, len(preds), len(preds))
        c = froc([preds[i] for i in idx], [gts[i] for i in idx])
        band.append([c.sensitivity_at(r) for r in grid])
    lo, hi = np.percentile(band, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=200)
    style(ax)
    ax.fill_between(grid, lo, hi, color=SERIES, alpha=0.15, linewidth=0)
    ax.plot(grid, main, color=SERIES, linewidth=2)
    for r in FP_RATES:
        s = curve.sensitivity_at(r)
        ax.axvline(r, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.plot([r], [s], "o", color=SERIES, markersize=7, markeredgecolor="white", markeredgewidth=2)
        ax.annotate(br(s), (r, s), textcoords="offset points", xytext=(6, -14), fontsize=9, color=INK)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4])
    ax.set_xticklabels(["1/32", "1/16", "1/8", "1/4", "1/2", "1", "2", "4"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Falsos positivos por imagem", color=INK, fontsize=10)
    ax.set_ylabel("Sensibilidade", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def calibration_figure(series, path):
    """series: lista de (rótulo, tabela, ece, cor)."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=200)
    style(ax)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.24, 0.31, "calibração perfeita", color=MUTED, fontsize=8, rotation=40)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 1)))
    for label, table, ece, color in series:
        ax.plot(table["mean_score"], table["frac_tp"], color=color, linewidth=2, label=f"{label}: ECE {br(ece)}")
        ax.plot(table["mean_score"], table["frac_tp"], "o", color=color, markersize=7,
                markeredgecolor="white", markeredgewidth=2)
    for _, r in series[0][1].iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["mean_score"], r["frac_tp"]), textcoords="offset points",
                    xytext=(5, 6), fontsize=7, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Escore médio da faixa", color=INK, fontsize=10)
    ax.set_ylabel("Fração de verdadeiros positivos", color=INK, fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--fig-dir", required=True)
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only-calibration", action="store_true", help="refaz só a calibração, mantendo o resto do summary.json")
    args = p.parse_args()

    run = Path(args.run)
    pred = pd.read_csv(run / "eval_test" / "predictions.csv")
    boxes = boxes_by_image(load_metadata(args.metadata))
    names = select(pd.read_csv(args.splits), split="test")
    gts = [boxes[n] for n in names]
    out = run / "analysis"
    out.mkdir(exist_ok=True)

    degenerate = ((pred.x2 - pred.x1) < 2) | ((pred.y2 - pred.y1) < 2)
    table, ece = calibration(pred)
    table_nd, ece_nd = calibration(pred[~degenerate])
    table.to_csv(out / "calibration.csv", index=False)
    table_nd.to_csv(out / "calibration_without_degenerate.csv", index=False)
    calibration_figure([("todas as predições", table, ece, SERIES),
                        ("sem degeneradas, post hoc", table_nd, ece_nd, SERIES2)],
                       Path(args.fig_dir) / "calibracao_teste.png")
    print(f"calibração (escore >= {MIN_CAL_SCORE}): ECE {ece:.3f}; sem degeneradas {ece_nd:.3f}")
    print(table.round(3).to_string(index=False))
    print(table_nd.round(3).to_string(index=False))
    if args.only_calibration:
        summary = json.loads((out / "summary.json").read_text())
        summary.update(ece=ece, ece_without_degenerate=ece_nd, calibration_min_score=MIN_CAL_SCORE)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return

    preds = predictions_by_image(pred, names)
    main_metrics = evaluate(preds, gts)
    ci = bootstrap(preds, gts, args.reps, args.seed)

    posthoc_metrics = evaluate(predictions_by_image(pred[~degenerate], names), gts)

    summary = {
        "n_images": len(names), "n_lesions": int(sum(len(g) for g in gts)), "bootstrap_reps": args.reps,
        "main": {m: main_metrics[m] for m in METRICS},
        "ci95": ci,
        "posthoc_without_degenerate": {m: posthoc_metrics[m] for m in METRICS},
        "degenerate_predictions": int(degenerate.sum()),
        "ece": ece, "ece_without_degenerate": ece_nd, "calibration_min_score": MIN_CAL_SCORE,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    fig_dir = Path(args.fig_dir)
    froc_figure(preds, gts, min(args.reps, 300), args.seed, fig_dir / "froc_teste.png")

    print(f"{'métrica':22s} {'teste':>7s}  {'IC 95%':>17s}  {'sem degeneradas (post hoc)':>26s}")
    for m in METRICS:
        lo, hi = ci[m]
        print(f"{m:22s} {main_metrics[m]:7.3f}  [{lo:.3f}, {hi:.3f}]  {posthoc_metrics[m]:26.3f}")
    print(f"\npredições degeneradas: {int(degenerate.sum())}")


if __name__ == "__main__":
    main()
