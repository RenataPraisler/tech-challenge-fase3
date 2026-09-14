"""Gráficos de avaliação do modelo: ROC, Precision-Recall, calibração e confusão."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay, average_precision_score, brier_score_loss,
    confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve,
)

logger = logging.getLogger(__name__)


def aplicar_estilo() -> None:
    """Estilo visual comum a todos os gráficos do projeto."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 11


def plot_benchmark(df_benchmark, save_dir="images/evaluation"):
    """Comparativo dos modelos em ROC-AUC, PR-AUC e Recall da classe 0."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    df_plot = df_benchmark.melt(
        id_vars=["Modelo"],
        value_vars=["ROC-AUC", "PR-AUC", "Recall (Classe 0)"],
        var_name="Métrica", value_name="Score",
    )
    sns.barplot(data=df_plot, x="Modelo", y="Score", hue="Métrica",
                ax=ax, palette="Blues_r")
    # A linha do acaso deixa explícito quanto de fato foi aprendido
    ax.axhline(0.5, color="firebrick", linestyle="--", lw=1,
               label="Acaso (ROC-AUC = 0,5)")
    ax.set_title("Comparativo de performance dos modelos (validação)", fontweight="bold")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=9)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/00_benchmark_modelos.png", dpi=300)
    plt.show()


def plot_evaluation_curves(model, X_test, y_test, nome_modelo="Modelo campeão",
                           save_dir="images/evaluation"):
    """ROC, Precision-Recall, curva de calibração e matriz de confusão."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0, 0].plot(fpr, tpr, color="#2b5c8f", lw=2,
                    label=f"ROC (AUC = {roc_auc_score(y_test, y_proba):.3f})")
    axes[0, 0].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Acaso")
    axes[0, 0].set_xlabel("Taxa de falsos positivos")
    axes[0, 0].set_ylabel("Taxa de verdadeiros positivos")
    axes[0, 0].set_title("Curva ROC", fontweight="bold")
    axes[0, 0].legend()

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    axes[0, 1].plot(rec, prec, color="#d95f02", lw=2,
                    label=f"PR (AUC = {average_precision_score(y_test, y_proba):.3f})")
    # A taxa base é a referência honesta da curva PR, não o 0,5 da ROC
    axes[0, 1].axhline(y_test.mean(), color="gray", linestyle="--",
                       label=f"Taxa base ({y_test.mean():.3f})")
    axes[0, 1].set_xlabel("Recall")
    axes[0, 1].set_ylabel("Precisão")
    axes[0, 1].set_title("Curva Precision-Recall", fontweight="bold")
    axes[0, 1].legend()

    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    axes[1, 0].plot(prob_pred, prob_true, marker="o", color="#7570b3",
                    label=f"Calibração (Brier = {brier_score_loss(y_test, y_proba):.3f})")
    axes[1, 0].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Perfeita")
    axes[1, 0].set_xlabel("Probabilidade prevista")
    axes[1, 0].set_ylabel("Frequência observada")
    axes[1, 0].set_title("Curva de calibração", fontweight="bold")
    axes[1, 0].legend()

    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, y_pred),
        display_labels=["Não alfabetizado (0)", "Alfabetizado (1)"],
    ).plot(ax=axes[1, 1], cmap="Blues", values_format=",d")
    axes[1, 1].set_title(f"Matriz de confusão — {nome_modelo}", fontweight="bold")
    axes[1, 1].grid(False)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/01_metricas_avaliacao_completas.png", dpi=300)
    plt.show()
