"""Interpretabilidade do modelo campeão com SHAP.

O ponto não trivial deste módulo é a amostragem. Explicar centenas de milhares
de registros é caro e desnecessário, mas amostrar mal distorce a conclusão --
e a conclusão do SHAP é o que vai para o relatório executivo.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEED = 42


def amostrar_para_shap(X_test, y_test, pesos, n_alvo: int = 8000,
                       min_por_celula: int = 30, estratos=("nome_regiao", "rede_label"),
                       random_state: int = SEED):
    """Amostra estratificada e ponderada pelo peso amostral do INEP.

    Três decisões, cada uma com um motivo:

    1. Só o conjunto de teste. Explicar o treino mostraria o que o modelo
       decorou, não o que aprendeu.
    2. Estratificação por alvo x região x rede, com piso por célula. Sorteio
       simples perderia os estratos raros -- a rede Privada tem 24 alunos em
       3,35 milhões; num sorteio de 8 mil, sairia zero.
    3. Peso proporcional a `peso_aluno`. O INEP atribui peso amostral a cada
       aluno; ignorá-lo faria o SHAP retratar o arquivo em vez da população de
       crianças. Como a Fase 2 usa esse peso para reproduzir o indicador
       oficial, descartá-lo aqui seria incoerente.
    """
    aux = X_test.copy()
    aux["_alvo"] = y_test
    aux["_peso"] = pd.Series(pesos, index=X_test.index).astype("float64")
    aux["_peso"] = aux["_peso"].fillna(aux["_peso"].median()).clip(lower=1e-6)

    chaves = ["_alvo", *[e for e in estratos if e in aux.columns]]
    partes = []
    for _, grupo in aux.groupby(chaves, observed=True):
        cota = max(min_por_celula, int(round(n_alvo * len(grupo) / len(aux))))
        partes.append(grupo.sample(n=min(cota, len(grupo)),
                                   weights=grupo["_peso"], random_state=random_state))
    amostra = pd.concat(partes)
    logger.info("Amostra para SHAP: %d registros de %d do teste",
                len(amostra), len(X_test))
    return amostra.drop(columns=["_alvo", "_peso"]), amostra["_alvo"]


def compute_shap_explanations(model, X):
    """Aplica o pré-processamento e calcula os valores SHAP no classificador.

    O TreeExplainer no modo padrão (`tree_path_dependent`) usa as estatísticas
    de cobertura das próprias árvores e dispensa conjunto de referência.
    """
    import shap

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    X_trans = preprocessor.transform(X)
    nomes = [c.replace("num__", "").replace("cat__", "")
             for c in preprocessor.get_feature_names_out()]
    X_df = pd.DataFrame(X_trans, columns=nomes, index=X.index)

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer(X_df)
    # Classificação binária pode devolver (n, features, 2); fica só a classe 1
    if len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
        shap_values = shap_values[:, :, 1]
    return explainer, shap_values, X_df


def verificar_convergencia(shap_values) -> float:
    """Correlação de Spearman entre os rankings de duas submostras disjuntas.

    Acima de 0,95 o ranking estabilizou e aumentar a amostra não muda a
    conclusão. É a diferença entre afirmar e demonstrar que a amostra bastou.
    """
    from scipy.stats import spearmanr

    meio = len(shap_values.values) // 2
    a = np.abs(shap_values.values[:meio]).mean(axis=0)
    b = np.abs(shap_values.values[meio:]).mean(axis=0)
    rho = float(spearmanr(a, b).statistic)
    logger.info("Convergência do ranking SHAP (Spearman): %.4f", rho)
    return rho


def ranking_importancia(shap_values, X_df, cols_sinteticas) -> pd.DataFrame:
    """Importância média |SHAP|, marcando o que veio de dado sintético."""
    imp = pd.DataFrame({
        "feature": X_df.columns,
        "importancia_shap": np.abs(shap_values.values).mean(axis=0),
    }).sort_values("importancia_shap", ascending=False).reset_index(drop=True)

    imp["origem"] = imp["feature"].apply(
        lambda f: "SINTÉTICA" if any(f == c or f.startswith(c + "_")
                                     for c in cols_sinteticas)
        else "real (Fase 2 / IBGE)"
    )
    imp["participacao_%"] = (
        imp["importancia_shap"] / imp["importancia_shap"].sum() * 100
    ).round(2)
    return imp


def plot_shap_summary(shap_values, titulo="", save_dir="images/shap", max_display=15):
    """Beeswarm (direção do efeito) e barra (importância média)."""
    import matplotlib.pyplot as plt
    import shap

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 7))
    shap.plots.beeswarm(shap_values, max_display=max_display, show=False)
    plt.title(f"SHAP — distribuição dos efeitos por variável {titulo}",
              fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/01_shap_summary_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.show()

    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    plt.title("Importância média |SHAP| por variável", fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/02_shap_importance_bar.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_shap_dependence(shap_values, X_df, feature, save_dir="images/shap"):
    """Dependence plot, com o ponto em que o efeito cruza o zero."""
    import matplotlib.pyplot as plt
    import shap

    if feature not in X_df.columns:
        logger.warning("Variável %s ausente após o pré-processamento", feature)
        return None

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    shap.plots.scatter(shap_values[:, feature], show=False)
    plt.axhline(0, color="firebrick", linestyle="--", lw=1)
    plt.title(f"SHAP Dependence — {feature}", fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/03_shap_dependence_{feature}.png", dpi=300, bbox_inches="tight")
    plt.show()

    aux = pd.DataFrame({"valor": X_df[feature].values,
                        "shap": shap_values[:, feature].values})
    aux["faixa"] = pd.qcut(aux["valor"], q=20, duplicates="drop")
    perfil = aux.groupby("faixa", observed=True).agg(
        valor_medio=("valor", "mean"), shap_medio=("shap", "mean")
    ).reset_index(drop=True)
    positivos = perfil[perfil["shap_medio"] > 0]
    if len(positivos) and len(perfil[perfil["shap_medio"] <= 0]):
        return float(positivos["valor_medio"].iloc[0])
    return None


def plot_shap_waterfall(shap_values, proba, save_dir="images/shap", max_display=12):
    """Waterfall dos casos de percentil 1 (alto risco) e 99 (baixo risco).

    Escolhidos por percentil, não sorteados: sortear daria dois casos medianos,
    que não ilustram nada.
    """
    import matplotlib.pyplot as plt
    import shap

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ordem = np.argsort(proba)
    pos_alto = int(ordem[max(0, int(0.01 * len(ordem)))])
    pos_baixo = int(ordem[min(len(ordem) - 1, int(0.99 * len(ordem)))])

    for titulo, pos, arquivo in [
        (f"Aluno de ALTO RISCO — probabilidade prevista: {proba[pos_alto]:.1%}",
         pos_alto, "04_waterfall_alto_risco.png"),
        (f"Aluno de BAIXO RISCO — probabilidade prevista: {proba[pos_baixo]:.1%}",
         pos_baixo, "05_waterfall_baixo_risco.png"),
    ]:
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[pos], max_display=max_display, show=False)
        plt.title(titulo, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/{arquivo}", dpi=300, bbox_inches="tight")
        plt.show()

    return pos_alto, pos_baixo
