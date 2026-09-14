"""Benchmark, sintonia de hiperparâmetros e serialização do modelo campeão."""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

SEED = 42

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:  # pragma: no cover - depende do ambiente
    HAS_LGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # pragma: no cover
    HAS_XGB = False

ORDEM_METRICAS = [
    "Modelo", "ROC-AUC", "PR-AUC", "Recall (Classe 0)", "F1 (Classe 0)",
    "F1-Macro", "Brier Score", "Acurácia",
]


def evaluate_model(model, X, y, threshold: float = 0.5) -> dict:
    """Métricas de avaliação de um modelo já treinado.

    Recall da classe 0 vem primeiro na leitura porque o custo social de um falso
    negativo - não sinalizar uma criança em risco - é maior que o do inverso.
    O Brier Score entra porque a probabilidade vira priorização de política
    pública, então precisa ser confiável em si mesma, não só ordenar bem.
    """
    y_proba = (model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba")
               else model.predict(X))
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "ROC-AUC": roc_auc_score(y, y_proba),
        "PR-AUC": average_precision_score(y, y_proba),
        "Brier Score": brier_score_loss(y, y_proba),
        "F1-Macro": f1_score(y, y_pred, average="macro"),
        "F1 (Classe 0)": f1_score(y, y_pred, pos_label=0),
        "Recall (Classe 0)": recall_score(y, y_pred, pos_label=0),
        "Precisão (Classe 0)": precision_score(y, y_pred, pos_label=0, zero_division=0),
        "Acurácia": accuracy_score(y, y_pred),
    }


def modelos_do_benchmark(random_state: int = SEED) -> dict:
    """Os cinco algoritmos comparados, do baseline ao gradient boosting."""
    modelos = {
        "Dummy (Estratificado)": DummyClassifier(
            strategy="stratified", random_state=random_state),
        "Regressão Logística": LogisticRegression(
            class_weight="balanced", max_iter=1000, C=1.0, random_state=random_state),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, class_weight="balanced",
            random_state=random_state, n_jobs=-1),
    }
    if HAS_LGBM:
        modelos["LightGBM"] = LGBMClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            class_weight="balanced", random_state=random_state, verbose=-1)
    if HAS_XGB:
        modelos["XGBoost"] = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            eval_metric="logloss", random_state=random_state, n_jobs=-1)
    return modelos


def run_benchmark(build_preprocessor_fn, X_train, y_train, X_val, y_val,
                  random_state: int = SEED):
    """Treina os cinco modelos e compara no conjunto de validação.

    O pré-processador é reconstruído a cada modelo (daí receber uma função e não
    um objeto): reaproveitar a mesma instância já ajustada carregaria as
    estatísticas de um fit anterior.
    """
    resultados, treinados = [], {}
    for nome, estimador in modelos_do_benchmark(random_state).items():
        logger.info("Treinando %s...", nome)
        pipe = Pipeline([
            ("preprocessor", build_preprocessor_fn()),
            ("classifier", estimador),
        ])
        pipe.fit(X_train, y_train)
        treinados[nome] = pipe
        metricas = evaluate_model(pipe, X_val, y_val)
        metricas["Modelo"] = nome
        resultados.append(metricas)

    df_res = (pd.DataFrame(resultados)[ORDEM_METRICAS]
              .sort_values("ROC-AUC", ascending=False)
              .reset_index(drop=True))
    return df_res, treinados


def grade_de_busca(nome_campeao: str, random_state: int = SEED):
    """Estimador e grade de hiperparâmetros do modelo campeão do benchmark."""
    if "Random Forest" in nome_campeao:
        return RandomForestClassifier(
            class_weight="balanced", random_state=random_state, n_jobs=-1), {
            "classifier__n_estimators": [100, 150],
            "classifier__max_depth": [6, 8, 10, 12],
            "classifier__min_samples_split": [2, 5, 10],
        }
    if "LightGBM" in nome_campeao and HAS_LGBM:
        return LGBMClassifier(
            class_weight="balanced", random_state=random_state, verbose=-1), {
            "classifier__n_estimators": [100, 150, 250],
            "classifier__max_depth": [3, 4, 5, 7],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__num_leaves": [15, 31, 63],
        }
    if "XGBoost" in nome_campeao and HAS_XGB:
        return XGBClassifier(
            eval_metric="logloss", random_state=random_state, n_jobs=-1), {
            "classifier__n_estimators": [100, 150, 250],
            "classifier__max_depth": [3, 4, 5, 7],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__subsample": [0.8, 1.0],
        }
    return LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=random_state), {
        "classifier__C": np.logspace(-3, 2, 10),
    }


def tune_hyperparameters(nome_campeao, build_preprocessor_fn, X_train, y_train,
                         n_iter: int = 10, cv_splits: int = 5,
                         random_state: int = SEED) -> RandomizedSearchCV:
    """Busca aleatória com validação cruzada estratificada, otimizando ROC-AUC."""
    estimador, grade = grade_de_busca(nome_campeao, random_state)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    pipe = Pipeline([
        ("preprocessor", build_preprocessor_fn()),
        ("classifier", estimador),
    ])
    busca = RandomizedSearchCV(
        pipe, param_distributions=grade, n_iter=n_iter, scoring="roc_auc",
        cv=cv, random_state=random_state, n_jobs=-1, verbose=1,
    )
    busca.fit(X_train, y_train)
    logger.info("Melhor ROC-AUC na validação cruzada: %.4f", busca.best_score_)
    return busca


def save_champion_model(model, filepath="artifacts/modelo_alfabetizacao_final.joblib") -> str:
    """Serializa o Pipeline inteiro, com pré-processamento acoplado."""
    caminho = Path(filepath)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, caminho)
    logger.info("Modelo campeão salvo em %s", caminho)
    return str(caminho)
