"""Carga da ABT da camada Gold, definição de features e pré-processamento.

Este módulo não transforma dado de origem: a ABT é construída pelo pipeline da
Fase 2 (`gold.abt_alfabetizacao`). Aqui só selecionamos a família de indicadores
correta, separamos treino/validação/teste e montamos o `ColumnTransformer`.
"""
from __future__ import annotations

import logging
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

PROJETO_PADRAO = os.environ.get("GCP_PROJECT_ID", "tech-challenge-alfabetiza-25")
DATASET_GOLD = os.environ.get("BQ_DATASET_GOLD", "gold")

SEED = 42
TARGET_COL = "alfabetizado"

# Separadas por origem de propósito: o SHAP precisa saber o que é real e o que
# é sintético para que a leitura do resultado seja honesta (ver README, seção 9).
FEATURES_NUM_REAIS = [
    "mun_taxa_alfabetizacao",
    "mun_media_portugues",
    "mun_gap_meta_2024",
    "escola_qtd_alunos_avaliados",
]
FEATURES_NUM_SINTETICAS = [
    "idhm_educacao",
    "idhm_renda",
    "pib_per_capita",
    "indice_gini",
    "taxa_abandono_fundamental",
]
FEATURES_CAT_REAIS = ["sigla_uf", "nome_regiao", "rede_label"]
FEATURES_CAT_SINTETICAS = ["porte_municipio"]

COLS_SINTETICAS = set(FEATURES_NUM_SINTETICAS + FEATURES_CAT_SINTETICAS)

# Nunca podem virar feature. proficiencia_audit_only define matematicamente o
# alvo (corte 743); peso_aluno é peso amostral, não atributo do aluno; o resto
# são identificadores.
COLS_PROIBIDAS = {
    "proficiencia", "proficiencia_audit_only", "id_aluno", "id_escola",
    "id_municipio", "nome_municipio", "ano", "peso_aluno", TARGET_COL,
}

# Pares (mesmo ano, ano anterior). A versão sem sufixo contém o alvo agregado
# do próprio município: correlação medida de 0,993 com a média do alvo. Só a
# versão _ant pode alimentar o modelo.
PARES_INDICADOR = [
    ("mun_taxa_alfabetizacao", "mun_taxa_alfabetizacao_ant"),
    ("mun_media_portugues", "mun_media_portugues_ant"),
    ("mun_nivel_alfabetizacao", "mun_nivel_alfabetizacao_ant"),
    ("mun_percentual_participacao", "mun_percentual_participacao_ant"),
    ("mun_gap_meta_2024", "mun_gap_meta_2024_ant"),
    ("uf_taxa_alfabetizacao", "uf_taxa_alfabetizacao_ant"),
    ("uf_media_portugues", "uf_media_portugues_ant"),
    ("razao_mun_uf_taxa", "razao_mun_uf_taxa_ant"),
]

SQL_ABT = """
SELECT *
FROM `{projeto}.{gold}.abt_alfabetizacao`
WHERE TRUE
  {filtro_modo}
  {filtro_ano}
"""


def listar_features(incluir_socio: bool = True) -> tuple[list[str], list[str]]:
    """Devolve (numéricas, categóricas) conforme o uso ou não das sintéticas."""
    if incluir_socio:
        return (
            FEATURES_NUM_SINTETICAS + FEATURES_NUM_REAIS,
            FEATURES_CAT_REAIS + FEATURES_CAT_SINTETICAS,
        )
    return list(FEATURES_NUM_REAIS), list(FEATURES_CAT_REAIS)


def load_data(client=None, projeto=PROJETO_PADRAO, gold=DATASET_GOLD,
              modo="ano_anterior", ano=None) -> pd.DataFrame:
    """Lê a ABT da camada Gold e seleciona a família de indicadores pedida.

    Args:
        modo: "ano_anterior" usa as colunas _ant (sem vazamento do alvo, é o
            padrão); "mesmo_ano" usa as do próprio ano, só para descrição.
    """
    if modo not in ("ano_anterior", "mesmo_ano"):
        raise ValueError('modo deve ser "ano_anterior" ou "mesmo_ano"')

    if client is None:
        from google.cloud import bigquery
        client = bigquery.Client(project=projeto)

    sql = SQL_ABT.format(
        projeto=projeto,
        gold=gold,
        filtro_modo=("AND mun_taxa_alfabetizacao_ant IS NOT NULL"
                     if modo == "ano_anterior" else ""),
        filtro_ano=f"AND ano = {int(ano)}" if ano is not None else "",
    )
    job = client.query(sql)
    df = job.result().to_dataframe()
    logger.info("ABT lida da Gold: %d linhas | %.0f MB escaneados | modo=%s",
                len(df), (job.total_bytes_processed or 0) / 1024 / 1024, modo)

    mesmo_ano = [par[0] for par in PARES_INDICADOR]
    ano_anterior = [par[1] for par in PARES_INDICADOR]
    if modo == "ano_anterior":
        df = df.drop(columns=mesmo_ano)
        df = df.rename(columns=dict(zip(ano_anterior, mesmo_ano)))
    else:
        df = df.drop(columns=ano_anterior)

    return df.drop(columns=["_processed_at"], errors="ignore").reset_index(drop=True)


def otimizar_memoria(df: pd.DataFrame) -> pd.DataFrame:
    """Converte texto para categoria e float64 para float32.

    Em milhões de linhas isso corta o uso de memória pela metade, o que é a
    diferença entre rodar e não rodar no Colab gratuito.
    """
    df = df.copy()
    for col in ("id_municipio", "nome_municipio", "sigla_uf", "nome_regiao",
                "rede_label", "porte_municipio"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in df.select_dtypes("float").columns:
        df[col] = df[col].astype("float32")
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype("int8")
    return df


def build_preprocessor(features_num: list[str], features_cat: list[str]) -> ColumnTransformer:
    """Imputação + escala para numéricas, imputação + one-hot para categóricas.

    O ColumnTransformer é devolvido solto de propósito: ele deve ser acoplado ao
    estimador dentro de um Pipeline, para que o `.fit()` nunca veja validação
    ou teste.
    """
    pipe_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    pipe_cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[("num", pipe_num, features_num), ("cat", pipe_cat, features_cat)],
        remainder="drop",
    )


def split_data(df: pd.DataFrame, features_num: list[str], features_cat: list[str],
               test_size=0.15, val_size=0.15, random_state=SEED):
    """Separa em treino (70%), validação (15%) e teste (15%), estratificado."""
    vazadas = COLS_PROIBIDAS.intersection(features_num + features_cat)
    if vazadas:
        raise ValueError(f"Coluna proibida na lista de features: {vazadas}")

    X = df[features_num + features_cat].copy()
    # O SimpleImputer não lida bem com dtype 'category'
    for col in features_cat:
        X[col] = X[col].astype(object)
    y = df[TARGET_COL].astype(int)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    val_ajustado = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ajustado,
        random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def amostrar_treino(X_train, y_train, n_max: int, random_state=SEED):
    """Reduz o treino para n_max linhas, mantendo a proporção do alvo.

    Medimos: 200 mil linhas dão ROC-AUC 0,6708 e 3,35 milhões dão 0,6769. A
    diferença não paga o tempo de máquina, e validação e teste seguem íntegros.
    """
    if len(X_train) <= n_max:
        return X_train, y_train
    idx, _ = train_test_split(
        X_train.index, train_size=n_max, random_state=random_state, stratify=y_train
    )
    return X_train.loc[idx], y_train.loc[idx]
