"""Testes de integridade da ABT e do pipeline de pré-processamento.

Os testes que tocam o BigQuery são pulados automaticamente quando não há
credencial disponível, para que a suíte rode em qualquer máquina.

    pytest -q
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing.pipeline import (  # noqa: E402
    COLS_PROIBIDAS, PARES_INDICADOR, TARGET_COL, build_preprocessor,
    listar_features, split_data,
)


def _tem_bigquery() -> bool:
    try:
        from google.cloud import bigquery
        bigquery.Client()
        return True
    except Exception:
        return False


precisa_bq = pytest.mark.skipif(
    not _tem_bigquery(), reason="sem credencial do BigQuery neste ambiente"
)


# --------------------------------------------------------------------------
# Testes que não dependem de nuvem
# --------------------------------------------------------------------------

def test_nenhuma_feature_proibida():
    """A blindagem contra leakage precisa valer nas duas configurações."""
    for incluir_socio in (True, False):
        num, cat = listar_features(incluir_socio)
        assert not COLS_PROIBIDAS.intersection(num + cat)


def test_proficiencia_nunca_e_feature():
    """A nota do teste define o alvo (corte 743) e não pode virar preditor."""
    num, cat = listar_features()
    assert "proficiencia" not in num + cat
    assert "proficiencia_audit_only" not in num + cat


def test_indicadores_do_modelo_tem_versao_defasada():
    """Todo indicador contextual precisa ter par do ano anterior.

    Sem isso, alguma variável entraria no modelo carregando o alvo agregado.
    """
    num, _ = listar_features()
    contextuais = [c for c in num if c.startswith(("mun_", "uf_", "razao_"))]
    com_par = {par[0] for par in PARES_INDICADOR}
    assert set(contextuais).issubset(com_par)


def test_split_rejeita_feature_proibida():
    df = pd.DataFrame({
        "mun_taxa_alfabetizacao": [1.0, 2.0, 3.0, 4.0],
        "peso_aluno": [1.0, 1.0, 1.0, 1.0],
        "sigla_uf": ["SP", "BA", "SP", "BA"],
        TARGET_COL: [0, 1, 0, 1],
    })
    with pytest.raises(ValueError, match="proibida"):
        split_data(df, ["mun_taxa_alfabetizacao", "peso_aluno"], ["sigla_uf"])


def test_preprocessador_so_aprende_no_treino():
    """O ColumnTransformer não pode ver validação nem teste no .fit()."""
    import numpy as np

    n = 400
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "mun_taxa_alfabetizacao": rng.normal(60, 10, n),
        "mun_media_portugues": rng.normal(750, 30, n),
        "mun_gap_meta_2024": rng.normal(0, 5, n),
        "escola_qtd_alunos_avaliados": rng.integers(5, 200, n),
        "sigla_uf": rng.choice(["SP", "BA", "RS"], n),
        "nome_regiao": rng.choice(["Sudeste", "Nordeste", "Sul"], n),
        "rede_label": rng.choice(["Municipal", "Estadual"], n),
        TARGET_COL: rng.integers(0, 2, n),
    })
    num, cat = listar_features(incluir_socio=False)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, num, cat)

    assert len(X_train) + len(X_val) + len(X_test) == n
    # Nenhum registro pode aparecer em dois conjuntos
    assert not set(X_train.index) & set(X_val.index)
    assert not set(X_train.index) & set(X_test.index)
    assert not set(X_val.index) & set(X_test.index)

    pre = build_preprocessor(num, cat)
    pre.fit(X_train)
    media_treino = pre.named_transformers_["num"].named_steps["scaler"].mean_
    pre2 = build_preprocessor(num, cat)
    pre2.fit(pd.concat([X_train, X_test]))
    # Se as médias fossem iguais, o teste teria entrado no ajuste
    assert not (media_treino == pre2.named_transformers_["num"]
                .named_steps["scaler"].mean_).all()


# --------------------------------------------------------------------------
# Testes contra a camada Gold
# --------------------------------------------------------------------------

@precisa_bq
def test_abt_gold_bate_com_a_silver():
    """A ABT precisa conter exatamente o público elegível da Silver."""
    from google.cloud import bigquery
    client = bigquery.Client()
    projeto = client.project
    r = list(client.query(f"""
        SELECT
          (SELECT COUNT(*) FROM `{projeto}.gold.abt_alfabetizacao`) AS na_gold,
          (SELECT COUNT(*) FROM `{projeto}.silver.dados_alunos`
           WHERE presenca = 1 AND preenchimento_caderno = 1) AS elegivel_silver
    """).result())[0]
    assert r.na_gold == r.elegivel_silver, (
        f"ABT com {r.na_gold:,} linhas contra {r.elegivel_silver:,} elegíveis na Silver"
    )


@precisa_bq
def test_abt_gold_alvo_e_chave():
    from google.cloud import bigquery
    client = bigquery.Client()
    r = list(client.query(f"""
        SELECT
          COUNTIF(alfabetizado IS NULL OR alfabetizado NOT IN (0, 1)) AS alvo_invalido,
          COUNT(*) - COUNT(DISTINCT CONCAT(CAST(ano AS STRING), '-', id_aluno))
            AS chave_duplicada,
          COUNTIF(sigla_uf IS NULL OR nome_regiao IS NULL) AS sem_territorio
        FROM `{client.project}.gold.abt_alfabetizacao`
    """).result())[0]
    assert r.alvo_invalido == 0
    assert r.chave_duplicada == 0
    assert r.sem_territorio == 0


@precisa_bq
def test_indicador_do_mesmo_ano_realmente_vaza():
    """Documenta o motivo de existirem as colunas _ant.

    Se um dia esta correlação cair, a decisão de usar o indicador defasado
    merece ser revisitada - e este teste é o lembrete.
    """
    from google.cloud import bigquery
    client = bigquery.Client()
    r = list(client.query(f"""
        WITH alvo AS (
          SELECT ano, id_municipio, AVG(alfabetizado) AS taxa_alvo
          FROM `{client.project}.gold.abt_alfabetizacao`
          GROUP BY ano, id_municipio HAVING COUNT(*) >= 30
        ),
        indicador AS (
          SELECT DISTINCT ano, id_municipio, mun_taxa_alfabetizacao
          FROM `{client.project}.gold.abt_alfabetizacao`
        )
        SELECT CORR(a.taxa_alvo * 100, i.mun_taxa_alfabetizacao) AS corr
        FROM alvo a JOIN indicador i USING (ano, id_municipio)
    """).result())[0]
    assert r.corr > 0.95, (
        f"correlação de {r.corr:.4f}: o indicador do mesmo ano deveria conter o alvo"
    )
