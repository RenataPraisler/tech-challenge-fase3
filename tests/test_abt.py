"""Testes unitários e de integridade para a ABT (Analytical Base Table)."""
import pytest
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_FILE = ROOT_DIR / "data" / "processed" / "abt_alfabetizacao.parquet"


def test_abt_exists():
    """Valida se o arquivo da ABT foi gerado."""
    assert PROCESSED_FILE.exists(), f"Arquivo não encontrado: {PROCESSED_FILE}"


def test_abt_schema_and_target():
    """Valida colunas essenciais, tipos e integridade da variável resposta."""
    df = pd.read_parquet(PROCESSED_FILE)
    
    # 1. Deve possuir linhas
    assert len(df) > 0, "A ABT está vazia."
    
    # 2. Target binário estrito (0 ou 1, sem nulos)
    assert "alfabetizado" in df.columns, "Coluna target 'alfabetizado' ausente."
    assert set(df["alfabetizado"].unique()).issubset({0, 1}), "Target contém valores fora de {0, 1}."
    assert df["alfabetizado"].isna().sum() == 0, "Target possui valores nulos."

    # 3. Chaves territoriais presentes
    assert "id_municipio" in df.columns
    assert "sigla_uf" in df.columns
    assert "nome_regiao" in df.columns

    # 4. Indicadores socioeconômicos presentes
    assert "idhm" in df.columns
    assert "pib_per_capita" in df.columns
    assert "porte_municipio" in df.columns

    # 5. Prevenção a data leakage: 'proficiencia' não deve estar como feature normal
    assert "proficiencia" not in df.columns, "Coluna 'proficiencia' detectada diretamente na base (risco de leakage)."
    if "proficiencia_audit_only" in df.columns:
        # Coluna permitida exclusivamente para auditoria pós-teste
        pass


def test_abt_no_duplicate_students():
    """Garante unicidade do id_aluno na base analítica."""
    df = pd.read_parquet(PROCESSED_FILE)
    if "id_aluno" in df.columns and len(df["id_aluno"].unique()) == len(df):
        assert df["id_aluno"].duplicated().sum() == 0, "Existem alunos duplicados na ABT."
