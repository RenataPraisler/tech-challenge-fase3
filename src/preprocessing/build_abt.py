"""Pipeline de construção da Tabela Analítica de Modelagem (ABT - Analytical Base Table).

Integra as fontes da Camada Gold da Fase 2 (microdados de alunos, avaliações municipais e estaduais,
metas oficiais de alfabetização) e fontes socioeconômicas complementares (IBGE e Atlas Brasil).

Garante blindagem contra Data Leakage, eliminando qualquer variável de resultado pós-teste
(como a proficiência contínua do aluno) da matriz de atributos preditores.
"""
from pathlib import Path
import logging
import sys
import pandas as pd
import numpy as np

# Ajuste de path para imports locais
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.preprocessing.ibge_reference import fetch_or_load_ibge_localidades, enrich_with_territory
from src.preprocessing.socioeconomic_data import generate_or_load_socioeconomic_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
EXTERNAL_DIR = ROOT_DIR / "data" / "external"

REDE_MAP = {
    1: "Federal",
    2: "Estadual",
    3: "Municipal",
    4: "Privada",
    5: "Pública",
    6: "Pública",
    0: "Total"
}


def load_raw_datasets():
    """Carrega os datasets brutos sincronizados da Fase 2."""
    logger.info("Carregando datasets da pasta raw: %s", RAW_DIR)

    df_alunos = pd.read_csv(RAW_DIR / "dados_alunos.csv")
    df_aval_mun = pd.read_csv(RAW_DIR / "br_inep_avaliacao_alfabetizacao_municipio.csv")
    df_meta_mun = pd.read_csv(RAW_DIR / "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_municipio.csv")
    df_aval_uf = pd.read_csv(RAW_DIR / "br_inep_avaliacao_alfabetizacao_uf.csv")
    df_meta_uf = pd.read_csv(RAW_DIR / "br_inep_avaliacao_alfabetizacao_meta_alfabetizacao_uf.csv")

    return {
        "alunos": df_alunos,
        "aval_mun": df_aval_mun,
        "meta_mun": df_meta_mun,
        "aval_uf": df_aval_uf,
        "meta_uf": df_meta_uf,
    }


def prepare_alunos_base(df_alunos: pd.DataFrame) -> pd.DataFrame:
    """Filtra e prepara o público elegível para avaliação no grão do aluno."""
    logger.info("Total inicial de alunos na base: %d", len(df_alunos))

    # Filtro de elegibilidade: apenas alunos presentes e que responderam ao caderno
    df = df_alunos[
        (df_alunos["presenca"] == 1) & (df_alunos["preenchimento_caderno"] == 1)
    ].copy()
    logger.info("Alunos elegíveis (presenca=1 e preenchimento_caderno=1): %d", len(df))

    # Padronização de tipos de identificadores
    df["id_aluno"] = df["id_aluno"].astype(str)
    df["id_escola"] = df["id_escola"].astype(str)
    df["id_municipio"] = df["id_municipio"].astype(str).str.split(".").str[0].str.zfill(7)
    df["ano"] = df["ano"].astype(int)

    # Mapeamento do rótulo da rede
    df["rede_label"] = df["rede"].map(REDE_MAP).fillna("Desconhecida")

    # Tratamento da variável resposta supervisionada
    df["alfabetizado"] = df["alfabetizado"].astype(int)

    # Cálculo do porte da escola (alunos avaliados por escola no dataset)
    escola_counts = df.groupby("id_escola")["id_aluno"].transform("count")
    df["escola_qtd_alunos_avaliados"] = escola_counts

    return df


def prepare_municipal_features(df_aval_mun: pd.DataFrame, df_meta_mun: pd.DataFrame) -> pd.DataFrame:
    """Consolida indicadores educacionais e metas no nível do município."""
    df_aval = df_aval_mun.copy()
    df_aval["id_municipio"] = df_aval["id_municipio"].astype(str).str.split(".").str[0].str.zfill(7)
    df_aval["rede_label"] = df_aval["rede"].map(REDE_MAP).fillna("Pública")

    # Média agregada por município para garantir matching completo
    mun_aval_agg = df_aval.groupby("id_municipio").agg(
        mun_taxa_alfabetizacao=("taxa_alfabetizacao", "mean"),
        mun_media_portugues=("media_portugues", "mean"),
    ).reset_index()

    df_meta = df_meta_mun.copy()
    df_meta["id_municipio"] = df_meta["id_meta_id"] = df_meta["id_municipio"].astype(str).str.split(".").str[0].str.zfill(7)
    
    mun_meta_agg = df_meta.groupby("id_municipio").agg(
        mun_meta_2024=("meta_alfabetizacao_2024", "mean"),
        mun_meta_2025=("meta_alfabetizacao_2025", "mean"),
        mun_meta_2030=("meta_alfabetizacao_2030", "mean"),
        mun_nivel_alfabetizacao=("nivel_alfabetizacao", "median"),
        mun_percentual_participacao=("percentual_participacao", "mean")
    ).reset_index()

    mun_features = mun_aval_agg.merge(mun_meta_agg, on="id_municipio", how="outer")
    mun_features["mun_gap_meta_2024"] = mun_features["mun_taxa_alfabetizacao"] - mun_features["mun_meta_2024"]

    return mun_features


def prepare_uf_features(df_aval_uf: pd.DataFrame, df_meta_uf: pd.DataFrame) -> pd.DataFrame:
    """Consolida médias educacionais e metas no nível estadual (UF)."""
    df_aval = df_aval_uf.copy()
    df_aval["sigla_uf"] = df_aval["sigla_uf"].astype(str).str.strip().str.upper()
    uf_aval_agg = df_aval.groupby("sigla_uf").agg(
        uf_taxa_alfabetizacao=("taxa_alfabetizacao", "mean"),
        uf_media_portugues=("media_portugues", "mean")
    ).reset_index()

    df_meta = df_meta_uf.copy()
    df_meta["sigla_uf"] = df_meta["sigla_uf"].astype(str).str.strip().str.upper()
    uf_meta_agg = df_meta.groupby("sigla_uf").agg(
        uf_meta_2024=("meta_alfabetizacao_2024", "mean")
    ).reset_index()

    uf_features = uf_aval_agg.merge(uf_meta_agg, on="sigla_uf", how="outer")
    return uf_features


def build_analytical_base_table() -> pd.DataFrame:
    """Executa a construção fim a fim da ABT."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carga dos dados brutos da Fase 2
    raw = load_raw_datasets()

    # 2. Filtragem e estruturação do grão aluno
    df_alunos = prepare_alunos_base(raw["alunos"])

    # 3. Enriquecimento territorial via IBGE
    loc_cache_path = EXTERNAL_DIR / "ibge_municipios.csv"
    df_localidades = fetch_or_load_ibge_localidades(loc_cache_path)
    df_enriched = enrich_with_territory(df_alunos, df_localidades)

    # 4. Enriquecimento com variáveis socioeconômicas (IDHM, PIB, Gini)
    socio_cache_path = EXTERNAL_DIR / "socioeconomico_municipios.csv"
    # Garante cobertura de todos os municípios presentes no dataset
    unique_muns = df_enriched[["id_municipio", "nome_regiao"]].drop_duplicates()
    df_socio = generate_or_load_socioeconomic_data(socio_cache_path, unique_muns)
    df_enriched = df_enriched.merge(df_socio, on="id_municipio", how="left")

    # 5. Cruzamento com indicadores educacionais do município
    df_mun_feat = prepare_municipal_features(raw["aval_mun"], raw["meta_mun"])
    df_enriched = df_enriched.merge(df_mun_feat, on="id_municipio", how="left")

    # 6. Cruzamento com indicadores da UF
    df_uf_feat = prepare_uf_features(raw["aval_uf"], raw["meta_uf"])
    df_enriched = df_enriched.merge(df_uf_feat, on="sigla_uf", how="left")

    # 7. Engenharia de atributos contextuais adicionais
    df_enriched["razao_mun_uf_taxa"] = (
        df_enriched["mun_taxa_alfabetizacao"] / (df_enriched["uf_taxa_alfabetizacao"] + 1e-5)
    ).round(4)

    # 8. Protocolo Rigoroso de Prevenção a Data Leakage:
    # A coluna `proficiencia` representa a pontuação no teste padronizado
    # e define matematicamente se o aluno é alfabetizado (corte 743).
    # Ela É MANTIDA apenas com sufixo `_leakage_risk_audit_only` para auditoria,
    # e NÃO entrará na seleção de features de treino.
    if "proficiencia" in df_enriched.columns:
        df_enriched.rename(columns={"proficiencia": "proficiencia_audit_only"}, inplace=True)

    # 9. Ordenação das colunas para máxima clareza analítica
    cols_order = [
        # Identificadores e Chaves
        "ano", "id_municipio", "nome_municipio", "sigla_uf", "nome_regiao", "id_escola", "id_aluno",
        # Variáveis Contextuais do Aluno e da Escola
        "caderno", "serie", "rede_label", "peso_aluno", "escola_qtd_alunos_avaliados",
        # Contexto Socioeconômico Municipal (IBGE / Atlas)
        "porte_municipio", "populacao_estimada", "idhm", "idhm_educacao", "idhm_renda",
        "idhm_longevidade", "pib_per_capita", "indice_gini", "taxa_abandono_fundamental",
        # Indicadores Educacionais Municipais (Camada Gold)
        "mun_taxa_alfabetizacao", "mun_media_portugues", "mun_meta_2024", "mun_meta_2025",
        "mun_meta_2030", "mun_nivel_alfabetizacao", "mun_percentual_participacao", "mun_gap_meta_2024",
        # Indicadores Estaduais
        "uf_taxa_alfabetizacao", "uf_media_portugues", "uf_meta_2024", "razao_mun_uf_taxa",
        # Auditoria (Excluído do treino por Data Leakage)
        "proficiencia_audit_only",
        # Target Supervisionado
        "alfabetizado"
    ]
    
    # Mantém apenas as colunas existentes ordenadas
    final_cols = [c for c in cols_order if c in df_enriched.columns]
    abt = df_enriched[final_cols].copy()

    # 10. Persistência dos dados
    csv_path = PROCESSED_DIR / "abt_alfabetizacao.csv"
    parquet_path = PROCESSED_DIR / "abt_alfabetizacao.parquet"

    logger.info("Salvando ABT em formato CSV: %s", csv_path)
    abt.to_csv(csv_path, index=False, encoding="utf-8")

    logger.info("Salvando ABT em formato Parquet: %s", parquet_path)
    abt.to_parquet(parquet_path, index=False)

    logger.info("ABT gerada com sucesso! Linhas: %d | Colunas: %d", len(abt), len(abt.columns))
    logger.info("Distribuição da classe Target (alfabetizado):\n%s", abt["alfabetizado"].value_counts(normalize=True).to_string())

    return abt


if __name__ == "__main__":
    build_analytical_base_table()
