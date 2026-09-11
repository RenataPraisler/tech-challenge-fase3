"""Módulo de enriquecimento territorial e dimensão IBGE.

Obtém e cacheia a tabela de referência de municípios, UFs e regiões
a partir da API pública do IBGE ou de arquivo local existente.
"""
from pathlib import Path
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)

IBGE_MUNICIPIOS_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"

# Colunas padrão da dimensão localidade
LOCALIDADE_COLUMNS = [
    "id_municipio",
    "nome_municipio",
    "sigla_uf",
    "nome_uf",
    "id_uf",
    "sigla_regiao",
    "nome_regiao",
]


def fetch_or_load_ibge_localidades(cache_path: Path) -> pd.DataFrame:
    """Carrega localidades do cache local ou busca da API do IBGE."""
    if cache_path.exists():
        logger.info("Carregando dimensão de localidades do cache: %s", cache_path)
        df_loc = pd.read_csv(cache_path, dtype={"id_municipio": str, "id_uf": str})
        df_loc["id_municipio"] = df_loc["id_municipio"].astype(str).str.zfill(7)
        return df_loc

    logger.info("Buscando dimensão de municípios na API do IBGE...")
    try:
        response = requests.get(IBGE_MUNICIPIOS_URL, timeout=30)
        response.raise_for_status()
        municipios = response.json()

        rows = []
        for m in municipios:
            uf = m["regiao-imediata"]["regiao-intermediaria"]["UF"]
            regiao = uf["regiao"]
            rows.append(
                {
                    "id_municipio": str(m["id"]).zfill(7),
                    "nome_municipio": m["nome"],
                    "sigla_uf": uf["sigla"],
                    "nome_uf": uf["nome"],
                    "id_uf": str(uf["id"]),
                    "sigla_regiao": regiao["sigla"],
                    "nome_regiao": regiao["nome"],
                }
            )

        df_loc = pd.DataFrame(rows)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df_loc.to_csv(cache_path, index=False, encoding="utf-8")
        logger.info("Dimensão salva com sucesso em: %s (%d municípios)", cache_path, len(df_loc))
        return df_loc

    except Exception as e:
        logger.warning("Falha ao consultar API do IBGE: %s. Gerando fallback sintetizado.", e)
        # Fallback estruturado caso não haja conexão externa no ambiente
        return _create_fallback_localidades(cache_path)


def _create_fallback_localidades(cache_path: Path) -> pd.DataFrame:
    """Cria uma base mínima baseada nos IDs de município conhecidos do IBGE."""
    # O prefixo de 2 dígitos do id_municipio identifica a UF no padrão IBGE
    uf_map = {
        "11": ("RO", "Rondônia", "NO", "Norte"),
        "12": ("AC", "Acre", "NO", "Norte"),
        "13": ("AM", "Amazonas", "NO", "Norte"),
        "14": ("RR", "Roraima", "NO", "Norte"),
        "15": ("PA", "Pará", "NO", "Norte"),
        "16": ("AP", "Amapá", "NO", "Norte"),
        "17": ("TO", "Tocantins", "NO", "Norte"),
        "21": ("MA", "Maranhão", "NE", "Nordeste"),
        "22": ("PI", "Piauí", "NE", "Nordeste"),
        "23": ("CE", "Ceará", "NE", "Nordeste"),
        "24": ("RN", "Rio Grande do Norte", "NE", "Nordeste"),
        "25": ("PB", "Paraíba", "NE", "Nordeste"),
        "26": ("PE", "Pernambuco", "NE", "Nordeste"),
        "27": ("AL", "Alagoas", "NE", "Nordeste"),
        "28": ("SE", "Sergipe", "NE", "Nordeste"),
        "29": ("BA", "Bahia", "NE", "Nordeste"),
        "31": ("MG", "Minas Gerais", "SE", "Sudeste"),
        "32": ("ES", "Espírito Santo", "SE", "Sudeste"),
        "33": ("RJ", "Rio de Janeiro", "SE", "Sudeste"),
        "35": ("SP", "São Paulo", "SE", "Sudeste"),
        "41": ("PR", "Paraná", "S", "Sul"),
        "42": ("SC", "Santa Catarina", "S", "Sul"),
        "43": ("RS", "Rio Grande do Sul", "S", "Sul"),
        "50": ("MS", "Mato Grosso do Sul", "CO", "Centro-Oeste"),
        "51": ("MT", "Mato Grosso", "CO", "Centro-Oeste"),
        "52": ("GO", "Goiás", "CO", "Centro-Oeste"),
        "53": ("DF", "Distrito Federal", "CO", "Centro-Oeste"),
    }
    logger.info("Criando tabela territorial baseada nos prefixos estaduais do IBGE")
    records = []
    for cod_uf, (sigla_uf, nome_uf, sigla_reg, nome_reg) in uf_map.items():
        records.append({
            "id_municipio": cod_uf + "00000",
            "nome_municipio": f"Município Referência {sigla_uf}",
            "sigla_uf": sigla_uf,
            "nome_uf": nome_uf,
            "id_uf": cod_uf,
            "sigla_regiao": sigla_reg,
            "nome_regiao": nome_reg,
        })
    df_fb = pd.DataFrame(records)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df_fb.to_csv(cache_path, index=False)
    return df_fb


def enrich_with_territory(df: pd.DataFrame, df_localidades: pd.DataFrame) -> pd.DataFrame:
    """Enriquece um DataFrame que contém id_municipio com dados territoriais."""
    df = df.copy()
    df["id_municipio_str"] = df["id_municipio"].astype(str).str.split(".").str[0].str.zfill(7)
    df["id_uf_calc"] = df["id_municipio_str"].str[:2]

    # Join com localidades
    merged = df.merge(
        df_localidades[["id_municipio", "nome_municipio", "sigla_uf", "nome_uf", "sigla_regiao", "nome_regiao"]],
        left_on="id_municipio_str",
        right_on="id_municipio",
        how="left",
        suffixes=("", "_ibge")
    )

    # Fallback para UFs e Regiões via código IBGE caso algum município não case
    uf_prefixes = {
        "11": ("RO", "Norte"), "12": ("AC", "Norte"), "13": ("AM", "Norte"),
        "14": ("RR", "Norte"), "15": ("PA", "Norte"), "16": ("AP", "Norte"),
        "17": ("TO", "Norte"), "21": ("MA", "Nordeste"), "22": ("PI", "Nordeste"),
        "23": ("CE", "Nordeste"), "24": ("RN", "Nordeste"), "25": ("PB", "Nordeste"),
        "26": ("PE", "Nordeste"), "27": ("AL", "Nordeste"), "28": ("SE", "Nordeste"),
        "29": ("BA", "Nordeste"), "31": ("MG", "Sudeste"), "32": ("ES", "Sudeste"),
        "33": ("RJ", "Sudeste"), "35": ("SP", "Sudeste"), "41": ("PR", "Sul"),
        "42": ("SC", "Sul"), "43": ("RS", "Sul"), "50": ("MS", "Centro-Oeste"),
        "51": ("MT", "Centro-Oeste"), "52": ("GO", "Centro-Oeste"), "53": ("DF", "Centro-Oeste")
    }

    for idx, row in merged.iterrows():
        if pd.isna(row["sigla_uf"]) or not row["sigla_uf"]:
            prefix = row["id_uf_calc"]
            if prefix in uf_prefixes:
                sigla, regiao = uf_prefixes[prefix]
                merged.at[idx, "sigla_uf"] = sigla
                merged.at[idx, "nome_regiao"] = regiao

    merged.drop(columns=["id_municipio_str", "id_uf_calc"], inplace=True)
    if "id_municipio_ibge" in merged.columns:
        merged.drop(columns=["id_municipio_ibge"], inplace=True)

    return merged
