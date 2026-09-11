# Guia de Configuração no BigQuery & Práticas FinOps (Custo Zero)

Este guia orienta como disponibilizar a estrutura de dados da **Fase 3** (a Tabela Analítica de Modelagem - ABT) no **Google BigQuery** garantindo **custo zero ($0,00)**, aproveitando os limites do **Google Cloud Free Tier** e aplicando boas práticas de engenharia e FinOps.

---

## 1. Por que isso não gasta dinheiro? (Regras do Free Tier)

O Google Cloud BigQuery oferece cotas gratuitas mensais permanentes (*Always Free Tier*):
1. **Armazenamento:** Primeiros **10 GB/mês** são 100% gratuitos.  
   *Nossa ABT de amostra ocupa ~300 KB, e mesmo a base completa com 3,87 milhões de linhas compactada em Parquet ocupa ~250 MB (menos de 3% da cota gratuita).*
2. **Consultas (Query Analysis):** Primeiro **1 TB/mês** de dados processados é gratuito.
3. **Carga de Dados (Load Jobs):** Ingestão de arquivos (Parquet, CSV, Avro) para criar tabelas no BigQuery tem **custo computacional ZERO**.

---

## 2. Opção 1: Upload Direto da ABT (Recomendada / Custo Computacional Zero)

Como a ABT já foi gerada localmente em formato otimizado (`data/processed/abt_alfabetizacao.parquet`), você pode carregá-la diretamente no BigQuery sem gastar nenhum byte da sua cota de queries.

### Pelo Console Web do Google Cloud:
1. Acesse o [Console do BigQuery](https://console.cloud.google.com/bigquery);
2. Selecione seu projeto (ex.: `tech-challenge-alfabetiza-25`);
3. No menu do Explorer, clique nos três pontinhos ao lado do seu dataset (ou crie um dataset `fase3` / `gold` se não existir);
4. Clique em **Create table** (Criar tabela):
   - **Create table from:** `Upload`
   - **Select file:** Selecione o arquivo local `data/processed/abt_alfabetizacao.parquet` (ou CSV);
   - **File format:** `PARQUET`
   - **Table name:** `abt_alfabetizacao`
   - **Partitioning:** Partition by field `ano` (se aplicável)
   - **Clustering:** Adicione `sigla_uf, id_municipio`
5. Clique em **Create Table**. O schema e os dados serão importados instantaneamente e de forma gratuita.

### Pelo Terminal (Google Cloud SDK / `bq`):
```bash
bq load \
  --source_format=PARQUET \
  --clustering_fields=sigla_uf,id_municipio \
  seu_projeto_id:gold.abt_alfabetizacao \
  data/processed/abt_alfabetizacao.parquet
```

---

## 3. Opção 2: Criando a Tabela via Consulta SQL (CTAS) no BigQuery

Se o seu grupo já possui os datasets da Fase 2 (`silver` e `gold`) carregados no BigQuery, você pode gerar a tabela analítica rodando a query abaixo diretamente no BigQuery.

> **Importante para FinOps:** Note o uso de `CLUSTER BY` e `PARTITION BY`. Isso garante que qualquer consulta futura de exploração ou visualização leia apenas as partições relevantes, consumindo frações minúsculas de megabytes.

```sql
-- DDL da ABT da Fase 3 no BigQuery com FinOps
CREATE OR REPLACE TABLE `SEU_PROJETO.gold.abt_alfabetizacao`
PARTITION BY RANGE_BUCKET(ano, GENERATE_ARRAY(2020, 2040, 1))
CLUSTER BY sigla_uf, id_municipio, alfabetizado AS
SELECT
  -- Chaves e Identificadores
  a.ano,
  a.id_municipio,
  loc.nome_municipio,
  loc.sigla_uf,
  loc.nome_uf,
  loc.nome_regiao,
  a.id_escola,
  a.id_aluno,
  
  -- Atributos do Aluno e da Escola
  a.caderno,
  a.serie,
  a.rede AS rede_label,
  a.peso_aluno,
  COUNT(a.id_aluno) OVER (PARTITION BY a.id_escola) AS escola_qtd_alunos_avaliados,
  
  -- Indicadores Educacionais Municipais (Camada Gold Fase 2)
  m.taxa_alfabetizacao_real AS mun_taxa_alfabetizacao,
  m.media_portugues_real AS mun_media_portugues,
  m.meta_taxa_alfabetizacao AS mun_meta_2024,
  m.desvio_meta_real AS mun_gap_meta_2024,
  m.nivel_alfabetizacao AS mun_nivel_alfabetizacao,
  
  -- Auditoria (Prevenção a Data Leakage: Não utilizar como feature de treino!)
  a.proficiencia AS proficiencia_audit_only,
  
  -- Variável Alvo Supervisionada
  a.alfabetizado

FROM `SEU_PROJETO.silver.dados_alunos` a
INNER JOIN `SEU_PROJETO.silver.dim_localidades` loc
  ON a.id_municipio = loc.id_municipio
LEFT JOIN `SEU_PROJETO.gold.mart_comparativo_municipio` m
  ON a.ano = m.ano AND a.id_municipio = m.id_municipio AND a.rede = m.rede
WHERE
  -- Filtro de elegibilidade: apenas alunos avaliados
  a.presenca = 1 
  AND a.preenchimento_caderno = 1;
```

---

## 4. Travas de Segurança Financeira (Garantia de Não Gastar)

Para evitar que qualquer membro da equipe rode por engano uma consulta descontrolada:

1. **Trava de Bytes Billed (Maximum Bytes Billed):**
   No BigQuery Web UI, antes de rodar qualquer query:
   - Clique em **More** > **Query settings**;
   - Em **Advanced options**, preencha o campo **Maximum bytes billed** com `1073741824` (1 GB).  
   *Se uma query tentar processar mais de 1 GB, o BigQuery cancelará a execução antes de faturar qualquer coisa.*

2. **Evitar `SELECT *`:**
   Sempre selecione apenas as colunas necessárias para treinar os modelos ou plotar gráficos.

3. **Verificar o validador de custos antes de rodar:**
   No canto superior direito do editor de consultas do BigQuery, observe sempre o ícone de validação verde:  
   `"This query will process 12.4 MB when run."`  
   Se o valor for pequeno (em KB ou MB), você está gastando frações ínfimas da sua cota gratuita de 1 TB.
