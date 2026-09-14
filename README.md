# Tech Challenge — Fase 3: Predição e Inteligência Analítica para Alfabetização no Brasil

**POSTECH / FIAP — AI Scientist**

Modelo supervisionado que estima a probabilidade de uma criança do 2º ano do Ensino
Fundamental ser considerada alfabetizada, a partir da camada Gold construída na Fase 2.

| | |
|---|---|
| **Base** | `gold.abt_alfabetizacao` — 3.354.661 alunos, 5.547 municípios, 42.802 escolas |
| **Modelo campeão** | LightGBM (`class_weight="balanced"`), sintonizado por `RandomizedSearchCV` |
| **Resultado no teste** | ROC-AUC **0,6712** · PR-AUC **0,7508** · Recall classe 0 **0,6497** |
| **Conjunto de teste** | 272.621 alunos inéditos |
| **Repositório da Fase 2** | [tech-challenge-fase2-alfabetizacao](https://github.com/LuizFernandoNascimento/tech-challenge-fase2-alfabetizacao) |

---

## 1. Contexto do Problema

O Compromisso Nacional Criança Alfabetizada estabelece que toda criança brasileira deve
estar alfabetizada ao final do 2º ano do Ensino Fundamental até 2030. O INEP mede isso
pelo **Indicador Criança Alfabetizada**: o percentual de estudantes que atingem 743 pontos
na escala de proficiência do Saeb.

Saber a taxa atual, porém, não ajuda a mudá-la. Um gestor público precisa saber **onde** e
**por quê** — quais municípios vão ficar para trás, e quais fatores estão associados a
isso — com antecedência suficiente para agir.

## 2. Objetivo Analítico

Prever, no grão do aluno, se a criança será classificada como alfabetizada, usando apenas
variáveis **disponíveis antes da avaliação**: contexto educacional do município, território,
rede de ensino e indicadores socioeconômicos.

A restrição do "antes da avaliação" é o que diferencia um modelo útil de um exercício
circular, e é o eixo de várias decisões documentadas abaixo.

## 3. Descrição da Base Utilizada

A base é a tabela **`gold.abt_alfabetizacao`**, construída pelo pipeline da Fase 2 como um
mart da camada Gold, ao lado dos comparativos:

```
gold/
├── mart_comparativo_municipio   (24.070)   → dashboards, BI
├── mart_comparativo_uf             (179)   → dashboards, BI
├── mart_comparativo_brasil           (8)   → dashboards, BI
├── mart_frescor_streaming            (2)   → observabilidade
└── abt_alfabetizacao         (3.354.661)   → treinamento de ML   ← consumida aqui
```

Este repositório **não transforma dado de origem**: ele consome a Gold. A construção da ABT
vive no repositório da Fase 2 (`src/gold/transform_gold.py`), e a cada execução o pipeline
valida que a contagem bate exatamente com o público elegível da Silver.

**Grão e cobertura**

| | |
|---|---|
| Grão | Aluno avaliado no 2º ano do EF, por ano |
| Elegibilidade | `presenca = 1` e `preenchimento_caderno = 1` |
| Linhas | 3.354.661 (1.502.809 em 2023 · 1.851.852 em 2024) |
| Alvo | `alfabetizado` (0 = não alfabetizado, 1 = alfabetizado) |
| Distribuição | 59,2% alfabetizados · 40,8% não alfabetizados |
| **Modelável** | **1.817.470** — alunos de 2024, ver seção 4 |

**Origem das variáveis**

| Grupo | Origem | Observação |
|---|---|---|
| Identificação, rede, caderno, série, peso | `silver.dados_alunos` (INEP) | real |
| Território (UF, região, município) | `silver.dim_localidades` (IBGE) | real |
| Indicadores e metas municipais e de UF | `silver.avaliacao_*` e `silver.meta_*` (INEP) | real |
| IDHM, PIB, Gini, abandono, porte | `silver.dim_socioeconomico_municipio` | **sintético** — ver seção 9 |

## 4. Etapas de Modelagem

### 4.1 Análise exploratória

[`notebooks/01_eda_alfabetizacao.ipynb`](notebooks/01_eda_alfabetizacao.ipynb) — sete
dimensões analíticas e cinco hipóteses testadas. As conclusões que mudaram a modelagem:

- **Distribuição do alvo** (59/41) descarta acurácia como métrica principal;
- **Multicolinearidade**: `idhm` é combinação linear das três dimensões que o compõem — foi
  retirado da matriz preditora;
- **Auditoria de vazamento** (Dimensão 7) — detalhada abaixo;
- **H5 (estabilidade por porte de escola)** validada: o desvio-padrão da taxa por escola é
  **26,9** nas escolas com menos de 20 alunos avaliados contra **18,3** nas com mais de 50.
  Consequência prática: ranquear escolas pequenas por taxa produz falsos extremos.

### 4.2 Tratamento de data leakage

Foram encontrados **dois** vazamentos, um evidente e um disfarçado.

**Evidente — `proficiencia`.** É a nota no Saeb, e o aluno é considerado alfabetizado se ela
for ≥ 743. Não é preditor, é o alvo reescrito. A EDA confirma separação perfeita no corte.
Na ABT ela existe apenas como `proficiencia_audit_only` e está na lista de colunas proibidas,
protegida por teste automatizado.

**Disfarçado — `mun_taxa_alfabetizacao`.** Parece contexto do município, mas vem da avaliação
**do mesmo ano do aluno** — e essa taxa é, por definição, a média do alvo naquele município,
com o próprio aluno dentro dela.

> Correlação medida entre a taxa municipal declarada e a média observada do alvo por
> município/ano: **0,993** (9.437 pares). Com o indicador do **ano anterior**, cai para
> **0,687** — que é inércia educacional real, não identidade matemática.

Numa primeira execução, essa variável concentrava **68% de toda a importância SHAP**: o
modelo praticamente só olhava para ela.

Por isso a Gold entrega os indicadores em duas versões, e a modelagem usa exclusivamente as
colunas com sufixo `_ant`:

| Indicador municipal usado | ROC-AUC | PR-AUC |
|---|---|---|
| Do mesmo ano (com vazamento) | 0,6806 | 0,7601 |
| **Do ano anterior (`_ant`)** | **0,6641** | **0,7438** |
| Nenhum indicador municipal | 0,6369 | 0,7169 |

O vazamento valia 0,017 de ROC-AUC. O modelo se sustenta sem ele.

O custo é restringir a modelagem a 2024 (não há 2022 como referência para 2023), e perder
Acre e Distrito Federal, cujos municípios não aparecem na avaliação municipal de 2023.
Restam 1.817.470 alunos, 24 UFs e 5.481 municípios.

### 4.3 Pipeline de pré-processamento

Tudo dentro de um `Pipeline` do scikit-learn, com o `ColumnTransformer` acoplado ao
estimador — é isso que garante que `.fit()` nunca veja validação ou teste:

| Tipo | Imputação | Transformação |
|---|---|---|
| Numéricas | `SimpleImputer(strategy="median")` | `StandardScaler()` |
| Categóricas | `SimpleImputer(strategy="most_frequent")` | `OneHotEncoder(handle_unknown="ignore")` |

### 4.4 Particionamento

70% treino · 15% validação · 15% teste, estratificado pelo alvo.

O treino usa uma amostra estratificada de 300.000 linhas; **validação e teste permanecem
completos** (272.621 cada). A justificativa é medida: 200 mil linhas dão ROC-AUC 0,6708
contra 0,6769 com 3,35 milhões — diferença dentro do ruído, que não paga o tempo de máquina
no Colab gratuito.

## 5. Escolha do Algoritmo

Cinco algoritmos comparados no conjunto de validação completo:

| Modelo | ROC-AUC | PR-AUC | Recall (Classe 0) | F1 (Classe 0) | Brier |
|---|---|---|---|---|---|
| **LightGBM** | **0,6709** | **0,7501** | **0,6499** | **0,5759** | 0,2276 |
| XGBoost | 0,6682 | 0,7475 | 0,3567 | 0,4442 | 0,2201 |
| Random Forest | 0,6669 | 0,7472 | 0,6481 | 0,5737 | 0,2291 |
| Regressão Logística | 0,6605 | 0,7405 | 0,6351 | 0,5650 | 0,2301 |
| Dummy (estratificado) | 0,4993 | 0,5979 | 0,4016 | 0,4012 | 0,4815 |

**LightGBM** foi escolhido. O ganho sobre a Regressão Logística é pequeno (0,010 de ROC-AUC),
o que por si só diz algo sobre o problema: a relação entre contexto municipal e alfabetização
é majoritariamente linear.

O XGBoost merece nota: ROC-AUC quase idêntico, mas **Recall da classe 0 de 0,357** contra
0,650 do LightGBM. Sem `class_weight="balanced"`, ele privilegia a classe majoritária no
corte de 0,5 — exatamente o erro que menos queremos. É um bom exemplo de por que ordenar o
benchmark só por ROC-AUC seria insuficiente.

Sintonia por `RandomizedSearchCV` com `StratifiedKFold(n_splits=5)`, otimizando ROC-AUC.

## 6. Métricas de Avaliação

A métrica norte **não é acurácia**. O custo social de um falso negativo — não sinalizar uma
criança em risco — é muito maior que o do falso positivo.

| Métrica | Por que está aqui |
|---|---|
| **Recall da classe 0** | Fração das crianças em risco que o modelo encontra |
| **PR-AUC** | Robusta ao desbalanceamento; referência é a taxa base, não 0,5 |
| **ROC-AUC** | Capacidade discriminatória geral |
| **Brier Score** | A probabilidade vira priorização de política pública, então precisa ser confiável em si mesma |
| Acurácia | Reportada apenas como contraste |

**Resultado no conjunto de teste** (272.621 alunos inéditos, nunca usados para treinar,
selecionar modelo ou sintonizar):

| Métrica | Valor |
|---|---|
| ROC-AUC | 0,6712 |
| PR-AUC | 0,7508 |
| Recall (Classe 0) | 0,6497 |
| Precisão (Classe 0) | 0,5186 |
| F1 (Classe 0) | 0,5768 |
| F1-Macro | 0,6135 |
| Brier Score | 0,2274 |
| Acurácia | 0,6170 |

Referência: a taxa base é 59,8%. Um modelo que chutasse sempre "alfabetizado" teria
**acurácia 0,598 e Recall da classe 0 igual a zero** — encontraria nenhuma das crianças em
risco. O modelo encontra 65%.

## 7. Interpretação dos Resultados

SHAP (`TreeExplainer`) sobre 8.073 alunos do conjunto de teste.

**Sobre a amostragem**, que não é trivial: os registros são estratificados por alvo × região
× rede, com piso por célula, e sorteados com probabilidade **proporcional a `peso_aluno`**.
O INEP atribui peso amostral a cada aluno; ignorá-lo faria o SHAP retratar o arquivo em vez
da população de crianças. Estratos raros também se perderiam — a rede Privada tem 24 alunos
em 3,35 milhões.

A suficiência da amostra foi **verificada, não presumida**: a correlação de Spearman entre os
rankings de importância de duas submostras disjuntas é **0,982**. Acima de 0,95, o ranking
estabilizou.

| # | Variável | Participação | Origem |
|---|---|---|---|
| 1 | `mun_media_portugues` | 32,2% | real |
| 2 | `mun_taxa_alfabetizacao` (ano anterior) | 14,8% | real |
| 3 | `sigla_uf_RS` | 6,2% | real |
| 4 | `sigla_uf_MG` | 6,2% | real |
| 5 | `mun_gap_meta_2024` | 4,4% | real |
| 6 | `rede_label_Estadual` | 3,4% | real |
| 7 | `escola_qtd_alunos_avaliados` | 3,3% | real |
| 9 | `pib_per_capita` | 2,5% | **sintética** |
| 11 | `idhm_renda` | 2,1% | **sintética** |
| 14 | `idhm_educacao` | 1,6% | **sintética** |

As variáveis sintéticas somam **9,8%** da importância total. Essa fatia da explicação é
artefato do dado gerado e não deve sustentar conclusão.

**A leitura de fundo:** nenhum atributo individual do aluno aparece no topo. O que pesa é o
desempenho histórico da rede em que a criança está matriculada. É um resultado de **inércia
institucional** — a criança tende a repetir o desempenho da sua rede, não o da sua biografia.

## 8. Insights Encontrados

**1. Quais fatores mais impactam a alfabetização?**
O desempenho histórico do município em Língua Portuguesa e sua taxa de alfabetização do ano
anterior respondem por **47% da importância** do modelo. Municípios com bom histórico
sustentam bom desempenho; o inverso também. A política pública eficaz atua sobre a rede, não
sobre a criança isolada.

**2. Quais municípios apresentam maior risco educacional?**
Entre os 1.736 municípios com pelo menos 30 alunos no conjunto de teste, os de maior risco
previsto concentram-se fortemente na **Bahia** — Casa Nova, Serrinha, Araci, Entre Rios,
Formosa do Rio Preto. Ranking completo em
[`reports/ranking_risco_municipios.csv`](reports/ranking_risco_municipios.csv).

**3. Quais regiões possuem padrões semelhantes?**

| Região | Municípios | Probabilidade média | Desvio-padrão |
|---|---|---|---|
| Norte | 185 | 41,4% | 0,104 |
| Nordeste | 572 | 50,2% | **0,224** |
| Sul | 296 | 52,5% | 0,130 |
| Centro-Oeste | 151 | 56,2% | 0,129 |
| Sudeste | 532 | 56,3% | 0,111 |

O dado mais acionável não é a média, é o **desvio-padrão**. O Nordeste tem o dobro da
dispersão interna do Sudeste: tratá-lo como bloco homogêneo desperdiça política pública. O
Norte, com a menor média e baixa dispersão, é o caso onde a ação regional ampla faz sentido.

**4. Como prever municípios que podem não atingir metas futuras?**
Agregando a probabilidade prevista por município, **ponderada por `peso_aluno`**, e cruzando
com o `gap` em relação à meta oficial. Municípios com gap negativo **e** probabilidade média
baixa são os de risco composto.

Ressalva honesta: o modelo projeta a continuidade do padrão atual. Serve para **priorizar**,
não para prever o futuro na presença de mudança de política.

**5. Quais variáveis possuem maior influência nos modelos?**
Ranking completo na seção 7. O achado estrutural é a ausência de variáveis individuais no
topo — o que é, em si, a principal limitação do trabalho (seção 9).

## 9. Limitações do Projeto

**1. Variáveis socioeconômicas sintéticas.** IDHM, PIB per capita, Gini, abandono e porte
vêm de `silver.dim_socioeconomico_municipio`, gerada por sorteio determinístico em torno das
médias regionais do Atlas do Desenvolvimento Humano. São reprodutíveis, mas **não observadas**
— nesta base, Brasília aparece como município de menos de 20 mil habitantes.

Medimos a contribuição delas: **0,000 de ROC-AUC** sobre as variáveis reais. Sozinhas dão
0,6386, contra 0,6318 de "só a região" — ou seja, carregam apenas o sinal da região de onde
foram sorteadas. Substituí-las por Atlas Brasil e IBGE/SIDRA é uma troca de uma tabela só, e
é a melhoria pendente de maior impacto.

**2. O modelo separa municípios, não alunos.** Todas as features são de contexto municipal,
estadual ou de rede. Dois alunos da mesma escola recebem praticamente a mesma previsão. O
ROC-AUC de 0,67 é o teto desse conjunto de variáveis, e não uma deficiência do algoritmo — a
Regressão Logística chega a 0,66. Para uso individual seriam necessários atributos do aluno
ou da turma (frequência, histórico, formação do professor), ausentes na base.

**3. Apenas um ano modelável.** O tratamento de vazamento consome 2023 como referência,
restando 2024. Não há série histórica para modelar tendência, nem como validar o modelo num
ano futuro — isso só será possível com os dados de 2025.

**4. Duas UFs fora.** Acre e Distrito Federal não têm avaliação municipal em 2023 para servir
de referência: 34.382 alunos, 1,9% do total de 2024. Conclusões regionais devem declarar essa
ausência.

**5. Agregação municipal.** As médias por município somam linhas de redes diferentes
(`Municipal`, `Estadual`, `Pública`, `Total`), sendo que `Pública` e `Total` já consolidam as
outras. Isso dá peso extra a municípios com mais linhas de recorte.

## 10. Aplicação Prática para Políticas Públicas

**Priorização de municípios.** O ranking de risco é uma lista ordenada e reprodutível, com
critério auditável. Substitui o rateio por região — que, como mostra a dispersão do Nordeste,
erra o alvo.

**Triagem antecipada.** Com Recall de 0,65 na classe 0, o modelo encontra dois terços das
crianças em risco antes da avaliação. Não substitui o diagnóstico pedagógico; define onde
aplicá-lo primeiro.

**Alocação de recursos.** As probabilidades são calibradas (Brier 0,227), então podem
ordenar repasses e formação continuada, não apenas classificar.

**Cuidado necessário:** o modelo prevê o desempenho da **rede**, não o potencial da criança.
Usá-lo para rotular alunos individualmente seria uma leitura incorreta do que ele mede, com
risco real de profecia autorrealizável.

## 11. Possíveis Evoluções Futuras

1. **Substituir a dimensão socioeconômica por fonte oficial** (Atlas Brasil, IBGE/SIDRA).
   Maior impacto por menor esforço — é uma tabela.
2. **Incorporar o Censo Escolar**: infraestrutura da escola, formação docente, razão
   aluno/turma. São as variáveis que faltam para o modelo distinguir alunos, e não só
   municípios.
3. **Validação temporal** com os dados de 2025: treinar em 2024, testar em 2025.
4. **Modelo hierárquico** (aluno dentro de escola dentro de município), que respeita a
   estrutura aninhada que a EDA evidenciou.
5. **API de escoragem** para integrar o ranking aos sistemas das secretarias.

---

## Estrutura do Repositório

```
tech-challenge-fase3/
├── data/
│   └── processed/                       # ABT em cache local (ignorada pelo git)
├── notebooks/
│   ├── 01_eda_alfabetizacao.ipynb       # EDA: 7 dimensões e 5 hipóteses
│   └── 02_modelagem_supervisionada_e_shap.ipynb
├── src/
│   ├── preprocessing/
│   │   └── pipeline.py                  # carga da Gold, features, split, ColumnTransformer
│   ├── modeling/
│   │   └── train.py                     # benchmark, tuning, serialização
│   ├── evaluation/
│   │   └── explainability.py            # amostragem, SHAP, convergência
│   └── visualization/
│       └── plots.py                     # ROC, PR, calibração, confusão
├── reports/
│   └── ranking_risco_municipios.csv     # saída acionável do modelo
├── images/
│   ├── eda/                             # 9 figuras
│   ├── evaluation/                      # benchmark e curvas
│   └── shap/                            # beeswarm, bar, dependence, waterfall
├── artifacts/
│   └── modelo_alfabetizacao_final.joblib
├── tests/
│   └── test_abt.py                      # 8 testes (4 locais, 4 contra a Gold)
├── requirements.txt
└── README.md
```

## Como Executar

### Google Colab (caminho mais curto)

1. Faça upload de `notebooks/01_eda_alfabetizacao.ipynb` e
   `notebooks/02_modelagem_supervisionada_e_shap.ipynb`;
2. Execute a primeira célula — ela autentica na conta Google e instala as dependências;
3. Execute as demais em sequência.

Tempo estimado no Colab gratuito: ~5 min para a EDA, 15 a 25 min para a modelagem (a
sintonia de hiperparâmetros é a parte demorada). Se a memória apertar, `ANO_FILTRO` e
`N_TREINO_MAX` estão no topo da primeira célula.

### Local

```bash
git clone https://github.com/RenataPraisler/tech-challenge-fase3.git
cd tech-challenge-fase3

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

gcloud auth application-default login
gcloud config set project tech-challenge-alfabetiza-25

pytest -q                          # 8 testes
jupyter lab notebooks/
```

### Custo

Cada execução da modelagem escaneia ~1,1 GB no BigQuery, dentro do 1 TB gratuito mensal.
Para reduzir, troque o `SELECT *` por uma lista explícita de colunas.

## Reprodutibilidade

- Sementes fixas (`random_state=42`) em splits, modelos, busca e amostragem do SHAP;
- Pré-processamento acoplado ao estimador, sem etapa manual entre `fit` e `predict`;
- Conjunto de teste isolado do início ao fim;
- Suficiência da amostra do SHAP verificada por correlação de Spearman entre submostras;
- A ABT é validada contra a Silver a cada execução do pipeline da Fase 2.
