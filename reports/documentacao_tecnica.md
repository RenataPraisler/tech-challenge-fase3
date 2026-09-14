# Documentação Técnica — Fase 3

Complementa o README com as decisões de implementação e o rastro de evidências.
O README responde *o que foi feito*; este documento responde *por que foi feito assim*.

---

## 1. Fronteira entre Fase 2 e Fase 3

A ABT é construída **no repositório da Fase 2**, como um mart da camada Gold
(`src/gold/transform_gold.py`). Este repositório apenas consome
`gold.abt_alfabetizacao`.

A razão é arquitetural. A camada Gold, segundo o enunciado da Fase 2, deve estar preparada
para três usos: dashboards, análises estatísticas e treinamento de modelos de machine
learning. Os quatro marts comparativos atendem os dois primeiros e são agregados, porque
respondem a perguntas sobre município. A ABT atende o terceiro e é no grão do aluno, porque
a pergunta que ela responde — *esta criança será alfabetizada?* — é sobre a criança.

Grão de Gold é grão da decisão, não sinônimo de agregação.

Consequência prática: a regra de elegibilidade, o tratamento de vazamento e os indicadores
defasados vivem na Gold, não na Silver. A Silver permanece como a entidade "aluno" limpa,
disponível para outro time que venha estudar outra pergunta — evasão, por exemplo — sem
herdar nossas escolhas de modelagem.

## 2. Descoberta e correção do vazamento de alvo

### 2.1 Como apareceu

Na primeira execução do pipeline completo, o SHAP atribuiu **68% de toda a importância** a
`mun_taxa_alfabetizacao`. Concentração dessa magnitude numa variável é sinal de atalho, não
de sinal.

### 2.2 Como foi confirmado

```sql
WITH alvo AS (
  SELECT ano, id_municipio, AVG(alfabetizado) * 100 AS taxa_alvo
  FROM `gold.abt_alfabetizacao`
  GROUP BY ano, id_municipio HAVING COUNT(*) >= 30
),
indicador AS (
  SELECT DISTINCT ano, id_municipio, mun_taxa_alfabetizacao
  FROM `gold.abt_alfabetizacao`
)
SELECT CORR(a.taxa_alvo, i.mun_taxa_alfabetizacao)
FROM alvo a JOIN indicador i USING (ano, id_municipio)
```

Resultado: **0,9930** sobre 9.437 pares município-ano.

A variável não descreve o município — ela é a média do alvo naquele município, com o próprio
aluno dentro dela. É *target leakage* por agregação, o tipo que não aparece numa checagem de
correlação com o target linha a linha (que dá apenas 0,31).

### 2.3 Como foi corrigido

A Gold passou a materializar duas famílias de indicadores. A consulta faz o mesmo `JOIN` duas
vezes, uma delas deslocada em um ano:

```sql
LEFT JOIN mun_aval ma  ON ma.id_municipio  = a.id_municipio AND ma.ano  = a.ano
LEFT JOIN mun_aval mal ON mal.id_municipio = a.id_municipio AND mal.ano = a.ano - 1
```

As colunas sem sufixo ficam para descrição; as com sufixo `_ant` são as únicas que entram no
modelo. Com o indicador defasado, a correlação cai para **0,687** — inércia educacional real.

### 2.4 Impacto medido

LightGBM, alunos de 2024, amostra de 400 mil:

| Indicador municipal | ROC-AUC | PR-AUC | Recall classe 0 |
|---|---|---|---|
| Mesmo ano (com vazamento) | 0,6806 | 0,7601 | 0,6412 |
| Ano anterior (`_ant`) | 0,6641 | 0,7438 | 0,6411 |
| Nenhum | 0,6369 | 0,7169 | 0,6895 |

O vazamento inflava 0,017 de ROC-AUC. Corrigi-lo custou pouco em métrica e muito em
defensabilidade.

### 2.5 Como não volta

`tests/test_abt.py::test_indicador_do_mesmo_ano_realmente_vaza` afirma que a correlação do
indicador do mesmo ano é > 0,95. É um teste invertido de propósito: ele documenta o motivo
das colunas `_ant` existirem. Se um dia falhar, a decisão merece ser revisitada.

`test_indicadores_do_modelo_tem_versao_defasada` garante que todo indicador contextual na
lista de features tem par defasado — impede que alguém acrescente uma variável nova sem
versão segura.

## 3. Amostragem para o SHAP

Quatro decisões, cada uma com consequência mensurável.

**Só o conjunto de teste.** Explicar o treino mostraria o que o modelo decorou.

**Estratificação por alvo × região × rede, com piso de 30 por célula.** Sorteio simples
perderia estratos raros: a rede Privada tem 24 alunos em 3,35 milhões; num sorteio de 8 mil,
sairia zero.

**Peso proporcional a `peso_aluno`.** O INEP atribui peso amostral a cada aluno. Amostrando
com probabilidade proporcional ao peso, o SHAP retrata a população de crianças; ignorando,
retrata o arquivo. Como a Fase 2 usa esse peso para reproduzir o indicador oficial com erro
de 0,03 p.p., descartá-lo aqui seria incoerente.

**Convergência verificada.** Correlação de Spearman entre os rankings de duas submostras
disjuntas: **0,982**. Acima de 0,95, aumentar a amostra não muda a conclusão. É a diferença
entre afirmar e demonstrar que 8 mil bastaram.

Sobre o *background*: o `TreeExplainer` no modo padrão (`tree_path_dependent`) usa as
estatísticas de cobertura das próprias árvores e dispensa conjunto de referência. Só o modo
`interventional` precisaria.

## 4. Amostragem do treino

O treino usa 300.000 linhas das 1.272.228 disponíveis. Validação e teste permanecem
completos, com 272.621 cada.

Justificativa medida: ROC-AUC de **0,6708** com 200 mil linhas contra **0,6769** com 3,35
milhões. A diferença de 0,006 está dentro do ruído e não paga o tempo de máquina no Colab
gratuito, onde a sintonia de hiperparâmetros são 50 ajustes do modelo.

A escolha vale para o treino, que é caro. A avaliação continua sendo feita sobre centenas de
milhares de alunos inéditos, que é onde a integridade importa.

## 5. Por que o XGBoost não foi escolhido

ROC-AUC praticamente empatado com o LightGBM (0,6682 contra 0,6709), mas **Recall da classe 0
de 0,357 contra 0,650**.

A causa é o balanceamento: o `LGBMClassifier` recebeu `class_weight="balanced"`, e o
`XGBClassifier` não tem esse parâmetro — precisaria de `scale_pos_weight`. Sem ele, no corte
de 0,5 o modelo privilegia a classe majoritária e deixa de sinalizar dois terços das crianças
em risco.

É um bom lembrete de por que o benchmark não pode ser lido só pela primeira coluna. Dois
modelos com ROC-AUC idêntico podem ter utilidade prática completamente diferente.

## 6. Dimensão socioeconômica sintética

`silver.dim_socioeconomico_municipio` é gerada em SQL, com semente derivada do código IBGE do
município (`FARM_FINGERPRINT` + Box-Muller), calibrada pelas médias regionais do Atlas do
Desenvolvimento Humano.

Ser gerada em SQL, e não em Python local, tem uma razão: reprodutibilidade sem estado. Rodar
a query duas vezes, em máquinas diferentes, produz exatamente os mesmos valores — verificado
por hash MD5 do conteúdo ordenado.

**Ela não deve sustentar conclusão.** Ablação sobre a base completa:

| Conjunto de features | ROC-AUC |
|---|---|
| Tudo (reais + socioeconômicas) | 0,6731 |
| Só as reais (+ UF / região / rede) | **0,6735** |
| Só as socioeconômicas | 0,6386 |
| Só contexto territorial (UF / região / rede) | 0,6318 |

Contribuição líquida: **−0,0004**. E "só sintéticas" (0,6386) mal supera "só a região"
(0,6318) porque foi assim que foram construídas — ruído em torno da média regional.

A tabela carrega o aviso na descrição do BigQuery e cada linha tem
`origem_dado = 'SINTETICO_CALIBRADO_REGIONAL'`. O notebook e o README repetem a ressalva, e o
ranking SHAP marca cada variável pela origem.

## 7. Qualidade de dados

A ABT é validada a cada execução do pipeline da Gold (Fase 2), com cinco checks que comparam
a tabela **com a fonte**, e não apenas consigo mesma:

| Check | O que protege |
|---|---|
| Paridade de linhas com o elegível da Silver | ABT construída sobre base parcial |
| Unicidade de (ano, id_aluno) | duplicação por erro de join |
| Alvo binário sem nulos | linha inútil entrando no treino |
| Cobertura territorial e socioeconômica | join silenciosamente vazio |
| Indicador do ano anterior presente | ABT existe mas não serve para treinar |

O primeiro é o mais importante. Um modelo treina e devolve métrica mesmo com a base errada —
foi exatamente o que aconteceu na primeira versão deste projeto, que rodou sobre 1.488 linhas
sem que nada acusasse.

## 8. Rastro de decisões

| Decisão | Alternativa descartada | Razão |
|---|---|---|
| ABT na Gold da Fase 2 | Construir localmente na Fase 3 | Enunciado exige consumo da Gold; evita divergência entre cópias |
| Grão de aluno | Modelar taxa municipal | Enunciado pede prever se *um aluno* será alfabetizado |
| Indicador defasado | Indicador do mesmo ano | Correlação de 0,993 com o alvo agregado |
| LightGBM | XGBoost | Recall da classe 0: 0,650 contra 0,357 |
| Amostra de treino | Base completa | 0,006 de ROC-AUC não paga horas de CPU |
| SHAP ponderado por `peso_aluno` | Amostra simples | Representar a população, não o arquivo |
| `CREATE OR REPLACE` sem partição | Particionar por ano | Evita `DROP` da tabela existente; ganho irrelevante nesta escala |
