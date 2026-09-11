# Tech Challenge — Fase 3: Predição e Inteligência Analítica para Alfabetização no Brasil

**Pós-Graduação em Inteligência Artificial para Ciência de Dados (POSTECH / FIAP)**  
**Projeto Integrador:** Fase 3  

---

## 1. Contexto do Problema
A alfabetização infantil é um dos pilares mais decisivos para o desenvolvimento socioeconômico e a equidade social no Brasil. Através do **Compromisso Nacional Criança Alfabetizada**, o país estabeleceu a meta de garantir que 100% das crianças estejam alfabetizadas até o término do 2º ano do Ensino Fundamental.  
Para além do monitoramento descritivo implementado na Fase 2 (pipeline de dados e camada Gold), esta Fase 3 desenvolve **inteligência analítica preditiva** para antecipar riscos pedagógicos, apoiar a alocação eficiente de recursos públicos e orientar intervenções tempestivas em redes municipais e estaduais de ensino.

## 2. Objetivo Analítico
Desenvolver um **modelo preditivo supervisionado** capaz de prever com alta sensibilidade e calibração se um aluno do 2º ano do Ensino Fundamental será considerado **alfabetizado** ou **não alfabetizado** (segundo a régua de proficiência oficial do Saeb / Alfabetiza Brasil), correlacionando:
- Indicadores educacionais e metas municipais/estaduais da Camada Gold;
- Características contextuais da rede e porte escolar;
- Variáveis socioeconômicas e demográficas territoriais.

## 3. Descrição da Base Utilizada
- **Fonte Primária (Camada Gold - Fase 2):**
  - Microdados de avaliação de alunos (`dados_alunos`);
  - Indicadores e metas municipais de alfabetização (`br_inep_avaliacao_alfabetizacao_municipio` e metas);
  - Indicadores agregados por Unidade Federativa e Brasil (`avaliacao_uf`, `meta_uf`);
  - Base territorial padronizada via códigos IBGE.
- **Tabela Analítica (ABT - Analytical Base Table):**
  - Grão: Aluno avaliado no ciclo de alfabetização;
  - Target binário: `alfabetizado` (0 = Não Alfabetizado, 1 = Alfabetizado);
  - Blindagem estrita contra *data leakage*: exclusão total da nota contínua de proficiência pós-teste da matriz de atributos preditores.

## 4. Etapas de Modelagem
1. **Engenharia de Recursos & Imputação:** Tratamento de valores ausentes e normalização integrada via Scikit-learn `Pipeline` e `ColumnTransformer`.
2. **Separação Estratificada:** Divisão estrita de treino, validação e teste com estratificação por target para mitigar vazamentos e manter representatividade.
3. **Treinamento Supervisionado:** Benchmark comparativo entre algoritmos lineares e baseados em árvores (Ensembles).
4. **Otimização de Hiperparâmetros:** Validação cruzada com busca sistemática.

## 5. Escolha do Algoritmo
*(Será consolidado na Fase 4 com o benchmark de modelos: Logistic Regression, Random Forest, LightGBM e XGBoost).*

## 6. Métricas de Avaliação
- **ROC-AUC & PR-AUC:** Capacidade discriminatória geral sobre classes desbalanceadas;
- **Recall (Classe 0 - Não Alfabetizados):** Priorização do erro do tipo II (evitar falsos negativos para não deixar alunos vulneráveis sem intervenção);
- **F1-Score e Acurácia Ponderada;**
- **Brier Score / Curva de Calibração:** Confiabilidade das probabilidades para suporte a decisões de risco.

## 7. Interpretação dos Resultados
*(Será consolidado após extração de Feature Importance e SHAP Values).*

## 8. Insights Encontrados
- Resposta detalhada às 5 perguntas estratégicas de negócio:
  1. Fatores determinantes para a alfabetização;
  2. Municípios com vulnerabilidade crítica;
  3. Padrões regionais semelhantes;
  4. Previsão de risco de não cumprimento das metas 2024-2030;
  5. Variáveis de maior impacto na predição.

## 9. Limitações do Projeto
- Dependência da cobertura e tempestividade das avaliações censitárias oficiais;
- Heterogeneidade de preenchimento de variáveis socioeconômicas entre diferentes municípios;
- Necessidade de atualização periódica dos pesos amostrais e calibração temporal.

## 10. Aplicação Prática para Políticas Públicas
- Painel de alerta antecipado (*Early Warning System*) para secretarias municipais e estaduais de educação;
- Priorização de repasses de programas como FUNDEB e formação continuada de professores;
- Matriz de intervenção focalizada em escolas com probabilidade de risco acentuado.

## 11. Possíveis Evoluções Futuras
- Incorporação de dados longitudinais (acompanhamento de coortes de alunos ao longo dos anos);
- Modelagem de grafos espaciais considerando efeitos de transbordamento entre municípios vizinhos;
- Deploy de API REST em nuvem para escoragem em tempo real de novas turmas avaliadas.

---

## Estrutura do Repositório
```text
tech-challenge-fase3/
├── data/
│   ├── raw/           # Dados de entrada da Fase 2
│   ├── processed/     # Tabela analítica de modelagem (ABT)
│   └── external/      # Dados socioeconômicos complementares
├── notebooks/         # Análises exploratórias e notebooks analíticos
├── src/
│   ├── preprocessing/ # Pipelines de tratamento e controle de data leakage
│   ├── modeling/      # Treinamento, validação e tuning
│   ├── evaluation/    # Diagnóstico de métricas e SHAP explainability
│   └── visualization/ # Geração de mapas, gráficos e plots
├── reports/           # Relatórios estratégicos de negócio
├── images/            # Evidências gráficas geradas
├── requirements.txt   # Dependências do projeto
├── README.md          # Documentação principal
└── .gitignore         # Regras de exclusão do Git
```

## Como Executar
*(Instruções detalhadas de setup de ambiente virtual e scripts de execução).*
