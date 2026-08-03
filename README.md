# 🪄 Datathon — Fase 05 | Passos Mágicos

> **FIAP PosTech — Data Analytics**


> Análise de dados e modelo preditivo de risco de defasagem escolar, a partir da pesquisa PEDE (2022-2024) da Associação Passos Mágicos.

---

## 💻 Autor

**Gustavo Henrique Lisboa do Nascimento** — FIAP PosTech Data Analytics

---

## 🔗 Links Rápidos

| Recurso | Link |
|---|---|
| 🚀 **Aplicação Streamlit** | [Acessar App](https://datathon---fase-5-bf6petnhp8kycpxnq3kr-v2.streamlit.app/) |
| 📓 **Notebook renderizado** | [Acessar Notebook](https://github.com/gnascimento9/Datathon---Fase-5/blob/main/notebooks/Datathon_Fase05_PassosMagicos.ipynb) |
| 📓 **Análise Técnica** | [Acessar Análise](https://github.com/gnascimento9/Datathon---Fase-5/blob/main/docs/analise_tecnica.pdf) |

---

## 🎯 O Problema

A Associação Passos Mágicos atua há mais de 35 anos na transformação da vida de crianças e jovens em vulnerabilidade social, por meio de educação, apoio psicológico/psicopedagógico e ampliação de horizontes. O desafio: usar os dados da pesquisa **PEDE** (Pesquisa Extensiva do Desenvolvimento Educacional) de 2022 a 2024 para responder 11 perguntas de negócio sobre a efetividade do programa e **construir um modelo preditivo capaz de identificar alunos em risco de defasagem escolar** antes da queda de desempenho.

---

## 🧪 Estratégia de Modelagem — Três Modelos, Uma Escolha Criteriosa

Um dos achados centrais deste projeto foi a necessidade de **evitar vazamento de dados (data leakage)**: a Defasagem de um aluno já é calculada diretamente da Fase atual, então prever a defasagem do *mesmo ano* usando indicadores do mesmo ano seria quase trapacear. Por isso, o problema foi redefinido para prever a defasagem do **ano seguinte** a partir dos indicadores do ano atual, com **validação temporal** (treino em 2022→2023, teste em 2023→2024).

| Modelo | AUC (holdout) | F1-score | Recall | Observação |
|---|---|---|---|---|
| **Regressão Logística** | 0,808 | 0,625 | 0,539 | Baseline interpretável |
| **Random Forest** | 0,864 | 0,707 | 0,662 | Captura relações não lineares |
| **XGBoost** | **0,870** | **0,723** | **0,679** | **Modelo escolhido** — boosting sequencial de árvores |

**Algoritmo escolhido:** `XGBoost` (`scale_pos_weight` para tratar o desbalanceamento entre alunos em risco e fora de risco), selecionado pelo melhor AUC no holdout temporal — o ganho sobre o Random Forest é pequeno, mas consistente em todas as métricas.

---

## 🛠️ Pipeline de Machine Learning

```
Dados Brutos
        │
        ▼
[1] Consolidação ──► Padronização de Fase/Pedra, união em dataset longitudinal
        │
        ▼
[2] Análise Exploratória ──► Resposta às 11 perguntas de negócio do desafio
        │
        ▼
[3] Definição do problema ──► Prever defasagem do ano seguinte (evita vazamento)
        │
        ▼
[4] Feature Engineering ──► Imputação, padronização, one-hot encoding
        │
        ▼
[5] Split Temporal ──► Treino (2022→2023) / Teste (2023→2024)
        │
        ▼
[6] Treinamento ──► Regressão Logística vs. Random Forest vs. XGBoost
        │
        ▼
[7] Avaliação ──► AUC, F1, Precisão, Recall, Matriz de Confusão
        │
        ▼
[8] Serialização ──► .pkl (joblib) + metadata.json
        │
        ▼
[9] Deploy ──► Streamlit App
```

---

## 📊 Principais Insights

1. **Defasagem severa caiu de 22,2% (2022) para 8,0% (2024)** — o sinal mais forte de que o programa está funcionando na adequação de nível.
2. A **Fase 3** (7º/8º ano) concentra o pior desempenho acadêmico (IDA) de toda a jornada — um ponto crítico na transição para o Fundamental II.
3. O indicador **IPP (psicopedagógico)** é o mais associado ao Ponto de Virada (IPV), mais até que o desempenho acadêmico ou o engajamento.
4. A **autoavaliação (IAA)** tem correlação quase nula com desempenho e engajamento reais — a percepção do aluno sobre si mesmo está desconectada da sua realidade escolar.
5. Na jornada de Pedra entre 2023 e 2024, **24% dos alunos subiram, 50% mantiveram e 26% desceram** de classificação — mais subida do que descida, mas por uma margem pequena.
6. A **idade correlaciona negativamente com o engajamento (IEG, r=-0,52)** mais fortemente do que qualquer outro par de indicadores acadêmicos — o fator idade/fase pesa mais que o tempo de permanência na associação.
7. Alunos de **escola privada** (majoritariamente via bolsa/apadrinhamento) têm INDE e IAN mais altos, mas **engajamento (IEG) menor**, que os de escola pública.

O notebook também traz uma seção de **análise exploratória complementar** — perfil demográfico, matriz de correlação completa entre todos os indicadores/notas/idade/tempo de associação, e o efeito de idade vs. tempo de permanência — além das 11 perguntas do enunciado.

---

## 📁 Estrutura do Repositório

```
Datathon-Fase-5-PassosMagicos
├── data/
│   ├── raw/PEDE_raw.xlsx
│   └── processed/pede_long.csv
├── notebooks/
│   └── Datathon_Fase05_PassosMagicos.ipynb
├── models/
│   ├── modelo_risco_defasagem.pkl
│   └── metadata.json
├── app/
│   └── app.py
└── docs/
```

---

## 🚀 Como Executar

### Opção 1 — Google Colab

1. Abra `notebooks/Datathon_Fase05_PassosMagicos.ipynb` no [Google Colab](https://colab.research.google.com).
2. Rode as células em ordem.
3. Ao final, baixe `modelo_risco_defasagem.pkl` e `metadata.json` e coloque-os em `models/`.

### Opção 2 — Localmente

```bash
# 1. Clone o repositório
git clone <url-do-repositorio>

# 2. Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate    # Windows

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Rode o notebook 
jupyter notebook notebooks/Datathon_Fase05_PassosMagicos.ipynb

# 5. Mova os dois artefatos gerados para models/
mv notebooks/modelo_risco_defasagem.pkl notebooks/metadata.json models/

# 6. Rode o app Streamlit
streamlit run app/app.py
```

Acesse `http://localhost:8501` no navegador.

---

## 📄 Licença

Este projeto está sob a licença MIT.

---

<p align="center">
  <em>Desenvolvido como Datathon da Fase 05 — FIAP PosTech Data Analytics</em>
</p>
