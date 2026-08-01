import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================================
# CONFIG E CONSTANTES
# ============================================================================
_APP_DIR = Path(__file__).resolve().parent
_ROOT = _APP_DIR.parent

st.set_page_config(
    page_title="Passos Mágicos — Risco de Defasagem",
    page_icon="🪄", layout="wide", initial_sidebar_state="expanded",
)

# Paleta de cores
AZUL = "#145089"
LARANJA = "#F58334"
VERMELHO = "#ED3237"
VERDE = "#1E9E5A"
AMARELO = "#FFC845"
CINZA = "#9AA5B1"

CORES_PEDRA = {
    "Quartzo": "#C9A227",
    "Ágata": "#2E9BD6",
    "Ametista": "#8E44AD",
    "Topázio": AMARELO,
}
ORDEM_PEDRA = ["Quartzo", "Ágata", "Ametista", "Topázio"]

CORES_DEFAS = {"Severa": VERMELHO, "Moderada": LARANJA, "Em fase/Avancado": VERDE}
ORDEM_DEFAS = ["Severa", "Moderada", "Em fase/Avancado"]

NOME_MODELO = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "logistic_regression": "Regressão Logística",
}


# ============================================================================
# DADOS E MODELO
# ============================================================================
@st.cache_resource
def carregar_artefatos():
    base = _ROOT / "models"
    modelo = joblib.load(base / "modelo_risco_defasagem.pkl")
    meta = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
    return modelo, meta


@st.cache_data
def carregar_dados():
    caminho = _ROOT / "data" / "processed" / "pede_long.csv"
    df = pd.read_csv(caminho)
    return df


modelo, meta = carregar_artefatos()
df = carregar_dados()
FEATS = meta["modelo"]["features"]


# ============================================================================
# HELPERS DE UI
# ============================================================================
def secao(rotulo, titulo):
    st.markdown(
        f"<div style='border-left:3px solid {LARANJA}; padding:4px 0 4px 18px; margin:28px 0 12px 0;'>"
        f"<p style='color:{CINZA}; font-size:12px; letter-spacing:1.5px; text-transform:uppercase; margin:0 0 4px 0;'>{rotulo}</p>"
        f"<h3 style='color:{AZUL}; margin:0; font-weight:700;'>{titulo}</h3></div>",
        unsafe_allow_html=True,
    )


def tema(fig, legenda=None):
    fig.update_layout(
        font=dict(family="Inter, Arial, sans-serif", size=12, color=CINZA),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=40),
        legend=dict(title=dict(text=legenda or ""), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor="rgba(154,165,177,0.15)", linecolor="rgba(154,165,177,0.35)")
    fig.update_yaxes(gridcolor="rgba(154,165,177,0.15)", linecolor="rgba(154,165,177,0.35)")
    return fig


def card_etapa(n, titulo, desc):
    return (
        f"<div style='display:flex; gap:16px; align-items:flex-start;"
        f" background-color:rgba(20,80,137,0.08); border-left:2px solid {AZUL};"
        f" padding:16px 18px; margin-bottom:14px; border-radius:0 6px 6px 0;'>"
        f"<div style='color:{LARANJA}; font-size:28px; font-weight:700; line-height:1; min-width:36px;'>{n}</div>"
        f"<div style='flex:1;'>"
        f"<p style='color:#E4E9EE; font-size:15px; font-weight:600; margin:0 0 6px 0; line-height:1.3;'>{titulo}</p>"
        f"<p style='color:#D1D9E0; font-size:14.5px; margin:0; line-height:1.7;'>{desc}</p>"
        f"</div></div>"
    )


def pergunta(n, texto, resposta, fig=None):
    with st.expander(f"**Pergunta {n}** — {texto}"):
        st.markdown(resposta)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# PÁGINA 1 — VISÃO GERAL
# ============================================================================
def pagina_visao_geral():
    st.markdown(
        f"<h1 style='color:{AZUL}; margin-bottom:6px; font-weight:700;'>Datathon — Passos Mágicos</h1>"
        f"<p style='color:{CINZA}; font-size:17px; margin-top:0; margin-bottom:24px;'>"
        "Risco de defasagem escolar a partir dos indicadores PEDE (2022-2024)</p>",
        unsafe_allow_html=True,
    )

    secao("Contexto", "O Desafio")
    st.markdown(
        """
        A Associação Passos Mágicos atua há mais de 35 anos na transformação da vida
        de crianças e jovens em vulnerabilidade social, por meio de educação, apoio
        psicológico/psicopedagógico e ampliação de horizontes.

        Este projeto usa os dados da pesquisa **PEDE** (Pesquisa Extensiva do
        Desenvolvimento Educacional) de 2022 a 2024 para responder às 11 perguntas de
        negócio sobre a efetividade do programa e construir um **modelo preditivo
        de risco de defasagem escolar**, apoiando a equipe pedagógica a agir antes
        da queda de desempenho.
        """
    )

    secao("Metodologia", "Abordagem Adotada")
    etapas = [
        ("01", "Consolidação da base",
         "União das 3 abas anuais (PEDE2022/23/24) em um único dataset longitudinal por aluno, com padronização de Fase, Pedra e demais campos."),
        ("02", "Análise exploratória",
         "Resposta às 11 perguntas de negócio do desafio, com estatísticas e gráficos sobre defasagem, desempenho, engajamento e ponto de virada."),
        ("03", "Definição do problema preditivo",
         "Uso dos indicadores do ano corrente para prever a defasagem do ano seguinte, evitando vazamento de dados."),
        ("04", "Validação temporal",
         "Treino nas transições 2022→2023 e teste (holdout) nas transições 2023→2024 — simula o uso real do modelo."),
        ("05", "Modelagem e seleção",
         "Comparação entre Regressão Logística, Random Forest e XGBoost; seleção pelo melhor AUC no holdout."),
    ]
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown("".join(card_etapa(*e) for e in etapas[:3]), unsafe_allow_html=True)
    with col_b:
        st.markdown("".join(card_etapa(*e) for e in etapas[3:]), unsafe_allow_html=True)

    secao("Resultado", "Indicadores do Projeto")
    m = meta["modelo"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alunos-ano analisados", f"{len(df):,}")
    c2.metric("Anos cobertos", df["ano"].nunique())
    c3.metric("AUC do modelo (holdout)", f"{m['test_auc']:.2f}")
    c4.metric("F1-score (holdout)", f"{m['test_f1']:.2f}")

    secao("Base de Dados", "Perfil de Defasagem por Ano")
    tab = df.groupby(["ano", "defasagem_cat"])["ra"].count().unstack()
    tab_pct = (tab.T / tab.sum(axis=1)).T.reset_index().melt(
        id_vars="ano", var_name="Categoria", value_name="pct")
    tab_pct["pct"] = tab_pct["pct"] * 100
    fig = px.bar(
        tab_pct, x="ano", y="pct", color="Categoria", barmode="stack",
        color_discrete_map=CORES_DEFAS, category_orders={"Categoria": ORDEM_DEFAS},
        labels={"ano": "Ano", "pct": "% de alunos"},
    )
    st.plotly_chart(tema(fig, legenda="Defasagem"), use_container_width=True)


# ============================================================================
# PÁGINA 2 — PERGUNTAS DE NEGÓCIO
# ============================================================================
def pagina_perguntas():
    st.markdown(
        f"<h1 style='color:{AZUL}; margin-bottom:6px; font-weight:700;'>Perguntas de Negócio</h1>"
        f"<p style='color:{CINZA}; font-size:15px; margin-top:0; margin-bottom:18px;'>"
        f"As 11 perguntas do desafio, respondidas sobre a base completa (2022-2024)</p>",
        unsafe_allow_html=True,
    )

    # --- Pergunta 1 ---
    tab1 = df.groupby(["ano", "defasagem_cat"])["ra"].count().unstack().reindex(columns=ORDEM_DEFAS, fill_value=0)
    tab1_pct = (tab1.T / tab1.sum(axis=1)).T * 100
    fig1 = px.bar(
        tab1_pct.reset_index().melt(id_vars="ano", var_name="Categoria", value_name="pct"),
        x="ano", y="pct", color="Categoria", barmode="stack",
        color_discrete_map=CORES_DEFAS, category_orders={"Categoria": ORDEM_DEFAS},
        labels={"ano": "Ano", "pct": "% de alunos"},
    )
    pergunta(
        1, "Qual é o perfil geral de defasagem dos alunos (IAN) e como ele evolui ao longo do ano?",
        "A proporção de alunos severamente defasados cai de 22,2% (2022) para 8,0% (2024), e o IAN "
        "médio sobe de 6,42 para 7,68 — o sinal mais forte de que o programa está funcionando na "
        "adequação de nível.",
        tema(fig1, legenda="Defasagem"),
    )

    # --- Pergunta 2 ---
    serie2 = df.groupby(["ano", "fase"])["ida"].mean().reset_index()
    fig2 = px.line(serie2, x="fase", y="ida", color="ano", markers=True,
                   labels={"fase": "Fase", "ida": "IDA médio", "ano": "Ano"})
    pergunta(
        2, "O desempenho acadêmico médio (IDA) está melhorando, estagnado ou caindo ao longo das fases e anos?",
        "O IDA oscila. A Fase 3 (7º/8º ano, IDA médio 5,39) concentra o pior desempenho acadêmico de "
        "toda a jornada — a transição para o Fundamental II é um ponto crítico.",
        tema(fig2, legenda="Ano"),
    )

    # --- Pergunta 3 ---
    pergunta(
        3, "O engajamento (IEG) tem relação direta com desempenho (IDA) e Ponto de Virada (IPV)?",
        "Há uma correlação moderada (IEG-IDA = 0,54; IEG-IPV = 0,56) — mais engajamento, mais chance "
        "de bom desempenho e de atingir o Ponto de Virada, mas sem determinismo devido à correlação.",
    )

    # --- Pergunta 4 ---
    pergunta(
        4, "A autoavaliação (IAA) é coerente com o desempenho (IDA) e o engajamento (IEG) reais?",
        "Correlação quase nula: a percepção do aluno sobre si mesmo está desconectada de seu "
        "desempenho e engajamento reais.",
    )

    # --- Pergunta 5 ---
    pergunta(
        5, "Há padrões psicossociais (IPS) que antecedem quedas de desempenho ou engajamento?",
        "Comparamos o IPS de um ano com a variação de IDA/IEG no ano seguinte. Correlações próximas de zero em todos.",
    )

    # --- Pergunta 6 ---
    pergunta(
        6, "As avaliações psicopedagógicas (IPP) confirmam ou contradizem a defasagem identificada pelo IAN?",
        "Correlação direta fraca (0,12), mas o IPP médio cai de forma consistente conforme a "
        "defasagem piora (7,68 → 7,55 → 7,11) — confirma parcialmente, na direção esperada.",
    )

    # --- Pergunta 7 ---
    corr7 = df[["ipv", "iaa", "ieg", "ips", "ipp", "ida"]].corr()["ipv"].drop("ipv").sort_values()
    fig7 = px.bar(x=corr7.values, y=corr7.index.str.upper(), orientation="h",
                  labels={"x": "Correlação com IPV", "y": ""}, color=corr7.values,
                  color_continuous_scale=[LARANJA, AZUL])
    fig7.update_layout(coloraxis_showscale=False)
    pergunta(
        7, "Quais comportamentos — acadêmicos, emocionais ou de engajamento — mais influenciam o IPV?",
        "O IPP (r=0,61) é o mais associado ao Ponto de Virada, seguido de engajamento (0,56) e "
        "desempenho acadêmico (0,56). Autoavaliação e psicossocial têm relação praticamente nula.",
        tema(fig7),
    )

    # --- Pergunta 8 ---
    feat8 = ["iaa", "ieg", "ips", "ipp", "ida", "ipv", "ian"]
    corr8 = df[["inde"] + feat8].corr()["inde"].drop("inde").sort_values(ascending=False)
    fig8 = px.bar(x=corr8.index.str.upper(), y=corr8.values,
                  labels={"x": "", "y": "Correlação com INDE"}, color=corr8.values,
                  color_continuous_scale=[LARANJA, AZUL])
    fig8.update_layout(coloraxis_showscale=False)
    pergunta(
        8, "Quais combinações de indicadores (IDA + IEG + IPS + IPP) elevam mais a nota global (INDE)?",
        "A correlação simples já aponta IDA (0,79), IEG (0,75) e IPV (0,72) como os indicadores mais "
        "associados ao INDE. A combinação que mais eleva o INDE é justamente IDA + IEG + IPV, pois "
        "pesam o dobro dos demais indicadores na composição da nota. Na prática, investir em "
        "desempenho acadêmico, engajamento e ponto de virada é a alavanca mais eficiente para elevar "
        "o INDE de um aluno.",
        tema(fig8),
    )

    # --- Pergunta 9 (modelo preditivo, ver página dedicada) ---
    pergunta(
        9, "Quais padrões permitem identificar alunos em risco? Construa um modelo preditivo de risco de defasagem.",
        "O XGBoost venceu a comparação entre Regressão Logística, Random Forest e XGBoost (AUC ≈ "
        "0,87, F1 ≈ 0,72) — o ganho do boosting sobre o Random Forest é pequeno, mas consistente em "
        "todas as métricas. O modelo final foi treinado sobre os indicadores do ano corrente para "
        "prever a defasagem no ano seguinte. Veja a aba Sistema Preditivo para simular um aluno.",
    )

    # --- Pergunta 10 ---
    serie10 = df.dropna(subset=["pedra"]).groupby(["ano", "pedra"])["inde"].mean().reset_index()
    fig10 = px.bar(serie10, x="pedra", y="inde", color="ano", barmode="group",
                   category_orders={"pedra": ORDEM_PEDRA},
                   labels={"pedra": "Pedra", "inde": "INDE médio", "ano": "Ano"})
    pergunta(
        10, "Os indicadores mostram melhora consistente por Pedra, confirmando o impacto real do programa?",
        "O INDE por Pedra é estável. O sinal real de efetividade está na Pergunta 1 e na jornada de "
        "Pedra 2023→2024: 24% subiram, 50% mantiveram e 26% desceram — mais subida do que descida, "
        "mas por uma margem pequena.",
        tema(fig10, legenda="Ano"),
    )


# ============================================================================
# PÁGINA 3 — PAINEL ANALÍTICO
# ============================================================================
def pagina_analitico():
    st.markdown(
        f"<h1 style='color:{AZUL}; margin-bottom:6px; font-weight:700;'>Painel Analítico</h1>"
        f"<p style='color:{CINZA}; font-size:15px; margin-top:0; margin-bottom:18px;'>"
        f"Explore os indicadores livremente, filtrando por ano, gênero e Pedra</p>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        f"<h4 style='color:{AZUL}; margin-bottom:10px; font-weight:700;'>Filtros do painel</h4>",
        unsafe_allow_html=True,
    )
    anos_opts = sorted(df["ano"].unique())
    anos_sel = st.sidebar.multiselect("Ano", anos_opts, anos_opts, key="flt_ano")
    gen_opts = sorted(df["genero"].dropna().unique())
    gen_sel = st.sidebar.multiselect("Gênero", gen_opts, gen_opts, key="flt_gen")
    pedra_opts = [p for p in ORDEM_PEDRA if p in df["pedra"].dropna().unique()]
    pedra_sel = st.sidebar.multiselect("Pedra", pedra_opts, pedra_opts, key="flt_pedra")

    df_f = df[
        df["ano"].isin(anos_sel)
        & df["genero"].isin(gen_sel)
        & (df["pedra"].isin(pedra_sel) | df["pedra"].isna())
    ]

    st.markdown(
        f"<div style='background-color:rgba(20,80,137,0.12); padding:12px 18px;"
        f" border-left:3px solid {AZUL}; border-radius:4px; margin-bottom:18px;'>"
        f"<span style='color:{CINZA}; font-size:13px;'>Amostra atual: </span>"
        f"<span style='color:{AZUL}; font-weight:700; font-size:15px;'>{len(df_f):,} registros aluno-ano</span>"
        f"<span style='color:{CINZA}; font-size:13px;'> selecionados pelos filtros aplicados</span></div>",
        unsafe_allow_html=True,
    )
    if df_f.empty:
        st.warning("Nenhum registro corresponde aos filtros. Ajuste a barra lateral.")
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("IAN médio", f"{df_f['ian'].mean():.2f}")
    k2.metric("IDA médio", f"{df_f['ida'].mean():.2f}")
    k3.metric("IEG médio", f"{df_f['ieg'].mean():.2f}")
    k4.metric("% severamente defasados", f"{(df_f['defasagem_cat']=='Severa').mean()*100:.1f}%")
    st.markdown("---")

    tab_f = df_f.groupby(["ano", "defasagem_cat"])["ra"].count().unstack().reindex(columns=ORDEM_DEFAS, fill_value=0)
    tab_f_pct = (tab_f.T / tab_f.sum(axis=1)).T.reset_index().melt(
        id_vars="ano", var_name="Categoria", value_name="pct")
    tab_f_pct["pct"] = tab_f_pct["pct"] * 100
    st.markdown("##### Perfil de defasagem por ano")
    fig_defas = px.bar(
        tab_f_pct, x="ano", y="pct", color="Categoria", barmode="stack",
        color_discrete_map=CORES_DEFAS, category_orders={"Categoria": ORDEM_DEFAS},
        labels={"ano": "Ano", "pct": "% de alunos"},
    )
    st.plotly_chart(tema(fig_defas, legenda="Defasagem"), use_container_width=True)

    st.markdown("##### IDA médio por fase e ano")
    serie = df_f.groupby(["ano", "fase"])["ida"].mean().reset_index()
    fig = px.line(serie, x="fase", y="ida", color="ano", markers=True,
                  labels={"fase": "Fase", "ida": "IDA médio", "ano": "Ano"})
    st.plotly_chart(tema(fig, legenda="Ano"), use_container_width=True)

    st.markdown("##### Correlação dos indicadores com o Ponto de Virada (IPV)")
    corr = df_f[["ipv", "iaa", "ieg", "ips", "ipp", "ida"]].corr()["ipv"].drop("ipv").sort_values()
    if corr.notna().any():
        fig = px.bar(x=corr.values, y=corr.index.str.upper(), orientation="h",
                     labels={"x": "Correlação com IPV", "y": ""}, color=corr.values,
                     color_continuous_scale=[LARANJA, AZUL])
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(tema(fig), use_container_width=True)
    else:
        st.info("Amostra insuficiente para calcular correlações com os filtros atuais.")

    st.markdown("##### INDE médio por Pedra e ano")
    serie = df_f.dropna(subset=["pedra"]).groupby(["ano", "pedra"])["inde"].mean().reset_index()
    fig = px.bar(serie, x="pedra", y="inde", color="ano", barmode="group",
                 category_orders={"pedra": ORDEM_PEDRA},
                 labels={"pedra": "Pedra", "inde": "INDE médio", "ano": "Ano"})
    st.plotly_chart(tema(fig, legenda="Ano"), use_container_width=True)

    st.markdown("##### Distribuição por Pedra (ano mais recente do filtro)")
    ultimo_ano = max(anos_sel) if anos_sel else df_f["ano"].max()
    contagem = df_f[df_f["ano"] == ultimo_ano]["pedra"].value_counts().reindex(ORDEM_PEDRA).reset_index()
    contagem.columns = ["Pedra", "Quantidade"]
    fig = px.bar(contagem, x="Pedra", y="Quantidade", color="Pedra",
                 color_discrete_map=CORES_PEDRA, category_orders={"Pedra": ORDEM_PEDRA}, text="Quantidade")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False)
    st.plotly_chart(tema(fig), use_container_width=True)


# ============================================================================
# PÁGINA 3 — SISTEMA PREDITIVO
# ============================================================================
def pagina_preditivo():
    st.markdown(
        f"<h1 style='color:{AZUL}; margin-bottom:6px; font-weight:700;'>Sistema Preditivo</h1>"
        f"<p style='color:{CINZA}; font-size:15px; margin-top:0; margin-bottom:18px;'>"
        f"Probabilidade de risco de defasagem no próximo ciclo</p>",
        unsafe_allow_html=True,
    )
    m = meta["modelo"]
    st.markdown(
        f"Modelo: **{NOME_MODELO.get(m['nome'], m['nome'])}** · "
        f"AUC (holdout 2023→2024): **{m['test_auc']:.2f}** · "
        f"F1: **{m['test_f1']:.2f}** · "
        f"Precisão: **{m['test_precisao']:.2f}** · "
        f"Recall: **{m['test_recall']:.2f}**"
    )

    with st.form("form_aluno"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("##### Indicadores acadêmicos")
            ida = st.slider("IDA - Desempenho Acadêmico", 0.0, 10.0, 6.0, 0.1)
            ieg = st.slider("IEG - Engajamento", 0.0, 10.0, 6.0, 0.1)
            iaa = st.slider("IAA - Autoavaliação", 0.0, 10.0, 6.0, 0.1)
        with c2:
            st.markdown("##### Indicadores gerais")
            ips = st.slider("IPS - Psicossocial", 0.0, 10.0, 6.0, 0.1)
            ipv = st.slider("IPV - Ponto de Virada", 0.0, 10.0, 6.0, 0.1)
            inde = st.slider("INDE - Índice Geral", 0.0, 10.0, 6.5, 0.1)
            ian = st.slider("IAN - Adequação de Nível", 0.0, 10.0, 7.0, 0.1)
        with c3:
            st.markdown("##### Perfil do aluno")
            fase = st.number_input("Fase atual (0=Alfa a 8=Universitário)", 0, 8, 3)
            idade = st.number_input("Idade", 6, 25, 13)
            defasagem = st.slider("Defasagem atual (Fase efetiva - Fase ideal)", -5, 2, 0)
            genero = st.radio("Gênero", ["Feminino", "Masculino"], horizontal=True)

        submit = st.form_submit_button("Calcular probabilidade de risco", type="primary", use_container_width=True)

    if not submit:
        return

    entrada = pd.DataFrame([{
        "ida": ida, "ieg": ieg, "iaa": iaa, "ips": ips, "ipv": ipv,
        "inde": inde, "ian": ian, "fase": fase, "idade": idade,
        "defasagem": defasagem, "genero": genero,
    }])[FEATS]

    proba = modelo.predict_proba(entrada)[0, 1]

    st.markdown("---")
    if proba >= 0.66:
        cor, rotulo = VERMELHO, "🔴 Risco ALTO"
    elif proba >= 0.33:
        cor, rotulo = LARANJA, "🟡 Risco MODERADO"
    else:
        cor, rotulo = VERDE, "🟢 Risco BAIXO"

    st.markdown(
        f"<div style='background-color:{cor}; padding:25px; border-radius:10px;"
        f" text-align:center; color:white;'>"
        f"<p style='margin:0; font-size:16px; opacity:0.9;'>Probabilidade de ficar defasado(a) no próximo ciclo</p>"
        f"<h1 style='margin:8px 0; color:white;'>{proba*100:.1f}%</h1>"
        f"<p style='margin:0; font-size:18px; font-weight:600;'>{rotulo}</p></div>",
        unsafe_allow_html=True,
    )


# ============================================================================
# SIDEBAR / NAVEGAÇÃO
# ============================================================================
st.sidebar.markdown(
    f"<h2 style='color:{AZUL}; margin-bottom:0; font-weight:700;'>🪄 Passos Mágicos</h2>"
    f"<p style='color:{LARANJA}; margin-top:4px; font-size:13px; font-weight:600;'>Risco de Defasagem Escolar</p>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

PAGINAS = {
    "Visão geral": pagina_visao_geral,
    "Perguntas de negócio": pagina_perguntas,
    "Painel analítico": pagina_analitico,
    "Sistema preditivo": pagina_preditivo,
}

pagina = st.sidebar.radio("Navegação", list(PAGINAS.keys()), key="nav")
st.sidebar.markdown("---")
st.sidebar.caption("Datathon · Fase 05")
st.sidebar.caption("Pós Tech FIAP — Data Analytics")
st.sidebar.caption("**Aluno:** Gustavo Henrique Lisboa do Nascimento")

PAGINAS[pagina]()
