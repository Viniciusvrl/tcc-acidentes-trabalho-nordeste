
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import modelo_random_forest as rf
import modelos_comparativos as mc

AQUI = Path(__file__).resolve().parent
SEED = 42
TIPOS = ["Típico", "Trajeto", "Doença"]
FAIXAS = ["14-24", "25-34", "35-44", "45-54", "55-64", "65+"]
VARS_AUSENCIA = ["agente_causador", "cbo", "cid", "cnae", "sexo", "idade", "natureza_lesao",
                 "parte_corpo", "emitente_cat"]
BLOCOS = {
    "perfil": ["sexo", "idade", "cbo_grande_grupo", "cnae_divisao", "uf_empregador", "mes", "dia_semana"],
    "evento": ["agente_causador", "natureza_lesao", "parte_corpo"],
    "registro": ["cid_capitulo", "emitente_cat"],
}
NUMERICAS = ["idade", "mes", "dia_semana"]
AZUL, CINZA, LARANJA, VERDE = "#2a78d6", "#9a9892", "#d9822b", "#3b8f5a"


def gravar(df: pd.DataFrame, pasta: Path, nome: str, index: bool = True) -> None:
    df.to_csv(pasta / nome, sep=";", decimal=",", encoding="utf-8-sig", index=index)


def wilson(k, n, z: float = 1.959964):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    meia = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centro - meia, centro + meia


def cramers_v(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    """V de Cramér e versão com correção de viés (Bergsma, 2013)."""
    t = pd.crosstab(a, b).to_numpy()
    n = t.sum()
    chi2 = stats.chi2_contingency(t, correction=False)[0]
    r, k = t.shape
    phi2 = chi2 / n
    v = np.sqrt(phi2 / min(r - 1, k - 1))
    phi2c = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    vc = np.sqrt(phi2c / min(rc - 1, kc - 1))
    return v, vc, max(r, k) if r != len(TIPOS) else k


def logistica(X: np.ndarray, y: np.ndarray, n_iter: int = 60, l2: float = 1e-6):
    """Regressão logística binária por Newton (IRLS); devolve coeficientes e erro padrão."""
    X = np.column_stack([np.ones(len(X)), X]).astype(float)
    b = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        w = p * (1 - p)
        H = X.T @ (X * w[:, None]) + l2 * np.eye(X.shape[1])
        passo = np.linalg.solve(H, X.T @ (y - p) - l2 * b)
        b += passo
        if np.abs(passo).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(X.shape[1]))
    return b, np.sqrt(np.diag(cov))


def razoes_de_chance(df: pd.DataFrame, alvo: np.ndarray, categoricas: dict[str, str | None],
                     min_n: int = 200) -> pd.DataFrame:
    """Razões de chance ajustadas. categoricas = {coluna: categoria de referência (None = a mais frequente)}."""
    partes, nomes = [], []
    for c, ref in categoricas.items():
        s = df[c].astype(str)
        cont = s.value_counts()
        s = s.where(s.map(cont) >= min_n, "(outras)")
        ref = ref if ref is not None else s.value_counts().index[0]
        d = pd.get_dummies(s, dtype=float).drop(columns=[ref])
        partes.append(d.to_numpy())
        nomes += [f"{c}={v} (ref. {ref})" for v in d.columns]
    X = np.column_stack(partes)
    b, ep = logistica(X, alvo.astype(float))
    b, ep = b[1:], ep[1:]
    return pd.DataFrame({"termo": nomes, "razao_de_chances": np.exp(b), "ic95_inf": np.exp(b - 1.96 * ep),
                         "ic95_sup": np.exp(b + 1.96 * ep)})


# =============================================================================== PARTE 1
def parte_descritiva(base: pd.DataFrame, rem: pd.DataFrame | None, pasta_saida: Path,
                     pasta: Path, log) -> None:
    b = base
    n = len(b)
    b["doenca"] = (b["tipo_acidente"] == "Doença").astype(int)

    # ---- 1. ausência por layout e por variável; base antes e depois da exclusão
    if rem is not None:
        antes = pd.concat([b[rem.columns.intersection(b.columns)], rem], ignore_index=True)
        log(f"Base antes da exclusão por ausência: {len(antes):,} | removidos {len(rem):,} | final {n:,}")
        t = antes.groupby("layout_origem").agg(
            registros=("ano", "size"), inicio=("arquivo", "min"), fim=("arquivo", "max"))
        for v in VARS_AUSENCIA:
            t[f"ausente_{v}_%"] = antes.groupby("layout_origem")[v].apply(lambda s: s.isna().mean() * 100)
        t["removidos"] = rem.groupby("layout_origem").size()
        t["removidos_%"] = t["removidos"] / t["registros"] * 100
        gravar(t.round(2), pasta, "ausencia_por_layout.csv")
        log("\nAusência por layout (% dos registros do Nordeste antes da exclusão):\n" + t.round(1).to_string())
        por_arq = antes.groupby(["layout_origem", "arquivo"]).agg(registros=("ano", "size"))
        por_arq["removidos"] = rem.groupby(["layout_origem", "arquivo"]).size()
        por_arq["removidos_%"] = (por_arq["removidos"].fillna(0) / por_arq["registros"] * 100).round(1)
        gravar(por_arq, pasta, "ausencia_por_arquivo.csv")
        log("\nArquivos com mais de 15% de remoção:\n" + por_arq[por_arq["removidos_%"] > 15].to_string())
        t2 = antes.groupby("ano")[VARS_AUSENCIA].apply(lambda g: g.isna().mean() * 100).round(1)
        gravar(t2, pasta, "ausencia_por_ano_variavel.csv")
        log("\nAusência por ano e variável (%):\n" + t2.to_string())

        linhas = []
        for nome, col, ordem in [("Tipo de acidente", "tipo_acidente", TIPOS), ("Sexo", "sexo", None),
                                 ("Faixa etária", "faixa_etaria", FAIXAS), ("Ano", "ano", None)]:
            a = antes[col].value_counts(normalize=True, dropna=True) * 100
            d = b[col].value_counts(normalize=True) * 100
            for k in (ordem or sorted(d.index)):
                linhas.append({"variavel": nome, "categoria": k, "antes_%": a.get(k, np.nan),
                               "depois_%": d.get(k, np.nan), "diferenca_pp": d.get(k, np.nan) - a.get(k, np.nan)})
        linhas.append({"variavel": "Óbito", "categoria": "sim", "antes_%": antes["indica_obito"].mean() * 100,
                       "depois_%": b["indica_obito"].mean() * 100,
                       "diferenca_pp": (b["indica_obito"].mean() - antes["indica_obito"].mean()) * 100})
        t3 = pd.DataFrame(linhas)
        gravar(t3.round(2), pasta, "antes_depois_exclusao.csv", index=False)
        log("\nComposição antes e depois da exclusão por ausência:\n" + t3.round(2).to_string(index=False))
        rr = rem.assign(doenca=(rem["tipo_acidente"] == "Doença"))
        log(f"\nRemovidos: doença {rr['doenca'].mean()*100:.1f}% (base final {b['doenca'].mean()*100:.1f}%) | "
            f"óbito {rem['indica_obito'].mean()*100:.2f}%")
        log("Entre os removidos, % de doença por variável ausente: "
            + str({v: round(rr.loc[rr[v].isna(), 'doenca'].mean() * 100, 1) for v in VARS_AUSENCIA}))
        log("Variável ausente entre as doenças removidas (%): "
            + str({v: round(rr.loc[rr['doenca'], v].isna().mean() * 100, 1) for v in VARS_AUSENCIA}))

    # ---- 2. idade
    log(f"\nIdade: mínimo {b['idade'].min()}, máximo {b['idade'].max()}, mediana {b['idade'].median()} | "
        f"menores de 16: {(b['idade'] < 16).sum()} | acima de 80: {(b['idade'] > 80).sum()} | "
        f"negativas: {(b['idade'] < 0).sum()}")

    # ---- 3. rótulos cortados em 20 caracteres
    dic = pasta_saida / "dicionario_rotulos.csv"
    if dic.exists():
        d = pd.read_csv(dic, sep=None, engine="python", encoding="utf-8-sig")
        for c, g in d.groupby("coluna"):
            amb = set(g.loc[g["ambiguo"] == 1, "rotulo_20"])
            log(f"Rótulos de {c}: {b[c].nunique()} categorias na base | {len(amb)} rótulos de 20 caracteres "
                f"reúnem descrições diferentes na fonte | {b[c].isin(amb).mean()*100:.1f}% dos registros da base")
    for c in ["agente_causador", "natureza_lesao", "parte_corpo", "cid_capitulo", "cnae_divisao",
              "cbo_grande_grupo", "emitente_cat", "uf_empregador", "cid", "cbo", "cnae"]:
        log(f"  categorias distintas em {c}: {b[c].nunique()}")

    # ---- 4. composição por ano: todos os meses x meses regulares; série mensal
    linhas = []
    for nome, g in [("todos", b), ("regulares", b[b["mes_estavel"] == 1])]:
        t = (pd.crosstab(g["ano"], g["tipo_acidente"], normalize="index") * 100)[TIPOS]
        t["registros"] = g.groupby("ano").size()
        t["obito_%"] = g.groupby("ano")["indica_obito"].mean() * 100
        t["meses"] = g.groupby("ano")["ym"].nunique()
        t.insert(0, "recorte", nome)
        linhas.append(t)
        tot = (g["tipo_acidente"].value_counts(normalize=True) * 100)[TIPOS].round(2).to_dict()
        log(f"\nComposição ({nome}, {len(g):,} registros, {g['ym'].nunique()} meses): {tot} | "
            f"óbito {g['indica_obito'].mean()*100:.2f}%")
    t = pd.concat(linhas)
    gravar(t.round(2), pasta, "composicao_por_ano.csv")
    log(t.round(1).to_string())
    for nome, g in [("todos", b), ("regulares", b[b["mes_estavel"] == 1])]:
        log(f"  [{nome}] masculino {(g['sexo']=='Masculino').mean()*100:.1f}% | "
            + str((g["faixa_etaria"].value_counts(normalize=True) * 100).reindex(FAIXAS).round(1).to_dict())
            + " | UF " + str((g["uf_empregador"].value_counts(normalize=True) * 100).round(1).to_dict()))
    jan = b.groupby("janela").agg(registros=("ano", "size"), meses=("ym", "nunique"),
                                  inicio=("ym", "min"), fim=("ym", "max"))
    jan["media_mensal"] = (jan["registros"] / jan["meses"]).round(0)
    jan = jan.join(pd.crosstab(b["janela"], b["layout_origem"]))
    gravar(jan, pasta, "janelas.csv")
    log("\nJanelas de cobertura (0 = incompleta):\n" + jan.to_string())

    mensal = b[b["ano"] >= 2020].groupby("ym").agg(registros=("ano", "size"), estavel=("mes_estavel", "first"))
    idx = pd.period_range("2020-01", "2025-11", freq="M").astype(str)
    mensal = mensal.reindex(idx).fillna({"registros": 0, "estavel": 0})
    gravar(mensal, pasta, "serie_mensal.csv")

    anual = (pd.crosstab(b["ano"], b["tipo_acidente"], normalize="index") * 100)[TIPOS].loc[2020:]
    anual_reg = (pd.crosstab(b.loc[b["mes_estavel"] == 1, "ano"], b.loc[b["mes_estavel"] == 1, "tipo_acidente"],
                             normalize="index") * 100)[TIPOS].loc[2020:]
    excl = None
    if rem is not None:
        excl = (rem.groupby("ano").size() / antes.groupby("ano").size() * 100).loc[2020:]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 3.1), dpi=200, gridspec_kw={"width_ratios": [1.9, 1]})
    x = np.arange(len(mensal))
    cores = [AZUL if e == 1 else CINZA for e in mensal["estavel"]]
    a1.bar(x, mensal["registros"], color=cores, width=0.85)
    a1.set_xticks([i for i, m in enumerate(idx) if m.endswith("-01")], [m[:4] for m in idx if m.endswith("-01")])
    a1.set_ylabel("Registros por mês do acidente", fontsize=8.5)
    a1.set_title("(a) Série mensal", fontsize=9, loc="left")
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(color=AZUL, label="cobertura regular (47 meses)"),
                       Patch(color=CINZA, label="cobertura incompleta")], fontsize=7.5, frameon=False,
              loc="upper left")
    anos = anual.index.astype(int)
    a2.plot(anos, anual["Trajeto"], "-o", color=AZUL, ms=3.5, label="Trajeto (%)")
    a2.plot(anos, anual["Doença"], "-s", color=LARANJA, ms=3.5, label="Doença (%)")
    a2.plot(anual_reg.index, anual_reg["Trajeto"], ":", color=AZUL, lw=1)
    a2.plot(anual_reg.index, anual_reg["Doença"], ":", color=LARANJA, lw=1)
    if excl is not None:
        a2.plot(excl.index, excl.values, "--^", color=CINZA, ms=3.5, label="Excluídos por ausência (%)")
    a2.set_title("(b) Composição e exclusão por ano", fontsize=9, loc="left")
    a2.set_ylabel("% dos registros do ano", fontsize=8.5)
    a2.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.62))
    for a in (a1, a2):
        a.tick_params(labelsize=8)
        for lado in ("top", "right"):
            a.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(pasta / "figura_serie_mensal_anual.png")
    plt.close(fig)

    # ---- 5. doenças e atraso na emissão da CAT
    e = b[b["dias_ate_emissao"].notna()].copy()
    log(f"\nRegistros com data de emissão: {len(e):,} ({len(e)/n*100:.1f}%)")
    q = e.groupby("tipo_acidente")["dias_ate_emissao"].describe(percentiles=[0.5, 0.75, 0.9, 0.95])[
        ["count", "50%", "75%", "90%", "95%", "max"]].reindex(TIPOS)
    q[">30 dias_%"] = e.groupby("tipo_acidente")["dias_ate_emissao"].apply(lambda s: (s > 30).mean() * 100)
    q[">180 dias_%"] = e.groupby("tipo_acidente")["dias_ate_emissao"].apply(lambda s: (s > 180).mean() * 100)
    gravar(q.round(2), pasta, "atraso_emissao_por_tipo.csv")
    log("Dias entre o acidente e a emissão da CAT:\n" + q.round(1).to_string())
    e["emissao_ano"] = pd.to_datetime(e["data_emissao_cat"], errors="coerce").dt.year
    log(f"Emissão em ano diferente do acidente: {(e['emissao_ano'] != e['ano']).sum()} de {len(e):,}")
    t = pd.DataFrame({
        "doenca_%_todas": b[b["ano"] >= 2020].groupby("ano")["doenca"].mean() * 100,
        "doenca_%_com_data_emissao": e[e["ano"] >= 2020].groupby("ano")["doenca"].mean() * 100,
        "doenca_%_emitidas_ate_30_dias": e[(e["ano"] >= 2020) & (e["dias_ate_emissao"] <= 30)]
        .groupby("ano")["doenca"].mean() * 100,
        "n_ate_30_dias": e[(e["ano"] >= 2020) & (e["dias_ate_emissao"] <= 30)].groupby("ano").size(),
    })
    gravar(t.round(2), pasta, "doencas_por_ano_prazo_emissao.csv")
    log("Proporção de doenças por ano, com o mesmo prazo de emissão:\n" + t.round(2).to_string())
    t = pd.crosstab(b["mes"], b["ano"], values=b["doenca"], aggfunc="mean").mul(100).round(1)
    gravar(t, pasta, "doencas_por_mes_do_ano.csv")
    sem = b[b["mes_estavel"] == 1].assign(semestre=lambda d: np.where(d["mes"] <= 6, "jan-jun", "jul-dez"))
    log("Doenças (%) por semestre do acidente, meses regulares:\n"
        + pd.crosstab(sem["ano"], sem["semestre"], values=sem["doenca"], aggfunc="mean").mul(100).round(2).to_string())

    inf_ = b["agente_causador"].str.startswith("Agente Infeccioso")
    s2 = b[(b["mes_estavel"] == 1) & ~inf_].assign(semestre=lambda d: np.where(d["mes"] <= 6, "jan-jun", "jul-dez"))
    log("Doenças (%) por semestre, meses regulares, sem registros de agente infeccioso:\n"
        + pd.crosstab(s2["ano"], s2["semestre"], values=s2["doenca"], aggfunc="mean").mul(100).round(2).to_string())
    e2 = e[(e["ano"] >= 2020) & (e["dias_ate_emissao"] <= 30) & ~e["agente_causador"].str.startswith("Agente Infeccioso")]
    log("Doenças (%) por ano, CAT emitida em até 30 dias e sem agente infeccioso: "
        + str((e2.groupby("ano")["doenca"].mean() * 100).round(2).to_dict()))
    log("Tipo (%) por ano sem registros de agente infeccioso:\n"
        + (pd.crosstab(b.loc[~inf_, "ano"], b.loc[~inf_, "tipo_acidente"], normalize="index") * 100)[TIPOS].round(1).to_string())

    # ---- 6. COVID-19 nas doenças
    do = b[b["doenca"] == 1]
    infecc = do["agente_causador"].str.startswith("Agente Infeccioso")
    covid_cid = do["cid"].astype(str).str[:3].isin(["U07", "B34", "B97"])
    t = pd.DataFrame({"doencas": do.groupby("ano").size(),
                      "agente_infeccioso_%": infecc.groupby(do["ano"]).mean() * 100,
                      "cid_U07_B34_B97_%": covid_cid.groupby(do["ano"]).mean() * 100})
    t["doenca_%_do_ano"] = b.groupby("ano")["doenca"].mean() * 100
    sem_inf = b[~(b["doenca"].eq(1) & b["agente_causador"].str.startswith("Agente Infeccioso"))]
    t["doenca_%_sem_agente_infeccioso"] = sem_inf.groupby("ano")["doenca"].mean() * 100
    gravar(t.round(2), pasta, "doencas_covid_por_ano.csv")
    log(f"\nDoenças com agente infeccioso: {infecc.mean()*100:.1f}% | com CID U07/B34/B97: {covid_cid.mean()*100:.1f}%")
    log(t.round(2).to_string())
    log("CID mais frequentes entre as doenças com agente infeccioso: "
        + str(do.loc[infecc, "cid"].astype(str).str[:3].value_counts().head(6).to_dict()))

    # ---- 7. óbito com intervalo de Wilson
    linhas = []
    for nome, col in [("Faixa etária", "faixa_etaria"), ("Estado", "uf_empregador"), ("Sexo", "sexo"),
                      ("Ocupação (grande grupo CBO)", "cbo_grande_grupo"), ("Tipo", "tipo_acidente"), ("Ano", "ano")]:
        g = b.groupby(col)["indica_obito"].agg(["sum", "size"])
        lo, hi = wilson(g["sum"], g["size"])
        for k in g.index:
            linhas.append({"variavel": nome, "categoria": k, "registros": g.loc[k, "size"], "obitos": g.loc[k, "sum"],
                           "taxa_%": g.loc[k, "sum"] / g.loc[k, "size"] * 100, "ic95_inf_%": lo[g.index.get_loc(k)] * 100,
                           "ic95_sup_%": hi[g.index.get_loc(k)] * 100})
    t = pd.DataFrame(linhas)
    gravar(t.round(3), pasta, "obito_wilson.csv", index=False)
    log("\nTaxa de óbito com intervalo de confiança de 95% (Wilson):\n" + t.round(2).to_string(index=False))
    cn = b.groupby("cnae_divisao")["indica_obito"].agg(["sum", "size"])
    cn = cn[cn["size"] > 5000]
    cn["taxa_%"] = cn["sum"] / cn["size"] * 100
    cn["inf"], cn["sup"] = [v * 100 for v in wilson(cn["sum"], cn["size"])]
    log("Óbito por divisão da CNAE (mais de 5.000 registros), maiores taxas:\n"
        + cn.sort_values("taxa_%", ascending=False).head(4).round(2).to_string())

    oc = razoes_de_chance(b, b["indica_obito"].to_numpy(),
                          {"uf_empregador": "BA", "sexo": "Feminino", "faixa_etaria": "25-34",
                           "cbo_grande_grupo": None, "tipo_acidente": "Típico"})
    gravar(oc.round(3), pasta, "obito_razoes_de_chance_ajustadas.csv", index=False)
    log("\nÓbito: razões de chance ajustadas (estado, sexo, faixa etária, ocupação e tipo):\n"
        + oc.round(2).to_string(index=False))

    # ---- 8. sexo e doença: estratificação e ajuste
    bruta = razoes_de_chance(b, b["doenca"].to_numpy(), {"sexo": "Masculino"})
    aj = razoes_de_chance(b, b["doenca"].to_numpy(),
                          {"sexo": "Masculino", "faixa_etaria": "25-34", "cnae_divisao": None, "cbo_grande_grupo": None})
    aj2 = razoes_de_chance(b, b["doenca"].to_numpy(),
                           {"sexo": "Masculino", "faixa_etaria": "25-34", "cnae_divisao": None,
                            "cbo_grande_grupo": None, "emitente_cat": "Empregador"})
    t = pd.concat([bruta.assign(modelo="bruta"), aj[aj["termo"].str.startswith("sexo")].assign(
        modelo="ajustada por idade, CNAE e CBO"), aj2[aj2["termo"].str.startswith("sexo")].assign(
        modelo="ajustada por idade, CNAE, CBO e emitente")])
    gravar(t.round(3), pasta, "doenca_sexo_razao_de_chances.csv", index=False)
    log("\nDoença x sexo (razão de chances, feminino contra masculino):\n" + t.round(2).to_string(index=False))
    div_num = pd.to_numeric(b["cnae_divisao"], errors="coerce")   # a divisão pode vir como texto (parquet) ou número (CSV)
    est = b.assign(grupo=np.select([div_num == 64, div_num == 86], ["64 financeiro", "86 saúde"],
                                   "demais"))
    t = est.groupby(["grupo", "sexo"])["doenca"].agg(["mean", "size"])
    t["mean"] *= 100
    log("Doença (%) por sexo, estratificada:\n" + t.round(2).to_string())
    ss = b[b["emitente_cat"] != "Sindicato"]
    log("Sem CATs de sindicatos, doença (%) por sexo: "
        + str((ss.groupby("sexo")["doenca"].mean() * 100).round(2).to_dict())
        + f" | mulheres entre as doenças: {(ss.loc[ss['doenca']==1,'sexo']=='Feminino').mean()*100:.1f}%")

    # ---- 9. partes do corpo (Tabela 2) e natureza
    def tab_tipo(col, minimo=0):
        t = (pd.crosstab(b[col], b["tipo_acidente"], normalize="index") * 100)[TIPOS]
        t.insert(0, "registros", b.groupby(col).size())
        t.insert(1, "%_base", t["registros"] / n * 100)
        g = b.groupby(col)["indica_obito"].agg(["sum", "size"])
        t["obitos"] = g["sum"]
        t["obito_%"] = g["sum"] / g["size"] * 100
        return t[t["registros"] >= minimo].sort_values("registros", ascending=False)
    t = tab_tipo("parte_corpo")
    gravar(t.round(2), pasta, "partes_do_corpo.csv")
    log("\nPartes do corpo (15 mais frequentes):\n" + t.head(15).round(2).to_string())
    for tp in TIPOS:
        s = b.loc[b["tipo_acidente"] == tp, "parte_corpo"].value_counts(normalize=True).head(6) * 100
        log(f"  Partes mais frequentes em {tp}: {s.round(1).to_dict()}")
    log("  Maiores taxas de óbito (partes com 500+ registros):\n"
        + t[t["registros"] >= 500].sort_values("obito_%", ascending=False).head(5).round(2).to_string())

    # ---- 10. sindicatos, CNAE e CBO
    sind = b["emitente_cat"] == "Sindicato"
    log(f"\nCATs de sindicatos: {sind.sum():,} ({sind.mean()*100:.2f}%) | doenças entre elas: "
        f"{b.loc[sind,'doenca'].mean()*100:.1f}% | divisão 64 entre as CATs de sindicatos: "
        f"{(pd.to_numeric(b.loc[sind,'cnae_divisao'], errors='coerce')==64).mean()*100:.1f}% | doenças comunicadas por sindicatos: "
        f"{sind[b['doenca']==1].mean()*100:.1f}%")
    log("Emitente das doenças (%): " + str((do["emitente_cat"].value_counts(normalize=True) * 100).round(1).to_dict()))
    for col in ["cnae_divisao", "cbo_grande_grupo", "sexo"]:
        a = tab_tipo(col)[["registros", "Doença"]]
        s = b[~sind]
        c = pd.DataFrame({"registros_sem_sindicato": s.groupby(col).size(),
                          "Doença_sem_sindicato": s.groupby(col)["doenca"].mean() * 100})
        t = a.join(c)
        t["doencas_%_do_total"] = b[b["doenca"] == 1].groupby(col).size() / b["doenca"].sum() * 100
        gravar(t.round(2), pasta, f"doenca_com_e_sem_sindicato_{col}.csv")
        log(f"Doença (%) com e sem as CATs de sindicatos, por {col}:\n"
            + t.sort_values("Doença", ascending=False).head(8).round(1).to_string())
    log(f"Total sem sindicatos: doença {b.loc[~sind,'doenca'].mean()*100:.2f}% de {(~sind).sum():,}")

    # ---- 11. Tabela 3: critério por resíduo padronizado
    obs = pd.crosstab(b["cnae_divisao"], b["tipo_acidente"])[TIPOS]
    chi2, _, _, esp = stats.chi2_contingency(obs.to_numpy(), correction=False)
    tot_l, tot_c = obs.sum(axis=1).to_numpy()[:, None], obs.sum(axis=0).to_numpy()[None, :]
    res = (obs.to_numpy() - esp) / np.sqrt(esp * (1 - tot_l / n) * (1 - tot_c / n))
    t = (obs.div(obs.sum(axis=1), axis=0) * 100).round(1)
    t.insert(0, "registros", obs.sum(axis=1))
    for i, tp in enumerate(TIPOS):
        t[f"residuo_{tp}"] = res[:, i].round(1)
    t["maior_residuo_abs"] = np.abs(res).max(1).round(1)
    t["contrib_qui2_%"] = (((obs.to_numpy() - esp) ** 2 / esp).sum(1) / chi2 * 100).round(1)
    t = t.sort_values("contrib_qui2_%", ascending=False)
    gravar(t, pasta, "cnae_residuos.csv")
    limiar = stats.norm.ppf(1 - 0.05 / (2 * obs.size))
    log(f"\nDivisões da CNAE: {len(obs)} | resíduo crítico com correção de Bonferroni ({obs.size} células): {limiar:.2f}")
    log("Divisões com maior contribuição para o qui-quadrado:\n" + t.head(12).to_string())

    # ---- 12. V de Cramér com correção de viés
    linhas = []
    for nome, col in [("Agente causador", "agente_causador"), ("Natureza da lesão", "natureza_lesao"),
                      ("Diagnóstico (capítulo do CID)", "cid_capitulo"), ("Parte do corpo", "parte_corpo"),
                      ("Emitente da CAT", "emitente_cat"), ("Atividade econômica (divisão CNAE)", "cnae_divisao"),
                      ("Ocupação (grande grupo CBO)", "cbo_grande_grupo"), ("Estado do empregador", "uf_empregador"),
                      ("Sexo", "sexo"), ("Ano", "ano"), ("Faixa etária", "faixa_etaria"), ("Mês", "mes"),
                      ("Dia da semana", "dia_semana")]:
        v, vc, _ = cramers_v(b[col], b["tipo_acidente"])
        vo, voc, _ = cramers_v(b[col], b["indica_obito"])
        linhas.append({"variavel": nome, "categorias": b[col].nunique(), "V_tipo": v, "V_tipo_corrigido": vc,
                       "V_obito": vo, "V_obito_corrigido": voc})
    t = pd.DataFrame(linhas)
    gravar(t.round(4), pasta, "cramer_v.csv", index=False)
    log("\nV de Cramér (com correção de viés de Bergsma):\n" + t.round(3).to_string(index=False))

    # ---- 13. categorias com maior taxa de óbito (checagem de vazamento)
    for col in ["cid_capitulo", "natureza_lesao", "parte_corpo", "agente_causador", "emitente_cat"]:
        g = b.groupby(col)["indica_obito"].agg(["sum", "size"])
        g["taxa_%"] = g["sum"] / g["size"] * 100
        g["%_dos_obitos"] = g["sum"] / g["sum"].sum() * 100
        g = g[g["size"] >= 100].sort_values("taxa_%", ascending=False).head(6)
        gravar(g.round(2), pasta, f"obito_categorias_{col}.csv")
        log(f"\nCategorias de {col} com maior taxa de óbito (100+ registros):\n" + g.round(2).to_string())
    cid3 = b.assign(c3=b["cid"].astype(str).str[:3]).groupby("c3")["indica_obito"].agg(["sum", "size"])
    cid3["taxa_%"] = cid3["sum"] / cid3["size"] * 100
    log("CID (3 caracteres) com mais óbitos:\n" + cid3.sort_values("sum", ascending=False).head(10).round(1).to_string())
    log(f"Registros com CID R95-R99 ou do capítulo R: {(b['cid_capitulo']=='R').sum()} | óbitos neles: "
        f"{b.loc[b['cid_capitulo']=='R','indica_obito'].sum()}")


# =============================================================================== PARTE 2
def bootstrap_matriz(cm: np.ndarray, n_rep: int = 2000, seed: int = SEED) -> dict:
    """Bootstrap do conjunto de teste a partir da matriz de confusão (reamostrar linhas do teste
    equivale a sortear as células da matriz com probabilidade proporcional às contagens)."""
    rng = np.random.default_rng(seed)
    n = cm.sum()
    am = rng.multinomial(n, (cm / n).ravel(), size=n_rep).reshape(n_rep, *cm.shape).astype(float)

    def metr(m):
        vp = np.diagonal(m, axis1=-2, axis2=-1)
        rev = vp / np.clip(m.sum(-1), 1, None)
        pre = vp / np.clip(m.sum(-2), 1, None)
        f1 = 2 * pre * rev / np.clip(pre + rev, 1e-12, None)
        return {"f1_macro": f1.mean(-1), "acuracia_balanceada": rev.mean(-1),
                **{f"f1_{i}": f1[..., i] for i in range(m.shape[-1])},
                **{f"precisao_{i}": pre[..., i] for i in range(m.shape[-1])},
                **{f"revocacao_{i}": rev[..., i] for i in range(m.shape[-1])}}
    ponto, dist = metr(cm.astype(float)), metr(am)
    return {k: (float(ponto[k]), float(np.percentile(dist[k], 2.5)), float(np.percentile(dist[k], 97.5)))
            for k in ponto}


def ic_das_matrizes(pasta_modelos: Path, pasta: Path, log) -> None:
    linhas = []
    for alg in ["regressao_logistica", "arvore_decisao", "random_forest", "xgboost"]:
        for versao in ["completa", "perfil"]:
            arq = pasta_modelos / alg / f"matriz_confusao_tipo_{versao}.csv"
            if not arq.exists():
                continue
            m = pd.read_csv(arq, sep=";", encoding="utf-8-sig", index_col=0)
            classes = [c.replace("previsto_", "") for c in m.columns]
            r = bootstrap_matriz(m.to_numpy())
            linha = {"algoritmo": alg, "variaveis": versao}
            for k, (p, lo, hi) in r.items():
                for i, c in enumerate(classes):
                    k = k.replace(f"_{i}", f"_{c}") if k.endswith(f"_{i}") else k
                linha[k], linha[k + "_inf"], linha[k + "_sup"] = p, lo, hi
            linhas.append(linha)
    if linhas:
        t = pd.DataFrame(linhas)
        gravar(t.round(4), pasta, "metricas_com_ic95_bootstrap.csv", index=False)
        cols = ["algoritmo", "variaveis", "f1_macro", "f1_macro_inf", "f1_macro_sup", "acuracia_balanceada",
                "precisao_Doença", "revocacao_Doença", "f1_Doença", "f1_Doença_inf", "f1_Doença_sup",
                "precisao_Trajeto", "revocacao_Trajeto", "precisao_Típico", "revocacao_Típico"]
        log("\nMétricas no teste com IC de 95% por bootstrap (2.000 reamostras):\n"
            + t[cols].round(3).to_string(index=False))


def montar(algoritmo: str, cols: list[str], params: dict, args):
    cats = [c for c in cols if c not in NUMERICAS]
    nums = [c for c in cols if c in NUMERICAS]
    if algoritmo == "random_forest":
        return rf.montar_modelo(cats, nums, args.arvores, SEED, args.n_jobs, **params)
    return mc.montar(algoritmo, cats, nums, args, **params)


def avaliar_tipo(modelo, Xte, yte) -> dict:
    pred = modelo.predict(Xte)
    classes = list(modelo.classes_)
    m = rf.metricas_tipo(yte, pred, classes)
    from sklearn.metrics import confusion_matrix, f1_score
    f1c = f1_score(yte, pred, labels=classes, average=None, zero_division=0)
    m |= {f"f1_{c}": v for c, v in zip(classes, f1c)}
    r = bootstrap_matriz(confusion_matrix(yte, pred, labels=classes), n_rep=1000)
    m["f1_macro_inf"], m["f1_macro_sup"] = r["f1_macro"][1], r["f1_macro"][2]
    return m, pred


def parte_modelos(base: pd.DataFrame, rem: pd.DataFrame | None, pasta_saida: Path, pasta: Path,
                  args, log) -> None:
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score, f1_score, make_scorer

    alg = args.algoritmo
    ic_das_matrizes(pasta_saida / "modelos", pasta, log)

    df = rf.preparar(base, log)
    arq_par = pasta_saida / "modelos" / alg / "melhores_parametros.json"
    par = json.loads(arq_par.read_text(encoding="utf-8"))["parametros"] if arq_par.exists() else {}
    def p(chave):
        for k, v in par.items():
            if k.endswith(chave):
                return v
        return {}
    todas = BLOCOS["perfil"] + BLOCOS["evento"] + BLOCOS["registro"]
    log(f"\nAlgoritmo das análises de sensibilidade: {alg} | hiperparâmetros: completa {p('tipo_completa')} | "
        f"perfil {p('tipo_perfil')}")

    def rodar(nome, treino, teste, cols, params, guardar=None):
        t0 = time.time()
        m = montar(alg, cols, params, args).fit(treino[cols], treino["tipo_acidente"])
        met, pred = avaliar_tipo(m, teste[cols], teste["tipo_acidente"])
        linha = {"cenario": nome, "n_treino": len(treino), "n_teste": len(teste), "n_variaveis": len(cols), **met}
        log(f"  {nome}: F1 macro {met['f1_macro']:.3f} [{met['f1_macro_inf']:.3f}; {met['f1_macro_sup']:.3f}] | "
            f"acur. bal. {met['acuracia_balanceada']:.3f} | F1 doença {met['f1_Doença']:.3f} | "
            f"treino {len(treino):,} teste {len(teste):,} ({time.time()-t0:.0f} s)")
        if guardar is not None:
            guardar.update(modelo=m, pred=pred)
        return linha

    tr, te = df[df["ym"] < "2024-01"], df[df["ym"] >= "2024-01"]
    log("\nComposição (%): treino " + str((tr["tipo_acidente"].value_counts(normalize=True) * 100).round(1).to_dict())
        + " | teste " + str((te["tipo_acidente"].value_counts(normalize=True) * 100).round(1).to_dict()))
    log(f"Meses de cobertura incompleta: treino {tr.loc[tr['mes_estavel']==0,'ym'].nunique()} meses "
        f"({(tr['mes_estavel']==0).mean()*100:.1f}% dos registros) | teste "
        f"{te.loc[te['mes_estavel']==0,'ym'].nunique()} meses ({(te['mes_estavel']==0).mean()*100:.1f}%)")

    # ---- A. referência, meses regulares e divisão alternativa
    linhas, ref = [], {}
    log("\nA. Referência, só meses regulares e divisão alternativa")
    for versao, cols in [("completa", todas), ("perfil", BLOCOS["perfil"])]:
        pr = p(f"tipo_{versao}")
        linhas.append(rodar(f"{versao} | referência (treino até 2023, teste 2024-2025)", tr, te, cols, pr,
                            ref.setdefault(versao, {})))
        pred = ref[versao]["pred"]
        est = (te["mes_estavel"] == 1).to_numpy()
        for nome_r, masc in [("teste só meses regulares", est), ("teste só meses incompletos", ~est)]:
            f1 = f1_score(te["tipo_acidente"][masc], pred[masc], average="macro")
            log(f"    mesmo modelo, {nome_r}: F1 macro {f1:.3f} (n={masc.sum():,})")
            linhas.append({"cenario": f"{versao} | referência, {nome_r}", "n_treino": len(tr),
                           "n_teste": int(masc.sum()), "n_variaveis": len(cols), "f1_macro": f1})
        linhas.append(rodar(f"{versao} | treino e teste só com meses regulares", tr[tr["mes_estavel"] == 1],
                            te[te["mes_estavel"] == 1], cols, pr))
        linhas.append(rodar(f"{versao} | treino até 2022, teste em 2023", df[df["ym"] < "2023-01"],
                            df[(df["ym"] >= "2023-01") & (df["ym"] < "2024-01")], cols, pr))
        # linha de base mais forte: classe mais frequente do treino em cada divisão da CNAE
        moda = tr.groupby("cnae_divisao")["tipo_acidente"].agg(lambda s: s.value_counts().index[0])
        pb = te["cnae_divisao"].map(moda).fillna(tr["tipo_acidente"].value_counts().index[0])
        if versao == "perfil":
            f1b = f1_score(te["tipo_acidente"], pb, average="macro")
            log(f"    linha de base por CNAE (classe mais frequente da divisão no treino): F1 macro {f1b:.3f}")
            linhas.append({"cenario": "linha de base | classe mais frequente por divisão da CNAE", "n_treino": len(tr),
                           "n_teste": len(te), "n_variaveis": 1, "f1_macro": f1b,
                           "acuracia_balanceada": rf.balanced_accuracy_score(te["tipo_acidente"], pb)})
            import itertools
            moda2 = tr.groupby(["cnae_divisao", "cbo_grande_grupo"])["tipo_acidente"].agg(
                lambda s: s.value_counts().index[0])
            chave = pd.MultiIndex.from_frame(te[["cnae_divisao", "cbo_grande_grupo"]])
            pb2 = pd.Series(moda2.reindex(chave).to_numpy(), index=te.index).fillna(pb)
            f1b2 = f1_score(te["tipo_acidente"], pb2, average="macro")
            log(f"    linha de base por CNAE e CBO: F1 macro {f1b2:.3f}")
            linhas.append({"cenario": "linha de base | classe mais frequente por CNAE e CBO", "n_treino": len(tr),
                           "n_teste": len(te), "n_variaveis": 2, "f1_macro": f1b2})
    gravar(pd.DataFrame(linhas).round(4), pasta, f"sensibilidade_{alg}.csv", index=False)

    # ---- B. blocos de variáveis
    log("\nB. Blocos de variáveis (perfil, descrição do evento, registro)")
    linhas = []
    combos = [("perfil", ["perfil"]), ("evento", ["evento"]), ("registro", ["registro"]),
              ("perfil + evento", ["perfil", "evento"]), ("perfil + registro", ["perfil", "registro"]),
              ("evento + registro", ["evento", "registro"]), ("perfil + evento + registro", ["perfil", "evento", "registro"]),
              ("completo sem emitente", None), ("completo sem CID", None)]
    for nome, blocos in combos:
        if blocos is None:
            fora = "emitente_cat" if "emitente" in nome else "cid_capitulo"
            cols = [c for c in todas if c != fora]
        else:
            cols = [c for b_ in blocos for c in BLOCOS[b_]]
        pr = p("tipo_perfil") if blocos == ["perfil"] else p("tipo_completa")
        linhas.append(rodar(nome, tr, te, cols, pr))
    gravar(pd.DataFrame(linhas).round(4), pasta, f"blocos_{alg}.csv", index=False)

    # ---- C. permutação por grupos de variáveis (teste completo, 10 repetições)
    log("\nC. Importância por permutação de grupos de variáveis (10 repetições, amostra de 20.000 do teste)")
    am = te.sample(min(20000, len(te)), random_state=SEED)
    rng = np.random.default_rng(SEED)
    linhas = []
    for versao, cols in [("completa", todas), ("perfil", BLOCOS["perfil"])]:
        m = ref[versao]["modelo"]
        f0 = f1_score(am["tipo_acidente"], m.predict(am[cols]), average="macro")
        grupos = {"perfil (7 variáveis)": BLOCOS["perfil"], "evento (agente, natureza, parte)": BLOCOS["evento"],
                  "registro (CID, emitente)": BLOCOS["registro"]} if versao == "completa" else {}
        grupos |= {c: [c] for c in cols}
        for g, gcols in grupos.items():
            quedas = []
            for _ in range(10):
                X = am[cols].copy()
                perm = rng.permutation(len(X))
                X[gcols] = X[gcols].to_numpy()[perm]
                quedas.append(f0 - f1_score(am["tipo_acidente"], m.predict(X), average="macro"))
            linhas.append({"variaveis": versao, "grupo": g, "queda_f1_macro": np.mean(quedas),
                           "desvio": np.std(quedas, ddof=1)})
    t = pd.DataFrame(linhas)
    gravar(t.round(4), pasta, f"permutacao_grupos_{alg}.csv", index=False)
    log(t.round(4).to_string(index=False))

    # ---- D. manter os ausentes como categoria
    if rem is not None:
        log("\nD. Registros com dado ausente mantidos, com a categoria '(ausente)'")
        tudo = pd.concat([base[rem.columns.intersection(base.columns)], rem], ignore_index=True)
        tudo = tudo[tudo["tipo_acidente"].notna()].copy()
        tudo["idade"] = tudo["idade"].fillna(tudo["idade"].median())
        for c in ["cnae_divisao", "cbo_grande_grupo"]:
            tudo[c] = tudo[c].map(lambda v: "(ausente)" if pd.isna(v) else str(int(v)))
        d2 = rf.preparar(tudo, log)
        tr2, te2 = d2[d2["ym"] < "2024-01"], d2[d2["ym"] >= "2024-01"]
        linhas = [rodar("completa | com ausentes como categoria", tr2, te2, todas, p("tipo_completa"))]
        gravar(pd.DataFrame(linhas).round(4), pasta, f"ausentes_como_categoria_{alg}.csv", index=False)

    # ---- E. óbito: versões sem CID e sem variáveis de gravidade
    log("\nE. Óbito: checagem de vazamento")
    linhas = []
    yo_tr, yo_te = tr["indica_obito"].to_numpy(), te["indica_obito"].to_numpy()
    cenarios = [("completa", todas), ("sem CID", [c for c in todas if c != "cid_capitulo"]),
                ("sem CID, natureza e parte do corpo", [c for c in todas if c not in
                                                        ("cid_capitulo", "natureza_lesao", "parte_corpo")]),
                ("perfil", BLOCOS["perfil"])]
    for alg_o in dict.fromkeys([alg, "regressao_logistica"]):
        par_o = json.loads((pasta_saida / "modelos" / alg_o / "melhores_parametros.json").read_text(
            encoding="utf-8"))["parametros"] if (pasta_saida / "modelos" / alg_o / "melhores_parametros.json").exists() else {}
        for nome, cols in cenarios:
            pr = next((v for k, v in par_o.items() if k.endswith("obito_perfil" if nome == "perfil" else "obito_completa")), {})
            m = montar(alg_o, cols, pr, args).fit(tr[cols], tr["indica_obito"])
            pb = m.predict_proba(te[cols])[:, list(m.classes_).index(1)]
            topo = np.argsort(-pb)[:int(round(0.05 * len(te)))]
            linha = {"algoritmo": alg_o, "cenario": nome, "pr_auc": average_precision_score(yo_te, pb),
                     "revocacao_top5%": yo_te[topo].sum() / yo_te.sum(), "precisao_top5%": yo_te[topo].mean(),
                     "prevalencia": yo_te.mean(), "n_obitos_teste": int(yo_te.sum())}
            linhas.append(linha)
            log(f"  {alg_o} | {nome}: PR-AUC {linha['pr_auc']:.3f} | revocação no topo de 5% "
                f"{linha['revocacao_top5%']:.3f} | precisão no topo de 5% {linha['precisao_top5%']:.3f}")
    gravar(pd.DataFrame(linhas).round(4), pasta, "obito_sem_cid.csv", index=False)


def versoes() -> str:
    import matplotlib as mpl
    import scipy
    import sklearn
    linhas = [f"Python {platform.python_version()}", f"pandas {pd.__version__}", f"numpy {np.__version__}",
              f"scipy {scipy.__version__}", f"scikit-learn {sklearn.__version__}", f"matplotlib {mpl.__version__}"]
    for nome in ("xgboost", "pyarrow"):
        try:
            linhas.append(f"{nome} {__import__(nome).__version__}")
        except ImportError:
            linhas.append(f"{nome} não instalado")
    return " | ".join(linhas)


def main() -> None:
    ap = argparse.ArgumentParser(description="Análises complementares do TCC")
    ap.add_argument("--entrada", type=Path, default=None, help="base_nordeste.parquet ou .csv")
    ap.add_argument("--removidos", type=Path, default=None, help="removidos_por_ausencia.csv")
    ap.add_argument("--saida", type=Path, default=None, help="pasta saida/ (padrão: a pasta da base)")
    ap.add_argument("--parte", choices=["descritiva", "modelos", "tudo"], default="tudo")
    ap.add_argument("--algoritmo", default="random_forest",
                    choices=["random_forest", "xgboost", "regressao_logistica", "arvore_decisao"])
    ap.add_argument("--arvores", type=int, default=500)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--xgb-substituto", action="store_true",
                    help="usa o HistGradientBoosting no lugar do XGBoost (só para testar o script)")
    args = ap.parse_args()

    arq = rf.achar_base(args.entrada)
    pasta_saida = args.saida if args.saida is not None else arq.parent
    pasta = pasta_saida / "complementar"
    pasta.mkdir(parents=True, exist_ok=True)
    log = rf.Log()
    log(f"Entrada: {arq}\nSaída: {pasta}\nVersões: {versoes()}\nSemente aleatória: {SEED}")
    base = rf.ler_base(arq)
    arq_rem = args.removidos or pasta_saida / "removidos_por_ausencia.csv"
    rem = pd.read_csv(arq_rem, sep=";", encoding="utf-8-sig", low_memory=False) if arq_rem.exists() else None
    if rem is None:
        log("AVISO: removidos_por_ausencia.csv não encontrado; as análises de ausência serão puladas.")
    t0 = time.time()
    if args.parte in ("descritiva", "tudo"):
        parte_descritiva(base.copy(), rem, pasta_saida, pasta, log)
        log.gravar(pasta / "log_complementar_descritiva.txt")
    if args.parte in ("modelos", "tudo"):
        parte_modelos(base.copy(), rem, pasta_saida, pasta, args, log)
        log.gravar(pasta / f"log_complementar_modelos_{args.algoritmo}.txt")
    log(f"\nConcluído em {(time.time()-t0)/60:.1f} min. Arquivos em {pasta}")


if __name__ == "__main__":
    main()
