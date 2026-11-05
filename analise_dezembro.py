from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Pastas padrão;
PASTA_PROJETO = Path(__file__).resolve().parent   # pasta onde estão os scripts
SAIDA_PADRAO = PASTA_PROJETO / "saida"
ENTRADA_PADRAO = SAIDA_PADRAO / "base_nordeste.parquet"
BASE_NACIONAL_PADRAO = SAIDA_PADRAO / "cat_limpa.parquet"   # opcional (parte 5)

CORTE_TEMPORAL = "2023-02"   # treino: antes; teste: a partir daqui (igual ao montar_base_ne.py)
COLS_CATEGORICAS = ["agente_causador", "natureza_lesao", "parte_corpo", "sexo", "emitente_cat",
                    "cnae_divisao", "cbo_grande_grupo", "cid_capitulo", "uf_empregador"]
COLS_NUMERICAS = ["idade", "mes", "dia_semana"]


def pct(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def mil(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", ".")


def carregar(caminho: Path) -> pd.DataFrame:
    df = pd.read_parquet(caminho)
    for c in ("data_acidente", "data_emissao_cat"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ("ano", "mes", "janela", "mes_estavel", "dias_ate_emissao", "indica_obito"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df



#Partes 1 a 3: tabela mensal


def tabela_mensal(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("ym")
    t = pd.DataFrame({
        "janela": g["janela"].first(),
        "mes_estavel": g["mes_estavel"].first(),
        "registros": g.size(),
        # só linhas com data de emissão conhecida (o layout D de jun-out/2023 não tem)
        "com_emissao_conhecida": g["dias_ate_emissao"].apply(lambda s: int(s.notna().sum())),
        # só calcula quando pelo menos 90% das linhas do mês têm data de emissão conhecida;
        # abaixo disso o resultado seria enviesado (sobrariam só as CATs tardias, de outros layouts)
        "pct_atraso_15d": g["dias_ate_emissao"].apply(
            lambda s: round((s > 15).sum() / s.notna().sum() * 100, 1) if s.notna().mean() >= 0.9 else np.nan),
        "pct_doenca": g["tipo_acidente"].apply(lambda s: round((s == "Doença").mean() * 100, 1)),
        "pct_trajeto": g["tipo_acidente"].apply(lambda s: round((s == "Trajeto").mean() * 100, 1)),
    })
    t["mes"] = [int(i[5:]) for i in t.index]
    return t


def razao_dezembro(t: pd.DataFrame) -> pd.DataFrame:

    linhas = []
    for ym in t.index[t["mes"] == 12]:
        j = t.loc[ym, "janela"]
        estavel = bool(t.loc[ym, "mes_estavel"] == 1)
        obs = int(t.loc[ym, "registros"])
        ano = int(ym[:4])
        nov, jan = f"{ano}-11", f"{ano + 1}-01"
        linha = {
            "mes": ym, "mes_estavel": int(estavel), "registros": obs,
            "novembro": int(t.loc[nov, "registros"]) if nov in t.index else np.nan,
            "janeiro_seguinte": int(t.loc[jan, "registros"]) if jan in t.index else np.nan,
            "pct_doenca_dez": t.loc[ym, "pct_doenca"],
        }
        nulos = {"media_outros_meses_da_janela": np.nan, "razao_dez_sobre_media": np.nan,
                 "pct_atraso_dez": np.nan, "pct_atraso_normal_da_janela": np.nan,
                 "pct_doenca_outros_meses": np.nan, "estimativa_cats_faltantes": np.nan,
                 "estimativa_pct_faltante": np.nan}
        if not estavel or j == 0:
            linhas.append({**linha, **nulos})
            continue
        outros = t[(t["janela"] == j) & (t.index != ym)]
        normal = outros[(outros["mes"] != 12) & outros["pct_atraso_15d"].notna()]
        p_normal = normal["pct_atraso_15d"].mean() / 100 if len(normal) else np.nan
        p_dez = t.loc[ym, "pct_atraso_15d"] / 100
        # estimativa: se as CATs pontuais estão completas, total = pontuais / (1 - atraso normal)
        if pd.notna(p_normal) and pd.notna(p_dez):
            total_est = obs * (1 - p_dez) / (1 - p_normal)
            faltantes = total_est - obs
        else:
            total_est, faltantes = np.nan, np.nan
        linhas.append({**linha,
            "media_outros_meses_da_janela": round(outros["registros"].mean()),
            "razao_dez_sobre_media": round(obs / outros["registros"].mean(), 2),
            "pct_atraso_dez": round(p_dez * 100, 1),
            "pct_atraso_normal_da_janela": round(p_normal * 100, 1) if pd.notna(p_normal) else np.nan,
            "pct_doenca_outros_meses": round(
                (outros["pct_doenca"] * outros["registros"]).sum() / outros["registros"].sum(), 1),
            "estimativa_cats_faltantes": round(faltantes) if pd.notna(faltantes) else np.nan,
            "estimativa_pct_faltante": round(faltantes / total_est * 100, 1) if pd.notna(faltantes) else np.nan,
        })
    return pd.DataFrame(linhas)


# Parte 5: acidente e emissão em anos diferentes (base nacional)


def anos_diferentes(caminho: Path) -> dict | None:
    if not caminho.exists():
        return None
    d = pd.read_parquet(caminho, columns=["data_acidente", "data_emissao_cat", "layout_origem"])
    d = d[d["layout_origem"] != "D"]   # no layout D a data de emissão não é conhecida
    for c in ("data_acidente", "data_emissao_cat"):
        d[c] = pd.to_datetime(d[c], errors="coerce")
    d = d.dropna()
    cruza = d["data_acidente"].dt.year != d["data_emissao_cat"].dt.year
    dez = d["data_acidente"].dt.month == 12
    return {"comparaveis": len(d), "anos_diferentes": int(cruza.sum()),
            "pct": round(cruza.mean() * 100, 4),
            "acidentes_de_dezembro": int(dez.sum()),
            "dezembro_emitidos_em_janeiro": int((dez & cruza & (d["data_emissao_cat"].dt.month == 1)).sum())}



# Parte 6: sensibilidade do modelo


def avaliar(df: pd.DataFrame, nome: str) -> dict:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    base = df[df["tipo_acidente"].notna()]
    X = base[COLS_CATEGORICAS + COLS_NUMERICAS].copy()
    for c in COLS_CATEGORICAS:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    for c in COLS_NUMERICAS:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    y = base["tipo_acidente"].astype(str)
    tr = base["ym"] < CORTE_TEMPORAL
    te = ~tr
    prep = ColumnTransformer([
        ("cat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="Nao informado")),
                          ("o", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005))]),
         COLS_CATEGORICAS),
        ("num", Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)),
                          ("s", StandardScaler())]), COLS_NUMERICAS)])
    m = Pipeline([("p", prep), ("c", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    m.fit(X[tr], y[tr])
    p = m.predict(X[te])
    rec = recall_score(y[te], p, labels=["Doença"], average=None)
    return {"cenario": nome, "treino": int(tr.sum()), "teste": int(te.sum()),
            "f1_macro": round(f1_score(y[te], p, average="macro"), 3),
            "acuracia_balanceada": round(balanced_accuracy_score(y[te], p), 3),
            "recall_doenca": round(float(rec[0]), 3),
            "pct_doenca_no_teste": round((y[te] == "Doença").mean() * 100, 2)}


def sensibilidade(df: pd.DataFrame) -> pd.DataFrame:
    dez = df["mes_dezembro"].astype(float) == 1
    ult = df["ultimo_mes_dados"].astype(float) == 1
    est = df["mes_estavel"].astype(float) == 1
    cenarios = [("A. todos os meses do Nordeste", df),
                ("B. todos os meses, sem dezembro", df[~dez]),
                ("C. só meses estáveis", df[est]),
                ("D. só meses estáveis, sem dezembro e sem o último mês", df[est & ~dez & ~ult])]
    return pd.DataFrame([avaliar(d, n) for n, d in cenarios])



#Gráfico


def grafico(t: pd.DataFrame, caminho: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    x = np.arange(len(t))
    dez = (t["mes"] == 12).to_numpy()
    est = (t["mes_estavel"] == 1).to_numpy()
    cor = np.select([dez & est, dez & ~est, est], ["#C0392B", "#E6B0AA", "#7F8C8D"], default="#D0D3D4")
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    for a, col, rot in zip(ax, ["registros", "pct_atraso_15d", "pct_doenca"],
                           ["Registros por mês", "% de CATs com mais de 15 dias de atraso", "% de doença"]):
        a.bar(x, t[col].fillna(0), color=cor)
        a.set_ylabel(rot, fontsize=9)
        a.grid(axis="y", alpha=0.3)
    ax[1].text(0.01, 0.9, "Meses com mais de 10% das linhas sem data de emissão (layout D, mai-out/2023) ficam em branco",
               transform=ax[1].transAxes, fontsize=8, color="#555")
    ax[0].legend(handles=[Patch(color="#7F8C8D", label="mês de cobertura estável"),
                          Patch(color="#D0D3D4", label="mês de cobertura incompleta"),
                          Patch(color="#C0392B", label="dezembro (estável)"),
                          Patch(color="#E6B0AA", label="dezembro (cobertura incompleta)")],
                 fontsize=8, loc="upper left")
    ax[2].set_xticks(x[::3])
    ax[2].set_xticklabels(t.index[::3], rotation=60, fontsize=8)
    fig.suptitle("Nordeste: registros, CATs tardias e doenças por mês do acidente", fontsize=11)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)




def main() -> None:
    ap = argparse.ArgumentParser(description="Evidências sobre dezembro para o TCC")
    ap.add_argument("--entrada", default=ENTRADA_PADRAO, type=Path)
    ap.add_argument("--nacional", default=BASE_NACIONAL_PADRAO, type=Path,
                    help="cat_limpa.parquet (opcional; usada na parte 5)")
    ap.add_argument("--saida", default=SAIDA_PADRAO, type=Path)
    ap.add_argument("--sem-modelo", action="store_true", help="pula a sensibilidade do modelo")
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    df = carregar(args.entrada)
    print(f"Base: {len(df):,} linhas")

    t = tabela_mensal(df)
    t.to_csv(args.saida / "dezembro_mensal.csv", encoding="utf-8-sig")
    r = razao_dezembro(t)
    r.to_csv(args.saida / "dezembro_resumo.csv", index=False, encoding="utf-8-sig")
    grafico(t, args.saida / "dezembro_grafico.png")

    linhas = ["ANÁLISE DE DEZEMBRO E DA COBERTURA (nada foi excluído da base)", ""]
    n_inst = int((t["mes_estavel"] == 0).sum())
    linhas += [f"Meses na base: {len(t)} | de cobertura estável: {int((t['mes_estavel'] == 1).sum())} | "
               f"de cobertura incompleta: {n_inst}", ""]
    for _, d in r.iterrows():
        if d["mes_estavel"] != 1:
            linhas += [f"{d['mes']}: {mil(d['registros'])} registros. Mês de cobertura incompleta nos arquivos de origem: "
                       "a queda reflete principalmente a cobertura, não o fim do ano; sem comparação.", ""]
            continue
        linhas += [
            f"{d['mes']}: {mil(d['registros'])} registros, {pct(d['razao_dez_sobre_media'] * 100, 0)}% da média dos outros "
            f"meses da mesma janela ({mil(d['media_outros_meses_da_janela'])}); "
            f"novembro {mil(d['novembro'])}, janeiro seguinte "
            f"{mil(d['janeiro_seguinte']) if pd.notna(d['janeiro_seguinte']) else 'fora da base'}.",
            f"   CATs com mais de 15 dias de atraso: {pct(d['pct_atraso_dez'])}% em dezembro contra "
            f"{pct(d['pct_atraso_normal_da_janela'])}% nos demais meses da janela.",
            f"   Doença: {pct(d['pct_doenca_dez'])}% em dezembro contra {pct(d['pct_doenca_outros_meses'])}% nos demais meses.",
            f"   Estimativa aproximada de CATs de dezembro fora da base: {mil(d['estimativa_cats_faltantes'])} "
            f"({pct(d['estimativa_pct_faltante'])}% do total esperado). A queda restante é compatível com sazonalidade.",
            ""]
    estaveis = t.index[t["mes_estavel"] == 1]
    ult = estaveis[-1] if len(estaveis) else t.index[-1]
    linhas.append(f"Último mês de cobertura estável ({ult}): {mil(t.loc[ult, 'registros'])} registros, "
                  f"{pct(t.loc[ult, 'pct_atraso_15d'])}% com mais de 15 dias de atraso.")
    linhas.append("")

    na = anos_diferentes(args.nacional)
    if na:
        linhas += [f"Base nacional (layout D excluído): {mil(na['anos_diferentes'])} de {mil(na['comparaveis'])} registros "
                   f"({pct(na['pct'], 4)}%) têm acidente e emissão em anos diferentes.",
                   f"Dos {mil(na['acidentes_de_dezembro'])} acidentes de dezembro, só {mil(na['dezembro_emitidos_em_janeiro'])} "
                   "foram emitidos em janeiro.", ""]
    else:
        linhas += [f"Base nacional não encontrada em {args.nacional}: parte 5 pulada.", ""]

    if not args.sem_modelo:
        s = sensibilidade(df)
        s.to_csv(args.saida / "dezembro_sensibilidade.csv", index=False, encoding="utf-8-sig")
        linhas.append("Sensibilidade do modelo (regressão logística, divisão temporal):")
        linhas.append(s.to_string(index=False))

    texto = "\n".join(linhas)
    (args.saida / "dezembro_resumo.txt").write_text(texto, encoding="utf-8")
    print(texto)
    print(f"\nArquivos gravados em {args.saida}")


if __name__ == "__main__":
    main()
