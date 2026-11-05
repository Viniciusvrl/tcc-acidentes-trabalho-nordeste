from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

try:
    import modelo_random_forest as rf
except ImportError:
    sys.exit("Coloque analise_descritiva.py na MESMA pasta do modelo_random_forest.py.")

VARIAVEIS = ["sexo", "faixa_etaria", "uf_empregador", "cbo_grande_grupo", "cnae_divisao",
             "agente_causador", "natureza_lesao", "parte_corpo", "cid_capitulo", "emitente_cat",
             "ano", "mes", "dia_semana"]
ROTULOS = dict(rf.NOMES_LEGIVEIS) | {"faixa_etaria": "Faixa etária", "ano": "Ano"}
MIN_GRUPO = 100      # categorias menores entram em "(outras)" nos testes de associação
MIN_OBITO = 200      # categorias menores não entram na tabela de taxa de óbito
CORES_TIPO = {"Típico": "#12386b", "Trajeto": "#2a78d6", "Doença": "#9ec5f0"}   # rampa de um só tom


def virgula(x: float, casas: int = 1) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def agrupar_raras(s: pd.Series, minimo: int) -> pd.Series:
    n = s.map(s.value_counts())
    return s.where(n >= minimo, "(outras)")


def cramer(tab: pd.DataFrame) -> tuple[float, float, int]:
    chi2, p, gl, _ = chi2_contingency(tab.to_numpy())
    n = tab.to_numpy().sum()
    k = min(tab.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if k > 0 else np.nan, float(p), int(gl)


def main() -> None:
    ap = argparse.ArgumentParser(description="Análise descritiva por grupo")
    ap.add_argument("--entrada", type=Path, default=None)
    ap.add_argument("--saida", type=Path, default=None)
    ap.add_argument("--so-cobertura-regular", action="store_true")
    args = ap.parse_args()

    log = rf.Log()
    base = rf.achar_base(args.entrada)
    pasta = (args.saida if args.saida is not None else base.parent) / "descritiva"
    pasta.mkdir(parents=True, exist_ok=True)

    df = rf.preparar(rf.ler_base(base), log)
    if args.so_cobertura_regular:
        df = df[df["mes_estavel"] == 1]
        log(f"Só meses de cobertura regular: {len(df):,} linhas")
    for c in VARIAVEIS:
        if c not in df.columns:
            sys.exit(f"Coluna ausente na base: {c}")
        df[c] = df[c].astype("string").fillna("(ausente)").astype(str)
    df["tipo"] = df["tipo_acidente"].astype(str)
    tipos = [t for t in ["Típico", "Trajeto", "Doença"] if t in set(df["tipo"])]

    #composição por ano
    comp = pd.crosstab(df["ano"], df["tipo"], normalize="index")[tipos] * 100
    comp.insert(0, "n", df.groupby("ano").size())
    comp.round(2).to_csv(pasta / "composicao_tipo_por_ano.csv", sep=";", decimal=",", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150, facecolor=rf.SUPERFICIE)
    fundo = np.zeros(len(comp))
    for t in tipos:
        ax.bar(comp.index, comp[t], bottom=fundo, color=CORES_TIPO[t], label=t, width=0.65)
        for x, v, b in zip(comp.index, comp[t], fundo):
            if v >= 4:
                ax.text(x, b + v / 2, virgula(v) + "%", ha="center", va="center", fontsize=8,
                        color="#ffffff" if t != "Doença" else rf.TEXTO)
        fundo += comp[t].to_numpy()
    ax.set_ylabel("% dos acidentes do ano", color=rf.TEXTO_2, fontsize=9)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.12))
    rf._estilo(ax)
    fig.tight_layout()
    fig.savefig(pasta / "composicao_tipo_por_ano.png", facecolor=rf.SUPERFICIE)
    plt.close(fig)
    log("Composição por ano (%):\n" + comp.round(1).to_string())

    #tabelas por variável
    assoc = []
    for v in VARIAVEIS:
        # perfil dentro de cada tipo (colunas somam 100%)
        a = pd.crosstab(df[v], df["tipo"], normalize="columns")[tipos] * 100
        a["n_total"] = df[v].value_counts()
        a.sort_values(tipos[0], ascending=False).round(2).to_csv(
            pasta / f"perfil_por_tipo_{v}.csv", sep=";", decimal=",", encoding="utf-8-sig")
        # tipo dentro de cada categoria (linhas somam 100%)
        b = pd.crosstab(df[v], df["tipo"], normalize="index")[tipos] * 100
        b["n"] = df[v].value_counts()
        b.sort_values("n", ascending=False).round(2).to_csv(
            pasta / f"tipo_dentro_de_{v}.csv", sep=";", decimal=",", encoding="utf-8-sig")
        # taxa de óbito por categoria
        g = df.groupby(v)["indica_obito"].agg(n="size", obitos="sum")
        g["taxa_obito_pct"] = g["obitos"] / g["n"] * 100
        g[g["n"] >= MIN_OBITO].sort_values("taxa_obito_pct", ascending=False).round(3).to_csv(
            pasta / f"obito_taxa_por_{v}.csv", sep=";", decimal=",", encoding="utf-8-sig")
        # associação (V de Cramér) com tipo e com óbito
        x = agrupar_raras(df[v], MIN_GRUPO)
        for alvo, col in [("tipo_acidente", "tipo"), ("indica_obito", "indica_obito")]:
            tab = pd.crosstab(x, df[col])
            vc, p, gl = cramer(tab)
            assoc.append({"variavel": v, "rotulo": ROTULOS.get(v, v), "alvo": alvo,
                          "v_cramer": vc, "p_valor": p, "gl": gl, "n_categorias": tab.shape[0]})
    assoc = pd.DataFrame(assoc)
    assoc.to_csv(pasta / "associacao_variaveis.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")

    fig, eixos = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=150, facecolor=rf.SUPERFICIE)
    for ax, (alvo, titulo) in zip(eixos, [("tipo_acidente", "Associação com o tipo de acidente"),
                                          ("indica_obito", "Associação com o óbito")]):
        d = assoc[assoc["alvo"] == alvo].sort_values("v_cramer")
        ax.barh(d["rotulo"], d["v_cramer"], color=rf.COR_SERIE, height=0.6)
        for y, v in enumerate(d["v_cramer"]):
            ax.text(v + 0.01, y, virgula(v, 2), va="center", fontsize=8, color=rf.TEXTO)
        ax.set_title(titulo, loc="left", fontsize=10, color=rf.TEXTO)
        ax.set_xlabel("V de Cramér (0 = nenhuma, 1 = total)", color=rf.TEXTO_2, fontsize=8.5)
        ax.set_xlim(0, max(0.5, d["v_cramer"].max() * 1.15))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: virgula(x, 1)))
        ax.xaxis.grid(True, color=rf.GRADE, linewidth=0.8)
        ax.set_axisbelow(True)
        rf._estilo(ax)
    fig.tight_layout()
    fig.savefig(pasta / "associacao_variaveis.png", facecolor=rf.SUPERFICIE)
    plt.close(fig)

    #resumo
    log("\nV de Cramér com o tipo de acidente:\n" + assoc[assoc["alvo"] == "tipo_acidente"]
        .sort_values("v_cramer", ascending=False)[["rotulo", "v_cramer", "n_categorias"]]
        .round(3).to_string(index=False))
    log("\nV de Cramér com o óbito:\n" + assoc[assoc["alvo"] == "indica_obito"]
        .sort_values("v_cramer", ascending=False)[["rotulo", "v_cramer", "n_categorias"]]
        .round(3).to_string(index=False))
    log(f"\nTipo de acidente (total): " + str((df["tipo"].value_counts(normalize=True) * 100).round(1).to_dict()))
    log(f"Óbitos: {int(df['indica_obito'].sum()):,} ({df['indica_obito'].mean() * 100:.2f}%)")
    for v in ["sexo", "faixa_etaria", "uf_empregador", "cbo_grande_grupo"]:
        log(f"\nTipo dentro de {v} (%):\n" + (pd.crosstab(df[v], df["tipo"], normalize="index")[tipos] * 100)
            .round(1).to_string())
    for t in tipos:
        top = df[df["tipo"] == t]["agente_causador"].value_counts(normalize=True).head(5) * 100
        log(f"\nAgentes causadores mais comuns em {t} (%): " + "; ".join(f"{k} {virgula(x)}" for k, x in top.items()))
        top = df[df["tipo"] == t]["cnae_divisao"].value_counts(normalize=True).head(5) * 100
        log(f"Divisões CNAE mais comuns em {t} (%): " + "; ".join(f"{k} {virgula(x)}" for k, x in top.items()))
    log.gravar(pasta / "resumo_descritiva.txt")
    log(f"\nPronto. Arquivos em {pasta}")


if __name__ == "__main__":
    main()
