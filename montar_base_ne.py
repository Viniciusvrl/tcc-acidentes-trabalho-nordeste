

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Pastas padrão;
PASTA_PROJETO = Path(__file__).resolve().parent   # pasta onde estão os scripts
SAIDA_PADRAO = PASTA_PROJETO / "saida"
ENTRADA_PADRAO = SAIDA_PADRAO / "cat_limpa.parquet"   # gerado pelo limpeza_cat.py


JANELAS = [
    ("2020-01", "2021-08"),
    ("2022-06", "2022-09"),
    ("2023-02", "2024-07"),
    ("2025-06", "2025-10"),
]

# Colunas que não servem
COLS_DESCARTAR = [
    "uf_acidente", "regiao_acidente", "uf_acidente_rotulo", "uf_empregador_rotulo",
    "data_despacho_beneficio", "data_afastamento", "tipo_empregador", "cnpj_empregador",
    "munic_empregador", "regiao_empregador", "origem_cadastro", "especie_beneficio",
    "filiacao_segurado",   # constantes ou quase constantes no Nordeste
]

COLS_CATEGORICAS = [
    "agente_causador", "natureza_lesao", "parte_corpo", "sexo", "emitente_cat",
    "cnae_divisao", "cbo_grande_grupo", "cid_capitulo", "uf_empregador",
]
COLS_NUMERICAS = ["idade", "mes", "dia_semana"]


COLS_COMPLETAS = ["agente_causador", "cbo", "cid", "cnae", "sexo", "idade", "natureza_lesao",
                  "parte_corpo", "emitente_cat", "tipo_acidente", "indica_obito",
                  "munic_empregador_cod", "uf_empregador"]

CORTE_TEMPORAL = "2023-02"   # treino: antes; teste: a partir daqui


def ler(caminho: Path) -> pd.DataFrame:
    return pd.read_parquet(caminho)


def montar(df: pd.DataFrame, log: list[str], so_janelas: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["ym"] = df["ano"].astype("Int64").astype(str) + "-" + df["mes"].astype("Int64").astype(str).str.zfill(2)
    log.append(f"Base limpa: {len(df):,} linhas")


    estavel = pd.Series(False, index=df.index)
    for ini, fim in JANELAS:
        estavel |= df["ym"].between(ini, fim)
    log.append(f"Linhas nos meses estáveis {JANELAS}: {int(estavel.sum()):,} de {len(df):,}")
    if so_janelas:
        df = df[estavel]
        estavel = estavel[df.index]
        log.append("Opção --so-janelas-estaveis: meses fora das janelas excluídos.")
    else:
        log.append("Todos os meses mantidos; os de cobertura incompleta ficam marcados (mes_estavel = 0).")

    sem_uf = int(df["uf_empregador"].isna().sum())
    ne = df["regiao_empregador"] == "Nordeste"
    df = df[ne].copy()
    estavel = estavel[df.index]
    log.append(f"Empregador no Nordeste: {len(df):,} linhas "
               f"(registros sem UF do empregador, fora do recorte: {sem_uf:,})")
    df["mes_estavel"] = estavel.astype("Int8")

    df = df.drop(columns=[c for c in COLS_DESCARTAR if c in df.columns])


    chave = [c for c in df.columns if c not in ("arquivo", "layout_origem")]
    dup = df.duplicated(subset=chave)
    log.append(f"Duplicatas removidas (colunas finais): {int(dup.sum()):,}")
    df = df[~dup]

    
    df["mes_dezembro"] = (df["mes"] == 12).astype("Int8")
    # último mês com cobertura regular (os meses seguintes, quase vazios, já saem como mes_estavel = 0)
    ym_ultimo = df.loc[df["mes_estavel"] == 1, "ym"].max() if (df["mes_estavel"] == 1).any() else df["ym"].max()
    df["ultimo_mes_dados"] = (df["ym"] == ym_ultimo).astype("Int8")
    log.append(f"Linhas em dezembro: {int(df['mes_dezembro'].sum()):,} | "
               f"no último mês com cobertura regular ({ym_ultimo}): {int(df['ultimo_mes_dados'].sum()):,}")

    janela_id = pd.Series(0, index=df.index, dtype="float")   # 0 = fora das janelas estáveis
    for i, (ini, fim) in enumerate(JANELAS, start=1):
        janela_id[df["ym"].between(ini, fim)] = i
    df["janela"] = janela_id.astype("Int8")
    return df.reset_index(drop=True)


def remover_ausentes(df: pd.DataFrame, log: list[str], tolerar: tuple[str, ...] = ()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove registros com dado faltante em COLS_COMPLETAS e mede o efeito sobre a composição da base."""
    cols = [c for c in COLS_COMPLETAS if c in df.columns and c not in tolerar]
    if tolerar:
        log.append(f"\nAusência tolerada (não remove o registro) em: {', '.join(tolerar)}")
    falta = df[cols].isna()
    sai = falta.any(axis=1)
    removidos, df2 = df[sai].copy(), df[~sai].copy()
    n0, n1 = len(df), len(df2)
    log.append(f"\nREMOÇÃO DE REGISTROS COM DADO FALTANTE (colunas: {', '.join(cols)})")
    log.append(f"Registros removidos: {n0 - n1:,} de {n0:,} ({(n0 - n1) / n0 * 100:.1f}%) | restam {n1:,}")
    log.append("Faltantes por coluna (um registro pode faltar em mais de uma):\n"
               + falta.sum().sort_values(ascending=False).to_string())

    def comp(d: pd.DataFrame) -> pd.Series:
        return d["tipo_acidente"].value_counts(normalize=True).mul(100).round(1)
    tab = pd.DataFrame({"antes_%": comp(df), "depois_%": comp(df2)})
    tab["variacao_pontos"] = (tab["depois_%"] - tab["antes_%"]).round(1)
    log.append("\nComposição do tipo de acidente antes e depois da remoção:\n" + tab.to_string())
    ob = pd.Series({"antes_%": df["indica_obito"].mean() * 100, "depois_%": df2["indica_obito"].mean() * 100}).round(2)
    log.append("\nTaxa de óbito (%) antes e depois: " + ", ".join(f"{k} {v}" for k, v in ob.items()))
    por_ano = pd.DataFrame({"antes": df.groupby("ano").size(), "depois": df2.groupby("ano").size()})
    por_ano["removidos_%"] = ((1 - por_ano["depois"] / por_ano["antes"]) * 100).round(1)
    log.append("\nRegistros por ano antes e depois:\n" + por_ano.to_string())
    por_uf = pd.DataFrame({"antes": df.groupby("uf_empregador").size(), "depois": df2.groupby("uf_empregador").size()})
    por_uf["removidos_%"] = ((1 - por_uf["depois"] / por_uf["antes"]) * 100).round(1)
    log.append("\nRegistros por UF do empregador antes e depois:\n" + por_uf.to_string())
    return df2.reset_index(drop=True), removidos


def resumo(df: pd.DataFrame, log: list[str]) -> None:
    log.append("\nLinhas por janela (0 = meses fora das janelas estáveis):\n" + df.groupby("janela").size().to_string())
    log.append("\nLinhas por mes_estavel (1 = cobertura regular, 0 = cobertura incompleta):\n"
               + df.groupby("mes_estavel").size().to_string())
    log.append("\nLinhas por mês do acidente (ym) e mes_estavel:\n"
               + df.groupby(["ym", "mes_estavel"]).size().to_string())
    log.append("\nLinhas por ano:\n" + df.groupby("ano").size().to_string())
    log.append("\nLinhas por UF do empregador:\n" + df["uf_empregador"].value_counts().to_string())
    log.append("\ntipo_acidente (inclui ausentes):\n"
               + df["tipo_acidente"].value_counts(dropna=False).to_string())
    log.append("\nindica_obito (inclui ausentes):\n"
               + df["indica_obito"].value_counts(dropna=False).to_string())
    falt = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    log.append("\n% de ausentes por coluna (>0):\n" + falt[falt > 0].to_string())


def preprocessador():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="Nao informado")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                                 sparse_output=True)),
    ])
    num = Pipeline([
        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
        ("escala", StandardScaler()),
    ])
    return ColumnTransformer([("cat", cat, COLS_CATEGORICAS), ("num", num, COLS_NUMERICAS)])


def modelo_exemplo(df: pd.DataFrame, alvo: str = "tipo_acidente") -> None:
    """Regressão logística com validação temporal: treina antes de CORTE_TEMPORAL, testa depois."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
    from sklearn.pipeline import Pipeline

    base = df[df[alvo].notna()].copy()
    X = base[COLS_CATEGORICAS + COLS_NUMERICAS].copy()
    for c in COLS_CATEGORICAS:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    for c in COLS_NUMERICAS:
        X[c] = pd.to_numeric(X[c]).astype(float)
    y = base[alvo].astype(str)

    treino = base["ym"] < CORTE_TEMPORAL
    teste = ~treino
    if treino.sum() == 0 or teste.sum() == 0:
        print(f"\nModelo: sem dados dos dois lados do corte {CORTE_TEMPORAL}; nada a avaliar.")
        return

    modelo = Pipeline([
        ("prep", preprocessador()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    modelo.fit(X[treino], y[treino])
    pred = modelo.predict(X[teste])
    print(f"\nModelo ({alvo}) | treino: {int(treino.sum()):,} (antes de {CORTE_TEMPORAL}) | "
          f"teste: {int(teste.sum()):,}")
    print(f"  F1 macro:            {f1_score(y[teste], pred, average='macro'):.3f}")
    print(f"  Acurácia balanceada: {balanced_accuracy_score(y[teste], pred):.3f}")
    print(classification_report(y[teste], pred, digits=3))


def main() -> None:
    ap = argparse.ArgumentParser(description="Base final do TCC: Nordeste, janelas estáveis")
    ap.add_argument("--entrada", default=ENTRADA_PADRAO, type=Path,
                    help=f"cat_limpa.parquet (padrão: {ENTRADA_PADRAO})")
    ap.add_argument("--saida", default=SAIDA_PADRAO, type=Path,
                    help=f"pasta de saída (padrão: {SAIDA_PADRAO})")
    ap.add_argument("--so-janelas-estaveis", action="store_true",
                    help="mantém só os meses das janelas estáveis (recorte antigo); por padrão mantém todos")
    ap.add_argument("--manter-ausentes", action="store_true",
                    help="não remove registros com dado faltante (por padrão remove)")
    ap.add_argument("--tolerar-ausencia-em", nargs="+", default=[], choices=COLS_COMPLETAS, metavar="COLUNA",
                    help="colunas cuja ausência NÃO remove o registro (ex.: agente_causador cbo)")
    ap.add_argument("--modelo", action="store_true", help="roda o modelo de exemplo")
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    df = montar(ler(args.entrada), log, so_janelas=args.so_janelas_estaveis)
    if args.manter_ausentes:
        log.append("\nOpção --manter-ausentes: registros com dado faltante mantidos.")
    else:
        df, removidos = remover_ausentes(df, log, tuple(args.tolerar_ausencia_em))
        removidos.to_csv(args.saida / "removidos_por_ausencia.csv", index=False, sep=";", encoding="utf-8-sig")
    resumo(df, log)
    df.to_parquet(args.saida / "base_nordeste.parquet", index=False)
    log.append(f"\nBase final: {len(df):,} linhas x {df.shape[1]} colunas -> {args.saida / 'base_nordeste.parquet'}")

    texto = "\n".join(log)
    (args.saida / "resumo_base.txt").write_text(texto, encoding="utf-8")
    print(texto)

    if args.modelo:
        modelo_exemplo(df)


if __name__ == "__main__":
    main()
