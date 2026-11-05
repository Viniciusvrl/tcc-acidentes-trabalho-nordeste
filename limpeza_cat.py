from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Pastas padrão;
PASTA_PROJETO = Path(__file__).resolve().parent   # pasta onde estão os scripts
ENTRADA_PADRAO = PASTA_PROJETO / "entrada"
SAIDA_PADRAO = PASTA_PROJETO / "saida"


# 1. Layouts

# Nomes que começam com "_" são colunas redundantes descartadas após o mapeamento.

LAYOUT_A = [  # 2020: 25 colunas, CBO e CID em código + descrição
    "agente_causador", "_mes_acidente", "cbo", "cbo_desc", "cid", "cid_desc",
    "cnae", "cnae_desc", "emitente_cat", "especie_beneficio", "filiacao_segurado",
    "indica_obito", "munic_empregador", "natureza_lesao", "origem_cadastro",
    "parte_corpo", "sexo", "tipo_acidente", "uf_acidente_rotulo",
    "uf_empregador_rotulo", "data_afastamento", "data_despacho_beneficio",
    "data_acidente", "data_nascimento", "data_emissao_cat",
]

LAYOUT_B = [  # 2021 a 2023: 24 colunas, CBO e CID em campo único "código-descrição"
    "agente_causador", "_mes_acidente", "cbo", "cid", "cnae", "cnae_desc",
    "emitente_cat", "especie_beneficio", "filiacao_segurado", "indica_obito",
    "munic_empregador", "natureza_lesao", "origem_cadastro", "parte_corpo", "sexo",
    "tipo_acidente", "uf_acidente_rotulo", "uf_empregador_rotulo", "_mes_acidente2",
    "data_despacho_beneficio", "data_acidente", "data_nascimento", "data_emissao_cat",
    "cnpj_empregador",
]

LAYOUT_C = [  # 2024 e 2025 (e ~125 mil linhas de 2023): 27 colunas
    "agente_causador", "data_acidente", "cbo", "cbo_desc", "cid", "cid_desc",
    "cnae", "cnae_desc", "emitente_cat", "especie_beneficio", "filiacao_segurado",
    "indica_obito", "munic_empregador", "natureza_lesao", "origem_cadastro",
    "parte_corpo", "sexo", "tipo_acidente", "uf_acidente_rotulo",
    "uf_empregador_rotulo", "data_afastamento", "data_despacho_beneficio",
    "_data_acidente2", "data_nascimento", "data_emissao_cat", "tipo_empregador",
    "cnpj_empregador",
]

LAYOUT_A2 = [  # ~4,8 mil linhas de 2020: como A, mas data completa na 2ª coluna e mês na 21ª
    "agente_causador", "_data_acidente2", "cbo", "cbo_desc", "cid", "cid_desc",
    "cnae", "cnae_desc", "emitente_cat", "especie_beneficio", "filiacao_segurado",
    "indica_obito", "munic_empregador", "natureza_lesao", "origem_cadastro",
    "parte_corpo", "sexo", "tipo_acidente", "uf_acidente_rotulo",
    "uf_empregador_rotulo", "_mes_acidente", "data_despacho_beneficio",
    "data_acidente", "data_nascimento", "data_emissao_cat",
]

LAYOUT_D = [  # ~106 mil linhas de 2023: 24 campos, como C até a UF, sem afastamento e sem CNPJ
    "agente_causador", "data_acidente", "cbo", "cbo_desc", "cid", "cid_desc",
    "cnae", "cnae_desc", "emitente_cat", "especie_beneficio", "filiacao_segurado",
    "indica_obito", "munic_empregador", "natureza_lesao", "origem_cadastro",
    "parte_corpo", "sexo", "tipo_acidente", "uf_acidente_rotulo",
    "uf_empregador_rotulo", "data_despacho_beneficio", "_data_acidente2",
    "data_nascimento", "data_emissao_cat",
]

LAYOUTS = {"A": LAYOUT_A, "A2": LAYOUT_A2, "B": LAYOUT_B, "C": LAYOUT_C, "D": LAYOUT_D}
N_COLS_LEITURA = 30  # folga para linhas com campos extras


# 2. Tabelas de apoio

UF_IBGE = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
    "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP", "41": "PR",
    "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}
UF_NOME_SIGLA = {
    "Rondônia": "RO", "Acre": "AC", "Amazonas": "AM", "Roraima": "RR", "Pará": "PA",
    "Amapá": "AP", "Tocantins": "TO", "Maranhão": "MA", "Piauí": "PI", "Ceará": "CE",
    "Rio Grande Norte": "RN", "Rio Grande do Norte": "RN", "Paraíba": "PB",
    "Pernambuco": "PE", "Alagoas": "AL", "Sergipe": "SE", "Bahia": "BA",
    "Minas Gerais": "MG", "Espírito Santo": "ES", "Rio de Janeiro": "RJ",
    "São Paulo": "SP", "Paraná": "PR", "Santa Catarina": "SC",
    "Rio Grande do Sul": "RS", "Mato Grosso do Sul": "MS", "Mato Grosso": "MT",
    "Goiás": "GO", "Distrito Federal": "DF",
}
REGIAO = {
    **dict.fromkeys(["RO", "AC", "AM", "RR", "PA", "AP", "TO"], "Norte"),
    **dict.fromkeys(["MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE", "BA"], "Nordeste"),
    **dict.fromkeys(["MG", "ES", "RJ", "SP"], "Sudeste"),
    **dict.fromkeys(["PR", "SC", "RS"], "Sul"),
    **dict.fromkeys(["MS", "MT", "GO", "DF"], "Centro-Oeste"),
}

# Valores que a CAT usa para "sem informação"
SENTINELAS = {
    "", "0", "00", "0000/00", "00/00/0000", "000000", "000000-Ignorado", "Ignorado",
    "Não Informado", "Nao Informado", "Zerado", "Indeterminado", "00.000.000.000.000",
    "Não Classificado",
}
RE_SENTINELA = re.compile(r"^\{\s*[ñn]\s*class", re.IGNORECASE)


COLS_TEXTO_TRUNCAR = [
    "agente_causador", "emitente_cat", "especie_beneficio", "filiacao_segurado",
    "natureza_lesao", "origem_cadastro", "parte_corpo", "sexo", "tipo_acidente",
    "tipo_empregador",
]
TAM_TRUNCAR = 20
# Colunas descritivas cujo rótulo completo será guardado num dicionário antes do corte
COLS_DICIONARIO = ["agente_causador", "natureza_lesao", "parte_corpo"]

COLS_DATA = ["data_acidente", "data_nascimento", "data_emissao_cat",
             "data_afastamento", "data_despacho_beneficio"]



# 3. Leitura


def listar_arquivos(entrada: Path) -> list[tuple[str, bytes]]:

    def eh_cat(nome: str) -> bool:
        n = nome.lower()
        return n.endswith(".csv") and "cat" in n and "limpa" not in n

    arquivos = []
    if entrada.suffix.lower() == ".zip":
        with zipfile.ZipFile(entrada) as z:
            for nome in sorted(z.namelist()):
                if eh_cat(Path(nome).name):
                    arquivos.append((Path(nome).name, z.read(nome)))
    else:
        for p in sorted(entrada.glob("*.csv")):
            if eh_cat(p.name):
                arquivos.append((p.name, p.read_bytes()))
    if not arquivos:
        sys.exit(f"Nenhum CSV encontrado em {entrada}")
    return arquivos


def layout_do_cabecalho(cabecalho: list[str]) -> str:
    nomeados = [c for c in cabecalho if c.strip()]
    n = len(nomeados)
    return {25: "A", 24: "B", 27: "C"}.get(n, "C" if n > 25 else "B")


RE_DATA_COMPLETA = re.compile(r"^\d{2}/\d{2}/\d{4}$")
RE_MES_ANO = re.compile(r"^\d{4}/\d{2}$")


def ler_arquivo(nome: str, conteudo: bytes, log: list[str]) -> pd.DataFrame:
    texto = conteudo.decode("latin1")
    cabecalho = texto.splitlines()[0].split(";")
    layout_base = layout_do_cabecalho(cabecalho)

    bruto = pd.read_csv(
        io.StringIO(texto), sep=";", header=None, skiprows=1, dtype=str,
        names=range(N_COLS_LEITURA), keep_default_na=False, engine="python",
    )

    bruto = bruto.fillna("").apply(lambda s: s.str.strip(" \t\r\n"))


    col1 = bruto[1]
    preenchido = bruto.ne("")
    n_campos = preenchido.iloc[:, ::-1].idxmax(axis=1) + 1  # posição do último campo não vazio
    data_completa = col1.str.match(RE_DATA_COMPLETA)
    mes_ano = col1.str.match(RE_MES_ANO)

    layout_linha = pd.Series(layout_base, index=bruto.index, dtype=object)
    if layout_base == "A":
        layout_linha[mes_ano] = "A"
        layout_linha[data_completa] = "A2"
    else:
        layout_linha[mes_ano] = "B"
        layout_linha[data_completa & (n_campos >= 25)] = "C"
        layout_linha[data_completa & (n_campos <= 24)] = "D"

    partes = []
    for lay, idx in layout_linha.groupby(layout_linha).groups.items():
        cols = LAYOUTS[lay]
        bloco = bruto.loc[idx, list(range(len(cols)))].copy()
        bloco.columns = cols
        bloco["layout_origem"] = lay
        partes.append(bloco)
    df = pd.concat(partes).sort_index()
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    df["arquivo"] = nome

    cont = layout_linha.value_counts().to_dict()
    log.append(f"{nome}: {len(df):,} linhas | layout do cabeçalho={layout_base} | linhas por layout={cont}")
    return df



# 4. Limpeza de valores


def corrigir_mojibake(s: pd.Series) -> pd.Series:
    """Conserta textos UTF-8 lidos como latin1 (ex.: 'NÃ£o' -> 'Não')."""
    def fix(v):
        if isinstance(v, str) and "Ã" in v:
            try:
                return v.encode("latin1").decode("utf8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                return v
        return v
    return s.map(fix)


def para_nan(s: pd.Series) -> pd.Series:
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    mask = s.isin(SENTINELAS) | s.str.match(RE_SENTINELA, na=False)
    return s.mask(mask)


def extrair_codigo_cbo(s: pd.Series) -> pd.Series:
    return s.str.extract(r"^(\d{6})", expand=False)


def extrair_codigo_cid(s: pd.Series) -> pd.Series:
    # "S61.0 Ferim..." e "S610" viram "S610"
    m = s.str.extract(r"^([A-Z]\d{2})\.?(\d?)", expand=True)
    return (m[0] + m[1].fillna("")).where(m[0].notna())


def extrair_codigo_cnae(s: pd.Series) -> pd.Series:
    cod = s.str.extract(r"^(\d{3,5})", expand=False)
    cod = cod.where(cod.str.lstrip("0") != "")
    return cod.str.zfill(4)


def converter_datas(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    texto = [c for c in df.columns if c not in ("arquivo", "layout_origem")]
    for c in texto:
        df[c] = para_nan(corrigir_mojibake(df[c]))

    df["cbo"] = extrair_codigo_cbo(df["cbo"])
    df["cid"] = extrair_codigo_cid(df["cid"])
    df["cnae"] = extrair_codigo_cnae(df["cnae"])
    df["munic_empregador_cod"] = df["munic_empregador"].str.extract(r"^(\d{6})", expand=False)
    df.loc[df["munic_empregador_cod"] == "000000", "munic_empregador_cod"] = np.nan

    completos = df[[c for c in COLS_DICIONARIO if c in df]].copy()   # antes do corte em 20
    for c in COLS_TEXTO_TRUNCAR:
        if c in df:
            df[c] = df[c].str.slice(0, TAM_TRUNCAR).str.strip()

    df["indica_obito"] = df["indica_obito"].map({"Sim": 1, "Não": 0, "Nao": 0}).astype("Int8")

    for c in COLS_DATA:
        if c in df:
            df[c] = converter_datas(df[c])


    df = df.drop(columns=["cbo_desc", "cid_desc", "cnae_desc"], errors="ignore")
    return df, completos


def construir_dicionario(completos: pd.DataFrame) -> pd.DataFrame:

    linhas = []
    for c in completos.columns:
        s = completos[c].dropna()
        corte = s.str.slice(0, TAM_TRUNCAR).str.strip()
        contagem = pd.DataFrame({"corte": corte, "completo": s}).value_counts().reset_index(name="n")
        for k, g in contagem.groupby("corte"):
            mais_longo = g.loc[g["completo"].str.len().idxmax(), "completo"]
            ambiguo = int(any(not mais_longo.startswith(v) for v in g["completo"]))
            linhas.append({"coluna": c, "rotulo_20": k, "rotulo_completo": mais_longo,
                           "n_linhas": int(g["n"].sum()), "n_variantes": len(g), "ambiguo": ambiguo})
    return pd.DataFrame(linhas).sort_values(["coluna", "n_linhas"], ascending=[True, False])


def linhas_estruturalmente_invalidas(df: pd.DataFrame) -> pd.Series:
    """Linhas cujo conteúdo não bate com o layout atribuído (deslocamento não previsto)."""
    sexo_ok = df["sexo"].isna() | df["sexo"].isin(["Masculino", "Feminino"])
    tipo_ok = df["tipo_acidente"].isna() | df["tipo_acidente"].isin(["Típico", "Trajeto", "Doença"])
    data_ok = df["data_acidente"].notna()
    return ~(sexo_ok & tipo_ok & data_ok)



# 5. Diagnóstico da UF


def diagnosticar_uf(df: pd.DataFrame, log: list[str]) -> tuple[pd.DataFrame, bool]:

   # Cruza o rótulo de 'UF Munic. Acidente' com a UF do empregador

    uf_emp = df["munic_empregador_cod"].str[:2].map(UF_IBGE)
    uf_acid = df["uf_acidente_rotulo"].map(UF_NOME_SIGLA)
    ct = pd.crosstab(uf_acid, uf_emp)
    resumo = pd.DataFrame({"linhas": ct.sum(axis=1)})
    resumo.index.name = "uf_acidente_rotulo"
    resumo["uf_empregador_mais_frequente"] = ct.idxmax(axis=1)
    resumo["proporcao_uf_mais_frequente"] = (ct.max(axis=1) / ct.sum(axis=1)).round(4)
    resumo["proporcao_mesma_uf"] = [
        round(ct.loc[uf, uf] / ct.loc[uf].sum(), 4) if uf in ct.columns else 0.0
        for uf in resumo.index
    ]
    resumo["rotulo_confere"] = resumo.index == resumo["uf_empregador_mais_frequente"]

    comparaveis = uf_acid.notna() & uf_emp.notna()
    diferentes = int((comparaveis & (uf_acid != uf_emp)).sum())
    total = int(comparaveis.sum())
    pct = diferentes / total * 100 if total else 0.0
    log.append(f"UF acidente (rótulo) x UF empregador (IBGE): {diferentes:,} de {total:,} "
               f"({pct:.1f}%) são diferentes.")

    grandes = resumo[resumo["linhas"] >= 100]
    trocados = grandes[(~grandes["rotulo_confere"]) & (grandes["proporcao_uf_mais_frequente"] > 0.95)]
    rotulos_trocados = bool(len(grandes) and len(trocados) >= len(grandes) / 2)
    if rotulos_trocados:
        log.append(
            "ATENÇÃO: os rótulos de 'UF Munic. Acidente' estão trocados "
            f"({len(trocados)} de {len(grandes)} rótulos apontam quase sempre para outra UF única). "
            "A coluna NÃO é usada: uf_acidente e regiao_acidente ficam vazias. "
            "A localização da análise é a UF do empregador (código IBGE do município)."
        )
    else:
        log.append("Rótulos de UF do acidente compatíveis com a UF do empregador: uf_acidente mantida.")
    return resumo.sort_values("linhas", ascending=False), rotulos_trocados



# 6. Variáveis derivadas


def derivar(df: pd.DataFrame, rotulos_trocados: bool) -> pd.DataFrame:
    # UF do empregador: vem do código IBGE do município, que é confiável.
    df["uf_empregador"] = df["munic_empregador_cod"].str[:2].map(UF_IBGE)
    df["regiao_empregador"] = df["uf_empregador"].map(REGIAO)
    # UF do local do acidente: só se o rótulo da CAT passou no diagnóstico.
    if rotulos_trocados:
        df["uf_acidente"] = pd.Series(pd.NA, index=df.index, dtype="object")
    else:
        df["uf_acidente"] = df["uf_acidente_rotulo"].map(UF_NOME_SIGLA)
    df["regiao_acidente"] = df["uf_acidente"].map(REGIAO)

    df["ano"] = df["data_acidente"].dt.year.astype("Int16")
    df["mes"] = df["data_acidente"].dt.month.astype("Int8")
    df["dia_semana"] = df["data_acidente"].dt.dayofweek.astype("Int8")  # 0 = segunda

    idade = (df["data_acidente"] - df["data_nascimento"]).dt.days / 365.25
    df["idade"] = idade.where(idade.between(14, 90)).round(1)
    df["faixa_etaria"] = pd.cut(
        df["idade"], bins=[13, 24, 34, 44, 54, 64, 90],
        labels=["14-24", "25-34", "35-44", "45-54", "55-64", "65+"],
    ).astype("string")

    # Layout D (arquivos de jun a out/2023): a coluna lida como data de emissão repete a data do
    # acidente em 100% das linhas, então a emissão desses registros é desconhecida (NaT).
    eh_d = df["layout_origem"] == "D"
    df.loc[eh_d, "data_emissao_cat"] = pd.NaT

    dias = (df["data_emissao_cat"] - df["data_acidente"]).dt.days
    df["dias_ate_emissao"] = dias.where(dias.between(0, 3650))

    df["cnae_divisao"] = df["cnae"].str[:2]
    df["cbo_grande_grupo"] = df["cbo"].str[:1]
    df["cid_capitulo"] = df["cid"].str[:1]
    return df


def diagnostico_emissao(df: pd.DataFrame) -> pd.DataFrame:
    """Por arquivo: como a data de emissão se comporta (ajuda a achar layout lido errado)."""
    g = df.groupby("arquivo")["dias_ate_emissao"]
    return pd.DataFrame({
        "linhas": df.groupby("arquivo").size(),
        "mediana_dias": g.median(),
        "pct_mesmo_dia": g.apply(lambda s: round((s == 0).mean() * 100, 1)),
        "pct_atraso_15d": g.apply(lambda s: round((s > 15).mean() * 100, 1)),
        "layout_mais_comum": df.groupby("arquivo")["layout_origem"].agg(lambda s: s.value_counts().index[0]),
    })



# 7. Relatórios de faltantes


def relatorio_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in ("arquivo", "layout_origem", "ano")]
    rel = df.groupby("ano")[cols].apply(lambda g: g.isna().mean() * 100).T.round(1)
    rel["total"] = (df[cols].isna().mean() * 100).round(1)
    rel["sugestao"] = rel["total"].map(sugerir)
    return rel.sort_values("total", ascending=False)


def sugerir(pct: float) -> str:
    if pct >= 60:
        return "excluir coluna"
    if pct >= 2:
        return "manter: categoria 'Não informado' ou imputação + indicador"
    if pct > 0:
        return "imputar ou excluir linhas (documentar)"
    return "sem faltantes"


def ausencia_vs_alvo(df: pd.DataFrame, alvo: str) -> pd.DataFrame:
    """Qui-quadrado entre 'variável está ausente?' e o alvo. p < 0,05 indica ausência informativa."""
    from scipy.stats import chi2_contingency

    linhas = []
    base = df[df[alvo].notna()]
    for c in base.columns:
        if c in (alvo, "arquivo", "layout_origem"):
            continue
        ausente = base[c].isna()
        if ausente.mean() < 0.005 or ausente.mean() > 0.995:
            continue
        tab = pd.crosstab(ausente, base[alvo])
        chi2, p, _, _ = chi2_contingency(tab)
        n = tab.values.sum()
        v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
        linhas.append({"variavel": c, "pct_ausente": round(ausente.mean() * 100, 1),
                       "chi2": round(chi2, 1), "p_valor": p, "v_de_cramer": round(v, 3)})
    return pd.DataFrame(linhas).sort_values("v_de_cramer", ascending=False)



# 8. Pré-processamento


COLS_CATEGORICAS_MODELO = [
    "agente_causador", "natureza_lesao", "parte_corpo", "sexo", "emitente_cat",
    "cnae_divisao", "cbo_grande_grupo", "cid_capitulo", "uf_empregador",
]
# dias_ate_emissao fora
COLS_NUMERICAS_MODELO = ["idade", "mes", "dia_semana"]

# Não entram no modelo: identificadores, colunas quase vazias e datas brutas
COLS_FORA_DO_MODELO = [
    "cnpj_empregador", "data_despacho_beneficio", "data_afastamento", "munic_empregador",
    "uf_acidente_rotulo", "uf_empregador_rotulo", "uf_acidente", "regiao_acidente",
    "data_acidente", "data_nascimento", "data_emissao_cat", "arquivo", "layout_origem",
]


def montar_preprocessador(cat_cols=None, num_cols=None):
    """
    ColumnTransformer que trata faltantes dentro do Pipeline.
    Use sempre com fit no treino (ou dentro de cross_val_score), nunca na base inteira.
      categóricas: NaN vira 'Nao informado'; categorias raras são agrupadas
      numéricas: mediana + coluna indicadora de valor imputado
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat_cols = cat_cols or COLS_CATEGORICAS_MODELO
    num_cols = num_cols or COLS_NUMERICAS_MODELO

    cat = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="Nao informado")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005,
                                 sparse_output=True)),
    ])
    num = Pipeline([
        ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
        ("escala", StandardScaler()),
    ])
    return ColumnTransformer([("cat", cat, cat_cols), ("num", num, num_cols)])


def exemplo_modelo(df: pd.DataFrame, alvo: str = "tipo_acidente", amostra: int = 100_000):
    """Exemplo curto: regressão logística com validação cruzada estratificada."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.pipeline import Pipeline

    base = df[df[alvo].notna()]
    if len(base) > amostra:
        base = base.sample(amostra, random_state=42)
    X = base[COLS_CATEGORICAS_MODELO + COLS_NUMERICAS_MODELO].copy()
    for c in COLS_CATEGORICAS_MODELO:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    for c in COLS_NUMERICAS_MODELO:
        X[c] = pd.to_numeric(X[c]).astype(float)
    y = base[alvo].astype(str)

    modelo = Pipeline([
        ("prep", montar_preprocessador()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    res = cross_validate(modelo, X, y, cv=cv, scoring=["f1_macro", "balanced_accuracy"], n_jobs=-1)
    print(f"\nExemplo ({alvo}, {len(base):,} linhas, 5 folds):")
    print(f"  F1 macro:           {res['test_f1_macro'].mean():.3f} ± {res['test_f1_macro'].std():.3f}")
    print(f"  Acurácia balanceada: {res['test_balanced_accuracy'].mean():.3f}")


# ---------------------------------------------------------------------------
# 9. Execução
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Limpeza das CATs 2020-2025")
    ap.add_argument("--entrada", default=ENTRADA_PADRAO, type=Path,
                    help=f"pasta com os CSVs ou arquivo .zip (padrão: {ENTRADA_PADRAO})")
    ap.add_argument("--saida", default=SAIDA_PADRAO, type=Path,
                    help=f"pasta de saída (padrão: {SAIDA_PADRAO})")
    ap.add_argument("--somente-nordeste", action="store_true",
                    help="mantém só registros cujo EMPREGADOR (UF pelo código IBGE) está no Nordeste")
    ap.add_argument("--alvo", default="tipo_acidente",
                    help="variável alvo para o teste de ausência (ex.: tipo_acidente, indica_obito)")
    ap.add_argument("--exemplo-modelo", action="store_true",
                    help="regressão logística com validação cruzada aleatória (só um teste rápido; "
                         "para o TCC use a validação temporal do montar_base_ne.py)")
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    # leitura
    arquivos = listar_arquivos(args.entrada)
    log.append(f"Entrada: {args.entrada} | {len(arquivos)} arquivos de CAT")
    dfs = [ler_arquivo(n, c, log) for n, c in arquivos]
    df = pd.concat(dfs, ignore_index=True)
    log.append(f"Total lido: {len(df):,} linhas")

    # limpeza
    df, completos = limpar(df)
    dic = construir_dicionario(completos)
    dic.to_csv(args.saida / "dicionario_rotulos.csv", index=False, encoding="utf-8-sig")
    log.append(f"Dicionário de rótulos: {len(dic):,} entradas, {int(dic['ambiguo'].sum())} ambíguas "
               f"-> {args.saida / 'dicionario_rotulos.csv'}")
    del completos
    invalidas = linhas_estruturalmente_invalidas(df)
    log.append(f"Linhas com estrutura inválida removidas: {invalidas.sum():,}")
    df = df[~invalidas]

    # duplicatas: comparadas sem as colunas de controle
    cols_conteudo = [c for c in df.columns if c not in ("arquivo", "layout_origem")]
    dup = df.duplicated(subset=cols_conteudo)
    por_arquivo = df.loc[dup, "arquivo"].value_counts().to_dict()
    log.append(f"Duplicatas removidas: {dup.sum():,} | por arquivo: {por_arquivo}")
    df = df[~dup].reset_index(drop=True)

    # UF: diagnostica o rótulo do acidente; a localização usada é a UF do empregador (IBGE)
    diag, rotulos_trocados = diagnosticar_uf(df, log)
    diag.to_csv(args.saida / "diagnostico_uf.csv", encoding="utf-8-sig")
    df = derivar(df, rotulos_trocados)

    # data de emissão: os arquivos só trazem CATs emitidas no mesmo ano do acidente?
    ok = df["data_acidente"].notna() & df["data_emissao_cat"].notna()
    cruza = int((df.loc[ok, "data_acidente"].dt.year != df.loc[ok, "data_emissao_cat"].dt.year).sum())
    log.append(f"Acidente e emissão da CAT em anos diferentes: {cruza:,} de {int(ok.sum()):,}")
    diagnostico_emissao(df).to_csv(args.saida / "diagnostico_emissao.csv", encoding="utf-8-sig")

    sem_uf = int(df["uf_empregador"].isna().sum())
    log.append(f"Linhas sem UF do empregador (município ausente ou inválido): {sem_uf:,}")
    reg = df["regiao_empregador"].value_counts(normalize=True).mul(100).round(1).to_dict()
    log.append(f"Distribuição por região do empregador (%): {reg}")
    if args.somente_nordeste:
        antes = len(df)
        df = df[df["regiao_empregador"] == "Nordeste"].reset_index(drop=True)
        log.append(f"Filtro Nordeste (pela UF do empregador): {antes:,} -> {len(df):,} linhas")
        log.append("UF por estado do Nordeste: "
                   f"{df['uf_empregador'].value_counts().to_dict()}")

    meses = {int(k): int(v) for k, v in df.groupby("ano")["mes"].nunique().items()}
    por_ano = {int(k): int(v) for k, v in df["ano"].value_counts().sort_index().items()}
    log.append(f"Linhas por ano: {por_ano}")
    log.append(f"Meses presentes por ano: {meses}")

    # relatórios
    rel = relatorio_faltantes(df)
    rel.to_csv(args.saida / "relatorio_faltantes.csv", encoding="utf-8-sig")
    if args.alvo in df and df[args.alvo].nunique() > 1:
        aus = ausencia_vs_alvo(df.drop(columns=COLS_FORA_DO_MODELO, errors="ignore"), args.alvo)
        aus.to_csv(args.saida / "ausencia_vs_alvo.csv", index=False, encoding="utf-8-sig")

    # salvar
    for c in df.columns:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].astype("string")
    df.to_parquet(args.saida / "cat_limpa.parquet", index=False)
    log.append(f"Base final: {len(df):,} linhas x {df.shape[1]} colunas -> {args.saida / 'cat_limpa.parquet'}")

    texto_log = "\n".join(log)
    (args.saida / "log_limpeza.txt").write_text(texto_log, encoding="utf-8")
    print(texto_log)
    print("\nFaltantes (% total, 15 maiores):")
    print(rel[["total", "sugestao"]].head(15).to_string())

    if args.exemplo_modelo:
        exemplo_modelo(df, alvo=args.alvo if args.alvo != "indica_obito" else "tipo_acidente")


if __name__ == "__main__":
    main()

