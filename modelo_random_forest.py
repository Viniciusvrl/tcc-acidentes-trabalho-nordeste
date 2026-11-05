

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janela: só grava os PNG
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401  (formatador de eixo)
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------- pastas
PASTA_PROJETO = Path(__file__).resolve().parent   # pasta onde estão os scripts
BASES_PADRAO = [   # o primeiro arquivo que existir é usado
    PASTA_PROJETO / "saida" / "base_nordeste.parquet",
    PASTA_PROJETO / "entrada" / "saida" / "base_nordeste.parquet",
    PASTA_PROJETO / "saida" / "base_nordeste.csv",
    PASTA_PROJETO / "entrada" / "saida" / "base_nordeste.csv",
]

# ----------------------------------------------------------------------------- variáveis
CATEGORICAS_COMPLETA = [
    "agente_causador", "natureza_lesao", "parte_corpo", "sexo", "emitente_cat",
    "cnae_divisao", "cbo_grande_grupo", "cid_capitulo", "uf_empregador",
]
NUMERICAS_COMPLETA = ["idade", "mes", "dia_semana"]
CATEGORICAS_PERFIL = ["sexo", "cnae_divisao", "cbo_grande_grupo", "uf_empregador"]
NUMERICAS_PERFIL = ["idade", "mes", "dia_semana"]

VARIAVEIS = {
    "completa": (CATEGORICAS_COMPLETA, NUMERICAS_COMPLETA),
    "perfil": (CATEGORICAS_PERFIL, NUMERICAS_PERFIL),
}
NOMES_LEGIVEIS = {
    "agente_causador": "Agente causador", "natureza_lesao": "Natureza da lesão",
    "parte_corpo": "Parte do corpo", "sexo": "Sexo", "emitente_cat": "Emitente da CAT",
    "cnae_divisao": "Atividade econômica (CNAE)", "cbo_grande_grupo": "Ocupação (CBO)",
    "cid_capitulo": "Diagnóstico (capítulo CID)", "uf_empregador": "Estado do empregador",
    "idade": "Idade", "mes": "Mês do acidente", "dia_semana": "Dia da semana",
}

# ----------------------------------------------------------------------------- cores (gráficos)
COR_SERIE = "#2a78d6"                    # azul: uma só série
RAMPA_AZUL = ["#f1f6fd", "#2a78d6", "#12386b"]   # sequencial de um só tom, claro -> escuro
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
SUPERFICIE = "#fcfcfb"
GRADE = "#e4e3df"


class Log:
    """Imprime e guarda tudo para gravar em log_random_forest.txt."""

    def __init__(self) -> None:
        self.linhas: list[str] = []

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.linhas.append(msg)

    def gravar(self, caminho: Path) -> None:
        caminho.write_text("\n".join(self.linhas), encoding="utf-8")


# ----------------------------------------------------------------------------- dados
def achar_base(arg: Path | None) -> Path:
    if arg is not None:
        if not arg.exists():
            sys.exit(f"Arquivo não encontrado: {arg}")
        return arg
    for c in BASES_PADRAO:
        if c.exists():
            return c
    sys.exit("Não encontrei a base. Procurei em:\n  " + "\n  ".join(map(str, BASES_PADRAO))
             + "\nUse --entrada para indicar o caminho do base_nordeste.parquet (ou .csv).")


def ler_base(caminho: Path) -> pd.DataFrame:
    if caminho.suffix.lower() == ".csv":
        return pd.read_csv(caminho, sep=";", encoding="utf-8-sig", low_memory=False)
    return pd.read_parquet(caminho)


def preparar(df: pd.DataFrame, log: Log) -> pd.DataFrame:
    precisa = set(CATEGORICAS_COMPLETA + NUMERICAS_COMPLETA
                  + ["tipo_acidente", "indica_obito", "ym", "ano", "mes_estavel", "data_acidente"])
    faltam = sorted(precisa - set(df.columns))
    if faltam:
        sys.exit(f"Colunas ausentes na base: {faltam}")
    df = df.copy()
    df["data_acidente"] = pd.to_datetime(df["data_acidente"], errors="coerce")
    df = df.sort_values("data_acidente", kind="stable").reset_index(drop=True)  # ordem do tempo
    for c in CATEGORICAS_COMPLETA:
        df[c] = df[c].astype("string").fillna("(ausente)").astype(str)
    for c in NUMERICAS_COMPLETA:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["indica_obito"] = pd.to_numeric(df["indica_obito"], errors="coerce").fillna(0).astype(int)
    log(f"Base: {len(df):,} linhas | {df['ym'].min()} a {df['ym'].max()}")
    return df


# ----------------------------------------------------------------------------- modelo
def montar_modelo(cats: list[str], nums: list[str], n_arvores: int, seed: int,
                  n_jobs: int, **params) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               dtype=np.float32), cats),
        ("num", "passthrough", nums),
    ])
    rf = RandomForestClassifier(
        n_estimators=n_arvores, class_weight="balanced_subsample",
        n_jobs=n_jobs, random_state=seed, **params)
    return Pipeline([("pre", pre), ("rf", rf)])


def ajustar_hiperparametros(X: pd.DataFrame, y: pd.Series, cats, nums, scoring: str,
                            n_amostra: int, n_iter: int, seed: int, n_jobs: int, log: Log) -> dict:
    """Busca aleatória com validação cruzada temporal, só com dados de treino (já em ordem de data)."""
    idx = np.linspace(0, len(X) - 1, min(n_amostra, len(X))).astype(int)  # amostra sistemática
    Xs, ys = X.iloc[idx], y.iloc[idx]
    espaco = {
        "rf__max_depth": [10, 20, 30, None],
        "rf__min_samples_leaf": [1, 3, 5, 10],
        "rf__max_features": ["sqrt", 0.3, 0.5],
    }
    busca = RandomizedSearchCV(
        montar_modelo(cats, nums, 150, seed, n_jobs=n_jobs),
        espaco, n_iter=n_iter, scoring=scoring, cv=TimeSeriesSplit(n_splits=3),
        random_state=seed, n_jobs=1, refit=False, error_score=np.nan)
    t0 = time.time()
    busca.fit(Xs, ys)
    melhor = {k.replace("rf__", ""): v for k, v in busca.best_params_.items()}
    log(f"  Hiperparâmetros (amostra de {len(Xs):,} linhas do treino, {n_iter} combinações, "
        f"{scoring} = {busca.best_score_:.4f}, {time.time() - t0:.0f} s): {melhor}")
    return melhor


# ----------------------------------------------------------------------------- métricas
def metricas_tipo(y_true, y_pred, classes) -> dict:
    d = {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "acuracia_balanceada": balanced_accuracy_score(y_true, y_pred),
        "acuracia": accuracy_score(y_true, y_pred),
    }
    rec = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    pre = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    for c, r, p in zip(classes, rec, pre):
        d[f"revocacao_{c}"] = r
        d[f"precisao_{c}"] = p
    return d


def metricas_obito(y_true, proba, limiar: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    pred = (proba >= limiar).astype(int)
    n_pos = int(y_true.sum())
    d = {"prevalencia": y_true.mean(), "n_obitos": n_pos}
    if n_pos == 0 or n_pos == len(y_true):
        return d | {"pr_auc": np.nan, "roc_auc": np.nan}
    d["pr_auc"] = average_precision_score(y_true, proba)
    d["roc_auc"] = roc_auc_score(y_true, proba)
    d[f"precisao_limiar_{limiar}"] = precision_score(y_true, pred, zero_division=0)
    d[f"revocacao_limiar_{limiar}"] = recall_score(y_true, pred, zero_division=0)
    d[f"f1_limiar_{limiar}"] = f1_score(y_true, pred, zero_division=0)
    top = max(1, int(round(0.05 * len(y_true))))          # 5% de registros de maior risco
    ordem = np.argsort(-proba)[:top]
    d["revocacao_nos_5pct_de_maior_risco"] = y_true[ordem].sum() / n_pos
    return d


# ----------------------------------------------------------------------------- gráficos
def _estilo(ax) -> None:
    ax.set_facecolor(SUPERFICIE)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=TEXTO_2, labelsize=9)


def grafico_matriz(cm: np.ndarray, classes: list[str], titulo: str, caminho: Path) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    pct = cm / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    mapa = LinearSegmentedColormap.from_list("azul", RAMPA_AZUL)
    fig, ax = plt.subplots(figsize=(5.4, 4.6), dpi=150, facecolor=SUPERFICIE)
    ax.imshow(pct, cmap=mapa, vmin=0, vmax=100)
    for i in range(len(classes)):
        for j in range(len(classes)):
            claro = pct[i, j] > 55
            txt_pct = f"{pct[i, j]:.1f}".replace(".", ",")
            txt_n = f"{cm[i, j]:,}".replace(",", ".")
            ax.text(j, i, f"{txt_pct}%\n({txt_n})",
                    ha="center", va="center", fontsize=9,
                    color="#ffffff" if claro else TEXTO)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Classe prevista", color=TEXTO_2, fontsize=9)
    ax.set_ylabel("Classe real", color=TEXTO_2, fontsize=9)
    ax.set_title(titulo, color=TEXTO, fontsize=10, loc="left")
    _estilo(ax)
    for lado in ax.spines.values():
        lado.set_visible(False)
    fig.text(0.01, 0.01, "% da classe real (soma de cada linha = 100%); entre parênteses, nº de registros",
             fontsize=7.5, color=TEXTO_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(caminho, facecolor=SUPERFICIE)
    plt.close(fig)


def grafico_importancia(imp: pd.DataFrame, rotulo_metrica: str, titulo: str, caminho: Path) -> None:
    imp = imp.sort_values("importancia_media")
    fig, ax = plt.subplots(figsize=(6.2, 0.38 * len(imp) + 1.3), dpi=150, facecolor=SUPERFICIE)
    y = np.arange(len(imp))
    ax.barh(y, imp["importancia_media"], xerr=imp["desvio"], height=0.55, color=COR_SERIE,
            error_kw={"ecolor": TEXTO_2, "elinewidth": 1, "capsize": 2})
    ax.set_yticks(y, [NOMES_LEGIVEIS.get(v, v) for v in imp["variavel"]])
    ax.set_xlabel(f"Queda em {rotulo_metrica} ao embaralhar a variável (teste)", color=TEXTO_2, fontsize=9)
    ax.set_title(titulo, color=TEXTO, fontsize=10, loc="left")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.3f}".replace(".", ",")))
    ax.xaxis.grid(True, color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(caminho, facecolor=SUPERFICIE)
    plt.close(fig)


# ----------------------------------------------------------------------------- execução de um cenário
def rodar_cenario(df: pd.DataFrame, alvo: str, versao: str, args, pasta: Path, log: Log,
                  resumo: list[dict], parametros: dict) -> None:
    cats, nums = VARIAVEIS[versao]
    cols = cats + nums
    eh_tipo = alvo == "tipo"
    col_alvo = "tipo_acidente" if eh_tipo else "indica_obito"
    scoring = "f1_macro" if eh_tipo else "average_precision"
    marca = f"{alvo}_{versao}"

    treino = df[df["ym"] < args.corte]
    teste = df[df["ym"] >= args.corte]
    if len(treino) == 0 or len(teste) == 0:
        sys.exit(f"--corte {args.corte} deixa treino ou teste vazio.")
    Xtr, ytr = treino[cols], treino[col_alvo]
    Xte, yte = teste[cols], teste[col_alvo]

    log(f"\n{'=' * 78}\nALVO: {col_alvo} | VARIÁVEIS: {versao} ({len(cols)}) | "
        f"treino {len(treino):,} (até {treino['ym'].max()}) | teste {len(teste):,} (de {teste['ym'].min()})\n{'=' * 78}")
    if eh_tipo:
        log("Composição no treino (%): " + str((ytr.value_counts(normalize=True) * 100).round(1).to_dict()))
        log("Composição no teste  (%): " + str((yte.value_counts(normalize=True) * 100).round(1).to_dict()))
    else:
        log(f"Óbitos: treino {int(ytr.sum()):,} ({ytr.mean() * 100:.2f}%) | teste {int(yte.sum()):,} ({yte.mean() * 100:.2f}%)")

    # --- hiperparâmetros (só treino)
    if args.sem_tuning:
        melhores = {"max_depth": None, "min_samples_leaf": 3, "max_features": "sqrt"}
        log(f"  Sem busca de hiperparâmetros; usando {melhores}")
    else:
        melhores = ajustar_hiperparametros(Xtr, ytr, cats, nums, scoring, args.amostra_tuning,
                                           args.n_iter, args.seed, args.n_jobs, log)
    parametros[marca] = {k: (None if v is None else v) for k, v in melhores.items()}

    # --- modelo final
    t0 = time.time()
    modelo = montar_modelo(cats, nums, args.arvores, args.seed, args.n_jobs, **melhores)
    modelo.fit(Xtr, ytr)
    log(f"  Modelo final: {args.arvores} árvores treinadas em {time.time() - t0:.0f} s")
    pred = modelo.predict(Xte)
    proba = modelo.predict_proba(Xte)
    classes = list(modelo.classes_)

    # --- linha de base: sempre prever a classe mais comum do treino
    base = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    pred_base = base.predict(Xte)

    recortes = {"todos_os_meses": np.ones(len(teste), bool),
                "so_cobertura_regular": (teste["mes_estavel"].to_numpy() == 1)}
    for nome_recorte, mascara in recortes.items():
        if mascara.sum() == 0:
            continue
        linha = {"alvo": col_alvo, "variaveis": versao, "teste": nome_recorte,
                 "n_treino": len(treino), "n_teste": int(mascara.sum())}
        if eh_tipo:
            linha |= metricas_tipo(yte[mascara], pred[mascara], classes)
            linha["f1_macro_linha_de_base"] = f1_score(yte[mascara], pred_base[mascara],
                                                       average="macro", zero_division=0)
        else:
            p1 = proba[:, list(classes).index(1)] if 1 in classes else np.zeros(len(teste))
            linha |= metricas_obito(yte[mascara].to_numpy(), p1[mascara])
        resumo.append(linha)
        if eh_tipo:
            log(f"  [{nome_recorte}] n={linha['n_teste']:,} | F1 macro {linha['f1_macro']:.3f} "
                f"(linha de base {linha['f1_macro_linha_de_base']:.3f}) | acurácia balanceada "
                f"{linha['acuracia_balanceada']:.3f} | acurácia {linha['acuracia']:.3f}")
        else:
            log(f"  [{nome_recorte}] n={linha['n_teste']:,} | PR-AUC {linha['pr_auc']:.3f} "
                f"(acaso = {linha['prevalencia']:.4f}) | ROC-AUC {linha['roc_auc']:.3f} | "
                f"revocação nos 5% de maior risco {linha.get('revocacao_nos_5pct_de_maior_risco', float('nan')):.3f}")

    # --- relatório por classe e matriz de confusão (teste completo)
    if eh_tipo:
        rel = pd.DataFrame(classification_report(yte, pred, labels=classes, output_dict=True,
                                                 zero_division=0)).T
        rel.to_csv(pasta / f"relatorio_classes_{marca}.csv", sep=";", decimal=",", encoding="utf-8-sig")
        log("\n" + rel.round(3).to_string())
        cm = confusion_matrix(yte, pred, labels=classes)
        pd.DataFrame(cm, index=[f"real_{c}" for c in classes],
                     columns=[f"previsto_{c}" for c in classes]).to_csv(
            pasta / f"matriz_confusao_{marca}.csv", sep=";", encoding="utf-8-sig")
        grafico_matriz(cm, classes, f"Tipo de acidente - variáveis {versao}", pasta / f"matriz_confusao_{marca}.png")
    else:
        cm = confusion_matrix(yte, pred, labels=[0, 1])
        pd.DataFrame(cm, index=["real_0", "real_1"], columns=["previsto_0", "previsto_1"]).to_csv(
            pasta / f"matriz_confusao_{marca}.csv", sep=";", encoding="utf-8-sig")
        grafico_matriz(cm, ["Sem óbito", "Óbito"], f"Óbito (limiar 0,5) - variáveis {versao}",
                       pasta / f"matriz_confusao_{marca}.png")

    # --- desempenho por ano (mostra a mudança de composição ao longo do tempo)
    linhas_ano = []
    for ano, g in teste.groupby("ano"):
        ix = teste.index.get_indexer(g.index)
        if eh_tipo:
            m = metricas_tipo(g[col_alvo], pred[ix], classes)
            linhas_ano.append({"ano": ano, "n": len(g), **{k: m[k] for k in
                               ("f1_macro", "acuracia_balanceada", "acuracia")},
                               **{f"pct_{c}": (g[col_alvo] == c).mean() * 100 for c in classes}})
        else:
            m = metricas_obito(g[col_alvo].to_numpy(), proba[ix, classes.index(1)])
            linhas_ano.append({"ano": ano, "n": len(g), "n_obitos": int(g[col_alvo].sum()),
                               "pr_auc": m.get("pr_auc", np.nan), "roc_auc": m.get("roc_auc", np.nan)})
    pd.DataFrame(linhas_ano).to_csv(pasta / f"desempenho_por_ano_{marca}.csv", sep=";",
                                    decimal=",", index=False, encoding="utf-8-sig")

    # --- importância por permutação (no teste; amostra para ser rápido)
    n_imp = min(args.amostra_importancia, len(teste))
    amostra = teste.sample(n_imp, random_state=args.seed)
    t0 = time.time()
    r = permutation_importance(modelo, amostra[cols], amostra[col_alvo], scoring=scoring,
                               n_repeats=args.repeticoes, random_state=args.seed, n_jobs=1)
    imp = pd.DataFrame({"variavel": cols, "importancia_media": r.importances_mean, "desvio": r.importances_std})
    imp["importancia_gini_rf"] = modelo.named_steps["rf"].feature_importances_  # mesma ordem de colunas
    imp = imp.sort_values("importancia_media", ascending=False)
    imp.to_csv(pasta / f"importancia_permutacao_{marca}.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
    rotulo = "F1 macro" if eh_tipo else "PR-AUC"
    grafico_importancia(imp, rotulo, f"Importância das variáveis - {col_alvo} ({versao})",
                        pasta / f"importancia_permutacao_{marca}.png")
    log(f"\n  Importância por permutação ({n_imp:,} registros de teste, {args.repeticoes} repetições, "
        f"{time.time() - t0:.0f} s):")
    log(imp[["variavel", "importancia_media", "desvio"]].round(4).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Random Forest para o TCC (tipo de acidente e óbito)")
    ap.add_argument("--entrada", type=Path, default=None, help="base_nordeste.parquet ou .csv")
    ap.add_argument("--saida", type=Path, default=None,
                    help="pasta de saída (padrão: a pasta onde está a base). Grava em <saida>/modelos/random_forest")
    ap.add_argument("--alvo", choices=["tipo", "obito", "ambos"], default="ambos")
    ap.add_argument("--variaveis", choices=["completa", "perfil", "ambas"], default="ambas")
    ap.add_argument("--corte", default="2024-01",
                    help="AAAA-MM: treina com acidentes anteriores e testa nos demais (padrão 2024-01)")
    ap.add_argument("--arvores", type=int, default=500, help="árvores do modelo final (padrão 500)")
    ap.add_argument("--sem-tuning", action="store_true", help="não faz a busca de hiperparâmetros")
    ap.add_argument("--n-iter", type=int, default=8, help="combinações testadas na busca (padrão 8)")
    ap.add_argument("--amostra-tuning", type=int, default=40000, help="linhas do treino usadas na busca")
    ap.add_argument("--amostra-importancia", type=int, default=20000, help="linhas do teste na importância")
    ap.add_argument("--repeticoes", type=int, default=5, help="repetições da importância por permutação")
    ap.add_argument("--n-jobs", type=int, default=-1, help="núcleos de CPU (padrão: todos)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rapido", action="store_true",
                    help="teste rápido: 40.000 linhas, 60 árvores, busca pequena (resultados NÃO servem para o TCC)")
    args = ap.parse_args()

    log = Log()
    base = achar_base(args.entrada)
    saida_raiz = args.saida if args.saida is not None else base.parent
    pasta = saida_raiz / "modelos" / "random_forest"
    pasta.mkdir(parents=True, exist_ok=True)
    log(f"Entrada: {base}\nSaída: {pasta}")

    df = preparar(ler_base(base), log)
    if args.rapido:
        df = df.iloc[np.linspace(0, len(df) - 1, 40000).astype(int)].reset_index(drop=True)
        args.arvores, args.n_iter, args.amostra_tuning = 60, 3, 12000
        args.amostra_importancia, args.repeticoes = 5000, 2
        log("MODO RÁPIDO: amostra de 40.000 linhas e poucas árvores. Só para testar o script.")

    alvos = ["tipo", "obito"] if args.alvo == "ambos" else [args.alvo]
    versoes = ["completa", "perfil"] if args.variaveis == "ambas" else [args.variaveis]

    resumo: list[dict] = []
    parametros: dict = {}
    t_ini = time.time()
    for alvo in alvos:
        for versao in versoes:
            rodar_cenario(df, alvo, versao, args, pasta, log, resumo, parametros)

    pd.DataFrame(resumo).to_csv(pasta / "metricas_resumo.csv", sep=";", decimal=",",
                                index=False, encoding="utf-8-sig")
    (pasta / "melhores_parametros.json").write_text(
        json.dumps({"corte": args.corte, "arvores": args.arvores, "parametros": parametros},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\nConcluído em {(time.time() - t_ini) / 60:.1f} min. Arquivos em {pasta}")
    log("Principal: metricas_resumo.csv (abra no Excel), matriz_confusao_*.png e importancia_permutacao_*.png")
    log.gravar(pasta / "log_random_forest.txt")


if __name__ == "__main__":
    main()
