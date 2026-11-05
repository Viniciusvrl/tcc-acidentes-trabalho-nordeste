

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.utils.class_weight import compute_sample_weight

try:
    import modelo_random_forest as rf  # funções compartilhadas (mesma pasta)
except ImportError:
    sys.exit("Coloque modelos_comparativos.py na MESMA pasta do modelo_random_forest.py.")

warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------------- XGBoost
class Impulsionado(ClassifierMixin, BaseEstimator):
    """XGBoost com rótulos de texto e pesos balanceados por classe (equivale ao class_weight='balanced').

    motor='hgb' usa o HistGradientBoosting do scikit-learn no lugar do XGBoost; existe só para
    testar o script onde o xgboost não está instalado e NÃO deve ser usado nos resultados do TCC.
    """

    def __init__(self, motor="xgboost", n_estimators=300, learning_rate=0.1, max_depth=6,
                 subsample=1.0, colsample_bytree=1.0, n_jobs=-1, random_state=42):
        self.motor = motor
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        self.classes_, y_enc = np.unique(np.asarray(y), return_inverse=True)
        peso = compute_sample_weight("balanced", y_enc)
        if self.motor == "xgboost":
            from xgboost import XGBClassifier
            self.modelo_ = XGBClassifier(
                n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                max_depth=self.max_depth, subsample=self.subsample,
                colsample_bytree=self.colsample_bytree, tree_method="hist",
                n_jobs=self.n_jobs, random_state=self.random_state, verbosity=0)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self.modelo_ = HistGradientBoostingClassifier(
                max_iter=self.n_estimators, learning_rate=self.learning_rate,
                max_depth=self.max_depth, random_state=self.random_state)
        self.modelo_.fit(X, y_enc, sample_weight=peso)
        return self

    def predict_proba(self, X):
        return self.modelo_.predict_proba(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    @property
    def feature_importances_(self):
        return getattr(self.modelo_, "feature_importances_", None)


# ----------------------------------------------------------------------------- algoritmos
def _pre_ordinal(cats, nums):
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32), cats),
        ("num", "passthrough", nums),
    ])


def _pre_onehot(cats, nums, escalar: bool, min_freq: int):
    """One-hot para as categóricas (e para mês/dia da semana, que não são lineares)."""
    cats_oh = cats + [c for c in ("mes", "dia_semana") if c in nums]
    nums_ = [c for c in nums if c not in ("mes", "dia_semana")]
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=min_freq,
                              sparse_output=True), cats_oh),
        ("num", StandardScaler(with_mean=False) if escalar else "passthrough", nums_),
    ])


def montar(algoritmo: str, cats, nums, args, **params) -> Pipeline:
    seed, n_jobs = args.seed, args.n_jobs
    if algoritmo == "regressao_logistica":
        clf = LogisticRegression(class_weight="balanced", max_iter=300, random_state=seed, **params)
        return Pipeline([("pre", _pre_onehot(cats, nums, True, 50)), ("clf", clf)])
    if algoritmo == "arvore_decisao":
        clf = DecisionTreeClassifier(class_weight="balanced", random_state=seed, **params)
        return Pipeline([("pre", _pre_ordinal(cats, nums)), ("clf", clf)])
    if algoritmo == "xgboost":
        motor = "hgb" if args.xgb_substituto else "xgboost"
        clf = Impulsionado(motor=motor, n_jobs=n_jobs, random_state=seed, **params)
        return Pipeline([("pre", _pre_ordinal(cats, nums)), ("clf", clf)])
    raise ValueError(algoritmo)


ALGORITMOS = {
    "regressao_logistica": {
        "rotulo": "Regressão Logística",
        "espaco": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        "padrao": {"C": 1.0},
    },
    "arvore_decisao": {
        "rotulo": "Árvore de Decisão",
        "espaco": {"clf__max_depth": [6, 10, 15, 25, None],
                   "clf__min_samples_leaf": [5, 20, 50, 100]},
        "padrao": {"max_depth": 15, "min_samples_leaf": 20},
    },
    "xgboost": {
        "rotulo": "XGBoost",
        "espaco": {"clf__max_depth": [4, 6, 8], "clf__learning_rate": [0.05, 0.1, 0.2],
                   "clf__n_estimators": [200, 400], "clf__subsample": [0.8, 1.0],
                   "clf__colsample_bytree": [0.8, 1.0]},
        "padrao": {"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300},
    },
}


def ajustar(algoritmo, X, y, cats, nums, scoring, args, log) -> dict:
    """Busca aleatória com validação cruzada temporal, só no treino (já em ordem de data)."""
    espaco = ALGORITMOS[algoritmo]["espaco"]
    tamanho = int(np.prod([len(v) for v in espaco.values()]))
    n_iter = min(args.n_iter, tamanho)
    idx = np.linspace(0, len(X) - 1, min(args.amostra_tuning, len(X))).astype(int)
    busca = RandomizedSearchCV(
        montar(algoritmo, cats, nums, args), espaco, n_iter=n_iter, scoring=scoring,
        cv=TimeSeriesSplit(n_splits=3), random_state=args.seed, n_jobs=1, refit=False,
        error_score=np.nan)
    t0 = time.time()
    busca.fit(X.iloc[idx], y.iloc[idx])
    melhor = {k.replace("clf__", ""): v for k, v in busca.best_params_.items()}
    log(f"  Hiperparâmetros (amostra de {len(idx):,} linhas do treino, {n_iter} combinações, "
        f"{scoring} = {busca.best_score_:.4f}, {time.time() - t0:.0f} s): {melhor}")
    return melhor


# ----------------------------------------------------------------------------- saídas específicas
def salvar_coeficientes(modelo, classes, marca, pasta, log) -> None:
    nomes = modelo.named_steps["pre"].get_feature_names_out()
    nomes = [n.split("__", 1)[1] for n in nomes]
    coef = modelo.named_steps["clf"].coef_
    if coef.shape[0] == 1:                       # binário: um só vetor (classe positiva)
        coef = coef
        cols = [f"coef_{classes[1]}"]
    else:
        cols = [f"coef_{c}" for c in classes]
    tab = pd.DataFrame(coef.T, index=nomes, columns=cols)
    for c in cols:
        tab[c.replace("coef_", "razao_chances_")] = np.exp(tab[c])
    tab.index.name = "variavel"
    tab.to_csv(pasta / f"coeficientes_{marca}.csv", sep=";", decimal=",", encoding="utf-8-sig")
    for c in cols[:3]:
        top = tab[c].sort_values(ascending=False).head(8)
        log(f"\n  Maiores coeficientes ({c}): " + "; ".join(f"{k} ({v:+.2f})".replace(".", ",")
                                                          for k, v in top.items()))


def arvore_rasa(Xtr, ytr, cats, nums, marca, pasta, args, titulo, log) -> None:
    """Árvore de profundidade 3 com one-hot: regras que dá para ler e citar no texto."""
    pre = _pre_onehot(cats, nums, False, 100)
    A = pre.fit_transform(Xtr)
    arv = DecisionTreeClassifier(max_depth=3, min_samples_leaf=200, class_weight="balanced",
                                 random_state=args.seed).fit(A, ytr)
    nomes = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    texto = export_text(arv, feature_names=nomes, decimals=1, show_weights=True)
    (pasta / f"regras_arvore_rasa_{marca}.txt").write_text(texto, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(16, 6.5), dpi=150, facecolor=rf.SUPERFICIE)
    plot_tree(arv, feature_names=nomes, class_names=[str(c) for c in arv.classes_], impurity=False,
              proportion=True, precision=1, fontsize=7, ax=ax)
    ax.set_title(titulo + " (profundidade 3, treino)", color=rf.TEXTO, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(pasta / f"regras_arvore_rasa_{marca}.png", facecolor=rf.SUPERFICIE)
    plt.close(fig)
    log("\n  Árvore rasa (profundidade 3):\n" + texto)


# ----------------------------------------------------------------------------- um cenário
def rodar_cenario(df, algoritmo, alvo, versao, args, pasta, log, resumo, parametros) -> None:
    cats, nums = rf.VARIAVEIS[versao]
    cols = cats + nums
    eh_tipo = alvo == "tipo"
    col_alvo = "tipo_acidente" if eh_tipo else "indica_obito"
    scoring = "f1_macro" if eh_tipo else "average_precision"
    marca = f"{alvo}_{versao}"
    rotulo_alg = ALGORITMOS[algoritmo]["rotulo"]

    treino = df[df["ym"] < args.corte]
    teste = df[df["ym"] >= args.corte]
    if len(treino) == 0 or len(teste) == 0:
        sys.exit(f"--corte {args.corte} deixa treino ou teste vazio.")
    Xtr, ytr, Xte, yte = treino[cols], treino[col_alvo], teste[cols], teste[col_alvo]

    log(f"\n{'=' * 78}\n{rotulo_alg.upper()} | ALVO: {col_alvo} | VARIÁVEIS: {versao} ({len(cols)}) | "
        f"treino {len(treino):,} | teste {len(teste):,}\n{'=' * 78}")

    if args.sem_tuning:
        melhores = dict(ALGORITMOS[algoritmo]["padrao"])
        log(f"  Sem busca de hiperparâmetros; usando {melhores}")
    else:
        melhores = ajustar(algoritmo, Xtr, ytr, cats, nums, scoring, args, log)
    parametros[f"{algoritmo}_{marca}"] = melhores

    t0 = time.time()
    modelo = montar(algoritmo, cats, nums, args, **melhores).fit(Xtr, ytr)
    log(f"  Modelo final treinado em {time.time() - t0:.0f} s")
    pred = modelo.predict(Xte)
    proba = modelo.predict_proba(Xte)
    classes = list(modelo.classes_)
    pred_base = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr).predict(Xte)

    recortes = {"todos_os_meses": np.ones(len(teste), bool),
                "so_cobertura_regular": (teste["mes_estavel"].to_numpy() == 1)}
    for nome_recorte, mascara in recortes.items():
        if mascara.sum() == 0:
            continue
        linha = {"algoritmo": rotulo_alg, "alvo": col_alvo, "variaveis": versao, "teste": nome_recorte,
                 "n_treino": len(treino), "n_teste": int(mascara.sum())}
        if eh_tipo:
            linha |= rf.metricas_tipo(yte[mascara], pred[mascara], classes)
            linha["f1_macro_linha_de_base"] = f1_score(yte[mascara], pred_base[mascara],
                                                       average="macro", zero_division=0)
            log(f"  [{nome_recorte}] n={linha['n_teste']:,} | F1 macro {linha['f1_macro']:.3f} "
                f"(base {linha['f1_macro_linha_de_base']:.3f}) | acur. balanceada "
                f"{linha['acuracia_balanceada']:.3f} | acurácia {linha['acuracia']:.3f}")
        else:
            p1 = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(teste))
            linha |= rf.metricas_obito(yte[mascara].to_numpy(), p1[mascara])
            log(f"  [{nome_recorte}] n={linha['n_teste']:,} | PR-AUC {linha['pr_auc']:.3f} "
                f"(acaso = {linha['prevalencia']:.4f}) | ROC-AUC {linha['roc_auc']:.3f} | revocação nos 5% "
                f"de maior risco {linha.get('revocacao_nos_5pct_de_maior_risco', float('nan')):.3f}")
        resumo.append(linha)

    # relatório por classe e matriz de confusão
    if eh_tipo:
        rel = pd.DataFrame(classification_report(yte, pred, labels=classes, output_dict=True,
                                                 zero_division=0)).T
        rel.to_csv(pasta / f"relatorio_classes_{marca}.csv", sep=";", decimal=",", encoding="utf-8-sig")
        log("\n" + rel.round(3).to_string())
        cm = confusion_matrix(yte, pred, labels=classes)
        nomes_cls = classes
        titulo = f"{rotulo_alg} - tipo de acidente, variáveis {versao}"
    else:
        cm = confusion_matrix(yte, pred, labels=[0, 1])
        nomes_cls = ["Sem óbito", "Óbito"]
        titulo = f"{rotulo_alg} - óbito (limiar 0,5), variáveis {versao}"
    pd.DataFrame(cm, index=[f"real_{c}" for c in nomes_cls],
                 columns=[f"previsto_{c}" for c in nomes_cls]).to_csv(
        pasta / f"matriz_confusao_{marca}.csv", sep=";", encoding="utf-8-sig")
    rf.grafico_matriz(cm, nomes_cls, titulo, pasta / f"matriz_confusao_{marca}.png")

    # desempenho por ano
    linhas_ano = []
    for ano, g in teste.groupby("ano"):
        ix = teste.index.get_indexer(g.index)
        if eh_tipo:
            m = rf.metricas_tipo(g[col_alvo], pred[ix], classes)
            linhas_ano.append({"ano": ano, "n": len(g), **{k: m[k] for k in
                               ("f1_macro", "acuracia_balanceada", "acuracia")}})
        else:
            m = rf.metricas_obito(g[col_alvo].to_numpy(), proba[ix, classes.index(1)])
            linhas_ano.append({"ano": ano, "n": len(g), "n_obitos": int(g[col_alvo].sum()),
                               "pr_auc": m.get("pr_auc", np.nan), "roc_auc": m.get("roc_auc", np.nan)})
    pd.DataFrame(linhas_ano).to_csv(pasta / f"desempenho_por_ano_{marca}.csv", sep=";", decimal=",",
                                    index=False, encoding="utf-8-sig")

    # importância por permutação (igual à do Random Forest)
    n_imp = min(args.amostra_importancia, len(teste))
    amostra = teste.sample(n_imp, random_state=args.seed)
    t0 = time.time()
    r = permutation_importance(modelo, amostra[cols], amostra[col_alvo], scoring=scoring,
                               n_repeats=args.repeticoes, random_state=args.seed, n_jobs=1)
    imp = pd.DataFrame({"variavel": cols, "importancia_media": r.importances_mean,
                        "desvio": r.importances_std}).sort_values("importancia_media", ascending=False)
    imp.to_csv(pasta / f"importancia_permutacao_{marca}.csv", sep=";", decimal=",", index=False,
               encoding="utf-8-sig")
    rf.grafico_importancia(imp, "F1 macro" if eh_tipo else "PR-AUC",
                           f"{rotulo_alg} - importância das variáveis, {col_alvo} ({versao})",
                           pasta / f"importancia_permutacao_{marca}.png")
    log(f"\n  Importância por permutação ({n_imp:,} registros, {args.repeticoes} repetições, "
        f"{time.time() - t0:.0f} s):")
    log(imp[["variavel", "importancia_media", "desvio"]].round(4).to_string(index=False))

    # saídas próprias de cada algoritmo
    if algoritmo == "regressao_logistica":
        salvar_coeficientes(modelo, classes, marca, pasta, log)
    elif algoritmo == "arvore_decisao":
        pd.DataFrame({"variavel": cols, "importancia_gini": modelo.named_steps["clf"].feature_importances_}
                     ).sort_values("importancia_gini", ascending=False).to_csv(
            pasta / f"importancia_modelo_{marca}.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")
        arvore_rasa(Xtr, ytr, cats, nums, marca, pasta, args, titulo, log)
    elif algoritmo == "xgboost":
        fi = modelo.named_steps["clf"].feature_importances_
        if fi is not None:
            pd.DataFrame({"variavel": cols, "importancia_ganho": fi}).sort_values(
                "importancia_ganho", ascending=False).to_csv(
                pasta / f"importancia_modelo_{marca}.csv", sep=";", decimal=",", index=False,
                encoding="utf-8-sig")


# ----------------------------------------------------------------------------- comparativo
def comparativo(raiz: Path, log) -> None:
    """Junta o metricas_resumo.csv de cada algoritmo que já rodou (inclui o Random Forest)."""
    partes = []
    for nome, rotulo in [("random_forest", "Random Forest")] + [(k, v["rotulo"]) for k, v in ALGORITMOS.items()]:
        f = raiz / nome / "metricas_resumo.csv"
        if f.exists():
            d = pd.read_csv(f, sep=";", decimal=",", encoding="utf-8-sig")
            if "algoritmo" not in d.columns:
                d.insert(0, "algoritmo", rotulo)
            partes.append(d)
    if not partes:
        return
    tab = pd.concat(partes, ignore_index=True)
    tab.to_csv(raiz / "comparativo_algoritmos.csv", sep=";", decimal=",", index=False, encoding="utf-8-sig")

    ordem = ["Regressão Logística", "Árvore de Decisão", "Random Forest", "XGBoost"]
    ordem = [o for o in ordem if o in set(tab["algoritmo"])]
    base = tab[tab["teste"] == "todos_os_meses"]

    # F1 macro do tipo de acidente
    t = pd.DataFrame() if "f1_macro" not in base.columns else base[base["alvo"] == "tipo_acidente"].pivot(index="algoritmo", columns="variaveis",
                                                    values="f1_macro").reindex(ordem)
    if not t.empty:
        linha_base = base[base["alvo"] == "tipo_acidente"]["f1_macro_linha_de_base"].dropna().iloc[0]
        fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=150, facecolor=rf.SUPERFICIE)
        x = np.arange(len(t))
        larg = 0.36
        series = [("completa", rf.COR_SERIE, "Variáveis completas"), ("perfil", "#9ec5f0", "Só perfil")]
        for k, (col, cor, nome) in enumerate(series):
            if col not in t.columns:
                continue
            barras = ax.bar(x + (k - 0.5) * larg, t[col], larg, color=cor, label=nome)
            for b, v in zip(barras, t[col]):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}".replace(".", ","),
                        ha="center", fontsize=8, color=rf.TEXTO)
        ax.axhline(linha_base, color=rf.TEXTO_2, linewidth=1, linestyle="--")
        ax.text(len(t) - 0.5, linha_base + 0.012, "linha de base", ha="right", fontsize=8, color=rf.TEXTO_2)
        ax.set_xticks(x, t.index, fontsize=8.5)
        ax.set_ylabel("F1 macro no teste", color=rf.TEXTO_2, fontsize=9)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1, 1.12), ncol=2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
        ax.yaxis.grid(True, color=rf.GRADE, linewidth=0.8)
        ax.set_axisbelow(True)
        rf._estilo(ax)
        fig.tight_layout()
        fig.savefig(raiz / "comparativo_f1_tipo.png", facecolor=rf.SUPERFICIE)
        plt.close(fig)

    # PR-AUC do óbito
    t = pd.DataFrame() if "pr_auc" not in base.columns else base[base["alvo"] == "indica_obito"].pivot(index="algoritmo", columns="variaveis",
                                                   values="pr_auc").reindex(ordem)
    if not t.empty:
        prev = base[base["alvo"] == "indica_obito"]["prevalencia"].iloc[0]
        fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=150, facecolor=rf.SUPERFICIE)
        x = np.arange(len(t))
        for k, (col, cor, nome) in enumerate([("completa", rf.COR_SERIE, "Variáveis completas"),
                                              ("perfil", "#9ec5f0", "Só perfil")]):
            if col not in t.columns:
                continue
            barras = ax.bar(x + (k - 0.5) * 0.36, t[col], 0.36, color=cor, label=nome)
            for b, v in zip(barras, t[col]):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}".replace(".", ","),
                        ha="center", fontsize=8, color=rf.TEXTO)
        ax.axhline(prev, color=rf.TEXTO_2, linewidth=1, linestyle="--")
        ax.text(len(t) - 0.5, prev + 0.006, "acaso (prevalência)", ha="right", fontsize=8, color=rf.TEXTO_2)
        ax.set_xticks(x, t.index, fontsize=8.5)
        ax.set_ylabel("PR-AUC no teste (óbito)", color=rf.TEXTO_2, fontsize=9)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right", bbox_to_anchor=(1, 1.12), ncol=2)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
        ax.yaxis.grid(True, color=rf.GRADE, linewidth=0.8)
        ax.set_axisbelow(True)
        rf._estilo(ax)
        fig.tight_layout()
        fig.savefig(raiz / "comparativo_prauc_obito.png", facecolor=rf.SUPERFICIE)
        plt.close(fig)

    log(f"\nComparativo gravado em {raiz / 'comparativo_algoritmos.csv'} "
        f"({tab['algoritmo'].nunique()} algoritmos: {', '.join(sorted(tab['algoritmo'].unique()))})")


# ----------------------------------------------------------------------------- main
def main(algoritmos_fixos: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Regressão Logística, Árvore de Decisão e XGBoost para o TCC")
    ap.add_argument("--entrada", type=Path, default=None, help="base_nordeste.parquet ou .csv")
    ap.add_argument("--saida", type=Path, default=None, help="pasta de saída (padrão: onde está a base)")
    ap.add_argument("--algoritmos", nargs="+", choices=list(ALGORITMOS), default=list(ALGORITMOS))
    ap.add_argument("--alvo", choices=["tipo", "obito", "ambos"], default="ambos")
    ap.add_argument("--variaveis", choices=["completa", "perfil", "ambas"], default="ambas")
    ap.add_argument("--corte", default="2024-01")
    ap.add_argument("--sem-tuning", action="store_true")
    ap.add_argument("--n-iter", type=int, default=8, help="combinações testadas na busca (padrão 8)")
    ap.add_argument("--amostra-tuning", type=int, default=40000)
    ap.add_argument("--amostra-importancia", type=int, default=20000)
    ap.add_argument("--repeticoes", type=int, default=5)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rapido", action="store_true", help="teste rápido (resultados NÃO servem para o TCC)")
    ap.add_argument("--so-comparativo", action="store_true", help="só refaz a tabela/gráficos comparativos")
    ap.add_argument("--xgb-substituto", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if algoritmos_fixos:
        args.algoritmos = list(algoritmos_fixos)

    log = rf.Log()
    base = rf.achar_base(args.entrada)
    raiz = (args.saida if args.saida is not None else base.parent) / "modelos"
    raiz.mkdir(parents=True, exist_ok=True)

    if args.so_comparativo:
        comparativo(raiz, log)
        return

    if "xgboost" in args.algoritmos and not args.xgb_substituto:
        try:
            import xgboost  # noqa: F401
        except ImportError:
            sys.exit("Falta instalar o XGBoost. No terminal:  pip install xgboost\n"
                     "(ou rode sem ele: --algoritmos regressao_logistica arvore_decisao)")

    log(f"Entrada: {base}\nSaída: {raiz}")
    df = rf.preparar(rf.ler_base(base), log)
    if args.rapido:
        df = df.iloc[np.linspace(0, len(df) - 1, 40000).astype(int)].reset_index(drop=True)
        args.n_iter, args.amostra_tuning, args.amostra_importancia, args.repeticoes = 3, 12000, 5000, 2
        log("MODO RÁPIDO: amostra de 40.000 linhas. Só para testar o script.")

    alvos = ["tipo", "obito"] if args.alvo == "ambos" else [args.alvo]
    versoes = ["completa", "perfil"] if args.variaveis == "ambas" else [args.variaveis]

    for algoritmo in args.algoritmos:
        pasta = raiz / algoritmo
        pasta.mkdir(parents=True, exist_ok=True)
        resumo: list[dict] = []
        parametros: dict = {}
        t_ini = time.time()
        for alvo in alvos:
            for versao in versoes:
                rodar_cenario(df, algoritmo, alvo, versao, args, pasta, log, resumo, parametros)
        pd.DataFrame(resumo).to_csv(pasta / "metricas_resumo.csv", sep=";", decimal=",",
                                    index=False, encoding="utf-8-sig")
        (pasta / "melhores_parametros.json").write_text(
            json.dumps({"corte": args.corte, "parametros": parametros}, ensure_ascii=False, indent=2,
                       default=str), encoding="utf-8")
        log(f"\n{ALGORITMOS[algoritmo]['rotulo']} concluído em {(time.time() - t_ini) / 60:.1f} min.")
        log.gravar(pasta / f"log_{algoritmo}.txt")

    comparativo(raiz, log)
    if len(args.algoritmos) > 1:
        log.gravar(raiz / "log_modelos_comparativos.txt")
    log("\nPronto. Principal: comparativo_algoritmos.csv e as pastas de cada algoritmo.")


if __name__ == "__main__":
    main()
