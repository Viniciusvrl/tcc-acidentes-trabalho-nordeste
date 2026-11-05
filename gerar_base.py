import argparse
import subprocess
import sys
import time
from pathlib import Path

AQUI = Path(__file__).resolve().parent   # onde estão os outros scripts
PASTA_PROJETO = AQUI / "entrada"          # CSVs de CAT
SAIDA_PADRAO = AQUI / "saida"


def executar(nome: str, script: str, args: list[str], opcional: bool = False) -> bool:
    caminho = AQUI / script
    if not caminho.exists():
        msg = f"Script não encontrado: {caminho}"
        if opcional:
            print(f"\n[aviso] {msg} (passo opcional, seguindo)")
            return False
        sys.exit(msg)
    print(f"\n{'=' * 70}\n{nome}\n{'=' * 70}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(caminho), *args])
    dt = time.time() - t0
    if r.returncode != 0:
        if opcional:
            print(f"\n[aviso] {nome} falhou (código {r.returncode}); como é opcional, a base gerada continua valendo.")
            return False
        sys.exit(f"\nErro no passo '{nome}' (código {r.returncode}). Corrija e rode de novo.")
    print(f"\n[ok] {nome} em {dt:.0f} s", flush=True)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera a base do TCC do início ao fim")
    ap.add_argument("--pasta", default=PASTA_PROJETO, type=Path,
                    help=f"pasta do projeto, com os CSVs de CAT (padrão: {PASTA_PROJETO})")
    ap.add_argument("--saida", default=SAIDA_PADRAO, type=Path,
                    help=f"pasta de saída (padrão: {SAIDA_PADRAO})")
    ap.add_argument("--pular-limpeza", action="store_true",
                    help="reaproveita saida/cat_limpa.parquet em vez de refazer a limpeza")
    ap.add_argument("--sem-analise", action="store_true", help="não roda a análise de dezembro")
    ap.add_argument("--so-janelas-estaveis", action="store_true",
                    help="no passo 2, mantém só os meses das janelas estáveis (por padrão mantém todos)")
    ap.add_argument("--manter-ausentes", action="store_true",
                    help="no passo 2, não remove registros com dado faltante (por padrão remove)")
    ap.add_argument("--tolerar-ausencia-em", nargs="+", default=[], metavar="COLUNA",
                    help="no passo 2, colunas cuja ausência não remove o registro (ex.: agente_causador cbo)")
    ap.add_argument("--sem-modelo", action="store_true",
                    help="na análise de dezembro, pula a sensibilidade do modelo")
    args = ap.parse_args()

    saida = args.saida
    limpa = saida / "cat_limpa.parquet"
    base = saida / "base_nordeste.parquet"
    t0 = time.time()

    if args.pular_limpeza:
        if not limpa.exists():
            sys.exit(f"--pular-limpeza pedido, mas {limpa} não existe. Rode sem essa opção.")
        print(f"Passo 1 pulado: usando {limpa}")
    else:
        executar("1/4  Limpeza e integração dos CSVs de CAT", "limpeza_cat.py",
                 ["--entrada", str(args.pasta), "--saida", str(saida)])

    extra_base = (["--so-janelas-estaveis"] if args.so_janelas_estaveis else []) \
        + (["--manter-ausentes"] if args.manter_ausentes else []) \
        + (["--tolerar-ausencia-em", *args.tolerar_ausencia_em] if args.tolerar_ausencia_em else [])
    executar("2/4  Base do Nordeste (todos os meses)", "montar_base_ne.py",
             ["--entrada", str(limpa), "--saida", str(saida), *extra_base])

    executar("3/4  Exportação para CSV", "exportar_csv.py",
             ["--entrada", str(base), "--saida", str(saida)])

    if not args.sem_analise:
        extra = ["--sem-modelo"] if args.sem_modelo else []
        executar("4/4  Análise de dezembro", "analise_dezembro.py",
                 ["--entrada", str(base), "--nacional", str(limpa), "--saida", str(saida), *extra],
                 opcional=True)

    print(f"\n{'=' * 70}\nConcluído em {time.time() - t0:.0f} s. Arquivos em {saida}")
    print("  base_nordeste.parquet / base_nordeste.csv   base final do TCC (sem registros com dado faltante)")
    print("  removidos_por_ausencia.csv                  registros retirados por dado faltante")
    print("  cat_limpa.parquet                           base nacional limpa")
    print("  resumo_base.txt, log_limpeza.txt            contagens de cada etapa")
    print("  diagnostico_*.csv, dicionario_rotulos.csv   diagnósticos e dicionário")
    if not args.sem_analise:
        print("  dezembro_*.csv/.png/.txt                    evidências sobre dezembro")


if __name__ == "__main__":
    main()
