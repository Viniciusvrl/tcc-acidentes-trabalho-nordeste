from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Pastas padrão
PASTA_PROJETO = Path(__file__).resolve().parent   # pasta onde estão os scripts
SAIDA_PADRAO = PASTA_PROJETO / "saida"
ENTRADA_PADRAO = SAIDA_PADRAO / "base_nordeste.parquet"   # gerado pelo montar_base_ne.py


def main() -> None:
    ap = argparse.ArgumentParser(description="Exporta a base final para CSV")
    ap.add_argument("--entrada", default=ENTRADA_PADRAO, type=Path,
                    help=f"base_nordeste.parquet (padrão: {ENTRADA_PADRAO})")
    ap.add_argument("--saida", default=SAIDA_PADRAO, type=Path,
                    help=f"pasta de saída (padrão: {SAIDA_PADRAO})")
    ap.add_argument("--amostra", default=1000, type=int, help="linhas da amostra (padrão: 1000)")
    args = ap.parse_args()

    if not args.entrada.exists():
        sys.exit(f"Arquivo não encontrado: {args.entrada}\n"
                 "Rode antes o montar_base_ne.py (ele gera o base_nordeste.parquet).")
    args.saida.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.entrada)
    completo = args.saida / "base_nordeste.csv"
    df.to_csv(completo, index=False, sep=";", encoding="utf-8-sig")
    print(f"Base completa: {len(df):,} linhas x {df.shape[1]} colunas -> {completo}")

    n = min(args.amostra, len(df))
    amostra = args.saida / f"amostra_{n}.csv"
    df.sample(n, random_state=42).to_csv(amostra, index=False, sep=";", encoding="utf-8-sig")
    print(f"Amostra: {n:,} linhas -> {amostra}")


if __name__ == "__main__":
    main()
