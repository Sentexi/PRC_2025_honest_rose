#!/usr/bin/env python3
# to_ger_excel.py
# Liest eine CSV und schreibt sie als deutsch-lesbare CSV (sep=';', decimal=',').

import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Convert CSV to German Excel-friendly CSV.")
    parser.add_argument("--file", required=True, help="Pfad zur Eingabe-CSV")
    args = parser.parse_args()

    in_path = Path(args.file)
    if not in_path.exists():
        raise FileNotFoundError(f"Eingabedatei nicht gefunden: {in_path}")

    # Einlesen mit Pandas (UTF-8 als Default; bei Bedarf anpassen)
    # Versuche numerische Spalten automatisch zu erkennen/konvertieren.
    df = pd.read_csv(in_path)

    # Alles, was wie eine Zahl aussieht (inkl. wissenschaftlicher Notation), in numerisch umwandeln
    # (Strings, die nicht numerisch sind, bleiben unverändert).
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # Ausgabepfad erzeugen
    out_path = in_path.with_name(in_path.stem + "_de.csv")

    # Schreiben für deutsches Excel:
    # - Semikolon als Trenner
    # - Dezimal-Komma
    # - UTF-8 mit BOM verbessert Excel-Kompatibilität
    df.to_csv(
        out_path,
        sep=";",
        index=False,
        decimal=",",
        encoding="utf-8-sig",
        na_rep=""  # leere Zellen statt 'NaN'
    )

    print(f"Geschrieben: {out_path}")

if __name__ == "__main__":
    main()
