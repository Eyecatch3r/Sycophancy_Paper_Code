"""CLI wrapper to enrich a CSV using DatasetEnricher.enrich_csv

Usage (Windows cmd):
    python -m scripts.enrich_csv_cli --input path/to/in.csv --output path/to/out.csv

This module is intentionally small so it can be imported in unit tests and its
main() called with patched sys.argv.
"""
import argparse


def main(argv=None):
    p = argparse.ArgumentParser(description="Enrich a CSV with OP text from CMV via ConvoKit")
    p.add_argument("--input", "-i", required=True, help="Path to input CSV file")
    p.add_argument("--output", "-o", required=False, help="Path to write enriched CSV (optional)")
    p.add_argument("--corpus", "-c", default="winning-args-corpus", help="ConvoKit corpus name")
    args = p.parse_args(argv)

    # Import here to avoid ModuleNotFoundError when this module is imported
    # while sys.path doesn't include the project root (e.g. PyCharm running tests)
    from data_loader import DatasetEnricher

    enriched = DatasetEnricher.enrich_csv(args.input, output_csv_path=args.output, corpus_name=args.corpus)
    return enriched


if __name__ == "__main__":
    main()
