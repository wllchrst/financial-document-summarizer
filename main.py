def main():
    from pipelines.extract_pipeline import extract_table_from_pdf

    result = extract_table_from_pdf(filepath='data/FinancialStatement-2024-Tahunan-EKAD.pdf')


if __name__ == "__main__":
    main()
