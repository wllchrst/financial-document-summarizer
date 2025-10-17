def main():
    from pipelines.extract_pipeline import read_pdf

    result = read_pdf(filepath='data/FinancialStatement-2024-Tahunan-EKAD.pdf')


if __name__ == "__main__":
    main()
