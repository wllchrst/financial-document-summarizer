from api import run_api
from pipelines.extraction_pipeline import extract_pdf
from pipelines.categorization_pipeline import categorize_data
from helpers.argument_helper import ArgumentHelper

API_ACTION = 'api'
EXTRACT_ACTION = 'extract'
CATEGORIZE_ACTION = 'categorize'
SUMMARIZE_ACTION = 'summarize'
FINANCIAL_STATEMENT_PATH = "data/FinancialStatement-2024-Tahunan-EKAD.pdf"


def main():
    args = ArgumentHelper.parse_main_script()

    if args.action == API_ACTION: run_api()

    code = args.code
    if args.action == EXTRACT_ACTION:
        extract_pdf(FINANCIAL_STATEMENT_PATH)
    elif args.action == CATEGORIZE_ACTION:
        categorize_data(False, filepath=None, code=code)
    elif args.action == SUMMARIZE_ACTION:
        pass


if __name__ == "__main__":
    main()
