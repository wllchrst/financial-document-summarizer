from pandas import describe_option

from llm.gemini_llm import GeminiLLM
from llm.ollama_llm import OllamaLLM
from typing import *


def load_model(use_gemini: bool):
    if use_gemini:
        return GeminiLLM()
    else:
        return OllamaLLM()


categories = {
    "Assets Current": ["Cash", "Inventory", "Receivable", "Prepaid", "Short-term"],
    "Assets Non Current": ["Property", "Equipment", "Intangible", "Long-term", "Investment"],
    "Liabilities Current": ["Payable", "Accrued", "Short-term loan", "Liability"],
    "Liabilities Non Current": ["Long-term", "Bond", "Lease liability", "Liability"],
    "Equity": ["Capital", "Retained earnings", "Reserves", "Share", "Dividend"],
    "Income Statement": ["Revenue", "Income", "Expense", "Cost", "Profit", "Loss"],
    "Others": []
}


class CategorizePrompt:
    def __init__(self, use_gemini: bool):
        self.llm = load_model(use_gemini)
        self.seperator = '=' * 125

    def get_initial_analysis(self,
                             desc_formatted: str,
                             title: str):
        system_prompt = 'You will be given a financial report title (code of account and code name) and column description in two language.' \
                        + 'Your task is to make an analysis that is going to be helpful in categorizing the financial data'
        prompt = f'Title:\n{title}\n\n' \
                 + f'Column Descriptions:\n{desc_formatted}'

        return self.llm.answer(prompt=prompt, system_prompt=system_prompt, with_logging=False), prompt

    def get_general_categorization(self,
                                   analysis: str,
                                   initial_prompt: str):
        labels = categories.keys()
        labels_formatted = "\n".join([f'- {label}' for label in labels])

        system_prompt = 'You will be given a financial report title (code of account and code name) and column description in two language and an analysis' \
                        + 'The analysis given is intended to be helpful in categorizing the financial data.' \
                        + "Your task is to use the analysis and categorize it to this labels:\n" \
                        + labels_formatted

        prompt = f'Analysis:\n{analysis}' + f'\n{self.seperator}\n' + initial_prompt

        categorization_analysis = self.llm.answer(prompt=prompt, system_prompt=system_prompt, with_logging=False)

        # LABEL EXTRACTION PROCESS
        system_prompt = 'You will be given an output of an LLM which categorized a financial data.' \
                        + " Read the output and summarize the LLM's stance with ONLY one of this labels:\n" \
                        + labels_formatted

        final_analysis = self.llm.answer(prompt=categorization_analysis, system_prompt=system_prompt,
                                         with_logging=False)

        for label in labels:
            if label.lower() in final_analysis.lower():
                return label

        # FALLBACK RESULT IF LLM ISN'T CAPABLE TO CLASSIFY
        return 'Others'

    def get_detail_categorization(self,
                                  general_label: str,
                                  analysis: str,
                                  initial_prompt: str):
        detail_labels = categories[general_label]
        if len(detail_labels) == 0:
            return ''

        labels_formatted = '\n'.join([f'- {l}' for l in detail_labels])

        system_prompt = 'You will be given a financial report title (code of account and code name) and column description in two language and an analysis' \
                        + 'The analysis given is intended to be helpful in categorizing the financial data.\n' \
                        + f"General category of the financial data is already determined which is '{general_label}'\n" \
                        + "Your task is to use the analysis and categorize it to a detailed labels below:\n" \
                        + labels_formatted

        prompt = f'Analysis:\n{analysis}' + f'\n{self.seperator}\n' + initial_prompt

        categorization_analysis = self.llm.answer(prompt=prompt, system_prompt=system_prompt, with_logging=False)

        system_prompt = 'You will be given an output of an LLM which categorized a financial data into a detail labels.' \
                        + " Read the output and summarize the LLM's stance with ONLY one of this detail labels:\n" \
                        + labels_formatted

        final_analysis = self.llm.answer(prompt=categorization_analysis, system_prompt=system_prompt,
                                         with_logging=False)

        for label in detail_labels:
            if label.lower() in final_analysis.lower():
                return label

        # FALLBACK IF THE LLM IS NOT SUCCESSFUL IN DETERMINING THE DETAIL LABELS
        return ''

    def classify(self, descriptions: List[str], title: str):
        desc_formatted = "\n".join([f'- {d}' for d in descriptions])
        initial_analysis, initial_prompt = self.get_initial_analysis(
            desc_formatted=desc_formatted,
            title=title
        )

        general_label = self.get_general_categorization(
            analysis=initial_analysis,
            initial_prompt=initial_prompt
        )

        detail_categorization = self.get_detail_categorization(
            general_label=general_label,
            analysis=initial_analysis,
            initial_prompt=initial_prompt
        )

        return general_label, detail_categorization
