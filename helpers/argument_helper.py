import argparse


class ArgumentHelper:
    @staticmethod
    def parse_main_script():
        parser = argparse.ArgumentParser()
        parser.add_argument("--action", help="Action commands are: 'api', 'extract', 'categorize', 'summarize'")
        parser.add_argument("--code",
                            help="Code of account that is going to be use for extraction, categorization and summarization",
                            default=None)
        return parser.parse_args()
