import uvicorn
import json

from fastapi import FastAPI
from pydantic import BaseModel
from pipelines.categorization_pipeline import categorize_code
from pipelines.extraction_pipeline import extract_pdf, get_title_codes


class CategorizeRequest(BaseModel):
    code_of_account: int


class ExtractionRequest(BaseModel):
    filepath: int


class API:
    def __init__(self):
        self.app = FastAPI(
            title="Class-based API",
            version="1.0.0"
        )
        self._register_routes()

    def _register_routes(self):
        @self.app.post("/")
        def read_root():
            return {"message": "Welcome to my FastAPI application! visit /docs to get swagger documentation"}

        @self.app.post("/categorize")
        def categorize_code(categorize_request: CategorizeRequest):
            df = categorize_code(use_gemini=False,
                                 filepath=None,
                                 code=categorize_request.code_of_account)

            return {"message": "Categorization completed", "data": df}

        @self.app.post("/extraction")
        def extract_financial_statement(extraction_request: ExtractionRequest):
            df = extract_pdf(filepath=extraction_request.filepath)

            return {"message": "Extraction completed", "data": df}

        @self.app.get("/code_of_accounts")
        def retrieve_code_of_accounts():
            codes = get_title_codes()
            return {"message": "Retrieve code of accounts completed", "data": codes}


app = API().app


def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    uvicorn.run("api:app", host=host, port=port, reload=reload)
