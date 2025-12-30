from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

def create_presidio_analyzer():
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [
                {
                    "lang_code": "en",
                    "model_name": "en_core_web_sm",
                }
            ],
        }
    ).create_engine()

    return AnalyzerEngine(nlp_engine=nlp_engine)
