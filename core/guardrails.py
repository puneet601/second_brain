from presidio_anonymizer import AnonymizerEngine, OperatorConfig
from core.presidio_engine import create_presidio_analyzer


from core.logger import logger

analyzer = create_presidio_analyzer()
anonymizer = AnonymizerEngine()

def redact_pii(text: str) -> str:
    """
    Redacts PII entities from text using Microsoft Presidio.
    Replaces sensitive entities with labeled placeholders.
    """
    if not text or not isinstance(text, str):
        return text

    # Step 1: Detect entities
    results = analyzer.analyze(text=text, language="en")

    if not results:
        return text

    # Step 2: Define operator configurations properly
    operators = {
        "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE_NUMBER>"}),
        "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CREDIT_CARD>"}),
        "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
        "LOCATION": OperatorConfig("replace", {"new_value": "<LOCATION>"}),
        "DATE_TIME": OperatorConfig("replace", {"new_value": "<DATE_TIME>"}),
        "NRP": OperatorConfig("replace", {"new_value": "<NRP>"}),
    }

    # Step 3: Apply anonymization
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators
    )
    logger.info(f"results from redact_pii {anonymized_result.text}")
    return anonymized_result.text


def detect_pii(text: str):
    results = analyzer.analyze(text=text, language="en")
    logger.info(f"results from pii_detect {results}")
    return [r.entity_type for r in results]
