from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine, OperatorConfig
from core.presidio_engine import create_presidio_analyzer
from core.logger import logger

# Initialize engines once
analyzer: AnalyzerEngine = create_presidio_analyzer()
anonymizer = AnonymizerEngine()

# Explicit PII whitelist (chat-safe)
PII_ENTITIES = {
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "IBAN",
    "IP_ADDRESS",
}

# Confidence threshold to avoid false positives
PII_SCORE_THRESHOLD = 0.85


def detect_pii(text: str):
    """
    Detects high-confidence PII entities in text.
    Returns Presidio entity objects.
    """
    if not text or not isinstance(text, str):
        return []

    results = analyzer.analyze(
        text=text,
        language="en",
        score_threshold=PII_SCORE_THRESHOLD,
    )

    pii_results = [
        r for r in results
        if r.entity_type in PII_ENTITIES
    ]

    logger.debug(f"PII detected: {pii_results}")
    return pii_results


def redact_pii(text: str) -> str:
    """
    Redacts ONLY high-confidence PII entities.
    Leaves names, locations, preferences, and context intact.
    """
    if not text or not isinstance(text, str):
        return text

    pii_results = detect_pii(text)

    if not pii_results:
        return text

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=pii_results,
        operators={
            "DEFAULT": OperatorConfig(
                operator_name="replace",
                params={"new_value": "<PII_REDACTED>"}
            )
        }
    )

    logger.info(f"Redacted text: {anonymized.text}")
    return anonymized.text
