# second_brain/core/pii_guardrail.py

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Initialize analyzers (you can make them singletons if reused often)
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def detect_pii(text: str):
    """
    Detect PII entities in the given text using Microsoft Presidio.
    """
    results = analyzer.analyze(text=text, language="en")
    pii_entities = [r.entity_type for r in results]
    return pii_entities

def redact_pii(text: str) -> str:
    """
    Returns a cleaned version with PII replaced by placeholders.
    """
    if not text or not isinstance(text, str):
        return text

    # Step 1: Analyze text for PII entities
    results = analyzer.analyze(text=text, language="en")

    if not results:
        return text  # No PII detected, return original

    # Step 2: Replace PII entities with labeled placeholders
    anonymized_result = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={
            "DEFAULT": {"type": "replace", "new_value": "<REDACTED>"},
            "EMAIL_ADDRESS": {"type": "replace", "new_value": "<EMAIL>"},
            "PHONE_NUMBER": {"type": "replace", "new_value": "<PHONE_NUMBER>"},
            "CREDIT_CARD": {"type": "replace", "new_value": "<CREDIT_CARD>"},
            "PERSON": {"type": "replace", "new_value": "<PERSON>"},
            "LOCATION": {"type": "replace", "new_value": "<LOCATION>"},
            "DATE_TIME": {"type": "replace", "new_value": "<DATE_TIME>"},
            "NRP": {"type": "replace", "new_value": "<NRP>"},  # e.g., national ID
        },
    )

    return anonymized_result.text

# Example (you can test this directly)
if __name__ == "__main__":
    sample = "My email is puneet@example.com and my phone number is +91 9876543210"
    print("Detected PII:", detect_pii(sample))
    print("Cleaned text:", anonymize_pii(sample))
