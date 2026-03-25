from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

class PIIRedactor:
    def __init__(self):
        print("Initializing PII Redactor (Presidio)...")
        # Initialize Presidio Analyzer and Anonymizer
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def redact(self, text: str) -> str:
        if not text:
            return ""
            
        # Analyze text for PII entities (names, locations, etc.)
        results = self.analyzer.analyze(text=text, entities=[], language='en')
        
        # Anonymize the detected PII
        anonymized_result = self.anonymizer.anonymize(text=text, analyzer_results=results)
        
        return anonymized_result.text
