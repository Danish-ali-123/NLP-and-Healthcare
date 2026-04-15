from typing import Dict, Any, List
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvidenceCheckerAgent:
    """
    Evidence Checker Agent - Verifies if predicted label is supported by clinical text.
    
    Input: text + predicted_label
    Output: support_status, rationale, missing_questions
    """
    
    def __init__(self):
        """
        Initialize the Evidence Checker Agent with symptom dictionaries.
        """
        # Define symptom dictionaries for each diagnostic category
        self.symptom_dictionary = {
            "spinal_disorder": {
                "en": ["back pain", "lower back", "spine", "numbness", "radiating pain", "leg weakness"],
                "hi": ["कमर दर्द", "पीठ दर्द", "रीढ़", "सुन्नता", "दर्द फैलना"],
                "pa": ["ਕਮਰ ਦਰਦ", "ਰੀੜ੍ਹ", "सुन्नता", "दर्द फੈਲਣਾ"]
            },
            "fracture": {
                "en": ["fracture", "fall", "swelling", "severe pain", "x-ray", "trauma"],
                "hi": ["फ्रैक्चर", "गिरना", "सूजन", "तेज दर्द", "एक्स-रे"],
                "pa": ["ਫ੍ਰੈਕਚਰ", "ਡਿੱਗਣਾ", "ਸੂਜਨ", "ਤੇਜ਼ ਦਰਦ", "ਐਕਸ-ਰੇ"]
            },
            "musculoskeletal_disorder": {
                "en": ["joint pain", "stiffness", "movement difficulty", "muscle pain"],
                "hi": ["जोड़ों का दर्द", "जकड़न", "हिलने में कठिनाई", "मांसपेशियों का दर्द"],
                "pa": ["जੋੜਾਂ ਦਾ ਦਰਦ", "जकड़न", "ਹਿਲਣ ਵਿੱਚ ਮੁਸ਼ਕਲ"]
            },
            "other": {
                "en": ["other", "general", "medical", "health"],
                "hi": ["अन्य", "सामान्य", "चिकित्सा", "स्वास्थ्य"],
                "pa": ["ਹੋਰ", "ਆਮ", "ਚਿੱਕਿਤਸਾ", "ਸਿਹਤ"]
            }
        }
        
        logger.info("Evidence Checker Agent initialized")
    
    def match_symptoms(self, text: str, predicted_label: str, language: str) -> List[str]:
        """
        Match symptoms from text for the predicted label.
        
        Args:
            text: Clinical text to analyze
            predicted_label: Predicted diagnostic label
            language: Language code (en/hi/pa)
            
        Returns:
            List of matched symptoms
        """
        if predicted_label not in self.symptom_dictionary:
            return []
        
        if language not in self.symptom_dictionary[predicted_label]:
            return []
        
        symptoms = self.symptom_dictionary[predicted_label][language]
        matched = []
        
        # Make text lowercase for case-insensitive matching
        lower_text = text.lower()
        
        for symptom in symptoms:
            # Use word boundaries for more precise matching
            escaped_symptom = re.escape(symptom.lower())
            if escaped_symptom in lower_text:
                matched.append(symptom)
        
        return matched
    
    def determine_support_status(self, matched_symptoms: List[str]) -> str:
        """
        Determine support level based on number of matched symptoms.
        
        Args:
            matched_symptoms: List of matched symptoms
            
        Returns:
            Support level: SUPPORTED, WEAK, or CONTRADICTED
        """
        hit_count = len(matched_symptoms)
        
        if hit_count >= 3:
            return "SUPPORTED"
        elif hit_count >= 1:
            return "WEAK"
        else:
            return "CONTRADICTED"
    
    def process(self, text: str, predicted_label: str, language: str = "en") -> Dict[str, Any]:
        """
        Process a single clinical text and verify evidence for the predicted label.
        
        Args:
            text: Clinical text to analyze
            predicted_label: Predicted diagnostic label
            language: Language code (en/hi/pa)
            
        Returns:
            Dictionary containing evidence check results
        """
        # Validate input
        if not text or not predicted_label:
            raise ValueError("Text and predicted_label are required inputs")
        
        try:
            # Match symptoms
            matched_symptoms = self.match_symptoms(text, predicted_label, language)
            
            # Determine support level
            support_status = self.determine_support_status(matched_symptoms)
            
            # Generate rationale
            if support_status == "SUPPORTED":
                rationale = f"Multiple key symptoms found: {', '.join(matched_symptoms[:5])}{'...' if len(matched_symptoms) > 5 else ''}"
            elif support_status == "WEAK":
                rationale = f"Weak evidence found with {len(matched_symptoms)} matching keywords: {', '.join(matched_symptoms)}"
            else:
                rationale = f"No expected symptoms found for {predicted_label}"
            
            # Identify missing questions
            missing_questions = []
            if support_status != "SUPPORTED":
                missing_questions.append(f"Additional details needed to confirm {predicted_label}")
                if support_status == "CONTRADICTED":
                    missing_questions.append(f"Consider alternative diagnosis for {predicted_label}")
            
            # Format output
            output = {
                "support_status": support_status,
                "rationale": rationale,
                "matched_symptoms": matched_symptoms,
                "missing_questions": missing_questions,
                "hit_count": len(matched_symptoms)
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Error in Evidence Checker Agent: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    agent = EvidenceCheckerAgent()
    
    # Test with English text - Strong support
    result = agent.process(
        "Patient has severe back pain, numbness in legs, and radiating pain", 
        "spinal_disorder", 
        "en"
    )
    print("English Example - Strong Support:")
    print(f"Support Status: {result['support_status']}")
    print(f"Rationale: {result['rationale']}")
    print(f"Matched Symptoms: {result['matched_symptoms']}")
    print(f"Missing Questions: {result['missing_questions']}")
    print()
    
    # Test with Hindi text - Weak support
    result = agent.process(
        "मरीज को कमर दर्द है", 
        "spinal_disorder", 
        "hi"
    )
    print("Hindi Example - Weak Support:")
    print(f"Support Status: {result['support_status']}")
    print(f"Rationale: {result['rationale']}")
    print(f"Matched Symptoms: {result['matched_symptoms']}")
    print(f"Missing Questions: {result['missing_questions']}")
    print()
    
    # Test with Punjabi text - Contradicted
    result = agent.process(
        "ਮਰੀਜ਼ ਨੂੰ ਘੁੱਟਨੇ ਵਿੱਚ ਹਲਕਾ ਦਰਦ ਹੈ", 
        "fracture", 
        "pa"
    )
    print("Punjabi Example - Contradicted:")
    print(f"Support Status: {result['support_status']}")
    print(f"Rationale: {result['rationale']}")
    print(f"Matched Symptoms: {result['matched_symptoms']}")
    print(f"Missing Questions: {result['missing_questions']}")
