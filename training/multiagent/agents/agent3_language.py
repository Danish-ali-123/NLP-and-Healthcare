from typing import Dict, Any, List
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LanguageConsistencyAgent:
    """
    Language Consistency Agent - Checks language-related issues in clinical text.
    
    Input: text + language + predicted_label
    Output: lang_risk {LOW, MED, HIGH}, issues_list, suggestion
    """
    
    def __init__(self):
        """
        Initialize the Language Consistency Agent with language-specific patterns.
        """
        # Define language-specific character ranges and patterns
        self.language_patterns = {
            'en': {
                'name': 'English',
                'script_range': [0x0041, 0x005A, 0x0061, 0x007A],  # ASCII letters
                'min_devanagari_ratio': 0.0,
                'min_gurmukhi_ratio': 0.0
            },
            'hi': {
                'name': 'Hindi',
                'script_range': [0x0900, 0x097F],  # Devanagari characters
                'min_devanagari_ratio': 0.5,
                'min_gurmukhi_ratio': 0.0
            },
            'pa': {
                'name': 'Punjabi',
                'script_range': [0x0A00, 0x0A7F],  # Gurmukhi characters
                'min_devanagari_ratio': 0.0,
                'min_gurmukhi_ratio': 0.5
            }
        }
        
        # Define expected medical keywords by language and label
        self.medical_keywords = {
            'spinal_disorder': {
                'en': ['back', 'spine', 'numbness'],
                'hi': ['कमर', 'रीढ़', 'सुन्नता'],
                'pa': ['ਕਮਰ', 'ਰੀੜ੍ਹ', 'ਸੁੰਨਤਾ']
            },
            'fracture': {
                'en': ['fracture', 'pain', 'swelling'],
                'hi': ['फ्रैक्चर', 'दर्द', 'सूजन'],
                'pa': ['ਫ੍ਰੈਕਚਰ', 'ਦਰਦ', 'ਸੂਜਨ']
            },
            'musculoskeletal_disorder': {
                'en': ['joint', 'muscle', 'stiffness'],
                'hi': ['जोड़', 'मांसपेशी', 'जकड़न'],
                'pa': ['ਜੋੜ', 'ਮਾਸਪੇਸ਼ੀ', 'ਜਕੜਨ']
            }
        }
        
        logger.info("Language Consistency Agent initialized")
    
    def is_char_in_range(self, char: str, char_range: List[int]) -> bool:
        """
        Check if a character is in the expected Unicode range.
        
        Args:
            char: Character to check
            char_range: List of Unicode ranges [start1, end1, start2, end2, ...]
            
        Returns:
            True if character is in range, False otherwise
        """
        char_code = ord(char)
        for i in range(0, len(char_range), 2):
            start = char_range[i]
            end = char_range[i+1]
            if start <= char_code <= end:
                return True
        return False
    
    def count_script_chars(self, text: str) -> Dict[str, int]:
        """
        Count characters from different scripts in the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with counts of characters from different scripts
        """
        counts = {
            'total': 0,
            'devanagari': 0,
            'gurmukhi': 0,
            'latin': 0,
            'other': 0
        }
        
        for char in text:
            if char.isalnum():
                counts['total'] += 1
                if 0x0900 <= ord(char) <= 0x097F:  # Devanagari
                    counts['devanagari'] += 1
                elif 0x0A00 <= ord(char) <= 0x0A7F:  # Gurmukhi
                    counts['gurmukhi'] += 1
                elif 0x0041 <= ord(char) <= 0x005A or 0x0061 <= ord(char) <= 0x007A:  # Latin
                    counts['latin'] += 1
                else:
                    counts['other'] += 1
        
        return counts
    
    def detect_mixed_scripts(self, text: str, language: str) -> Dict[str, Any]:
        """
        Detect mixed scripts in text.
        
        Args:
            text: Clinical text to analyze
            language: Language code (en/hi/pa)
            
        Returns:
            Dictionary with mixed script detection results
        """
        script_counts = self.count_script_chars(text)
        
        if script_counts['total'] == 0:
            return {
                'mixed_script': False,
                'rationale': 'No alphanumeric characters to analyze'
            }
        
        # Calculate ratios
        devanagari_ratio = script_counts['devanagari'] / script_counts['total']
        gurmukhi_ratio = script_counts['gurmukhi'] / script_counts['total']
        latin_ratio = script_counts['latin'] / script_counts['total']
        
        issues = []
        
        # Check expected script ratios
        if language == 'hi' and devanagari_ratio < self.language_patterns['hi']['min_devanagari_ratio']:
            issues.append(f"Low Devanagari content ({devanagari_ratio:.1%}) for Hindi text")
        
        if language == 'pa' and gurmukhi_ratio < self.language_patterns['pa']['min_gurmukhi_ratio']:
            issues.append(f"Low Gurmukhi content ({gurmukhi_ratio:.1%}) for Punjabi text")
        
        # Check for high romanization
        if (language in ['hi', 'pa']) and latin_ratio > 0.7:
            issues.append(f"High romanization ({latin_ratio:.1%} Latin characters)")
        
        return {
            'mixed_script': len(issues) > 0,
            'rationale': "\n".join(issues) if issues else "Script ratios are acceptable",
            'issues': issues
        }
    
    def check_expected_keywords(self, text: str, language: str, predicted_label: str) -> Dict[str, Any]:
        """
        Check if expected medical keywords are present for the predicted label.
        
        Args:
            text: Clinical text to analyze
            language: Language code (en/hi/pa)
            predicted_label: Predicted diagnostic label
            
        Returns:
            Dictionary with keyword check results
        """
        if predicted_label not in self.medical_keywords:
            return {
                'missing_keywords': False,
                'rationale': f"No expected keywords defined for label: {predicted_label}"
            }
        
        if language not in self.medical_keywords[predicted_label]:
            return {
                'missing_keywords': False,
                'rationale': f"No expected keywords defined for {language} and label: {predicted_label}"
            }
        
        expected_keywords = self.medical_keywords[predicted_label][language]
        
        # Count matching keywords
        matched_keywords = []
        for keyword in expected_keywords:
            if keyword in text.lower():
                matched_keywords.append(keyword)
        
        if len(matched_keywords) == 0:
            return {
                'missing_keywords': True,
                'rationale': f"No expected medical keywords found for {predicted_label}",
                'expected_keywords': expected_keywords
            }
        
        return {
            'missing_keywords': False,
            'rationale': f"Found {len(matched_keywords)} expected keywords: {', '.join(matched_keywords)}",
            'matched_keywords': matched_keywords
        }
    
    def detect_unusual_tokens(self, text: str) -> Dict[str, Any]:
        """
        Detect unusual tokens or patterns in text.
        
        Args:
            text: Clinical text to analyze
            
        Returns:
            Dictionary with unusual token detection results
        """
        issues = []
        
        # Check for excessive punctuation
        if len(re.findall(r'[^\w\s]', text)) > len(text) * 0.1:
            issues.append("Excessive punctuation detected")
        
        # Check for very long words
        words = text.split()
        long_words = [word for word in words if len(word) > 20]
        if long_words:
            issues.append(f"Unusually long words detected: {', '.join(long_words[:2])}")
        
        return {
            'unusual_tokens': len(issues) > 0,
            'rationale': "\n".join(issues) if issues else "No unusual tokens detected",
            'issues': issues
        }
    
    def calculate_risk_level(self, issues_list: List[str]) -> str:
        """
        Calculate overall language risk level based on detected issues.
        
        Args:
            issues_list: List of detected issues
            
        Returns:
            Risk level: LOW, MED, or HIGH
        """
        num_issues = len(issues_list)
        
        if num_issues == 0:
            return "LOW"
        elif num_issues == 1:
            return "MED"
        else:
            return "HIGH"
    
    def process(self, text: str, language: str, predicted_label: str) -> Dict[str, Any]:
        """
        Process a single clinical text and check for language consistency issues.
        
        Args:
            text: Clinical text to analyze
            language: Language code (en/hi/pa)
            predicted_label: Predicted diagnostic label
            
        Returns:
            Dictionary containing language consistency check results
        """
        # Validate input
        if not text or not language or not predicted_label:
            raise ValueError("Text, language, and predicted_label are required inputs")
        
        if language not in ['en', 'hi', 'pa']:
            raise ValueError(f"Unsupported language: {language}. Supported languages: en/hi/pa")
        
        try:
            # Detect issues
            mixed_script_result = self.detect_mixed_scripts(text, language)
            keywords_result = self.check_expected_keywords(text, language, predicted_label)
            unusual_tokens_result = self.detect_unusual_tokens(text)
            
            # Compile all issues
            issues_list = []
            
            if mixed_script_result['mixed_script']:
                issues_list.extend(mixed_script_result['issues'])
            
            if keywords_result['missing_keywords']:
                issues_list.append(keywords_result['rationale'])
            
            if unusual_tokens_result['unusual_tokens']:
                issues_list.extend(unusual_tokens_result['issues'])
            
            # Calculate risk level
            lang_risk = self.calculate_risk_level(issues_list)
            
            # Generate suggestion
            if lang_risk == "HIGH":
                suggestion = f"Text has significant language issues. Consider review by a {self.language_patterns[language]['name']} speaker."
            elif lang_risk == "MED":
                suggestion = f"Text has some language issues. Consider verification."
            else:
                suggestion = f"Text has acceptable language consistency."
            
            # Format output
            output = {
                "lang_risk": lang_risk,
                "issues_list": issues_list,
                "suggestion": suggestion,
                "details": {
                    "mixed_script_check": mixed_script_result,
                    "keywords_check": keywords_result,
                    "unusual_tokens_check": unusual_tokens_result
                }
            }
            
            return output
            
        except Exception as e:
            logger.error(f"Error in Language Consistency Agent: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    agent = LanguageConsistencyAgent()
    
    # Test with English text - Low risk
    result = agent.process(
        "Patient has severe back pain", 
        "en", 
        "spinal_disorder"
    )
    print("English Example - Low Risk:")
    print(f"Language Risk: {result['lang_risk']}")
    print(f"Issues: {result['issues_list']}")
    print(f"Suggestion: {result['suggestion']}")
    print()
    
    # Test with Hindi text - High risk (mostly romanized)
    result = agent.process(
        "Patient ko kamar dard hai", 
        "hi", 
        "spinal_disorder"
    )
    print("Hindi Example - High Risk (Romanized):")
    print(f"Language Risk: {result['lang_risk']}")
    print(f"Issues: {result['issues_list']}")
    print(f"Suggestion: {result['suggestion']}")
    print()
    
    # Test with Punjabi text - Medium risk (missing keywords)
    result = agent.process(
        "ਮਰੀਜ਼ ਨੂੰ ਦਰਦ ਹੈ", 
        "pa", 
        "fracture"
    )
    print("Punjabi Example - Medium Risk (Missing Keywords):")
    print(f"Language Risk: {result['lang_risk']}")
    print(f"Issues: {result['issues_list']}")
    print(f"Suggestion: {result['suggestion']}")
