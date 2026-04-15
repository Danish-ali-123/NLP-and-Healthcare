from typing import Dict, Any, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HILGateAgent:
    """
    Human-in-the-Loop (HIL) Gate Agent - Decides if prediction should be AUTO_ACCEPT or REQUIRE_REVIEW.
    
    Input: confidence + evidence_status + lang_risk
    Output: decision, reason_codes
    """
    
    def __init__(self):
        """
        Initialize the HIL Gate Agent with decision rules.
        """
        # Define decision rules
        self.decision_rules = {
            "REQUIRE_REVIEW": [
                # High priority review conditions
                {"condition": lambda c, e, l: c < 0.60, "reason_code": "LOW_CONFIDENCE", "priority": "HIGH"},
                {"condition": lambda c, e, l: e == "CONTRADICTED", "reason_code": "EVIDENCE_CONTRADICTED", "priority": "HIGH"},
                {"condition": lambda c, e, l: l == "HIGH", "reason_code": "HIGH_LANG_RISK", "priority": "HIGH"},
                # Medium priority review conditions
                {"condition": lambda c, e, l: e == "WEAK", "reason_code": "WEAK_EVIDENCE", "priority": "MEDIUM"},
                {"condition": lambda c, e, l: l == "MED", "reason_code": "MEDIUM_LANG_RISK", "priority": "MEDIUM"}
            ]
        }
        
        logger.info("HIL Gate Agent initialized")
    
    def make_decision(self, confidence: float, evidence_status: str, lang_risk: str) -> Dict[str, Any]:
        """
        Make HIL decision based on input parameters.
        
        Args:
            confidence: Model confidence score (0-1)
            evidence_status: Evidence support status (SUPPORTED/WEAK/CONTRADICTED)
            lang_risk: Language risk level (LOW/MED/HIGH)
            
        Returns:
            Dictionary containing decision and reason codes
        """
        # Validate input
        if not (0 <= confidence <= 1):
            raise ValueError(f"Confidence must be between 0 and 1, got {confidence}")
        
        if evidence_status not in ["SUPPORTED", "WEAK", "CONTRADICTED"]:
            raise ValueError(f"Invalid evidence_status: {evidence_status}. Must be SUPPORTED/WEAK/CONTRADICTED")
        
        if lang_risk not in ["LOW", "MED", "HIGH"]:
            raise ValueError(f"Invalid lang_risk: {lang_risk}. Must be LOW/MED/HIGH")
        
        # Check REQUIRE_REVIEW conditions
        reason_codes = []
        priorities = []
        
        for rule in self.decision_rules["REQUIRE_REVIEW"]:
            if rule["condition"](confidence, evidence_status, lang_risk):
                reason_codes.append(rule["reason_code"])
                priorities.append(rule["priority"])
        
        if reason_codes:
            # Determine overall priority
            priority = "HIGH" if "HIGH" in priorities else "MEDIUM"
            
            return {
                "decision": "REQUIRE_REVIEW",
                "reason_codes": reason_codes,
                "priority": priority,
                "reason_text": self._generate_reason_text(reason_codes)
            }
        
        # If no review conditions met, AUTO_ACCEPT
        return {
            "decision": "AUTO_ACCEPT",
            "reason_codes": ["ALL_CRITERIA_MET"],
            "priority": "LOW",
            "reason_text": "All criteria met for auto-acceptance"
        }
    
    def _generate_reason_text(self, reason_codes: List[str]) -> str:
        """
        Generate human-readable reason text from reason codes.
        
        Args:
            reason_codes: List of reason codes
            
        Returns:
            Human-readable reason text
        """
        reason_map = {
            "LOW_CONFIDENCE": "Confidence score below threshold (0.60)",
            "EVIDENCE_CONTRADICTED": "Evidence contradicts predicted diagnosis",
            "HIGH_LANG_RISK": "High language consistency risk detected",
            "WEAK_EVIDENCE": "Weak evidence for predicted diagnosis",
            "MEDIUM_LANG_RISK": "Medium language consistency risk detected",
            "ALL_CRITERIA_MET": "All acceptance criteria met"
        }
        
        reasons = [reason_map[code] for code in reason_codes]
        return "; ".join(reasons)
    
    def process(self, confidence: float, evidence_status: str, lang_risk: str) -> Dict[str, Any]:
        """
        Process input parameters and generate HIL decision.
        
        Args:
            confidence: Model confidence score (0-1)
            evidence_status: Evidence support status (SUPPORTED/WEAK/CONTRADICTED)
            lang_risk: Language risk level (LOW/MED/HIGH)
            
        Returns:
            Dictionary containing HIL decision results
        """
        try:
            # Make decision
            decision = self.make_decision(confidence, evidence_status, lang_risk)
            
            # Add input parameters to output for audit
            decision["input_parameters"] = {
                "confidence": confidence,
                "evidence_status": evidence_status,
                "lang_risk": lang_risk
            }
            
            return decision
            
        except Exception as e:
            logger.error(f"Error in HIL Gate Agent: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    agent = HILGateAgent()
    
    # Test case 1: AUTO_ACCEPT
    result = agent.process(0.85, "SUPPORTED", "LOW")
    print("Case 1 - AUTO_ACCEPT:")
    print(f"Decision: {result['decision']}")
    print(f"Priority: {result['priority']}")
    print(f"Reason Codes: {result['reason_codes']}")
    print(f"Reason Text: {result['reason_text']}")
    print()
    
    # Test case 2: REQUIRE_REVIEW - LOW_CONFIDENCE
    result = agent.process(0.55, "SUPPORTED", "LOW")
    print("Case 2 - REQUIRE_REVIEW (Low Confidence):")
    print(f"Decision: {result['decision']}")
    print(f"Priority: {result['priority']}")
    print(f"Reason Codes: {result['reason_codes']}")
    print(f"Reason Text: {result['reason_text']}")
    print()
    
    # Test case 3: REQUIRE_REVIEW - CONTRADICTED EVIDENCE
    result = agent.process(0.75, "CONTRADICTED", "LOW")
    print("Case 3 - REQUIRE_REVIEW (Contradicted Evidence):")
    print(f"Decision: {result['decision']}")
    print(f"Priority: {result['priority']}")
    print(f"Reason Codes: {result['reason_codes']}")
    print(f"Reason Text: {result['reason_text']}")
    print()
    
    # Test case 4: REQUIRE_REVIEW - HIGH_LANG_RISK
    result = agent.process(0.80, "SUPPORTED", "HIGH")
    print("Case 4 - REQUIRE_REVIEW (High Language Risk):")
    print(f"Decision: {result['decision']}")
    print(f"Priority: {result['priority']}")
    print(f"Reason Codes: {result['reason_codes']}")
    print(f"Reason Text: {result['reason_text']}")
    print()
    
    # Test case 5: REQUIRE_REVIEW - WEAK EVIDENCE
    result = agent.process(0.70, "WEAK", "LOW")
    print("Case 5 - REQUIRE_REVIEW (Weak Evidence):")
    print(f"Decision: {result['decision']}")
    print(f"Priority: {result['priority']}")
    print(f"Reason Codes: {result['reason_codes']}")
    print(f"Reason Text: {result['reason_text']}")
