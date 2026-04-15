import pandas as pd
import logging
import os
import json
from typing import Dict, Any, List
from pathlib import Path
import sys

# Add the parent directory to path to import agents
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import agents
from agents.agent1_clinical import ClinicalAnalyzerAgent
from agents.agent2_evidence import EvidenceCheckerAgent
from agents.agent3_language import LanguageConsistencyAgent
from agents.agent4_hil_gate import HILGateAgent

class PipelineRunner:
    """
    Pipeline Runner - Orchestrates the execution of all four agents in sequence.
    """
    
    def __init__(self):
        """
        Initialize the Pipeline Runner with all required agents.
        """
        logger.info("Initializing pipeline agents...")
        
        # Initialize all agents
        self.clinical_agent = ClinicalAnalyzerAgent()
        self.evidence_agent = EvidenceCheckerAgent()
        self.language_agent = LanguageConsistencyAgent()
        self.hil_agent = HILGateAgent()
        
        logger.info("All agents initialized successfully")
    
    def load_test_data(self, test_path: str) -> pd.DataFrame:
        """
        Load test data from CSV file.
        
        Args:
            test_path: Path to test CSV file
            
        Returns:
            DataFrame containing test data
        """
        logger.info(f"Loading test data from: {test_path}")
        
        # Read CSV file
        test_data = pd.read_csv(test_path)
        logger.info(f"Loaded {len(test_data)} samples")
        
        # Validate required columns
        required_columns = ['id', 'text', 'language', 'true_label']
        for col in required_columns:
            if col not in test_data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return test_data
    
    def process_sample(self, sample: pd.Series, ablation_mode: str = None) -> Dict[str, Any]:
        """
        Process a single sample through the entire pipeline.
        
        Args:
            sample: Single row from test data DataFrame
            ablation_mode: Ablation mode (
                None: Full pipeline with all agents,
                'base_model': Only clinical analyzer,
                'evidence_only': Clinical analyzer + evidence checker,
                'language_only': Clinical analyzer + language consistency checker
            )
            
        Returns:
            Dictionary containing all agent outputs and pipeline result
        """
        sample_id = sample['id']
        text = sample['text']
        language = sample['language']
        true_label = sample['true_label']
        
        logger.debug(f"Processing sample: {sample_id} ({language}) with ablation_mode: {ablation_mode}")
        
        # Step 1: Clinical Analysis (always run)
        clinical_result = self.clinical_agent.process(text, language)
        predicted_label = clinical_result['predicted_label']
        confidence = clinical_result['confidence']
        
        # Step 2: Evidence Checking (conditionally run based on ablation)
        if ablation_mode in ['base_model', 'language_only']:
            # Skip evidence checking
            evidence_result = {
                "support_status": "SUPPORTED",
                "rationale": "Evidence checking skipped (ablation mode)",
                "matched_symptoms": [],
                "missing_questions": [],
                "hit_count": 0
            }
        else:
            evidence_result = self.evidence_agent.process(text, predicted_label, language)
        
        evidence_status = evidence_result['support_status']
        
        # Step 3: Language Consistency (conditionally run based on ablation)
        if ablation_mode in ['base_model', 'evidence_only']:
            # Skip language consistency checking
            language_result = {
                "lang_risk": "LOW",
                "issues_list": [],
                "suggestion": "Language consistency checking skipped (ablation mode)",
                "details": {}
            }
        else:
            language_result = self.language_agent.process(text, language, predicted_label)
        
        lang_risk = language_result['lang_risk']
        
        # Step 4: HIL Gate (conditionally run based on ablation)
        if ablation_mode == 'base_model':
            # Skip HIL gate - AUTO_ACCEPT all in base model mode
            hil_result = {
                "decision": "AUTO_ACCEPT",
                "reason_codes": ["BASE_MODEL_AUTO_ACCEPT"],
                "priority": "LOW",
                "reason_text": "Base model mode - all predictions auto-accepted",
                "input_parameters": {
                    "confidence": confidence,
                    "evidence_status": evidence_status,
                    "lang_risk": lang_risk
                }
            }
        elif ablation_mode in ['evidence_only', 'language_only']:
            # Skip HIL gate - AUTO_ACCEPT all in partial modes
            hil_result = {
                "decision": "AUTO_ACCEPT",
                "reason_codes": ["PARTIAL_MODE_AUTO_ACCEPT"],
                "priority": "LOW",
                "reason_text": "Partial mode - all predictions auto-accepted",
                "input_parameters": {
                    "confidence": confidence,
                    "evidence_status": evidence_status,
                    "lang_risk": lang_risk
                }
            }
        else:
            # Full pipeline - run HIL gate
            hil_result = self.hil_agent.process(confidence, evidence_status, lang_risk)
        
        # Compile all results
        pipeline_result = {
            'id': sample_id,
            'text': text,
            'language': language,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'confidence': confidence,
            'evidence_status': evidence_status,
            'lang_risk': lang_risk,
            'hil_decision': hil_result['decision'],
            'hil_reason_codes': hil_result['reason_codes'],
            'hil_priority': hil_result['priority'],
            'hil_reason_text': hil_result['reason_text'],
            'ablation_mode': ablation_mode,
            'agent_results': {
                'clinical': clinical_result,
                'evidence': evidence_result,
                'language': language_result,
                'hil': hil_result
            }
        }
        
        return pipeline_result
    
    def run_pipeline(self, test_path: str, output_dir: str, ablation_mode: str = None):
        """
        Run the entire pipeline on the test dataset.
        
        Args:
            test_path: Path to test CSV file
            output_dir: Directory to save output files
            ablation_mode: Ablation mode (None, 'base_model', 'evidence_only', 'language_only', 'full_pipeline')
        """
        # Handle full_pipeline mode (same as no ablation)
        if ablation_mode == 'full_pipeline':
            ablation_mode = None
        
        logger.info(f"Starting pipeline execution with ablation_mode: {ablation_mode}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load test data
        test_data = self.load_test_data(test_path)
        
        # Process samples
        pipeline_results = []
        audit_logs = []
        
        for idx, sample in test_data.iterrows():
            try:
                # Process sample through pipeline
                result = self.process_sample(sample, ablation_mode)
                
                pipeline_results.append(result)
                audit_logs.append(result)
                
                if (idx + 1) % 100 == 0:  # Log every 100 samples for faster execution
                    logger.info(f"Processed {idx + 1}/{len(test_data)} samples")
                    
            except Exception as e:
                logger.error(f"Error processing sample {sample['id']}: {str(e)}")
                continue
        
        # Save results
        self._save_results(pipeline_results, audit_logs, output_dir, ablation_mode)
        
        logger.info(f"Pipeline execution completed. Processed {len(pipeline_results)} samples")
    
    def _save_results(self, pipeline_results: List[Dict[str, Any]], audit_logs: List[Dict[str, Any]], output_dir: str, ablation_mode: str):
        """
        Save pipeline results to output files.
        
        Args:
            pipeline_results: List of pipeline results
            audit_logs: List of audit logs
            output_dir: Directory to save output files
            ablation_mode: Ablation mode used
        """
        # Determine suffix based on ablation mode
        suffix = f"_{ablation_mode}" if ablation_mode else ""
        
        # Save pipeline predictions CSV
        logger.info(f"Saving pipeline predictions to output directory")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(pipeline_results)
        
        # Select columns for output CSV
        output_columns = [
            'id', 'language', 'predicted_label', 'confidence',
            'evidence_status', 'lang_risk', 'hil_decision',
            'hil_reason_codes', 'hil_priority', 'hil_reason_text',
            'true_label', 'ablation_mode'
        ]
        
        # Save to CSV
        predictions_path = os.path.join(output_dir, f'pipeline_predictions{suffix}.csv')
        results_df[output_columns].to_csv(
            predictions_path,
            index=False,
            encoding='utf-8'
        )
        logger.info(f"Saved pipeline predictions to: {predictions_path}")
        
        # Save audit logs as JSONL
        audit_path = os.path.join(output_dir, f'audit_logs{suffix}.jsonl')
        logger.info(f"Saving audit logs to: {audit_path}")
        
        with open(audit_path, 'w', encoding='utf-8') as f:
            for log in audit_logs:
                f.write(json.dumps(log, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved audit logs to: {audit_path}")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the Agent-Assisted Diagnostic Validation Pipeline')
    
    # Input arguments
    parser.add_argument('--test_path', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--output_dir', type=str, default='../results',
                        help='Directory to save output files')
    parser.add_argument('--ablation_mode', type=str, default=None,
                        choices=['base_model', 'evidence_only', 'language_only', 'full_pipeline'],
                        help='Ablation mode to test pipeline components: '
                             'base_model - only clinical analyzer, '
                             'evidence_only - clinical + evidence, '
                             'language_only - clinical + language, '
                             'full_pipeline - all components (same as default)')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = PipelineRunner()
    pipeline.run_pipeline(args.test_path, args.output_dir, args.ablation_mode)
