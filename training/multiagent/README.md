# Agent-Based Validation Framework for Clinical Diagnosis

## Overview

This implementation provides a clean, production-quality pipeline for agent-based validation of clinical diagnosis predictions. The framework follows a single-pass, deterministic workflow with comprehensive audit logging, designed for hospital/CDSS (Clinical Decision Support System) environments.

## Goal

- Take test data (id, text, language, true_label)
- Run trained classifier (IndicBERT-HPA for HI/PA/EN) → label + confidence + top-k probs
- Run lightweight validation agents → evidence + language consistency
- Apply HIL (Human-in-the-Loop) gating rules → AUTO_ACCEPT or REQUIRE_REVIEW
- Save per-sample outputs + aggregated metrics
- Support ablation studies: Base model only vs +Evidence vs +Language vs +HIL gate (full)

## Folder Structure

```
multiagent/
├── agents/
│   ├── agent1_clinical.py      # Clinical Analyzer Agent
│   ├── agent2_evidence.py       # Evidence Checker Agent
│   ├── agent3_language.py       # Language Consistency Agent
│   └── agent4_hil_gate.py       # HIL Gate Agent
├── pipeline/
│   ├── run_pipeline.py          # Pipeline Orchestrator
│   └── evaluate_pipeline.py     # Evaluation Script
├── test_sample.csv              # Sample test data
└── README.md                    # This file
```

## Agents

### Agent 1: Clinical Analyzer
- **Input**: text, language
- **Output**: predicted_label, confidence, probs/top_k, model_name_used
- Uses trained IndicBERT-HPA models for each language (EN/HI/PA)

### Agent 2: Evidence Checker
- **Input**: text + predicted_label + language
- Uses symptom dictionary matching with hit_count thresholds:
  - STRONG (≥3 matches) → SUPPORTED
  - WEAK (1-2 matches) → WEAK
  - 0 matches → CONTRADICTED
- **Output**: support_status, rationale, matched_symptoms, missing_questions

### Agent 3: Language Consistency Agent
- **Input**: text + language + predicted_label
- Detects: mixed scripts, romanization, unusual tokens, missing expected medical keywords
- **Output**: lang_risk {LOW, MED, HIGH}, issues_list, suggestion

### Agent 4: HIL Gate
- **Input**: confidence + evidence_status + lang_risk
- **Rules**:
  - If confidence < 0.60 OR evidence=CONTRADICTED OR lang_risk=HIGH → REQUIRE_REVIEW (HIGH priority)
  - If evidence=WEAK OR lang_risk=MED → REQUIRE_REVIEW (MEDIUM priority)
  - Else → AUTO_ACCEPT
- **Output**: decision, reason_codes, priority, reason_text

## Pipeline Execution

### Prerequisites

- Python 3.7+
- pandas
- numpy
- scikit-learn

### Running the Pipeline

```bash
# Run the full pipeline with default settings
python -m pipeline.run_pipeline --test_path test_sample.csv --output_dir results

# Run with ablation (e.g., no evidence checking)
python -m pipeline.run_pipeline --test_path test_sample.csv --output_dir results --ablation_mode no_evidence

# Available ablation modes:
# - no_evidence: Skip evidence checking
# - no_language: Skip language consistency checking  
# - no_hil: Skip HIL gate (always AUTO_ACCEPT)
```

### Pipeline Input Format

The test CSV file should have the following columns:
- `id`: Unique identifier for each sample
- `text`: Clinical text to analyze
- `language`: Language code (en/hi/pa)
- `true_label`: Ground truth label

### Pipeline Output

The pipeline generates two main output files:

1. **pipeline_predictions.csv**: Per-sample results with all agent outputs
2. **audit_logs.jsonl**: Detailed audit logs in JSONL format

## Evaluation

### Running Evaluation

```bash
# Evaluate a single pipeline run
python -m pipeline.evaluate_pipeline --results_path results/pipeline_predictions.csv --output_dir results

# Run ablation study comparison
python -m pipeline.evaluate_pipeline --ablation_study --ablation_files results/pipeline_predictions.csv results/pipeline_predictions_no_evidence.csv --output_dir results
```

### Metrics Generated

#### Model Performance Metrics
- Accuracy
- F1 Macro
- Balanced Accuracy
- Confusion Matrix
- AUROC (if probabilities available)
- AUPRC (if probabilities available)

#### HIL Performance Metrics
- **Coverage**: % of samples AUTO_ACCEPTED
- **Escalations**: % of samples requiring REVIEW
- **AUTO_ACCEPT Accuracy**: Accuracy on auto-accepted samples
- **System Accuracy After Review**: Accuracy if escalated samples are reviewed

#### Language-Specific Metrics
All metrics are computed separately for each language (EN/HI/PA)

## Ablation Studies

The framework supports ablation studies to evaluate the impact of each component:

| Ablation Mode | Description |
|---------------|-------------|
| Full Pipeline | All components enabled (Clinical + Evidence + Language + HIL) |
| no_evidence | Disable Evidence Checker |
| no_language | Disable Language Consistency Checker |
| no_hil | Disable HIL Gate (always AUTO_ACCEPT) |

## Usage Examples

### Example 1: Basic Pipeline Run

```bash
python -m pipeline.run_pipeline --test_path test_sample.csv --output_dir results
```

### Example 2: Run with Ablation

```bash
python -m pipeline.run_pipeline --test_path test_sample.csv --output_dir results --ablation_mode no_evidence
```

### Example 3: Evaluate Results

```bash
python -m pipeline.evaluate_pipeline --results_path results/pipeline_predictions.csv --output_dir results
```

## Configuration

The pipeline uses a modular design that allows easy configuration:

1. **Agent Configuration**: Each agent can be configured individually
2. **Thresholds**: Decision thresholds can be adjusted in agent implementations
3. **Symptom Dictionary**: Can be extended with additional symptoms and languages
4. **Decision Rules**: HIL gate rules can be modified based on clinical requirements

## Audit Logging

All pipeline decisions are logged in JSONL format with:
- Sample identifiers and metadata
- Complete agent outputs
- Decision rationale and reason codes
- Timestamps for tracking

This provides a comprehensive audit trail for clinical governance and compliance.

## Design Constraints

- **No loops**: Single-pass workflow
- **No agent-to-agent dialogue**: Deterministic execution
- **Auditable**: Complete logging of all decisions
- **Healthcare-safe**: Deterministic, rule-based validation
- **Scalable**: Modular design allows easy extension

## Future Enhancements

- Add optional LLM-based evidence verification as fallback
- Support for more languages
- Integration with actual IndicBERT-HPA model inference
- Real-time monitoring dashboard
- Automated quality assurance checks

## License

This implementation is provided for clinical research and educational purposes.

## Contact

For questions or issues, please contact the development team.