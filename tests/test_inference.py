"""
Test script for the inference engine.
"""
import os
import json
import pytest
from pathlib import Path
from datetime import datetime
from src.inference import InferenceEngine, AssessmentResult, LLMMetrics

# Sample test data
SAMPLE_DOCUMENT = """
Risk Assessment Report

Company: Test Corp
Date: 2024-03-20

Risk Factors:
1. Inadequate cybersecurity measures
2. Non-compliance with GDPR requirements
3. Insufficient disaster recovery planning

Current Status:
- Security protocols need updating
- Data processing procedures require review
- Backup systems need improvement

Recommendations:
1. Implement multi-factor authentication
2. Conduct GDPR compliance audit
3. Develop comprehensive disaster recovery plan
"""

@pytest.fixture
def sample_document_path(tmp_path):
    """Create a temporary sample document for testing."""
    doc_path = tmp_path / "sample_assessment.txt"
    doc_path.write_text(SAMPLE_DOCUMENT)
    return doc_path

@pytest.fixture
def inference_engine():
    """Create an inference engine instance with test API keys."""
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not anthropic_api_key:
        pytest.skip("Missing required Anthropic API key")
    
    return InferenceEngine(anthropic_api_key=anthropic_api_key)

def test_assessment_result_schema():
    """Test the AssessmentResult schema validation."""
    # Valid assessment result
    valid_result = AssessmentResult(
        risk_level="HIGH",
        risk_factors=["Inadequate cybersecurity", "GDPR non-compliance"],
        mitigation_strategies=["Implement MFA", "Conduct compliance audit"],
        compliance_status="PARTIAL",
        required_actions=["Update security protocols", "Review data processing"],
        assessment_date="2024-03-20"
    )
    
    # Test invalid risk level
    with pytest.raises(ValueError):
        AssessmentResult(
            risk_level="INVALID",
            risk_factors=["Test"],
            mitigation_strategies=["Test"],
            compliance_status="COMPLIANT",
            required_actions=["Test"],
            assessment_date="2024-03-20"
        )

def test_llm_metrics_schema():
    """Test the LLMMetrics schema validation."""
    # Valid metrics
    valid_metrics = LLMMetrics(
        model_name="claude-sonnet-4-20250514",
        confidence_score=0.85,
        processing_time=1.5,
        token_count=1000
    )
    
    # Test invalid confidence score
    with pytest.raises(ValueError):
        LLMMetrics(
            model_name="claude-sonnet-4-20250514",
            confidence_score=1.5,  # Should be between 0 and 1
            processing_time=1.5,
            token_count=1000
        )

def test_process_document(inference_engine, sample_document_path, tmp_path):
    """Test document processing and result generation."""
    # Process the document
    assessment, metrics = inference_engine.process_document(sample_document_path)
    
    # Test assessment result
    assert isinstance(assessment, AssessmentResult)
    assert assessment.risk_level in ["LOW", "MEDIUM", "HIGH"]
    assert isinstance(assessment.risk_factors, list)
    assert isinstance(assessment.mitigation_strategies, list)
    assert assessment.compliance_status in ["COMPLIANT", "PARTIAL", "NON-COMPLIANT"]
    assert isinstance(assessment.required_actions, list)
    assert isinstance(assessment.assessment_date, str)
    
    # Test metrics
    assert isinstance(metrics, LLMMetrics)
    assert 0 <= metrics.confidence_score <= 1
    assert metrics.processing_time > 0
    assert metrics.token_count > 0

def test_save_results(inference_engine, sample_document_path, tmp_path):
    """Test saving results to files."""
    # Process the document
    assessment, metrics = inference_engine.process_document(sample_document_path)
    
    # Save results
    output_dir = tmp_path / "test_output"
    inference_engine.save_results(assessment, metrics, output_dir)
    
    # Check if files exist
    assert (output_dir / "assessment_result.json").exists()
    assert (output_dir / "llm_metrics.csv").exists()
    
    # Verify JSON content
    with open(output_dir / "assessment_result.json") as f:
        saved_assessment = json.load(f)
        assert isinstance(saved_assessment, dict)
        assert "risk_level" in saved_assessment
        assert "risk_factors" in saved_assessment
        assert "mitigation_strategies" in saved_assessment
        assert "compliance_status" in saved_assessment
        assert "required_actions" in saved_assessment
        assert "assessment_date" in saved_assessment

def test_error_handling(inference_engine, tmp_path):
    """Test error handling for invalid inputs."""
    # Test with non-existent file
    with pytest.raises(FileNotFoundError):
        inference_engine.process_document(tmp_path / "nonexistent.txt")
    
    # Test with empty file
    empty_file = tmp_path / "empty.txt"
    empty_file.touch()
    with pytest.raises(ValueError):
        inference_engine.process_document(empty_file)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 