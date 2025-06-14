from pathlib import Path
from typing import List, Dict, Any
import time
import csv
import os
import json
from dataclasses import dataclass
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from src.data import process_file


class AssessmentResult(BaseModel):
    
    Seniority: int = Field(description="Seniority in years")
    Policies_in_force: int = Field(description="Number of policies currently in force")
    Max_policies: int = Field(description="Maximum number of policies allowed")
    Max_products: int = Field(description="Maximum number of products allowed")
    Lapse: int = Field(description="Number of lapses")
    Date_lapse: str = Field(description="Date of last lapse (YYYY-MM-DD)")
    Payment: float = Field(description="Payment amount")
    Premium: float = Field(description="Premium amount")
    Cost_claims_year: float = Field(description="Cost of claims in the year")
    N_claims_year: int = Field(description="Number of claims in the year")
    N_claims_history: int = Field(description="Number of claims in history")
    R_Claims_history: float = Field(description="Ratio of claims in history")
    Type_risk: str = Field(description="Type of risk")
    Area: str = Field(description="Area or region")
    Second_driver: str = Field(description="Second driver information")
    Year_matriculation: int = Field(description="Year of matriculation")
    Power: float = Field(description="Vehicle power")
    Cylinder_capacity: float = Field(description="Cylinder capacity")
    Value_vehicle: float = Field(description="Value of the vehicle")
    N_doors: int = Field(description="Number of doors")
    Type_fuel: str = Field(description="Type of fuel")
    Length: float = Field(description="Vehicle length")
    Weight: float = Field(description="Vehicle weight")

class LLMMetrics(BaseModel):
    model_name: str = Field(description="Name of the LLM used")
    confidence_score: float = Field(description="Confidence score between 0 and 1")
    processing_time: float = Field(description="Time taken to process in seconds")
    token_count: int = Field(description="Number of tokens processed")

class InferenceEngine:
    def __init__(self, anthropic_api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-latest",
            temperature=0.1,
            anthropic_api_key=anthropic_api_key
        )

        assessment_schema = json.dumps(AssessmentResult.model_json_schema())
        metrics_schema = json.dumps(LLMMetrics.model_json_schema())
    
        self.assessment_prompt = PromptTemplate(
            template="""You are a risk assessment expert. Analyze the following document and provide a structured assessment focusing on risk and compliance.
            Your response must be a valid JSON object that strictly follows this schema:
            
            {format_instructions}
            
            Document content:
            {document_content}
            
            Remember to format your response as a valid JSON object that matches the schema exactly.
            """,
            input_variables=["document_content"],
            partial_variables={"format_instructions": assessment_schema}
        )

        self.metrics_prompt = PromptTemplate(
            template="""Based on your analysis, provide confidence metrics in this exact JSON format:
            
            {format_instructions}
            
            Document content:
            {document_content}
            
            Remember to format your response as a valid JSON object that matches the schema exactly.
            """,
            input_variables=["document_content"],
            partial_variables={"format_instructions": metrics_schema}
        )
        
        self.assessment_parser = PydanticOutputParser(pydantic_object=AssessmentResult)
        self.metrics_parser = PydanticOutputParser(pydantic_object=LLMMetrics)
    
    def process_document(self, file_path: str | Path) -> tuple[AssessmentResult, LLMMetrics]:

        chunks = process_file(file_path)
        if not chunks:
            raise ValueError(f"No content could be extracted from {file_path}")
        
        document_content = "\n\n".join(chunk.page_content for chunk in chunks)
        if not document_content.strip():
            raise ValueError(f"Empty document content extracted from {file_path}")
        
        assessment_prompt = self.assessment_prompt.format(document_content=document_content)
        metrics_prompt = self.metrics_prompt.format(document_content=document_content)
        
        start_time = time.time()
        assessment_response = self.llm.invoke(assessment_prompt)
        assessment_result = self.assessment_parser.parse(assessment_response.content)
        metrics_response = self.llm.invoke(metrics_prompt)
        metrics_result = self.metrics_parser.parse(metrics_response.content)
        metrics_result.processing_time = time.time() - start_time
        return assessment_result, metrics_result
    
    def save_results(self, assessment: AssessmentResult, metrics: LLMMetrics, 
                    output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        assessment_path = output_dir / "assessment_result.json"
        with open(assessment_path, 'w') as f:
            json.dump(assessment.model_dump(), f, indent=2)
        
        metrics_path = output_dir / "llm_metrics.csv"
        fieldnames = ["model_name", "confidence_score", "processing_time", "token_count"]
        
        with open(metrics_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(metrics.model_dump())

def main():
    load_dotenv()
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        raise ValueError("Missing required Anthropic API key in environment variables")
    engine = InferenceEngine(anthropic_api_key=anthropic_api_key)
    file_path = "data/input/car-insurance-question-set.pdf"
    assessment, metrics = engine.process_document(file_path)
    output_dir = "data/output"
    engine.save_results(assessment, metrics, output_dir)
    print(f"Results saved to {output_dir}/")

if __name__ == "__main__":
    main()