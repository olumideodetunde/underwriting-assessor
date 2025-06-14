import os
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

# Load environment variables (for API keys, etc.)
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

class LLMJudge:
    def __init__(self, anthropic_api_key: str = None, model: str = "claude-3-5-sonnet-latest"):
        self.llm = ChatAnthropic(
            model=model,
            temperature=0.1,
            anthropic_api_key=anthropic_api_key or ANTHROPIC_API_KEY
        )
        self.prompt_template = PromptTemplate(
            template="""
You are an expert evaluator. Given a SOURCE DATA and an LLM OUTPUT, rate the OUTPUT on the following metrics (0-10, with 10 best):
- Coherence: Is the output logically consistent and well-structured?
- Correctness: Is the output factually accurate based on the source data?
- Relevance: Does the output address the key points in the source data?

Provide a JSON object with the following fields:
{{
  "coherence": <int>,
  "correctness": <int>,
  "relevance": <int>,
  "justification": "<short explanation for your ratings>"
}}

SOURCE DATA:
{source_data}

LLM OUTPUT:
{llm_output}

Remember: This is a reference-free evaluation. Only use the source data and the output for your judgment.
""",
            input_variables=["source_data", "llm_output"]
        )

    def evaluate(self, source_data: str, llm_output: str) -> dict:
        prompt = self.prompt_template.format(source_data=source_data, llm_output=llm_output)
        response = self.llm.invoke(prompt)
        # Try to parse the response as JSON
        import json
        try:
            result = json.loads(response.content)
        except Exception:
            result = {"error": "Could not parse judge response as JSON", "raw_response": response.content}
        return result

if __name__ == "__main__":
    # Example usage
    source = "The capital of France is Paris."
    output = "Paris is the capital city of France."
    judge = LLMJudge()
    result = judge.evaluate(source, output)
    print("Evaluation Result:", result) 