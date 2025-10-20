# src/api/api_exposed.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import logging

# Import your analyzers
from src.lexicon_absa import LexiconABSA
from src.lexicon_absa2_opinion_to_sentiment import LexiconOpinionFirstABSA
from src.llm_absa import ABSA
from src.llm_3_absa import LLMABSA

# --- Logging setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ABSA_API")

# --- Request & Response models ---
class TextInput(BaseModel):
    text: str

class AspectResult(BaseModel):
    aspect: str
    sentiment: str
    confidence: float
    text_span: List[int]

class AnalyzerOutput(BaseModel):
    analyzer: str
    results: List[AspectResult]

# --- Initialize FastAPI app with lifespan ---
app = FastAPI(
    title="Unified ABSA API",
    description="Runs all ABSA implementations on input text",
    version="1.0"
)

# Placeholder for analyzers
analyzers = []

@app.on_event("startup")
async def load_analyzers():
    """Load ABSA analyzers at startup."""
    global analyzers
    logger.info("Loading ABSA analyzers...")

    # Initialize all analyzers safely
    try:
        analyzers = [
            ("LexiconABSA (rule-based)", LexiconABSA()),
            ("LexiconOpinionFirstABSA (opinion-first)", LexiconOpinionFirstABSA()),
            ("PyABSA (Transformer-based)", ABSA()),  # PyABSA may download model if missing
            ("LLMABSA (Local Ollama LLM)", LLMABSA()),
        ]
        logger.info("All analyzers loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load analyzers: %s", e)
        raise RuntimeError("Analyzer initialization failed") from e

# --- API endpoint ---
@app.post("/analyze", response_model=List[AnalyzerOutput])
async def analyze_text(input_data: TextInput):
    text = input_data.text
    logger.info("Received text for analysis: %s", text)

    response = []
    for name, analyzer in analyzers:
        try:
            results = analyzer.analyze(text)
            response.append(
                AnalyzerOutput(
                    analyzer=name,
                    results=[
                        AspectResult(
                            aspect=r.aspect,
                            sentiment=r.sentiment,
                            confidence=r.confidence,
                            text_span=r.text_span
                        )
                        for r in results
                    ]
                )
            )
        except Exception as e:
            logger.warning("[%s] Analysis failed: %s", name, e)
            response.append(AnalyzerOutput(analyzer=name, results=[]))

    return response
