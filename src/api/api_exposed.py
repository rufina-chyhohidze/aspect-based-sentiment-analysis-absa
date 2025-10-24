import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import logging
from contextlib import asynccontextmanager

# Fix Python path to find modules in parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # This goes up to src/
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ABSA_API")


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


# Global variable for analyzers
analyzers = []


# Import analyzers with error handling
def import_analyzers():
    analyzer_classes = []

    # Try to import each analyzer with proper error handling
    import_attempts = [
        ("LexiconABSA", "lexicon_absa", "LexiconABSA"),
        ("LexiconOpinionFirstABSA", "lexicon_absa2_opinion_to_sentiment", "LexiconOpinionFirstABSA"),
        ("PyABSA", "ML_ABSA", "ABSA"),
        ("LLMABSA", "LLMABSA", "LLMABSA"),
    ]

    for name, module_name, class_name in import_attempts:
        try:
            module = __import__(module_name, fromlist=[class_name])
            analyzer_class = getattr(module, class_name)
            analyzer_classes.append((name, analyzer_class))
            logger.info(f" Successfully imported {name}")
        except ImportError as e:
            logger.warning(f" Failed to import {name} from {module_name}: {e}")
            # Print debug info
            logger.info(f"Current Python path: {sys.path}")
            logger.info(f"Looking for module in: {parent_dir}")
        except AttributeError as e:
            logger.warning(f" Class {class_name} not found in {module_name}: {e}")
        except Exception as e:
            logger.warning(f" Unexpected error importing {name}: {e}")

    return analyzer_classes


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    global analyzers
    logger.info("Loading ABSA analyzers...")

    try:
        analyzer_classes = import_analyzers()

        for name, analyzer_class in analyzer_classes:
            try:
                analyzer_instance = analyzer_class()
                analyzers.append((f"{name}", analyzer_instance))
                logger.info(f" Successfully initialized {name}")
            except Exception as e:
                logger.error(f" Failed to initialize {name}: {e}")

        logger.info(f"Loaded {len(analyzers)} out of {len(analyzer_classes)} analyzers successfully.")
        yield

    except Exception as e:
        logger.exception("Failed to load analyzers: %s", e)
        raise RuntimeError("Analyzer initialization failed") from e
    finally:
        # Shutdown code (if needed)
        logger.info("Shutting down ABSA API...")


app = FastAPI(
    title="Unified ABSA API",
    description="Runs all ABSA implementations on input text",
    version="1.0",
    lifespan=lifespan
)


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
            logger.info(f"[{name}] Found {len(results)} aspects")
        except Exception as e:
            logger.warning("[%s] Analysis failed: %s", name, e)
            response.append(AnalyzerOutput(analyzer=name, results=[]))

    return response


@app.get("/")
async def root():
    return {"message": "Unified ABSA API is running"}


@app.get("/health")
async def health_check():
    analyzer_status = {}
    for name, analyzer in analyzers:
        try:
            # Simple test to check if analyzer is working
            test_result = analyzer.analyze("The food was great.")
            analyzer_status[name] = f"healthy (found {len(test_result)} aspects)"
        except Exception as e:
            analyzer_status[name] = f"unhealthy: {str(e)}"

    return {
        "status": "running",
        "analyzers_loaded": len(analyzers),
        "analyzers": analyzer_status
    }