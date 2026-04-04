import os

MEILI_HOST: str = os.environ.get("MEILI_HOST", "http://127.0.0.1:7700")
MEILI_API_KEY: str = os.environ.get("MEILI_API_KEY", "masterKey123")

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

GENERATION_MODEL: str = os.environ.get("GENERATION_MODEL", "gpt-5.4")

PROMPT_VERSION: str = "1.0.0"

OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "./corpus_output")
# Parallel workers in corpus_pipeline (each holds one in-flight generation+validation).
PIPELINE_CONCURRENCY: int = int(os.environ.get("PIPELINE_CONCURRENCY", "4"))
