FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download uk_core_news_sm

COPY dic_data/ ./dic_data/
COPY batching.py paragraph_pipeline.py gradio_app.py ./

EXPOSE 7860

CMD ["python", "gradio_app.py"]
