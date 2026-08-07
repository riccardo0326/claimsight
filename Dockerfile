FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU torch first (smaller than the CUDA default), then the rest.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Prefetch DocVQA + embedding + Vision model weights so first claim does not stall.
ENV HF_HOME=/cache/huggingface
RUN python -c "from transformers import pipeline; pipeline('document-question-answering', model='impira/layoutlm-document-qa')" \
 && python -c "from llama_index.embeddings.huggingface import HuggingFaceEmbedding; HuggingFaceEmbedding(model_name='sentence-transformers/all-MiniLM-L6-v2')" \
 && python -c "from transformers import pipeline; pipeline('zero-shot-object-detection', model='google/owlvit-base-patch32')" \
 && python -c "from transformers import pipeline; pipeline('zero-shot-image-classification', model='openai/clip-vit-base-patch32')" \
 && python -c "from transformers import BlipProcessor, BlipForQuestionAnswering; BlipProcessor.from_pretrained('Salesforce/blip-vqa-base'); BlipForQuestionAnswering.from_pretrained('Salesforce/blip-vqa-base')"

COPY . .

ENV PYTHONPATH=/app
ENV STORAGE_DIR=/data/storage
ENV HF_HOME=/cache/huggingface

RUN mkdir -p /data/storage

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
