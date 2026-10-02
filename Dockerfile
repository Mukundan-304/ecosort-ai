FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt \
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install transformers
# bake detector weights into the image so containers start without internet access
RUN python -c "from transformers import OwlViTProcessor, OwlViTForObjectDetection as M; n='google/owlvit-base-patch32'; OwlViTProcessor.from_pretrained(n); M.from_pretrained(n)"
COPY src src
COPY app app
COPY configs configs
COPY data/samples data/samples
COPY models models
RUN useradd -m eco && mkdir -p data/feedback data/store && chown -R eco /app /opt/hf
USER eco
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD python -c "import urllib.request as u;u.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit","run","app/streamlit_app.py","--server.port=8501","--server.address=0.0.0.0","--server.headless=true"]
