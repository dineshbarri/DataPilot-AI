FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN addgroup --system datapilot && adduser --system --ingroup datapilot datapilot
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY datapilot ./datapilot
COPY streamlit_app.py .
COPY .streamlit ./.streamlit
RUN mkdir -p /app/artifacts && chown -R datapilot:datapilot /app
USER datapilot
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]

