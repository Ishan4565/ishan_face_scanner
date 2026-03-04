FROM python:3.11-bookworm

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt --verbose > /tmp/pip.log 2>&1 || (cat /tmp/pip.log && exit 1)

COPY --chown=user app.py .
COPY --chown=user dataset/ ./dataset/

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
