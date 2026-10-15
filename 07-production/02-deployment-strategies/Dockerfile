FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml uv.lock service.py ./
COPY 02-deployment-strategies/app.py ./deployment_app.py
RUN pip install --no-cache-dir uv==0.12.20 && uv sync --frozen --no-dev
ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:${PATH}"
USER 65532
EXPOSE 8000
CMD ["uvicorn", "deployment_app:app", "--host", "0.0.0.0", "--port", "8000"]
