FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir uv

RUN uv pip install --system -e .

EXPOSE 7860

CMD ["uv", "run", "src/app.py"]