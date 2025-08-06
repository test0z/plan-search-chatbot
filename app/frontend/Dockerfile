FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 

# Node.js와 npm 설치 (YouTube MCP 서버 설치를 위해)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir uv

RUN uv pip install --system -e .

# YouTube MCP 서버 설치
RUN npm install -g youtube-data-mcp-server

EXPOSE 7860

# entrypoint.sh 스크립트 생성 및 실행 권한 부여
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# 환경변수 확인 및 로깅\n\
echo "Starting Chainlit application..."\n\
echo "PORT: 7860"\n\
echo "HOST: 0.0.0.0"\n\
\n\
# YouTube MCP 서버 설치 확인\n\
if command -v youtube-data-mcp-server >/dev/null 2>&1; then\n\
    echo "✅ YouTube MCP Server found: $(which youtube-data-mcp-server)"\n\
else\n\
    echo "❌ YouTube MCP Server not found, installing..."\n\
    npm install -g youtube-data-mcp-server\n\
fi\n\
\n\
# CHAINLIT_AUTH_SECRET이 설정되지 않은 경우 자동 생성\n\
if [ -z "$CHAINLIT_AUTH_SECRET" ]; then\n\
    export CHAINLIT_AUTH_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")\n\
    echo "Generated CHAINLIT_AUTH_SECRET for this session"\n\
else\n\
    echo "Using provided CHAINLIT_AUTH_SECRET"\n\
fi\n\
\n\
# 백엔드 연결 확인\n\
if [ ! -z "$SK_API_URL" ]; then\n\
    echo "Backend API URL: $SK_API_URL"\n\
fi\n\
\n\
# Chainlit 애플리케이션 실행 (Socket.IO 설정 포함)\n\
echo "Executing: chainlit run src/app_chainlit.py --port 7860 --host 0.0.0.0"\n\
exec chainlit run src/app_chainlit.py --port 7860 --host 0.0.0.0 --headless' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]