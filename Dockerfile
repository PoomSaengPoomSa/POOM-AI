
# ---- Build stage ----
FROM python:3.11-slim AS builder
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# TODO: requirements.txt 완성 후 아래 주석 해제
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
 
# ---- Runtime stage ----
FROM python:3.11-slim
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /app
 
# 현재 구조 기준 복사 (폴더 추가되면 여기에 COPY 추가)
COPY agent/ ./agent/
COPY ml/ ./ml/
COPY sql/ ./sql/
COPY llm/ ./llm/
COPY app/ ./app/
COPY POOM-BACK/app /POOM-BACK/app
 
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser
 
EXPOSE 8001
# TODO: 진입점 확정 후 수정
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
