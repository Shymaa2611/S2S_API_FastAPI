FROM python:3.10-slim-bullseye

ENV PYTHONUNBUFFERED True
ENV S2S_API /app
WORKDIR $S2S_API
COPY . ./

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
