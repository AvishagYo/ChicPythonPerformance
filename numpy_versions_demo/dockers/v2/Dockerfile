FROM python:3.10-slim

WORKDIR /app
COPY benchmark.py .

RUN pip install numpy==1.26

CMD ["python", "benchmark.py"]

