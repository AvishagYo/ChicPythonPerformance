FROM python:3.10-slim
RUN apt-get update && \
    apt-get install -y linux-perf python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN pip install numpy
COPY benchmark.py /benchmark.py
COPY script_regular.py /script_regular.py

CMD ["python", "/script_regular.py"]

