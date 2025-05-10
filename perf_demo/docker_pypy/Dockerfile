FROM pypy:3.10-slim
RUN apt-get update && \
    apt-get install -y linux-perf python3-pip && \
    rm -rf /var/lib/apt/lists/*

COPY benchmark.py /benchmark.py
COPY script_pypy.py /script_pypy.py

CMD ["pypy3", "/script_pypy.py"]

