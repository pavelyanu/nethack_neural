FROM docker.io/pavelyanu/nethack_neural:dev

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git

WORKDIR /app

RUN git clone https://github.com/pavelyanu/nethack_neural.git

RUN pip install -e ./nethack_neural

CMD ["python", "/app/nethack_neural/nethack_neural/main.py"]
