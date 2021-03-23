FROM rappdw/docker-java-python

WORKDIR /app

RUN pip install torch transformers vncorenlp

COPY . .

CMD [ "python", "/app/infer.py" ]