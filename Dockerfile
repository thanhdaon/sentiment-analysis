FROM rappdw/docker-java-python

WORKDIR /app

RUN pip install torch transformers vncorenlp
RUN pip install flask waitress
COPY . .

CMD [ "python", "/app/server.py" ]