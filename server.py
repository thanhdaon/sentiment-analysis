import flask
import werkzeug
from waitress import serve
from model import predict, predict_many

app = flask.Flask(__name__)


@app.errorhandler(werkzeug.exceptions.InternalServerError)
def handle_bad_request(e):
    return {
        "code": e.code,
        "name": e.name,
        "description": e.description,
    }, 400


@app.route('/predict', methods=['POST'])
def handle_predict():
    if flask.request.json is None:
        raise Exception("body empty")

    sentence = flask.request.json["sentence"]

    if not isinstance(sentence, str):
        raise Exception("sentence must be string!")

    if sentence == "":
        raise Exception("sentence empty")

    return {
        "data": predict(sentence)
    }


@app.route('/predict-many', methods=['POST'])
def handle_predict_many():
    if flask.request.json is None:
        raise Exception("body empty")

    sentences = flask.request.json["sentences"]

    if not isinstance(sentences, list):
        raise Exception("sentences must be list!")

    if len(sentences) == 0:
        raise Exception("sentences empty!")

    probs, preds = predict_many(sentences)
    return {
        "data": {
            "probs": probs.tolist(),
            "preds": preds.tolist(),
        }
    }


serve(app, host="0.0.0.0", port=8000)
