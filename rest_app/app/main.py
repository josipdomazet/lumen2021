import pickle
import json
import graphviz

from flask import request, send_from_directory, jsonify
from flask import Flask

from util import create_features

app = Flask(__name__)
app.config["SECRET_KEY"] = "019a5f06d22c4ea89ce4b2177c4bc98b"
file = None

with open('./resources/chaid.model', 'rb') as handle:
    model = pickle.load(handle)


@app.route("/", methods=["GET"])
def home():
    return send_from_directory("static", "home.html")


@app.route("/scoring", methods=["POST"])
def score():
    payload = request.json

    input_row, message = create_features(payload)
    if input_row is None:
        return message, 400

    result = model.predict(input_row)
    if result is None:
        return "Chosen example could not be scored.", 400

    segment, segment_pairs, imputed_pairs = result
    output = dict()
    output["supernode_pairs"] = segment.supernode_pairs
    output["segment_pairs"] = segment_pairs
    output["gm_cutoffs"] = segment.gm_cutoffs
    return jsonify(output), 200


if __name__ == "__main__":
    app.run(debug=True)
