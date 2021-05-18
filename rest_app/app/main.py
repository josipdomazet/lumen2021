from PIL import Image
from flask import render_template, flash, request, jsonify
from flask import Flask
import pandas as pd
import base64
from io import BytesIO

app = Flask(__name__)
app.config["SECRET_KEY"] = "019a5f06d22c4ea89ce4b2177c4bc98b"
file = None


@app.route("/", methods=["GET", "POST"])
@app.route("/scoring", methods=["GET", "POST"])
def scoring():
    return render_template("home.html")


@app.route("/get-table", methods=["GET"])
def get_table():
    filename = "LUMEN_DS_SAMPLE.csv"
    if filename.split(".")[-1] == "csv":
        df = pd.read_csv("../../dataset/" + filename, sep="|", encoding="utf-16")
        if len(df.columns) == 33:
            html = df.to_html(classes="table", max_cols=33, table_id="table_id")
            return jsonify(html)
    return jsonify(200)


if __name__ == "__main__":
    app.run(debug=True)
