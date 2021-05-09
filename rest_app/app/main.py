import os
from PIL import Image
from flask import render_template, flash, redirect, url_for, request, abort
from flask import Flask
import pandas as pd
import base64
from io import BytesIO

app = Flask(__name__)
app.config["SECRET_KEY"] = "019a5f06d22c4ea89ce4b2177c4bc98b"
file = None


def create_tree_from_prediction(df):
    image = BytesIO()
    image_original = Image.open("../../plots/tree-250.png")
    image_original.save(image, format="PNG")
    return "data:image/png;base64,{}".format(
        base64.b64encode(image.getvalue()).decode()
    )


@app.route("/", methods=["GET", "POST"])
@app.route("/scoring", methods=["GET", "POST"])
def scoring():
    global file
    if request.method == "POST":
        file = request.files["file"]
        if file:
            if file.filename.split(".")[-1] == "csv":
                df = pd.read_csv(file.stream, sep="|", encoding="utf-16")
                if len(df.columns) == 33:
                    html_string = '''
                        <html>
                          <head><title>HTML Pandas Dataframe with CSS</title></head>
                          <link rel="stylesheet" type="text/css" href="static/table.css"/>
                          <body>
                            {table}
                          </body>
                        </html>.
                    '''
                    html = html_string.format(table=df.to_html(classes="table", max_rows=10, max_cols=33))
                    image = create_tree_from_prediction(df)
                    return render_template("home.html", table=html, header_1="Data Set", header_2="Select Line By ID",
                                           image=image)
            else:
                flash("You didn't load a valid .csv file! Try again!", "danger")
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
