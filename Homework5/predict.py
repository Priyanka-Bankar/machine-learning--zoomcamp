import pickle
from flask import Flask
from flask import request
from flask import jsonify


def loadfile(model_file: str):
    with open(model_file, "rb") as f_in:
        return pickle.load(f_in)


dv = loadfile("./dv.bin")
model = loadfile("./model1.bin")

app = Flask("give_subs")


@app.route("/predict", methods=["POST"])
def predict():
    client = request.get_json()

    X = dv.transform([client])

    y_pred = model.predict_proba(X)[0, 1]
    get_subs = y_pred > 0.5

    result = {"give_subs_probability": float(y_pred), "subs": bool(get_subs)}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
