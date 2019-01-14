from flask import Flask, url_for
from flask import json
from flask import request
from flask import Response
from testA import predictModelA
from testB import predictModelB
from testC import predictModelC
import csv
from keras.layers import subtract
from keras.backend import abs
from keras.models import load_model

app = Flask(__name__)

def cosine_loss(y_true, y_pred):
    return abs(subtract([y_true, y_pred]))

truth = dict()
loaded_model = load_model("model_lstm_eph50.h5", custom_objects={'cosine_loss': cosine_loss})
loaded_model.compile(loss=cosine_loss, optimizer='adam', metrics=['accuracy'])
print("Loaded model from disk")

@app.route('/predict', methods = ["POST"])
def api_articles():
    r_dict = request.json
    if r_dict["modelId"] == "A":
        score = predictModelA(r_dict["question"], r_dict["questionResponse"], loaded_model, truth)
    elif r_dict["modelId"] == "B":
        score = predictModelB(r_dict["question"], r_dict["questionResponse"], loaded_model, truth)
    else:
        score = predictModelC(r_dict["question"], r_dict["questionResponse"], loaded_model, truth)
    data = {
        "score": score,
        "probability": None
    }
    js = json.dumps(data)

    return Response(js, status=200, mimetype='application/json')


if __name__ == '__main__':
    is_first = True
    with open('Weightless_dataset_train_A.csv', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)

        for row in reader:
            if is_first:
                truth["Question"] = [row["Question"]]
                truth["Best_Responses"] = [{row["Response"]}]
                truth["Text.used.to.make.inference"] = [row["Text.used.to.make.inference"]]
                is_first = False
            else:
                truth["Question"].append(row["Question"])
                truth["Best_Responses"].append({row["Response"]})
                truth["Text.used.to.make.inference"].append(row["Text.used.to.make.inference"])

    with open('Weightless_dataset_train.csv', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')

        for row in reader:
            if row["Glenn.s.rating"] == "1,0" and row["Amber.s.rating"] == "1,0" and row["Final.rating"] == "1,0":
                idx = truth["Question"].index(row["Question"])
                truth["Best_Responses"][idx].add(row["Response"])

    for i in range(len(truth["Best_Responses"])):
        truth["Best_Responses"][i] = list(truth["Best_Responses"][i])

    app.run(port=8080, debug = False, threaded = False)