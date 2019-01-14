import csv
import string
import numpy as np
from keras.layers import subtract
from keras.backend import abs
from keras.models import load_model
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.sequence import pad_sequences

def cosine_loss(y_true, y_pred):
    return abs(subtract([y_true, y_pred]))

def predictModelA(question, response, model, truth, vocab_size=5000):
    idx = truth["Question"].index(question)
    sequences1 = [hashing_trick(response, vocab_size, hash_function='md5')]
    sentence_one_pad = pad_sequences(sequences1, maxlen=50, padding="post")

    sequences2 = [hashing_trick(truth["Best_Responses"][idx][0], vocab_size, hash_function='md5')]
    sentence_two_pad = pad_sequences(sequences2, maxlen=50, padding="post")

    sentence_similarity = model.predict([sentence_one_pad, sentence_two_pad], batch_size=1, verbose=0)[0]

    if sentence_similarity < 0.2:
        sim_result_a = 0.0
    elif sentence_similarity < 0.4:
        sim_result_a = 0.5
    else:
        sim_result_a = 1.0

    sequences2 = [hashing_trick(truth["Text.used.to.make.inference"][idx], vocab_size, hash_function='md5')]
    sentence_two_pad = pad_sequences(sequences2, maxlen=50, padding="post")
    sentence_similarity = model.predict([sentence_one_pad, sentence_two_pad], batch_size=1, verbose=0)[0]

    if sentence_similarity < 0.2:
        sim_result_t = 0.0
    elif sentence_similarity < 0.4:
        sim_result_t = 0.5
    else:
        sim_result_t = 1.0

    if sim_result_a == 0 and sim_result_t == 0 or sim_result_a == 0.5 and sim_result_t == 0:
        comb_similarity = 0.0
    elif sim_result_a == 0 and sim_result_t == 0.5 or sim_result_a == 0 and sim_result_t == 1 or sim_result_a == 0.5 and sim_result_t == 0.5 or \
            sim_result_a == 1 and sim_result_t == 0 or sim_result_a == 1 and sim_result_t == 0.5:
        comb_similarity = 0.5
    else:
        comb_similarity = 1.0

    return comb_similarity

def main():
    loaded_model = load_model("model_lstm_eph50.h5", custom_objects={'cosine_loss': cosine_loss})
    loaded_model.compile(loss=cosine_loss, optimizer='adam', metrics=['accuracy'])
    print("Loaded model from disk")

    table = dict.fromkeys(map(ord, string.punctuation))
    truth = dict()
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

    train_data = dict()
    is_first = True
    with open('Weightless_dataset_train.csv', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')

        for row in reader:
            if is_first:
                train_data["Question"] = [row["Question"]]
                train_data["Response"] = [row["Response"]]
                train_data["Grade"] = [row["Final.rating"]]
                is_first = False
            else:
                train_data["Question"].append(row["Question"])
                train_data["Response"].append(row["Response"])
                train_data["Grade"].append(row["Final.rating"])

    for i in range(len(truth["Best_Responses"])):
        truth["Best_Responses"][i] = list(truth["Best_Responses"][i])

    tp = np.zeros(3)
    fp = np.zeros(3)
    fn = np.zeros(3)

    for i in range(len(train_data["Question"])):
        ques = train_data["Question"][i]
        resp = train_data["Response"][i]

        sim = predictModelA(ques, resp, loaded_model, truth)

        if train_data["Grade"][i] == "0,0":
            if sim == 0:
                tp[0] += 1
            elif sim == 0.5:
                fn[0] += 1
                fp[1] += 1
            else:
                fn[0] += 1
                fp[2] += 1
        elif train_data["Grade"][i] == "0,5":
            if sim == 0:
                fn[1] += 1
                fp[0] += 1
            elif sim == 0.5:
                tp[1] += 1
            else:
                fn[1] += 1
                fp[2] += 1

        else:
            if sim == 0:
                fn[2] += 1
                fp[0] += 1
            elif sim == 0.5:
                fn[2] += 1
                fp[1] += 1
            else:
                tp[2] += 1

    print("True Positives: " + str(tp))
    print("False Negatives: " + str(fn))
    print("False Positives: " + str(fp))

    microPr = float(tp[0] + tp[1] + tp[2]) / (tp[0] + tp[1] + tp[2] + fp[0] + fp[1] + fp[2])
    microRc = float(tp[0] + tp[1] + tp[2]) / (tp[0] + tp[1] + tp[2] + fn[0] + fn[1] + fn[2])
    microF = 2 * microPr * microRc / (microPr + microRc)

    print("Micro:\nPrecision: %f, Recall: %f, F-score: %f" % (microPr, microRc, microF))

    macroPr = (float(tp[0]) / (tp[0] + fp[0]) + float(tp[1]) / (tp[1] + fp[1]) + float(tp[2]) / (tp[2] + fp[2])) / 3
    macroRc = (float(tp[0]) / (tp[0] + fn[0]) + float(tp[1]) / (tp[1] + fn[1]) + float(tp[2]) / (tp[2] + fn[2])) / 3
    macroF = 2 * macroPr * macroRc / (macroPr + macroRc)

    print("Macro:\nPrecision: %f, Recall: %f, F-score: %f" % (macroPr, macroRc, macroF))

if __name__ == "__main__":
    main()