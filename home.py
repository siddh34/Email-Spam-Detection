import json
import time
from flask import Flask, render_template, jsonify, request
import numpy as np
import joblib
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

app = Flask(__name__)
input_text = ""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process-text", methods=["POST"])
def process_text():
    global input_text
    input_text = request.form.get("input-text")
    print(input_text)

    # Load CountVectorizer and TfidfVectorizer data
    vectorizer_count = joblib.load("vectorizer_count.joblib")
    vectorizer_tfidf = joblib.load("vectorizer_tfidf.joblib")
    # Load the models for future use

    loaded_knn_count_model = joblib.load("models/knn_count_model.joblib")
    loaded_knn_tfidf_model = joblib.load("models/knn_tfidf_model.joblib")

    loaded_lr_count_model = joblib.load("models/lr_count_model.joblib")
    loaded_lr_tfidf_model = joblib.load("models/lr_tfidf_model.joblib")

    loaded_nb_count_model = joblib.load("models/nb_count_model.joblib")
    loaded_nb_tfidf_model = joblib.load("models/nb_tfidf_model.joblib")

    loaded_rf_count_model = joblib.load("models/rf_count_model.joblib")
    loaded_rf_tfidf_model = joblib.load("models/rf_tfidf_model.joblib")

    loaded_svc_count_model = joblib.load("models/svc_count_model.joblib")
    loaded_svc_tfidf_model = joblib.load("models/svc_tfidf_model.joblib")

    new_text = [input_text]
    new_text_count = vectorizer_count.transform(new_text)
    new_text_tfidf = vectorizer_tfidf.transform(new_text)

    # Use the trained models to predict whether the new text is spam or ham
    lr_count_prediction = loaded_lr_count_model.predict(new_text_count)
    lr_tfidf_prediction = loaded_lr_tfidf_model.predict(new_text_tfidf)
    nb_count_prediction = loaded_nb_count_model.predict(new_text_count)
    nb_tfidf_prediction = loaded_nb_tfidf_model.predict(new_text_tfidf)
    knn_count_prediction = loaded_knn_count_model.predict(new_text_count)
    knn_tfidf_prediction = loaded_knn_tfidf_model.predict(new_text_tfidf)
    rf_count_prediction = loaded_rf_count_model.predict(new_text_count)
    rf_tfidf_prediction = loaded_rf_tfidf_model.predict(new_text_tfidf)
    svc_count_prediction = loaded_svc_count_model.predict(new_text_count)
    svc_tfidf_prediction = loaded_svc_tfidf_model.predict(new_text_tfidf)

    lr_count_scores = loaded_lr_count_model.predict_proba(new_text_count)
    lr_tfidf_scores = loaded_lr_tfidf_model.predict_proba(new_text_tfidf)

    nb_count_scores = loaded_nb_count_model.predict_proba(new_text_count)
    nb_tfidf_scores = loaded_nb_tfidf_model.predict_proba(new_text_tfidf)

    knn_count_scores = loaded_knn_count_model.predict_proba(new_text_count)
    knn_tfidf_scores = loaded_knn_tfidf_model.predict_proba(new_text_tfidf)

    rf_count_scores = loaded_rf_count_model.predict_proba(new_text_count)
    rf_tfidf_scores = loaded_rf_tfidf_model.predict_proba(new_text_tfidf)
    print(rf_tfidf_scores, type(rf_tfidf_scores))
    svc_count_scores = loaded_svc_count_model.predict_proba(new_text_count)
    svc_tfidf_scores = loaded_svc_tfidf_model.predict_proba(new_text_tfidf)

    zeros_prob = []
    ones_prob = []
    lr_count_zero, lr_count_one = np.split(lr_count_scores[0], 2)
    lr_tfidf_zero, lr_tfidf_one = np.split(lr_tfidf_scores[0], 2)
    nb_count_zero, nb_count_one = np.split(nb_count_scores[0], 2)
    nb_tfidf_zero, nb_tfidf_one = np.split(nb_tfidf_scores[0], 2)
    knn_count_zero, knn_count_one = np.split(knn_count_scores[0], 2)
    knn_tfidf_zero, knn_tfidf_one = np.split(knn_tfidf_scores[0], 2)

    rf_count_zero, rf_count_one = np.split(rf_count_scores[0], 2)
    rf_tfidf_zero, rf_tfidf_one = np.split(rf_tfidf_scores[0], 2)

    svc_count_zero, svc_count_one = np.split(svc_count_scores[0], 2)
    svc_tfidf_zero, svc_tfidf_one = np.split(svc_tfidf_scores[0], 2)

    # , svc_count_zero, svc_tfidf_zero
    # , svc_count_one, svc_tfidf_one
    zeros_prob.extend(
        [
            lr_count_zero,
            lr_tfidf_zero,
            nb_count_zero,
            nb_tfidf_zero,
            knn_count_zero,
            knn_tfidf_zero,
            rf_count_zero,
            rf_tfidf_zero,
            svc_count_zero,
            svc_tfidf_zero,
        ]
    )
    ones_prob.extend(
        [
            lr_count_one,
            lr_tfidf_one,
            nb_count_one,
            nb_tfidf_one,
            knn_count_one,
            knn_tfidf_one,
            rf_count_one,
            rf_tfidf_one,
            svc_count_one,
            svc_tfidf_one,
        ]
    )
    zeros_prob = [round(score.tolist()[0], 2) for score in zeros_prob]
    ones_prob = [round(label.tolist()[0], 2) for label in ones_prob]
    accumulate_result = []
    accumulate_result.extend(
        [
            lr_count_prediction,
            lr_tfidf_prediction,
            nb_count_prediction,
            nb_tfidf_prediction,
            knn_count_prediction,
            knn_tfidf_prediction,
            rf_count_prediction,
            rf_tfidf_prediction,
            svc_count_prediction,
            svc_tfidf_prediction,
        ]
    )
    accumulate_result = [round(label.tolist()[0], 2) for label in accumulate_result]
    final_result = (
        "not spam"
        if accumulate_result.count(0) >= accumulate_result.count(1)
        else "spam"
    )
    print(zeros_prob, ones_prob, accumulate_result, final_result)

    # create a response dictionary with the scores
    response = {
        "zeros_prob": zeros_prob,
        "ones_prob": ones_prob,
        "final_result": final_result,
    }

    return jsonify(response)


@app.route("/correctprediction", methods=["POST"])
def correctprediction():
    global input_text
    data = request.get_json()  # retrieve the data sent from JavaScript
    # process the data using Python code
    result = data["prediction"]
    print(input_text)
    print(result)
    vectorizer_count = joblib.load("vectorizer_count.joblib")
    vectorizer_tfidf = joblib.load("vectorizer_tfidf.joblib")
    if input_text != "" and result:
        new_text_count = vectorizer_count.transform(input_text)
        new_text_tfidf = vectorizer_tfidf.transform(input_text)
        loaded_nb_count_model = joblib.load("models/nb_count_model.joblib")
        loaded_nb_tfidf_model = joblib.load("models/nb_tfidf_model.joblib")
        loaded_nb_count_model.partial_fit(new_text_count, result, classes=[0, 1])
        loaded_nb_tfidf_model.partial_fit(new_text_tfidf, result, classes=[0, 1])
        joblib.dump("models/nb_count_model.joblib")
        joblib.dump("models/nb_tfidf_model.joblib")
    return jsonify(success=True)


@app.route("/getEmailFromGmail", methods=["POST"])
def getEmailFromGmail():
    data = request.get_json()
    email = data.get("email")
    clientSecret = data.get("clientSecret")
    clientId = data.get("clientId")
    projectId = None

    cred_json = {
        "web": {
            "client_id": clientId,
            "project_id": projectId,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": clientSecret,
        }
    }

    with open("cred.json", "w") as f:
        json.dump(cred_json, f)

    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

    flow = InstalledAppFlow.from_client_secrets_file("./cred.json", SCOPES)
    creds = flow.run_local_server(port=0)

    creds = Credentials.from_authorized_user_file("./cred.json")
    service = build("gmail", "v1", credentials=creds)

    results = service.users().labels().list(userId="me").execute()
    labels = results.get("labels", [])

    results = (
        service.users()
        .messages()
        .list(userId="me", labelIds=["INBOX"], maxResults=1)
        .execute()
    )

    messages = results.get("messages", [])
    if messages[0] == None:
        return jsonify({"message": "failure"})

    msg = service.users().messages().get(userId="me", id=messages[0]["id"]).execute()

    return jsonify({"message": "success", "data": f"{msg}"})

@app.route("/sarthak", methods=["GET"])
def sarthak():
    return "<h1>Sarthak running code</h1>"

if __name__ == "__main__":
    app.run(debug=True)