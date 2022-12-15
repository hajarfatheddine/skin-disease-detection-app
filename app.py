from flask import Flask, redirect, url_for, request
from diseaseDetection import detection
from labelsList import labels
import py_eureka_client.eureka_client as eureka_client

app = Flask(__name__)

labelList = "model/labels.txt"
rest_port = 5000
eureka_client.init(eureka_server="http://localhost:8761",
                   app_name="ml-service",
                   instance_port=rest_port)


@app.route('/labels', methods=["GET"])
def getAllLabels():
    return labels(labelList)


@app.route("/classify", methods=["GET", "POST"])
def getImageClass():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            return detection(image)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)