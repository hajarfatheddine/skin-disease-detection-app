from flask import Flask, redirect, url_for, request, render_template, jsonify
from diseaseDetection import detection
from labelsList import labels

app = Flask(__name__)

labelList = "model/labels.txt"





@app.route('/index', methods=["GET"])
def test():
    return labels(labelList)


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        user = request.form['nm']
        return redirect(url_for('success', name=user))
    else:
        user = request.args.get('nm')
        return redirect(url_for('success', name=user))


app.config["IMAGE_UPLOADS"] = ""


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            name = image.filename
            # image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            return detection(image)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)