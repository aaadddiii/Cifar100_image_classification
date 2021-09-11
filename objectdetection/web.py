from flask import Flask, json, request, jsonify
from objectdetection.helper.helper import get_result

app = Flask(__name__)

@app.route('/health', methods=["GET"])
def health_check():
    return jsonify({"status":"ok"})

@app.route('/detect', methods=["POST"])
def detect():
    image_base64 = request.json.get("image")
    result_dict = get_result(image_base64)
    return jsonify(result_dict)