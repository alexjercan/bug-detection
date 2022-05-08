from flask import request, jsonify
from app import app, ses


@app.route("/")
def home():
    return "hello world!"


@app.route("/api/inference", methods=["POST"])
def inference():
    data = request.get_json()
    source_code = data["source_code"]

    print(data)
    print(source_code)

    error_description, token_class, source_code = ses.run(source_code)

    return jsonify(
        {
            "error_description": error_description,
            "token_class": token_class,
            "source_code": source_code,
        }
    )
