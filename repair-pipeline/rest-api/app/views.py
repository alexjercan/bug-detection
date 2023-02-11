from app import app, ses
from flask import jsonify, request


@app.route("/")
def home():
    return "hello world!"


@app.route("/api/inference", methods=["POST"])
def inference():
    data = request.get_json()
    source_code = data["source_code"]
    beam_size = data.get("beam_size", 1)

    print(data)
    print(source_code)

    error_description, token_class, source_code = ses.run(source_code, beam_size)

    return jsonify(
        {
            "error_description": error_description,
            "token_class": token_class,
            "source_code": source_code,
        }
    )
