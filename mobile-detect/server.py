# Mobile Detect – server
# En lille webserver, der husker status (mobil fundet: ja/nej) og viser den på en hjemmeside.
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder="templates")

# Serverens hukommelse: True = mobil fundet (rød), False = alt OK (hvid)
status_data = {"status": False}


# Forsiden (hjemmesiden)
@app.route("/")
def index():
    return render_template("index.html", status=status_data["status"])


# Her sender webcam.py ny status til
@app.route("/update", methods=["POST"])
def update_status():
    data = request.get_json(silent=True) or {}
    if "status" not in data:
        return jsonify({"error": "Mangler 'status' i beskeden"}), 400

    status_data["status"] = bool(data["status"])
    print(f"Status opdateret: {status_data['status']}")
    return jsonify({"message": "Status opdateret"}), 200


# Her henter hjemmesiden den nuværende status
@app.route("/status", methods=["GET"])
def get_status():
    return jsonify(status_data)


if __name__ == "__main__":
    # host="0.0.0.0" gør siden synlig for andre på netværket.
    app.run(host="0.0.0.0", port=8080, debug=False)
