
"""
Independent Practice Sandbox API Server
========================================
This server is 100% isolated from the main MediAssist bot.
Use it to practice writing APIs, handling query params, JSON payloads, and testing in Postman!
"""

from flask import Flask, jsonify, request

app = Flask(__name__)

# In-memory database for testing CRUD operations
in_memory_notes = [
    {"id": 1, "title": "Learn Flask APIs", "done": True},
    {"id": 2, "title": "Test endpoints in Postman", "done": False}
]


# -------------------------------------------------------------
# 1. Simple GET Endpoint (Status / Health Check)
# -------------------------------------------------------------
@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({
        "status": "online",
        "service": "Practice Sandbox API",
        "message": "Your independent practice server is up and running!"
    }), 200


# -------------------------------------------------------------
# 2. Dynamic URL Parameter Endpoint (e.g. /api/greet/Parth)
# -------------------------------------------------------------
@app.route("/api/greet/<name>", methods=["GET"])
def greet_user(name):
    return jsonify({
        "greeting": f"Hello, {name}! Welcome to API engineering.",
        "tip": "Try passing different names in the URL in Postman!"
    }), 200


# -------------------------------------------------------------
# 3. POST Endpoint Receiving JSON Body (Calculator)
# -------------------------------------------------------------
@app.route("/api/calculate", methods=["POST"])
def calculate():
    data = request.get_json() or {}
    
    num1 = data.get("num1")
    num2 = data.get("num2")
    operation = data.get("operation", "add")  # "add", "subtract", "multiply", "divide"

    # Input validation
    if num1 is None or num2 is None:
        return jsonify({
            "error": "Missing input! Please provide both 'num1' and 'num2' in your JSON body."
        }), 400

    if operation == "add":
        result = num1 + num2
    elif operation == "subtract":
        result = num1 - num2
    elif operation == "multiply":
        result = num1 * num2
    elif operation == "divide":
        if num2 == 0:
            return jsonify({"error": "Cannot divide by zero!"}), 400
        result = num1 / num2
    else:
        return jsonify({"error": f"Unsupported operation '{operation}'. Use add, subtract, multiply, or divide."}), 400

    return jsonify({
        "num1": num1,
        "num2": num2,
        "operation": operation,
        "result": result
    }), 200


# -------------------------------------------------------------
# 4. CRUD Practice: Get All Notes (GET) or Add Note (POST)
# -------------------------------------------------------------
@app.route("/api/notes", methods=["GET", "POST"])
def manage_notes():
    if request.method == "POST":
        data = request.get_json() or {}
        title = data.get("title", "").strip()
        if not title:
            return jsonify({"error": "Note title is required"}), 400

        new_note = {
            "id": len(in_memory_notes) + 1,
            "title": title,
            "done": data.get("done", False)
        }
        in_memory_notes.append(new_note)
        return jsonify({
            "message": "Note created successfully!",
            "note": new_note
        }), 201

    # GET request: return all notes
    return jsonify({
        "total_notes": len(in_memory_notes),
        "notes": in_memory_notes
    }), 200


if __name__ == "__main__":
    print("==========================================================")
    print("Independent Practice Server starting on http://localhost:8000")
    print("Completely isolated from MediAssist (port 5050)")
    print("==========================================================")
    app.run(host="localhost", port=8000, debug=True)
