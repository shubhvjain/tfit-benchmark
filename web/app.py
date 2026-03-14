# web/app.py
import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from search import coregulators

app = Flask(__name__)

@app.route("/")
def index():
    # This sends the file directly without processing {{ }} tags
    return app.send_static_file("index.html")

mode_registry = {
    "coregulators": coregulators
}

@app.route("/api/search", methods=["POST"])
def search():
    data = request.get_json()
    
    if data["mode"] not in mode_registry.keys():
        return jsonify({
            "success": False,
            "message": "Invalid mode",  
            "results": []
        })
    
    results = mode_registry[data["mode"]](data)  # Pass search parameters
    #print(results)
    return jsonify({
        "success": True,
        "results": results
    })

if __name__ == "__main__":
    app.run(debug=True)