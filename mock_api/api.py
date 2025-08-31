
from flask import Flask, jsonify
import json
import os

app = Flask(__name__)

# Load product data
DATA_FILE = os.path.join(os.path.dirname(__file__), "products.json")

with open(DATA_FILE, "r") as f:
    PRODUCTS = json.load(f)

# API route to fetch products
@app.route("/products", methods=["GET"])
def get_products():
    return jsonify(PRODUCTS)

if __name__ == "__main__":
    app.run(port=5001, debug=True)
