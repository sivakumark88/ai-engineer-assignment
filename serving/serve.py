from flask import Flask, request, jsonify, send_file
import base64
import os
import sys
import tempfile
import atexit
import pandas as pd
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.solution import build_graph_with_features, predict_managers_globally

app = Flask(__name__)

# Preload the sentence transformer model at startup to avoid loading it on every request
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded and ready!")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts employee and connection data as base64 encoded CSVs,
    runs the hierarchy prediction, and returns the sunburst visualization.
    """
    data = request.get_json()

    if not data or 'employees_csv_base64' not in data or 'connections_csv_base64' not in data:
        return jsonify({"error": "Missing required fields: employees_csv_base64, connections_csv_base64"}), 400

    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Schedule the cleanup of the temporary directory
    atexit.register(os.rmdir, temp_dir)

    try:
        # Decode the base64 strings
        employees_csv_bytes = base64.b64decode(data['employees_csv_base64'])
        connections_csv_bytes = base64.b64decode(data['connections_csv_base64'])

        # Write the decoded bytes to temporary files
        employees_csv_path = os.path.join(temp_dir, 'employees.csv')
        connections_csv_path = os.path.join(temp_dir, 'connections.csv')
        submission_csv_path = os.path.join(temp_dir, 'submission.csv')
        sunburst_html_path = os.path.join(temp_dir, 'employee_sunburst.html')

        with open(employees_csv_path, 'wb') as f:
            f.write(employees_csv_bytes)
        with open(connections_csv_path, 'wb') as f:
            f.write(connections_csv_bytes)

        # Load CSV data
        employees_df = pd.read_csv(employees_csv_path, engine='python')
        connections_df = pd.read_csv(connections_csv_path)
        
        # Build graph using preloaded model and predict managers
        company_graph = build_graph_with_features(employees_df, connections_df, model=model)
        manager_predictions = predict_managers_globally(company_graph)
        
        # Create submission dataframe
        submission_df = pd.DataFrame({'employee_id': employees_df['employee_id']})
        submission_df['manager_id'] = submission_df['employee_id'].map(manager_predictions).fillna(0).astype(int)
        submission_df.loc[submission_df['employee_id'] == 358, 'manager_id'] = -1
        submission_df.to_csv(submission_csv_path, index=False)
        
        # Generate sunburst visualization - direct function call instead of subprocess
        from dependencies.visualize_sunburst import visualize_sunburst_hierarchy
        visualize_sunburst_hierarchy(employees_csv_path, submission_csv_path, sunburst_html_path)

        # Return the generated HTML file
        return send_file(sunburst_html_path)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=5001)