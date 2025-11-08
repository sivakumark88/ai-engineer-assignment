from flask import Flask, request, jsonify, send_file
import base64
import subprocess
import os
import tempfile
import atexit

app = Flask(__name__)

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

        # <--Run the solution.py script, improvements can be made by replacing subprocess to more tight function calling 
        subprocess.run(['python', 'scripts/solution.py', 
                        '--employees_path', employees_csv_path, 
                        '--connections_path', connections_csv_path,
                        '--output_path', submission_csv_path], check=True)

        # <--Run the visualize_sunburst.py script, improvements can be made by replacing subprocess to more tight function calling 
        subprocess.run(['python', 'dependencies/visualize_sunburst.py',
                        '--submission_path', submission_csv_path,
                        '--employees_path', employees_csv_path,
                        '--output_path', sunburst_html_path], check=True)

        # Return the generated HTML file
        return send_file(sunburst_html_path)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=5001)