from flask import Flask, request, redirect, jsonify, session, send_from_directory, render_template
from werkzeug.utils import secure_filename

from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

from flask import abort

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import requests
import json
import markdown2

app = Flask(__name__)
auth = HTTPBasicAuth()
app.config['BASIC_AUTH_FORCE'] = True
app.config["CACHE_TYPE"] = "null"
app.config['TEMPLATES_AUTO_RELOAD'] = True

with open('./static/report_template.txt', 'r') as f:
    REPORT_TEMPLATE = f.read()

with open('./secret_password.txt', 'r') as f:
    PASSWORD = f.read()

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# @app.before_request
def limit_remote_addr():
    client_ip = str(request.remote_addr)
    if client_ip.startswith('1.171.3.103'): return
    if not client_ip.startswith('140.113.'): abort(404)

@auth.verify_password
def verify_password(username, password):
    return check_password_hash(PASSWORD, password)

@app.route('/')
@auth.login_required
def root():
    return redirect("/static/index.html")

@app.route('/static/index.html')
@auth.login_required
def index():
    return app.send_static_file('index.html')

@app.route("/submit_ekg", methods=["POST"])
@auth.login_required
def receive_ekg():
    r = requests.post('http://localhost:9999/submit_ekg', files={'ekg_raw_file': request.files['ekg_raw_file']})

    if r.status_code != 200:
        abort(r.status_code)

    # generate report
    results = r.text
    results = json.loads(results)
    report = REPORT_TEMPLATE.format( request.files['ekg_raw_file'].filename,
                                results['ekg_plot'],
                                'red' if results['abnormally_score'] > 0.5 else 'blue',
                                results['abnormally_score'],
                                results['abnormally_explainer_plot'],
                                *results['hazard_score'],
                                *results['hazard_explainer_plot']
                                )

    html_report = markdown2.markdown(report, extras=['tables', 'header-ids'])
    html_report = '<head><link rel="stylesheet" href="github.css"></head>\n' +\
                        '<div class="markdown-body">' + html_report + '</div>'


    return html_report

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8080)
