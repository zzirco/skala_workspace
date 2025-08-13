# app.py
import os
from flask import Flask, request, jsonify, render_template
from readme_agent import run_agent

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"ok": False, "error": "invalid JSON body"}), 400

    try:
        repo_url = (data.get("repo_url") or "").strip()
        git_name = (data.get("git_commit_name") or "").strip()
        git_email = (data.get("git_commit_email") or "").strip()
        github_token = (data.get("github_token") or "").strip()
        final_filename = (data.get("final_filename") or "README.md").strip()
        open_pr = bool(data.get("open_pr", True))
        branch_name = (data.get("branch_name") or None)

        if not repo_url:
            return jsonify({"ok": False, "error": "repo_url is required"}), 400
        if not git_name or not git_email:
            return jsonify({"ok": False, "error": "git name/email required"}), 400

        os.environ["GIT_COMMIT_NAME"]  = git_name
        os.environ["GIT_COMMIT_EMAIL"] = git_email
        if github_token:
            os.environ["GITHUB_TOKEN"] = github_token

        result = run_agent(
            repo_url=repo_url,
            final_filename=final_filename,
            branch_name=branch_name,
            open_pr=open_pr,
        )
        return jsonify({"ok": True, "result": result}), 200
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    
@app.errorhandler(500)
def handle_500(err):
    return jsonify({"ok": False, "error": "Internal Server Error", "detail": str(err)}), 500

@app.errorhandler(404)
def handle_404(err):
    return jsonify({"ok": False, "error": "Not Found"}), 404
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
