#!/usr/bin/env python3
"""Run a .sql file on the SQL warehouse. Substitutes ${catalog}/${schema}. Usage: run_sql.py file.sql [--print]"""
import json, subprocess, sys, time, os, re
PROFILE = os.environ.get("PROFILE", "fevm-agentbricks-demo")
WH = os.environ.get("WAREHOUSE_ID", "7046f7a2659b62ba")
CAT = os.environ.get("CATALOG", "agentbricks_demo_catalog"); SCH = os.environ.get("SCHEMA", "axfood_demo")
def api(method, path, body=None):
    cmd = ["databricks", "api", method, path, "--profile", PROFILE]
    if body is not None: cmd += ["--json", json.dumps(body)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode: raise SystemExit(r.stderr or r.stdout)
    return json.loads(r.stdout)
def run(stmt):
    d = api("post", "/api/2.0/sql/statements", {"warehouse_id": WH, "statement": stmt, "wait_timeout": "50s", "on_wait_timeout": "CONTINUE"})
    while d["status"]["state"] in ("PENDING", "RUNNING"):
        time.sleep(5); d = api("get", f"/api/2.0/sql/statements/{d['statement_id']}")
    if d["status"]["state"] != "SUCCEEDED": raise SystemExit("FAILED: " + json.dumps(d["status"].get("error")))
    return d
text = open(sys.argv[1]).read().replace("${catalog}", CAT).replace("${schema}", SCH)
text = re.sub(r"^\s*--.*$", "", text, flags=re.M)
for stmt in [s.strip() for s in re.split(r";\s*(?:\n|$)", text) if s.strip()]:
    t0 = time.time(); d = run(stmt)
    print(f"OK {time.time()-t0:5.1f}s  {stmt[:70].replace(chr(10),' ')}")
