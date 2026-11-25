"""Add the architecture slide and the deployed-objects slide to a COPY of the Axfood deck."""
import sys, json
sys.path.insert(0, "/Users/marcelina.polak/.vibe/marketplace/plugins/fe-google-tools/skills/google-slides/resources")
import gslides_builder as g

PRES = "1upM6-UBI_1hoB9gIaTIUzSEweWoZYqWhsqyfU9t4wVA"   # the copy, never the original

def page(pid):
    p = g.api_call("GET", f"https://slides.googleapis.com/v1/presentations/{PRES}")
    return next(s for s in p["slides"] if s["objectId"] == pid)

def text_of(e):
    return "".join(t.get("textRun", {}).get("content", "") for t in e.get("shape", {}).get("text", {}).get("textElements", [])).strip()

def set_text(shape, new):
    reqs = []
    if text_of({"shape": shape}) != "":
        reqs.append({"deleteText": {"objectId": shape["objectId"], "textRange": {"type": "ALL"}}})
    reqs.append({"insertText": {"objectId": shape["objectId"], "insertionIndex": 0, "text": new}})
    return reqs

# 1. Architecture slide: duplicate slide 19 (reference architecture), swap title, subtitle, image, caption
if "--skip-dup" not in sys.argv: g.duplicate_slide(PRES, "sl19aa", "arch_overview")
s = page("arch_overview")
els = s["pageElements"]
title, sub, cap = els[0], els[1], els[3]
img = next(e for e in els if "image" in e)
IN = 914400
reqs = [{"deleteObject": {"objectId": img["objectId"]}}]
for e, t in ((title, "One job runs the whole ML lifecycle"), (sub, "Pipeline, features, training, registry, batch scoring and serving on one governed platform"),
             (cap, "Synthetic data. Built with Databricks Solution Builder.")):
    e2 = dict(e); reqs += set_text(e2, t)
# image is inserted by hand: org policy blocks public Drive links, so Slides cannot fetch it
g.batch_update(PRES, reqs)

# 2. Deployed objects slide: duplicate the text-only demo walkthrough slide (slide 29)
g.duplicate_slide(PRES, "sl23aa", "deployed_objects")
s = page("deployed_objects")
els = s["pageElements"]
body = [
    "Sales pipeline: a Spark Declarative Pipeline, bronze to silver to gold with quality rules, serverless",
    "Lifecycle job: five tasks (prepare data, build features, train and register, batch score, serve), serverless, retries and failure alerts are settings",
    "Daily scoring job: one notebook task at 06:00, created in one click from the notebook",
    "Last full run: all five tasks succeeded, with the model served on a scale to zero endpoint",
    "The same objects as code: three YAML files in a Declarative Automation Bundle, deployed with one command",
]
reqs = []
reqs += set_text(els[0], "What is deployed: one pipeline, one job")
reqs += set_text(els[1], "The real objects behind the demo. No Docker image, no DAG file")
reqs += set_text(els[2], "\n".join(body))
g.batch_update(PRES, reqs)
print("ok")
