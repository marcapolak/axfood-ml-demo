"""Draw the architecture overview as native, editable shapes on slide 'arch_overview' of the deck COPY."""
import sys
sys.path.insert(0, "/Users/marcelina.polak/.vibe/marketplace/plugins/fe-google-tools/skills/google-slides/resources")
import gslides_builder as g
PRES = "1upM6-UBI_1hoB9gIaTIUzSEweWoZYqWhsqyfU9t4wVA"; PG = "arch_overview"
IN = 914400; E = lambda v: int(v * IN)
TEAL = {"red": 0.106, "green": 0.192, "blue": 0.224}; CORAL = {"red": 1.0, "green": 0.373, "blue": 0.275}
WHITE = {"red": 1, "green": 1, "blue": 1}; PALE = {"red": 0.91, "green": 0.945, "blue": 0.949}
reqs = []; n = [0]

def box(x, y, w, h, text, fill=WHITE, fg=TEAL, line=TEAL, size=12, bold=True, dash=False, shape="ROUND_RECTANGLE", nofill=False):
    n[0] += 1; oid = f"ax_b{n[0]}"
    reqs.append({"createShape": {"objectId": oid, "shapeType": shape, "elementProperties": {"pageObjectId": PG,
        "size": {"width": {"magnitude": E(w), "unit": "EMU"}, "height": {"magnitude": E(h), "unit": "EMU"}},
        "transform": {"scaleX": 1, "scaleY": 1, "translateX": E(x), "translateY": E(y), "unit": "EMU"}}}})
    props = {"outline": {"outlineFill": {"solidFill": {"color": {"rgbColor": line}}}, "weight": {"magnitude": 1.25, "unit": "PT"}}}
    fields = "outline.outlineFill.solidFill.color,outline.weight"
    if dash: props["outline"]["dashStyle"] = "DASH"; fields += ",outline.dashStyle"
    if nofill: props["shapeBackgroundFill"] = {"propertyState": "NOT_RENDERED"}; fields += ",shapeBackgroundFill.propertyState"
    else: props["shapeBackgroundFill"] = {"solidFill": {"color": {"rgbColor": fill}}}; fields += ",shapeBackgroundFill.solidFill.color"
    reqs.append({"updateShapeProperties": {"objectId": oid, "shapeProperties": props, "fields": fields}})
    if text:
        reqs.append({"insertText": {"objectId": oid, "text": text}})
        reqs.append({"updateTextStyle": {"objectId": oid, "textRange": {"type": "ALL"}, "style": {
            "foregroundColor": {"opaqueColor": {"rgbColor": fg}}, "fontSize": {"magnitude": size, "unit": "PT"}, "bold": bold,
            "fontFamily": "DM Sans"}, "fields": "foregroundColor,fontSize,bold,fontFamily"}})
        reqs.append({"updateParagraphStyle": {"objectId": oid, "textRange": {"type": "ALL"}, "style": {"alignment": "CENTER"}, "fields": "alignment"}})
        reqs.append({"updateShapeProperties": {"objectId": oid, "shapeProperties": {"contentAlignment": "MIDDLE"}, "fields": "contentAlignment"}})
    return oid

def arrow(x1, y1, x2, y2, dashed=False, both=False):
    n[0] += 1; oid = f"ax_a{n[0]}"
    w, h = abs(x2 - x1), max(abs(y2 - y1), 0.001)
    sy = 1 if y2 >= y1 else -1
    reqs.append({"createLine": {"objectId": oid, "lineCategory": "STRAIGHT", "elementProperties": {"pageObjectId": PG,
        "size": {"width": {"magnitude": E(w) or 1, "unit": "EMU"}, "height": {"magnitude": E(h) or 1, "unit": "EMU"}},
        "transform": {"scaleX": 1, "scaleY": sy, "translateX": E(min(x1, x2)), "translateY": E(y1), "unit": "EMU"}}}})
    lp = {"lineFill": {"solidFill": {"color": {"rgbColor": TEAL}}}, "weight": {"magnitude": 1.5, "unit": "PT"}, "endArrow": "FILL_ARROW"}
    f = "lineFill.solidFill.color,weight,endArrow"
    if dashed: lp["dashStyle"] = "DASH"; f += ",dashStyle"
    if both: lp["startArrow"] = "FILL_ARROW"; f += ",startArrow"
    reqs.append({"updateLineProperties": {"objectId": oid, "lineProperties": lp, "fields": f}})

X = {"s": 0.6, "p": 3.0, "b": 5.6, "r": 8.2, "c": 10.8}; W = {"s": 2.0, "p": 2.2, "b": 2.2, "r": 2.2, "c": 2.0}
Y = [2.45, 3.6, 4.75]; H = 0.85; mid = lambda i: Y[i] + H / 2

# platform outline + governance bar
box(2.85, 2.2, 10.1, 3.6, "", line=TEAL, nofill=True, dash=True)
box(2.85, 5.92, 10.1, 0.5, "Unity Catalog: one governance model for data, features, models and jobs (permissions, lineage, audit)", fill=TEAL, fg=WHITE, size=12)
# column headers
for k, t in (("s", "Sources"), ("p", "Data"), ("b", "Build"), ("r", "Run"), ("c", "Consume")):
    box(X[k], 2.0 - 0.0, W[k], 0.3, "", nofill=True, line=WHITE) if False else None
# boxes
box(X["s"], Y[0], W["s"], H, "Weekly sales"); box(X["s"], Y[1], W["s"], H, "Store and product data"); box(X["s"], Y[2], W["s"], H, "Git + Declarative Automation Bundles", size=11, fill=PALE)
box(X["p"], Y[1], W["p"], H, "Sales pipeline\nbronze, silver, gold")
box(X["b"], Y[0], W["b"], H, "Feature Store"); box(X["b"], Y[1], W["b"], H, "Model Training\nMLflow runs"); box(X["b"], Y[2], W["b"], H, "UC Model Registry\n@champion", fill=CORAL, fg=WHITE, line=CORAL)
box(X["r"], Y[0], W["r"], H, "Lifecycle job\nserverless, retries, alerts"); box(X["r"], Y[1], W["r"], H, "Model Serving\nscale to zero"); box(X["r"], Y[2], W["r"], H, "Online features\nLakebase")
box(X["c"], Y[0], W["c"], H, "Genie"); box(X["c"], Y[1], W["c"], H, "AI/BI Dashboard"); box(X["c"], Y[2], W["c"], H, "Apps, real time")
# arrows
arrow(2.6, mid(0), 3.0, mid(1)); arrow(2.6, mid(1), 3.0, mid(1))
arrow(5.2, mid(1), 5.6, mid(0))
arrow(6.7, Y[0] + H, 6.7, Y[1]); arrow(6.7, Y[1] + H, 6.7, Y[2])
arrow(7.8, mid(2), 8.2, mid(1))
arrow(9.3, Y[2], 9.3, Y[1] + H)
arrow(7.8, mid(0), 8.2, mid(0), dashed=True, both=True)
arrow(10.4, mid(0), 10.8, mid(0)); arrow(10.4, mid(0), 10.8, mid(1))
arrow(10.4, mid(1), 10.8, mid(2))
g.batch_update(PRES, reqs)
print("drawn", n[0], "objects")
