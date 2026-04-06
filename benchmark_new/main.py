import os
import json
from difflib import SequenceMatcher

def _soft_compare(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()

def _hard_compare(a, b):
    return int(a == b)

def _normalize(a):
    return a.replace("״", '"')


page_id = 3

predictions_json_path = f"/Users/delmedigo/Dev/FineTree/outputs/gui_batch_runs/20260403_214637/pdfs/דוח_כספי_2020_-_א.ב.ן._-_סיוע_בנגב_ובדרום_ע-ר/pages/page_000{page_id}.png.json"
ground_truth_json_path = "data/annotations/דוח_כספי_2020_-_א.ב.ן._-_סיוע_בנגב_ובדרום_ע-ר.json"

with open(predictions_json_path, "rb") as f:
    prediction_data = json.loads(f.read())

prediction = json.loads(prediction_data.get("assistant_text"))

with open(ground_truth_json_path, "rb") as f:
    ground_truth_doc = json.loads(f.read())

ground_truth = ground_truth_doc.get("pages")[page_id-1]

def compare(gt, pred, method = "soft", field = "entity_name", verbose = True):
    gt = gt.get(field)
    pred = pred.get(field)

    if method == "soft":
        score = round(_soft_compare(gt, pred), 3)
    else:
        score = round(_hard_compare(gt, pred), 3)

    if verbose:
        print("----------------------")
        print(field)
        print("---- Ground Truth ----")
        print(gt)
        print("---- Predicted -------")
        print(pred)
        print("Score:", score)
        print("----------------------", "\n")
    
    return float(score)

import time

config = dict(
    entity_name = dict(comparison_type = "soft", weight = 0.2),
    page_num = dict(comparison_type = "hard", weight = 0.2),
    page_type = dict(comparison_type = "hard", weight = 0.2),
    statement_type = dict(comparison_type = "hard", weight = 0.2),
    title = dict(comparison_type = "soft", weight = 0.2),
)

if __name__ == "__main__":
    sleep = 0
    comparison_section = "meta"
    verbose = True
    is_soft = True
    fields = ['entity_name', 'page_num', 'page_type', 'statement_type', 'title']

    gt = ground_truth.get(comparison_section)
    pred = prediction.get(comparison_section)

    scores = []
    for field, cfg in config.items():
        print("field", field)
        comparison_type = cfg.get("comparison_type")
        weight = cfg.get("weight")
        scores.append(compare(gt, pred, comparison_type, field))

    print("Final score:", round(sum(scores) / len(scores), 3))