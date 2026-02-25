import argparse
import json
import re
import sys
from pathlib import Path

from pydantic import ValidationError

sys.path.append(str(Path(__file__).resolve().parents[1]))
from schemas import BBox, ExportBundle, Fact  # noqa: E402


def extract_doc_id_and_page(task: dict, result: dict) -> tuple[str, int]:
    data = task.get("data", {}) if isinstance(task, dict) else {}
    image_path = str(data.get("image") or data.get("img") or "")

    doc_id = "unknown_doc"
    page = 0

    match = re.search(r"/pages/([^/]+)/page_(\d+)\.", image_path)
    if match:
        doc_id = match.group(1)
        page = int(match.group(2))

    value_dict = result.get("value", {}) if isinstance(result, dict) else {}
    page = int(value_dict.get("page") or result.get("page") or page or 0)

    return doc_id, page


def parse_result(task: dict, result: dict) -> Fact | None:
    value_dict = result.get("value", {}) if isinstance(result, dict) else {}
    doc_id, page = extract_doc_id_and_page(task, result)

    text_values = value_dict.get("text") or value_dict.get("labels") or value_dict.get("rectanglelabels") or []
    if isinstance(text_values, list):
        value = str(text_values[0]) if text_values else ""
    else:
        value = str(text_values)

    choices = value_dict.get("choices") or []
    path_nodes = [str(x) for x in choices] if isinstance(choices, list) else []

    bbox = BBox(
        x=float(value_dict.get("x", 0.0)),
        y=float(value_dict.get("y", 0.0)),
        width=float(value_dict.get("width", 0.0)),
        height=float(value_dict.get("height", 0.0)),
        unit=str(value_dict.get("unit", "%")),
        page=page,
    )

    return Fact(doc_id=doc_id, page=page, value=value, path_nodes=path_nodes, bbox=bbox)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse Label Studio export JSON into structured facts.")
    parser.add_argument("export_path", help="Path to Label Studio export JSON inside annotation/export/")
    args = parser.parse_args()

    export_path = Path(args.export_path)
    if not export_path.exists() or export_path.suffix.lower() != ".json":
        raise SystemExit(f"Invalid export path: {export_path}")

    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise SystemExit("Expected export JSON to be a list of tasks")

    facts: list[Fact] = []
    total_results = 0
    skipped = 0

    for task in tasks:
        annotations = task.get("annotations") or []
        for ann in annotations:
            results = ann.get("result") or []
            for result in results:
                total_results += 1
                try:
                    fact = parse_result(task, result)
                    if fact is not None:
                        facts.append(fact)
                    else:
                        skipped += 1
                except (ValidationError, ValueError, TypeError):
                    skipped += 1

    bundle = ExportBundle(facts=facts)

    output_path = Path("data/processed") / f"{export_path.stem}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

    print(f"Total results: {total_results}")
    print(f"Parsed facts: {len(facts)}")
    print(f"Skipped invalid: {skipped}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
