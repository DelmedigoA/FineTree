from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..evaluation_specs import get_evaluator_spec
from ..models import FactsChannelSummary, FactsSummary, RunResult, to_jsonable


def _weighted_mean(values: list[tuple[float, float]], *, normalize_weights: bool) -> float:
    if not values:
        return 0.0
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return 0.0
    if normalize_weights:
        return float(sum(score * (weight / total_weight) for score, weight in values))
    return float(sum(score * weight for score, weight in values))


def _safe_divide(numerator: float, denominator: float, *, empty_value: float) -> float:
    if denominator <= 0:
        return empty_value
    return float(numerator / denominator)


def summarize_facts_run(
    run_result: RunResult,
    *,
    run_dir: Path,
    manifest: dict[str, Any] | None = None,
) -> FactsSummary:
    evaluator = get_evaluator_spec("facts")
    manifest = manifest or {}
    pages = [page for document in run_result.documents for page in document.pages]
    page_count = len(pages)
    channels: list[FactsChannelSummary] = []
    weighted_channels: list[tuple[float, float]] = []

    for section in evaluator.sections:
        for field_spec in section.fields:
            if field_spec.aggregate_on not in {None, "facts"}:
                raise ValueError(f"Unsupported facts aggregation scope for {field_spec.key}: {field_spec.aggregate_on}")

            field_results = [page.facts_result[field_spec.field] for page in pages if field_spec.field in page.facts_result]
            if not field_results:
                continue

            total_gt_facts = sum(int(page.gt_fact_count) for page in pages)
            total_pred_facts = sum(int(page.pred_fact_count) for page in pages)
            total_matches = sum(int(result.exact_metrics.matches) for result in field_results)

            if field_spec.field == "value":
                total_matched_pairs = sum(int(result.details.get("matched_pair_count", 0)) for result in field_results)
                total_correct_matched_pairs = sum(
                    float(result.details.get("value_accuracy", 0.0)) * int(result.details.get("matched_pair_count", 0))
                    for result in field_results
                )
                alignment_precision = _safe_divide(total_matches, total_pred_facts, empty_value=1.0 if total_gt_facts == 0 else 0.0)
                alignment_recall = _safe_divide(total_matches, total_gt_facts, empty_value=1.0 if total_pred_facts == 0 else 0.0)
                alignment_f1 = (
                    float((2 * alignment_precision * alignment_recall) / (alignment_precision + alignment_recall))
                    if (alignment_precision + alignment_recall)
                    else 0.0
                )

                value_group = field_spec.metric_groups.get("value")
                alignment_group = field_spec.metric_groups.get("alignment")
                value_weight = field_spec.weight / 2.0
                alignment_weight = field_spec.weight / 2.0

                if value_group is not None and "accuracy" in value_group.metrics:
                    if value_group.on not in {None, "matched_pairs"}:
                        raise ValueError(f"Unsupported facts.value metric scope: {value_group.on}")
                    accuracy_score = _safe_divide(total_correct_matched_pairs, total_matched_pairs, empty_value=1.0 if total_gt_facts == 0 and total_pred_facts == 0 else 0.0)
                    channels.append(
                        FactsChannelSummary(
                            field=field_spec.field,
                            channel="value",
                            metric="accuracy",
                            score=accuracy_score,
                            weight=value_weight,
                            fact_count=total_matched_pairs,
                        )
                    )
                    weighted_channels.append((accuracy_score, value_weight))

                if alignment_group is not None:
                    metric_map = {
                        "precision": alignment_precision,
                        "recall": alignment_recall,
                        "f1": alignment_f1,
                    }
                    for metric in alignment_group.metrics:
                        if metric not in metric_map:
                            continue
                        score = float(metric_map[metric])
                        channels.append(
                            FactsChannelSummary(
                                field=field_spec.field,
                                channel="alignment",
                                metric=metric,
                                score=score,
                                weight=alignment_weight if metric == "f1" else 0.0,
                                fact_count=max(total_gt_facts, total_pred_facts),
                            )
                        )
                        if metric == "f1":
                            weighted_channels.append((score, alignment_weight))
                continue

            if "accuracy" in field_spec.metrics:
                denominator = max(total_gt_facts, total_pred_facts)
                if field_spec.field == "path" and field_spec.compare_methods:
                    metric_groups = field_spec.metric_groups
                    scored_channels: list[tuple[float, float]] = []
                    if "strict" in metric_groups:
                        total_method_matches = sum(
                            int(result.details.get("path_method_summaries", {}).get("path_all_levels_threshold", {}).get("matches", 0))
                            for result in field_results
                        )
                        accuracy_score = _safe_divide(total_method_matches, denominator, empty_value=1.0)
                        channels.append(FactsChannelSummary(field=field_spec.field, channel="strict", metric="accuracy", score=accuracy_score, weight=0.0, fact_count=denominator))
                        scored_channels.append((accuracy_score, 1.0))
                    if "leaf" in metric_groups:
                        total_method_matches = sum(
                            int(result.details.get("path_method_summaries", {}).get("path_last_leaf_threshold", {}).get("matches", 0))
                            for result in field_results
                        )
                        accuracy_score = _safe_divide(total_method_matches, denominator, empty_value=1.0)
                        channels.append(FactsChannelSummary(field=field_spec.field, channel="leaf", metric="accuracy", score=accuracy_score, weight=0.0, fact_count=denominator))
                        scored_channels.append((accuracy_score, 1.0))
                    if "soft_overlap" in metric_groups:
                        total_shared_nodes = sum(
                            int(result.details.get("path_method_summaries", {}).get("path_soft_overlap", {}).get("shared_nodes", 0))
                            for result in field_results
                        )
                        total_gt_nodes = sum(
                            int(result.details.get("path_method_summaries", {}).get("path_soft_overlap", {}).get("gt_nodes", 0))
                            for result in field_results
                        )
                        total_pred_nodes = sum(
                            int(result.details.get("path_method_summaries", {}).get("path_soft_overlap", {}).get("pred_nodes", 0))
                            for result in field_results
                        )
                        precision = _safe_divide(total_shared_nodes, total_pred_nodes, empty_value=1.0 if total_gt_nodes == 0 else 0.0)
                        recall = _safe_divide(total_shared_nodes, total_gt_nodes, empty_value=1.0 if total_pred_nodes == 0 else 0.0)
                        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
                        channels.append(FactsChannelSummary(field=field_spec.field, channel="soft_overlap", metric="precision", score=precision, weight=0.0, fact_count=total_pred_nodes))
                        channels.append(FactsChannelSummary(field=field_spec.field, channel="soft_overlap", metric="recall", score=recall, weight=0.0, fact_count=total_gt_nodes))
                        channels.append(FactsChannelSummary(field=field_spec.field, channel="soft_overlap", metric="f1", score=f1, weight=0.0, fact_count=max(total_gt_nodes, total_pred_nodes)))
                        scored_channels.append((f1, 1.0))
                    if scored_channels:
                        channel_weight = field_spec.weight / float(len(scored_channels))
                        scored_index = 0
                        for index, channel in enumerate(channels):
                            if channel.field != field_spec.field:
                                continue
                            if channel.channel == "strict" and channel.metric == "accuracy":
                                channels[index] = FactsChannelSummary(**{**channel.__dict__, "weight": channel_weight})
                                weighted_channels.append((channel.score, channel_weight))
                            elif channel.channel == "leaf" and channel.metric == "accuracy":
                                channels[index] = FactsChannelSummary(**{**channel.__dict__, "weight": channel_weight})
                                weighted_channels.append((channel.score, channel_weight))
                            elif channel.channel == "soft_overlap" and channel.metric == "f1":
                                channels[index] = FactsChannelSummary(**{**channel.__dict__, "weight": channel_weight})
                                weighted_channels.append((channel.score, channel_weight))
                else:
                    accuracy_score = _safe_divide(total_matches, denominator, empty_value=1.0)
                    channels.append(
                        FactsChannelSummary(
                            field=field_spec.field,
                            channel="overall",
                            metric="accuracy",
                            score=accuracy_score,
                            weight=field_spec.weight,
                            fact_count=denominator,
                        )
                    )
                    weighted_channels.append((accuracy_score, field_spec.weight))

    global_aggregation = evaluator.global_aggregation
    overall_score = _weighted_mean(
        weighted_channels,
        normalize_weights=bool(global_aggregation.normalize_weights) if global_aggregation is not None else True,
    )
    return FactsSummary(
        benchmark_version=evaluator.benchmark_version,
        evaluator=evaluator.name,
        label=evaluator.label,
        run_dir=Path(run_dir),
        dataset_version_id=None if manifest.get("dataset_version_id") in (None, "") else str(manifest.get("dataset_version_id")),
        dataset_name=None if manifest.get("dataset_name") in (None, "") else str(manifest.get("dataset_name")),
        split=None if manifest.get("split") in (None, "") else str(manifest.get("split")),
        global_aggregation_method=None if global_aggregation is None else global_aggregation.method,
        overall_score=overall_score,
        page_count=page_count,
        fact_count=sum(int(page.gt_fact_count) for page in pages),
        channels=tuple(channels),
    )


def write_facts_summary(path: Path, summary: FactsSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_facts_summary(summary: FactsSummary) -> str:
    headers = ("field", "channel", "metric", "score", "facts")
    rows = [
        (channel.field, channel.channel, channel.metric, f"{channel.score:.6f}", str(channel.fact_count))
        for channel in summary.channels
    ]
    widths = [max([len(headers[index])] + [len(row[index]) for row in rows]) for index in range(len(headers))]
    path_fact_rows = [
        (channel.field, channel.channel, channel.metric, f"{channel.score:.6f}", str(channel.fact_count))
        for channel in summary.channels
        if channel.field == "path" and channel.channel in {"strict", "leaf"}
    ]
    path_node_rows = [
        (channel.field, channel.channel, channel.metric, f"{channel.score:.6f}", str(channel.fact_count))
        for channel in summary.channels
        if channel.field == "path" and channel.channel == "soft_overlap"
    ]
    other_rows = [
        (channel.field, channel.channel, channel.metric, f"{channel.score:.6f}", str(channel.fact_count))
        for channel in summary.channels
        if not (channel.field == "path" and channel.channel in {"strict", "leaf", "soft_overlap"})
    ]

    def _render_table(title: str, table_rows: list[tuple[str, str, str, str, str]], *, count_label: str = "facts") -> list[str]:
        if not table_rows:
            return []
        table_headers = (headers[0], headers[1], headers[2], headers[3], count_label)
        table_widths = [max([len(table_headers[index])] + [len(row[index]) for row in table_rows]) for index in range(len(table_headers))]
        lines = [
            f"  {title}",
            f"  {' '.join(header.ljust(table_widths[index]) for index, header in enumerate(table_headers))}",
            f"  {' '.join('-' * width for width in table_widths)}",
        ]
        lines.extend(f"  {' '.join(value.ljust(table_widths[index]) for index, value in enumerate(row))}" for row in table_rows)
        return lines

    lines = [
        summary.label,
        f"  Overall Score: {summary.overall_score:.6f}",
        f"  Facts Evaluated: {summary.fact_count}",
        "",
        "  Metrics",
    ]
    lines.extend(_render_table("Value And Fact-Level Metrics", other_rows, count_label="facts"))
    if path_fact_rows:
        lines.append("")
        lines.extend(_render_table("Path Facts", path_fact_rows, count_label="facts"))
    if path_node_rows:
        lines.append("")
        lines.extend(_render_table("Path Nodes", path_node_rows, count_label="nodes"))
    return "\n".join(lines)
