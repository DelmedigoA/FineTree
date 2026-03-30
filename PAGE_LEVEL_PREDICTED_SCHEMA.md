# Page Level Predicted Schema

This file is generated from `src/finetree_annotator/schema_contract.py`.
Do not edit it manually.

Refresh command:
```bash
PYTHONPATH=src ./.env/bin/python scripts/sync_page_level_predicted_schema.py
```

Schema version: `4`

Selected page meta keys:
- `entity_name`
- `page_num`
- `page_type`
- `statement_type`
- `title`

Selected fact keys:
- `bbox`
- `value`
- `fact_num`
- `equations`
- `natural_sign`
- `row_role`
- `comment_ref`
- `note_flag`
- `note_name`
- `note_num`
- `note_ref`
- `period_type`
- `period_start`
- `period_end`
- `duration_type`
- `recurring_period`
- `path`
- `path_source`
- `currency`
- `scale`
- `value_type`
- `value_context`

Exact page-level schema:
```jsonc
{
          "meta": {
            "entity_name": "<string or null>",
        "page_num": "<string or null>",
        "page_type": "title|contents|declaration|statements|other",
        "statement_type": "balance_sheet|income_statement|cash_flow_statement|statement_of_changes_in_equity|notes_to_financial_statements|board_of_directors_report|auditors_report|statement_of_activities|other_declaration|null",
        "title": "<string or null>"
          },
          "facts": [
            {
              "bbox": [<x>, <y>, <w>, <h>],
      "value": "<string>",
      "fact_num": <integer >= 1>,
      "equations": [
  {
    "equation": "<string>",
    "fact_equation": "<string or null>"
  }
]|null,
      "natural_sign": "positive|negative|null",
      "row_role": "detail|total",
      "comment_ref": "<string or null>",
      "note_flag": <true|false>,
      "note_name": "<string or null>",
      "note_num": "<string or null>",
      "note_ref": "<string or null>",
      "period_type": "instant|duration|expected|null",
      "period_start": "<YYYY-MM-DD|null>",
      "period_end": "<YYYY-MM-DD|null>",
      "duration_type": "recurrent|null",
      "recurring_period": "daily|quarterly|monthly|yearly|null",
      "path": ["<string>", "..."],
      "path_source": "observed|inferred|null",
      "currency": "ILS|USD|EUR|GBP|null",
      "scale": 1|1000|1000000|null,
      "value_type": "amount|percent|ratio|count|points|null",
      "value_context": "textual|tabular|mixed|null"
            }
          ]
        }
```
