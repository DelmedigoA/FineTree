{
    'benchmark_version': 'v1.0',

    'global_aggregation': {
        'method': 'weighted_mean',        # weighted average across fields
        'normalize_weights': True         # normalize weights to sum to 1
    },

    'report': {
        'per_field': True,                # show score per field
        'per_document': False,             # show score per document
        'include_std': False            # include std (stability / variance)
    },

    'normalize_config': {
        'strip': True,
        'quotes_unify': True,
        'lowercase': True
    },

    'fields': {

        'meta.entity_name': {
            'compare_method': 'string_similarity',   # soft similarity (e.g., SequenceMatcher)
            'averaging_rule': 'micro',               # average over all pages
            'normalize': True,                       # apply normalization
            'null_handling': 'regular',              # None == None is correct
            'metrics': ['mean'],              # mean similarity + variability
            'weight': 1.0                            # contribution to final score
        },

        'meta.page_num': {
            'compare_method': 'exact_match',         # strict equality
            'averaging_rule': 'micro',               # per-page aggregation
            'null_handling': 'regular',
            'metrics': ['accuracy'],                 # % correct
            'weight': 1.0
        },

        'meta.page_type': {
            'compare_method': 'exact_match',
            'averaging_rule': 'micro',
            'null_handling': 'regular',
            'metrics': ['accuracy'],
            'weight': 1.0
        },

        'meta.statement_type': {
            'compare_method': 'exact_match',
            'averaging_rule': 'micro',
            'null_handling': 'regular',
            'metrics': ['accuracy'],
            'weight': 1.0
        },

        'meta.title': {
            'compare_method': 'string_similarity',
            'averaging_rule': 'micro',
            'normalize': True,
            'null_handling': 'regular',
            'metrics': ['mean'],
            'weight': 1.0
        },
    }
}