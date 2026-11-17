"""Enforce attribution-only inputs for proposed warning methods."""
ATTRIBUTION_PREFIXES = ('ig_', 'gradcam_', 'delta_ig_', 'delta_gradcam_',
                        'derivative_ig_', 'derivative_gradcam_')


def validate_attribution_columns(columns):
    if not columns:
        raise ValueError('An attribution warning model needs attribution features.')
    invalid = [column for column in columns if not column.startswith(ATTRIBUTION_PREFIXES)]
    if invalid:
        raise ValueError(f'Attribution-only warning model rejects non-attribution inputs: {invalid}')
    return list(columns)


def is_attribution_method(name):
    text = name.lower()
    return any(token in text for token in ('attribution', 'fusion', 'ig ', 'gradcam '))


def warning_method_role(name):
    if not is_attribution_method(name):
        return 'baseline'
    if any(token in name.lower() for token in ('confidence', 'msp', 'odin')):
        return 'historical_hybrid'
    return 'attribution'
