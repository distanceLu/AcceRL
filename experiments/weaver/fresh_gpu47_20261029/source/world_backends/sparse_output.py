"""Explicit terminal-frame decode geometry; it never changes transition count."""
def sparse_last_geometry(result,horizon,*,allowed):
    keys=('observation_indices','prediction_horizon')
    if not any(key in result for key in keys):
        return False
    indices=result.get('observation_indices')
    if (not allowed or type(horizon) is not int or horizon<1
            or type(result.get('prediction_horizon')) is not int
            or result['prediction_horizon']!=horizon
            or not isinstance(indices,(list,tuple)) or len(indices)!=1
            or type(indices[0]) is not int or indices[0]!=horizon-1):
        raise ValueError('Sparse last observation requires explicit native horizon-only geometry')
    return True
