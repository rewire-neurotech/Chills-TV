 
# Shim for old scikit-learn pickles expecting a private symbol
try:
    import importlib
    m = importlib.import_module("sklearn.compose._column_transformer")
    if not hasattr(m, "_RemainderColsList"):
        class _RemainderColsList(list):
            ...
        m._RemainderColsList = _RemainderColsList
except Exception:
    pass
