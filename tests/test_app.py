"""
Tests for Streamlit application imports and components.
"""

import importlib


def test_streamlit_app_import():
    """
    Test that app.streamlit_app can be imported without errors.
    """
    app_module = importlib.import_module("app.streamlit_app")
    assert hasattr(app_module, "main")
    assert hasattr(app_module, "load_scaler_and_models")
