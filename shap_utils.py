# 1. Import Statements
import io, base64, logging, numpy as np, pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 3. Logging Configuration
logger = logging.getLogger("shap_utils")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 6.  Processing Functions
def _is_tree_model(model) -> bool:
    name = type(model).__name__.lower()
    mod  = getattr(type(model), "__module__", "").lower()
    return any(k in name or k in mod for k in [
        "randomforest", "extratrees", "decisiontree", "xgb", "lightgbm", "catboost"
    ])

def _ensure_background(model, X_background):
    if isinstance(X_background, pd.DataFrame) and X_background.shape[1] > 0:
        return X_background
    n = getattr(model, "n_features_in_", 0) or (len(getattr(model, "feature_names_in_", [])) or 0)
    cols = list(getattr(model, "feature_names_in_", [])) or [f"f{i}" for i in range(n)]
    return pd.DataFrame(np.zeros((50, len(cols))), columns=cols)

def build_universal_explainer(model, X_background: pd.DataFrame | None = None):
    """
    Prefer TreeExplainer with an explicit background set (probability output).
    Fall back to generic Explainer if needed.
    """
    Xb = _ensure_background(model, X_background)

    if _is_tree_model(model):
        # Try TreeExplainer with explicit background (handles the interventional path)
        try:
            try:
                return shap.TreeExplainer(
                    model, data=Xb, model_output="probability", feature_perturbation="interventional"
                )
            except TypeError:
                # Older/newer SHAP variants may use 'background=' instead of 'data='
                return shap.TreeExplainer(
                    model, background=Xb, model_output="probability", feature_perturbation="interventional"
                )
        except Exception as e:
            logger.warning(f"TreeExplainer failed, falling back to generic Explainer: {e}")

    # Generic path (works for anything)
    return shap.Explainer(model, Xb)

def _call_explainer(explainer, X):
    # Try modern API with/without check_additivity depending on SHAP version
    try:
        return explainer(X, check_additivity=False)
    except TypeError:
        return explainer(X)

def _explain_positive_class(explainer, X: pd.DataFrame) -> shap.Explanation:
    """
    Always return a shap.Explanation for the positive class in binary classifiers.
    """
    # Modern API
    try:
        exp = _call_explainer(explainer, X)
        if isinstance(exp, shap.Explanation):
            # If multi-output (n_samples, n_features, n_outputs), select class 1
            if getattr(exp, "values", None) is not None and exp.values.ndim == 3:
                exp = shap.Explanation(
                    values=exp.values[:, :, 1],
                    base_values=np.array(exp.base_values)[:, 1]
                        if np.ndim(exp.base_values) == 2 else exp.base_values,
                    data=exp.data,
                    feature_names=exp.feature_names
                )
            return exp
    except Exception as e:
        logger.info(f"Modern SHAP API failed, trying legacy shap_values(): {e}")

    # Legacy API
    shap_vals = explainer.shap_values(X)
    expected  = explainer.expected_value

    if isinstance(shap_vals, list):         # binary => list of 2 arrays
        sv = np.array(shap_vals[1])
        base = expected[1] if isinstance(expected, (list, np.ndarray)) else expected
    else:
        sv = np.array(shap_vals)
        base = expected[0] if isinstance(expected, (list, np.ndarray)) else expected

    if sv.ndim == 1:
        sv = sv.reshape(1, -1)

    if np.ndim(base) == 0:
        base_values = np.repeat(base, sv.shape[0])
    else:
        base = np.array(base)
        base_values = base if base.shape[0] == sv.shape[0] else np.repeat(base[0], sv.shape[0])

    return shap.Explanation(
        values=sv,
        base_values=base_values,
        data=X.to_numpy(),
        feature_names=list(X.columns)
    )

def shap_waterfall_base64(exp: shap.Explanation, row_index: int = 0, max_display: int = 10) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 6))
    try:
        shap.plots.waterfall(exp[row_index], max_display=max_display, show=False)
    except Exception:
        shap.waterfall_plot(exp[row_index], max_display=max_display, show=False)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def shap_top_k_insights(exp: shap.Explanation, row_index: int = 0, k: int = 3):
    vals = exp.values[row_index]
    data = exp.data[row_index]
    feats = exp.feature_names
    triples = list(zip(feats, data, vals))
    triples.sort(key=lambda t: abs(t[2]), reverse=True)
    out = []
    for feat, value, shap_val in triples[:k]:
        if shap_val > 0:
            text = f"Your value for <strong>{feat}</strong> ({value}) materially increases the predicted risk."
            css  = "high-risk"
        elif shap_val < 0:
            text = f"Your value for <strong>{feat}</strong> ({value}) materially lowers the predicted risk."
            css  = "low-risk"
        else:
            text = f"Your value for <strong>{feat}</strong> ({value}) has a neutral effect in this prediction."
            css  = "secondary"
        out.append({"feature": feat, "value": value, "shap": float(shap_val), "text": text, "class": css})
    return out

# 7. Post-processing Function(s)
def explain_row_universal(model, explainer, X_row_df: pd.DataFrame):
    if explainer is None:
        explainer = build_universal_explainer(model, X_row_df)
    exp = _explain_positive_class(explainer, X_row_df)
    b64 = shap_waterfall_base64(exp, 0, max_display=min(10, X_row_df.shape[1]))
    insights = shap_top_k_insights(exp, 0, k=3)
    return exp, b64, insights, explainer
