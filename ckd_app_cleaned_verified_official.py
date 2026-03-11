"""
CKD Progression Risk Calculator
Streamlit clinical decision-support tool.
Trained on T1DiabetesGranada dataset (713 patients).
Models: Random Forest - Model A (8 vars) and Model B (14 vars).
Pipeline v4.1 -- select-by auc, seed=42
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import shap

    SHAP_AVAILABLE = True
    SHAP_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    shap = None
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = str(exc)

BASE = Path(__file__).parent

MODEL_A_REQUIRED = [
    "Age",
    "Sex",
    "Diabetes_duration_years",
    "eGFR",
    "log_ACR",
    "Glycated hemoglobin (A1c)",
    "Uric acid",
    "Hypertension",
]
MODEL_B_REQUIRED = [
    "Age",
    "Sex",
    "Diabetes_duration_years",
    "eGFR",
    "log_ACR",
    "Glycated hemoglobin (A1c)",
    "Glucose",
    "Triglycerides",
    "HDL cholesterol",
    "Uric acid",
    "Potassium",
    "Sodium",
    "Hypertension",
    "Retinopathy",
]

RISK_LOW = 0.05
RISK_MOD = 0.20

st.set_page_config(
    page_title="CKD Risk Calculator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}
.main { background-color: #F0F4F8; }
.stApp { background-color: #F0F4F8; }
h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; }

.title-block {
    background: linear-gradient(135deg, #1F4E79 0%, #2E75B6 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
}
.title-block h1 {
    color: white;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
}
.title-block p {
    color: rgba(255,255,255,0.88);
    font-size: 0.95rem;
    margin: 0;
}

.risk-low {
    background: linear-gradient(135deg, #D4EDDA, #C3E6CB);
    border-left: 5px solid #28A745;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    color: #1A1A1A !important;
}
.risk-moderate {
    background: linear-gradient(135deg, #FFF3CD, #FFEEBA);
    border-left: 5px solid #FFC107;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    color: #1A1A1A !important;
}
.risk-high {
    background: linear-gradient(135deg, #F8D7DA, #F5C6CB);
    border-left: 5px solid #DC3545;
    border-radius: 8px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    color: #1A1A1A !important;
}
.risk-title {
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
}
.risk-prob {
    font-size: 3rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    margin: 0.3rem 0;
    line-height: 1;
}
.risk-ci {
    font-size: 0.9rem;
    opacity: 0.8;
    font-family: 'IBM Plex Mono', monospace;
    margin: 0.2rem 0 0.8rem 0;
}
.risk-action {
    font-size: 0.95rem;
    font-weight: 500;
    margin: 0.5rem 0 0 0;
}

.driver-box {
    background: white;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.driver-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #6C757D !important;
    margin-bottom: 0.8rem;
}
.driver-item {
    display: flex;
    align-items: center;
    padding: 0.4rem 0;
    font-size: 0.95rem;
    color: #1A1A1A !important;
    border-bottom: 1px solid #F0F0F0;
}
.driver-item * { color: inherit !important; }
.driver-item:last-child { border-bottom: none; }
.driver-up { color: #DC3545 !important; font-weight: 700; margin-right: 0.5rem; }
.driver-down { color: #28A745 !important; font-weight: 700; margin-right: 0.5rem; }

.metric-card {
    background: white;
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin: 0.4rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border-left: 3px solid #2E75B6;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #6C757D;
    margin-bottom: 0.15rem;
}
.metric-value {
    font-size: 1.02rem;
    font-weight: 700;
    color: #1F4E79;
    font-family: 'IBM Plex Mono', monospace;
}

.disclaimer {
    background: #FFF8E1;
    border: 1px solid #FFE082;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-size: 0.8rem;
    color: #5D4037;
    margin-top: 1rem;
}

.section-header {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #6C757D !important;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 2px solid #E9ECEF;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: white;
    padding: 0.5rem;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    font-weight: 600;
    font-size: 0.9rem;
}

/* Sidebar theme - force ALL text white */
div[data-testid="stSidebar"] { background: #1F4E79; }
div[data-testid="stSidebar"] * { color: white !important; }
div[data-testid="stSidebar"] .metric-card { background: rgba(255,255,255,0.96) !important; border-left-color: #4DB6AC !important; }
div[data-testid="stSidebar"] .metric-card * { color: #1A1A1A !important; }
div[data-testid="stSidebar"] .metric-label { color: #4F5B66 !important; }
div[data-testid="stSidebar"] .metric-value { color: #1F4E79 !important; }
div[data-testid="stSidebar"] .sidebar-chart-label {
    color: #F8FBFF !important;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    margin: 0.6rem 0 0.3rem 0;
}

/* Expander - force white background and dark text */
div[data-testid="stExpander"] { background-color: white !important; }
div[data-testid="stExpander"] > details { background-color: white !important; }
div[data-testid="stExpander"] > details > summary { color: #1A1A1A !important; background: white !important; }
div[data-testid="stExpander"] > details > div { background-color: white !important; color: #1A1A1A !important; }
div[data-testid="stExpander"] details summary span { color: #1A1A1A !important; }
div[data-testid="stExpander"] details > div * { color: #1A1A1A !important; background-color: white !important; }
[data-testid="stExpanderDetails"] { background-color: white !important; color: #1A1A1A !important; }
[data-testid="stExpanderDetails"] * { color: #1A1A1A !important; }

/* Main content only - force dark text */
[data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"] p,
[data-testid="stAppViewContainer"] .main [data-testid="stMarkdownContainer"] strong,
[data-testid="stAppViewContainer"] .main [data-testid="stWidgetLabel"] p,
[data-testid="stAppViewContainer"] .main [data-testid="stWidgetLabel"] span,
[data-testid="stAppViewContainer"] .main label {
    color: #1A1A1A !important;
}

/* Streamlit main section fallback selectors */
section[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stMain"] [data-testid="stMarkdownContainer"] strong,
section[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
section[data-testid="stMain"] [data-testid="stWidgetLabel"] span,
section[data-testid="stMain"] label,
section[data-testid="stMain"] li,
section[data-testid="stMain"] span {
    color: #1A1A1A !important;
}
section[data-testid="stMain"] .stTabs [data-baseweb="tab"] {
    color: #1A1A1A !important;
}
section[data-testid="stMain"] .stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #E53935 !important;
}

/* Sidebar markdown text must stay light for contrast */
[data-testid="stSidebarUserContent"] .stMarkdown,
[data-testid="stSidebarUserContent"] .stMarkdown *,
[data-testid="stSidebarUserContent"] p,
[data-testid="stSidebarUserContent"] li,
[data-testid="stSidebarUserContent"] span,
[data-testid="stSidebarUserContent"] strong,
[data-testid="stSidebarUserContent"] a,
[data-testid="stSidebarUserContent"] h1,
[data-testid="stSidebarUserContent"] h2,
[data-testid="stSidebarUserContent"] h3,
[data-testid="stSidebarUserContent"] h4,
[data-testid="stSidebarUserContent"] h5,
[data-testid="stSidebarUserContent"] h6 {
    color: #F8FBFF !important;
}

/* Keep metric-card text dark for readability on light cards */
[data-testid="stSidebarUserContent"] .metric-card,
[data-testid="stSidebarUserContent"] .metric-card * {
    color: #1A1A1A !important;
}
[data-testid="stSidebarUserContent"] .metric-card .metric-label {
    color: #4F5B66 !important;
}
[data-testid="stSidebarUserContent"] .metric-card .metric-value {
    color: #1F4E79 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


def _normalize_feature_list(feats):
    if isinstance(feats, pd.Index):
        feats = feats.tolist()
    elif isinstance(feats, np.ndarray):
        feats = feats.tolist()
    elif isinstance(feats, tuple):
        feats = list(feats)
    if not isinstance(feats, list):
        raise TypeError("Feature artifact must be a list-like object.")
    return [str(f) for f in feats]


def _validate_pipeline(model, label):
    if not hasattr(model, "named_steps"):
        raise TypeError(f"{label}: loaded object is not an sklearn Pipeline")
    if "clf" not in model.named_steps:
        raise KeyError(f"{label}: pipeline must contain a 'clf' step")
    _ = model[:-1]


def _validate_feature_contract(model, feats, required, label):
    if len(set(feats)) != len(feats):
        raise ValueError(f"{label}: duplicate feature names found")
    expected_n = getattr(model.named_steps["clf"], "n_features_in_", None)
    if expected_n is not None and len(feats) != int(expected_n):
        raise ValueError(f"{label}: feature count mismatch (features={len(feats)}, model={expected_n})")
    if set(feats) != set(required):
        missing = sorted(set(required) - set(feats))
        extra = sorted(set(feats) - set(required))
        raise ValueError(f"{label}: feature mismatch. Missing={missing}; Extra={extra}")


@st.cache_resource(show_spinner="Loading CKD models...")
def load_models():
    files = {
        "model_A": BASE / "ckd_model_A.pkl",
        "feats_A": BASE / "ckd_features_A.pkl",
        "model_B": BASE / "ckd_model_B.pkl",
        "feats_B": BASE / "ckd_features_B.pkl",
    }
    missing = [p.name for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))

    model_A = joblib.load(files["model_A"])
    feats_A = _normalize_feature_list(joblib.load(files["feats_A"]))
    model_B = joblib.load(files["model_B"])
    feats_B = _normalize_feature_list(joblib.load(files["feats_B"]))
    return model_A, feats_A, model_B, feats_B


try:
    model_A, feats_A, model_B, feats_B = load_models()
    _validate_pipeline(model_A, "Model A")
    _validate_pipeline(model_B, "Model B")
    _validate_feature_contract(model_A, feats_A, MODEL_A_REQUIRED, "Model A")
    _validate_feature_contract(model_B, feats_B, MODEL_B_REQUIRED, "Model B")
except Exception as exc:
    st.error(
        "Failed to load/validate model artifacts.\n\n"
        f"Details: {type(exc).__name__}: {exc}\n\n"
        "Place ckd_model_A.pkl, ckd_features_A.pkl, ckd_model_B.pkl, ckd_features_B.pkl next to this app."
    )
    st.stop()


# FIX: underscore prefix on _model tells Streamlit not to hash the sklearn Pipeline
@st.cache_resource(show_spinner="Preparing SHAP explainers...")
def get_explainer(_model):
    if not SHAP_AVAILABLE:
        return None
    return shap.TreeExplainer(_model.named_steps["clf"])


explainer_A = get_explainer(model_A)
explainer_B = get_explainer(model_B)


def _safe_transform(model, x_input):
    return model[:-1].transform(x_input)


def _compute_shap_output(explainer, x_t):
    try:
        return explainer(x_t)
    except TypeError:
        return explainer.shap_values(x_t)


def _extract_shap_row(shap_out):
    if SHAP_AVAILABLE and isinstance(shap_out, shap.Explanation):
        values = np.asarray(shap_out.values)
    else:
        values = shap_out

    if isinstance(values, list):
        arr = np.asarray(values[1] if len(values) > 1 else values[0])
        return np.asarray(arr[0], dtype=float)

    arr = np.asarray(values)
    if arr.ndim == 3:
        return np.asarray(arr[0, :, 1] if arr.shape[-1] > 1 else arr[0, :, 0], dtype=float)
    if arr.ndim == 2:
        return np.asarray(arr[0], dtype=float)
    raise ValueError(f"Unexpected SHAP output shape: {arr.shape}")


def _extract_base_value(explainer, shap_out):
    if SHAP_AVAILABLE and isinstance(shap_out, shap.Explanation):
        bv = np.asarray(shap_out.base_values)
        if bv.ndim == 2:
            return float(bv[0, 1] if bv.shape[1] > 1 else bv[0, 0])
        if bv.ndim == 1:
            return float(bv[1] if bv.size > 1 else bv[0])
        if bv.ndim == 0:
            return float(bv)
    ev = getattr(explainer, "expected_value", 0.0)
    ev_arr = np.asarray(ev)
    if ev_arr.ndim == 0:
        return float(ev_arr)
    return float(ev_arr.ravel()[1] if ev_arr.size > 1 else ev_arr.ravel()[0])


def predict_with_ci(model, x_input):
    prob = float(model.predict_proba(x_input)[0, 1])
    clf = model.named_steps["clf"]

    if not hasattr(clf, "estimators_"):
        prob = float(np.clip(prob, 0.0, 1.0))
        return prob, prob, prob

    x_t = _safe_transform(model, x_input)
    tree_probs = [float(tree.predict_proba(x_t)[0, 1]) for tree in clf.estimators_]
    if not tree_probs:
        prob = float(np.clip(prob, 0.0, 1.0))
        return prob, prob, prob

    tree_arr = np.asarray(tree_probs, dtype=float)
    ci_low = float(np.percentile(tree_arr, 2.5))
    ci_high = float(np.percentile(tree_arr, 97.5))
    prob = float(np.clip(prob, 0.0, 1.0))
    ci_low = float(np.clip(ci_low, 0.0, 1.0))
    ci_high = float(np.clip(ci_high, 0.0, 1.0))
    if ci_low > ci_high:
        ci_low, ci_high = ci_high, ci_low
    return prob, ci_low, ci_high


def risk_category(prob):
    if prob < RISK_LOW:
        return "Low", "risk-low", "&#x1F7E2;"
    if prob < RISK_MOD:
        return "Moderate", "risk-moderate", "&#x1F7E1;"
    return "High", "risk-high", "&#x1F534;"


def get_shap_drivers(explainer, model, x_input, feature_names, x_t=None):
    if explainer is None:
        return None, "SHAP is not available in this environment."
    try:
        if x_t is None:
            x_t = _safe_transform(model, x_input)
        shap_out = _compute_shap_output(explainer, x_t)
        sv = _extract_shap_row(shap_out)
        drivers = sorted(zip(feature_names, sv), key=lambda x: abs(x[1]), reverse=True)[:3]
        return drivers, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def shap_waterfall_fig(explainer, model, x_input, feature_names, x_t=None):
    if explainer is None:
        return None, "SHAP is not available in this environment."

    fig = None
    try:
        if x_t is None:
            x_t = _safe_transform(model, x_input)
        shap_out = _compute_shap_output(explainer, x_t)
        sv = _extract_shap_row(shap_out)
        base = _extract_base_value(explainer, shap_out)
        exp = shap.Explanation(
            values=sv,
            base_values=base,
            data=x_input.iloc[0].values,
            feature_names=feature_names,
        )
        fig, _ax = plt.subplots(figsize=(9, 5))
        shap.waterfall_plot(exp, show=False, max_display=10)
        plt.tight_layout()
        return fig, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    finally:
        if fig is None:
            plt.close()


def global_importance_fig(model, feature_names, color, title=None):
    clf = model.named_steps["clf"]
    fig, ax = plt.subplots(figsize=(4, max(3, len(feature_names) * 0.35)))
    fig.patch.set_facecolor("#1F4E79")
    ax.set_facecolor("#1F4E79")

    if not hasattr(clf, "feature_importances_"):
        ax.text(0.5, 0.5, "Feature importances unavailable", ha="center", va="center", color="white")
        ax.set_axis_off()
        return fig

    importances = np.asarray(clf.feature_importances_, dtype=float)
    idx = np.argsort(importances)
    ax.barh([feature_names[i] for i in idx], importances[idx], color=color, alpha=0.85)
    if title:
        ax.set_title(title, fontsize=9, color="white", pad=6)
    ax.set_xlabel("Importance", fontsize=8, color="white")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.2))
    plt.tight_layout()
    return fig


def _to_model_input(data_dict, feature_order):
    x = pd.DataFrame([data_dict]).astype(float)
    return x[feature_order]


def _render_risk(prob, ci_low, ci_high, category, css_class, emoji_html):
    action = {
        "Low": "Routine annual monitoring recommended.",
        "Moderate": "Increased surveillance - 6-month nephrology review recommended.",
        "High": "Intensive nephrology follow-up recommended. Consider intervention.",
    }[category]
    st.markdown(
        f"""
<div class="{css_class}">
  <div class="risk-title">{emoji_html} {category} Risk</div>
  <div class="risk-prob">{prob*100:.1f}%</div>
  <div class="risk-ci">95% CI: {ci_low*100:.1f}% - {ci_high*100:.1f}%
    <span title="Estimated from variation across all RF trees. Wider interval indicates higher uncertainty for this input profile." style="cursor:help;margin-left:4px;opacity:0.75">&#9432;</span>
  </div>
  <div class="risk-action">{action}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.progress(min(max(int(round(prob * 100)), 0), 100))


def _render_drivers_and_shap(drivers, drivers_err, fig, fig_err):
    if drivers:
        html = '<div class="driver-box"><div class="driver-title">Top Risk Drivers for This Patient</div>'
        for feat, val in drivers:
            arrow = '<span class="driver-up">&#8593;</span>' if val > 0 else '<span class="driver-down">&#8595;</span>'
            direction = "increases" if val > 0 else "decreases"
            html += f'<div class="driver-item">{arrow} <strong>{feat}</strong>&nbsp;- {direction} risk ({val:+.3f})</div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)
    elif drivers_err:
        st.info(f"SHAP drivers unavailable: {drivers_err}")

    with st.expander("📊 Why did the model give this score? (click to expand)"):
        st.markdown(
            """
<div style="background:#F0F4F8;border-radius:8px;padding:1rem 1.2rem;margin-bottom:1rem;color:#1A1A1A;">
<strong style="color:#1F4E79;">How to read this chart:</strong><br><br>
Each bar shows how much a specific measurement <em>pushed the risk score up or down</em> for this patient.<br><br>
<span style="color:#DC3545;font-weight:600;">Red bars →</span> this value <strong>increased</strong> the predicted risk.<br>
<span style="color:#2E75B6;font-weight:600;">Blue bars →</span> this value <strong>decreased</strong> the predicted risk.<br><br>
Longer bars = bigger impact on the score. The starting point (base value) is the average risk across all patients in the training data.
</div>
""",
            unsafe_allow_html=True,
        )
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        elif fig_err:
            st.info(f"Chart unavailable: {fig_err}")
        else:
            st.info("Chart unavailable for this input.")


with st.sidebar:
    st.markdown("### Model Performance")

    st.markdown("**Model A - Minimal**")
    st.markdown("8 predictors · Random Forest")
    st.markdown('<div class="metric-card"><div class="metric-label">Test AUC</div><div class="metric-value">0.857</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><div class="metric-label">95% CI</div><div class="metric-value">0.775 – 0.921</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><div class="metric-label">Brier Score</div><div class="metric-value">0.129</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-chart-label">Feature Importance (Model A)</div>', unsafe_allow_html=True)
    fig_a = global_importance_fig(model_A, feats_A, "#64B5F6", "Model A Feature Importance")
    st.pyplot(fig_a, use_container_width=True)
    plt.close(fig_a)

    st.markdown("---")
    st.markdown("**Model B - Standard ★ Primary**")
    st.markdown("14 predictors · Random Forest")
    st.markdown('<div class="metric-card"><div class="metric-label">Test AUC</div><div class="metric-value">0.867</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><div class="metric-label">95% CI</div><div class="metric-value">0.798 – 0.923</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="metric-card"><div class="metric-label">Brier Score</div><div class="metric-value">0.125</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-chart-label">Feature Importance (Model B)</div>', unsafe_allow_html=True)
    fig_b = global_importance_fig(model_B, feats_B, "#4DB6AC", "Model B Feature Importance")
    st.pyplot(fig_b, use_container_width=True)
    plt.close(fig_b)

    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown("- 713 T1D patients\n- Granada, Spain\n- 25.5% CKD positive\n- Pipeline v4.1, seed=42")
    st.markdown(
        '<div class="disclaimer" style="background:rgba(255,255,200,0.15);border-color:rgba(255,255,200,0.3);color:rgba(255,255,255,0.85)">Research tool only. Not a substitute for clinical judgment. Not validated for clinical deployment.</div>',
        unsafe_allow_html=True,
    )
    if not SHAP_AVAILABLE:
        st.warning(f"SHAP import failed. Explanations disabled. ({SHAP_IMPORT_ERROR})")


st.markdown(
    """
<div class="title-block">
  <h1>CKD Progression Risk Calculator</h1>
  <p>Predicts future CKD progression risk in Type 1 Diabetes patients from baseline variables. Outcome definition follows KDIGO thresholds after month 12 from baseline.</p>
</div>
""",
    unsafe_allow_html=True,
)


tab_A, tab_B = st.tabs(["Model A - Minimal (8 variables)", "Model B - Standard (14 variables)"])

with tab_A:
    st.markdown("**Use when only basic labs are available.** Requires 8 variables.")
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<div class="section-header">Patient Demographics</div>', unsafe_allow_html=True)
        a_age = st.slider("Age (years)", 13, 90, 45, key="a_age")
        if a_age > 82:
            st.warning("Age > 82 is outside the training data range. Predictions may be less reliable.")
        a_sex = st.selectbox("Sex", ["Female", "Male"], key="a_sex")
        a_dur = st.slider("Diabetes Duration (years)", 0, 60, 15, key="a_dur")

        st.markdown('<div class="section-header">Kidney Function</div>', unsafe_allow_html=True)
        a_egfr = st.slider("eGFR (mL/min/1.73m2)", 15, 160, 90, key="a_egfr")
        a_acr = st.slider("Baseline ACR (mg/g)", 0, 500, 10, key="a_acr")

        st.markdown('<div class="section-header">Labs and Comorbidities</div>', unsafe_allow_html=True)
        a_hba1c = st.slider("HbA1c (%)", 5.0, 14.0, 7.5, step=0.1, key="a_hba1c")
        a_uric = st.slider("Uric Acid (mg/dL)", 2.0, 10.0, 5.0, step=0.1, key="a_uric")
        a_htn = st.selectbox("Hypertension", ["No", "Yes"], key="a_htn")

        predict_A = st.button("Predict CKD Risk", key="btn_A", use_container_width=True, type="primary")

    with col_out:
        if predict_A:
            warns = []
            if a_acr == 0:
                warns.append("ACR = 0 mg/g is unusually low. Confirm this is not a missing value.")
            if a_egfr >= 150:
                warns.append("eGFR >= 150 is at the extreme upper range. Please verify.")
            if a_hba1c >= 13.0:
                warns.append("HbA1c >= 13% is very high. Ensure this is accurate.")
            if a_hba1c < 5.5:
                warns.append("HbA1c < 5.5% is unusually low for T1D. Verify this value.")
            if a_dur > 50:
                warns.append("Diabetes duration > 50 years is outside most training data range.")
            if a_egfr < 30 and a_acr < 30:
                warns.append("Low eGFR with normal ACR is unusual. Verify both values.")
            if a_dur >= a_age:
                warns.append(f"Diabetes duration ({a_dur} yr) >= age ({a_age} yr) is not possible. Check both values.")
            for w in warns:
                st.warning(f"[Model A] {w}")

            input_dict = {
                "Age": a_age,
                "Sex": 1 if a_sex == "Male" else 0,
                "Diabetes_duration_years": a_dur,
                "eGFR": a_egfr,
                "log_ACR": np.log1p(a_acr),
                "Glycated hemoglobin (A1c)": a_hba1c,
                "Uric acid": a_uric,
                "Hypertension": 1 if a_htn == "Yes" else 0,
            }
            try:
                x_in = _to_model_input(input_dict, feats_A)
                prob, ci_low, ci_high = predict_with_ci(model_A, x_in)
                cat, css, emoji = risk_category(prob)
                x_t = _safe_transform(model_A, x_in)
            except Exception as exc:
                st.error(f"Prediction failed for Model A: {type(exc).__name__}: {exc}")
                st.stop()

            _render_risk(prob, ci_low, ci_high, cat, css, emoji)
            drivers, derr = get_shap_drivers(explainer_A, model_A, x_in, feats_A, x_t=x_t)
            fig, ferr = shap_waterfall_fig(explainer_A, model_A, x_in, feats_A, x_t=x_t)
            _render_drivers_and_shap(drivers, derr, fig, ferr)
        else:
            st.markdown('<div style="background:white;border-radius:10px;padding:3rem 2rem;text-align:center;color:#9DA8B5;margin-top:2rem;box-shadow:0 1px 4px rgba(0,0,0,0.06)"><div style="font-size:1rem;font-weight:600;">Enter patient values and click Predict</div><div style="font-size:0.85rem;margin-top:0.5rem">Risk score, uncertainty interval, and SHAP explanation will appear here</div></div>', unsafe_allow_html=True)


with tab_B:
    st.markdown("**Use when full metabolic panel is available.** Primary model.")
    col_in2, col_out2 = st.columns([1, 1], gap="large")

    with col_in2:
        st.markdown('<div class="section-header">Patient Demographics</div>', unsafe_allow_html=True)
        b_age = st.slider("Age (years)", 13, 90, 45, key="b_age")
        if b_age > 82:
            st.warning("Age > 82 is outside the training data range. Predictions may be less reliable.")
        b_sex = st.selectbox("Sex", ["Female", "Male"], key="b_sex")
        b_dur = st.slider("Diabetes Duration (years)", 0, 60, 15, key="b_dur")

        st.markdown('<div class="section-header">Kidney Function</div>', unsafe_allow_html=True)
        b_egfr = st.slider("eGFR (mL/min/1.73m2)", 15, 160, 90, key="b_egfr")
        b_acr = st.slider("Baseline ACR (mg/g)", 0, 500, 10, key="b_acr")

        st.markdown('<div class="section-header">Metabolic Labs</div>', unsafe_allow_html=True)
        b_hba1c = st.slider("HbA1c (%)", 5.0, 14.0, 7.5, step=0.1, key="b_hba1c")
        b_glucose = st.slider("Glucose (mg/dL)", 50, 400, 120, key="b_glucose")
        b_trig = st.slider("Triglycerides (mg/dL)", 30, 500, 120, key="b_trig")
        b_hdl = st.slider("HDL Cholesterol (mg/dL)", 20, 120, 55, key="b_hdl")
        b_uric = st.slider("Uric Acid (mg/dL)", 2.0, 10.0, 5.0, step=0.1, key="b_uric")

        st.markdown('<div class="section-header">Electrolytes</div>', unsafe_allow_html=True)
        b_pot = st.slider("Potassium (mEq/L)", 2.5, 7.0, 4.2, step=0.1, key="b_pot")
        b_sod = st.slider("Sodium (mEq/L)", 125, 150, 140, key="b_sod")

        st.markdown('<div class="section-header">Comorbidities</div>', unsafe_allow_html=True)
        b_htn = st.selectbox("Hypertension", ["No", "Yes"], key="b_htn")
        b_ret = st.selectbox("Retinopathy", ["No", "Yes"], key="b_ret")

        predict_B = st.button("Predict CKD Risk", key="btn_B", use_container_width=True, type="primary")

    with col_out2:
        if predict_B:
            warns = []
            if b_acr == 0:
                warns.append("ACR = 0 mg/g is unusually low. Confirm this is not a missing value.")
            if b_egfr >= 150:
                warns.append("eGFR >= 150 is at the extreme upper range. Please verify.")
            if b_hba1c >= 13.0:
                warns.append("HbA1c >= 13% is very high. Ensure this is accurate.")
            if b_hba1c < 5.5:
                warns.append("HbA1c < 5.5% is unusually low for T1D. Verify this value.")
            if b_trig >= 400:
                warns.append("Triglycerides >= 400 mg/dL is severely elevated. Verify the value.")
            if b_pot >= 6.0:
                warns.append("Potassium >= 6.0 mEq/L indicates hyperkalemia. Confirm lab result.")
            if b_dur > 50:
                warns.append("Diabetes duration > 50 years is outside most training data range.")
            if b_egfr < 30 and b_acr < 30:
                warns.append("Low eGFR with normal ACR is unusual. Verify both values.")
            if b_dur >= b_age:
                warns.append(f"Diabetes duration ({b_dur} yr) >= age ({b_age} yr) is not possible. Check both values.")
            if b_hdl > 100:
                warns.append("HDL > 100 mg/dL is unusually high. Verify this value.")
            for w in warns:
                st.warning(f"[Model B] {w}")

            input_dict = {
                "Age": b_age,
                "Sex": 1 if b_sex == "Male" else 0,
                "Diabetes_duration_years": b_dur,
                "eGFR": b_egfr,
                "log_ACR": np.log1p(b_acr),
                "Glycated hemoglobin (A1c)": b_hba1c,
                "Glucose": b_glucose,
                "Triglycerides": b_trig,
                "HDL cholesterol": b_hdl,
                "Uric acid": b_uric,
                "Potassium": b_pot,
                "Sodium": b_sod,
                "Hypertension": 1 if b_htn == "Yes" else 0,
                "Retinopathy": 1 if b_ret == "Yes" else 0,
            }
            try:
                x_in = _to_model_input(input_dict, feats_B)
                prob, ci_low, ci_high = predict_with_ci(model_B, x_in)
                cat, css, emoji = risk_category(prob)
                x_t = _safe_transform(model_B, x_in)
            except Exception as exc:
                st.error(f"Prediction failed for Model B: {type(exc).__name__}: {exc}")
                st.stop()

            _render_risk(prob, ci_low, ci_high, cat, css, emoji)
            drivers, derr = get_shap_drivers(explainer_B, model_B, x_in, feats_B, x_t=x_t)
            fig, ferr = shap_waterfall_fig(explainer_B, model_B, x_in, feats_B, x_t=x_t)
            _render_drivers_and_shap(drivers, derr, fig, ferr)
        else:
            st.markdown('<div style="background:white;border-radius:10px;padding:3rem 2rem;text-align:center;color:#9DA8B5;margin-top:2rem;box-shadow:0 1px 4px rgba(0,0,0,0.06)"><div style="font-size:1rem;font-weight:600;">Enter patient values and click Predict</div><div style="font-size:0.85rem;margin-top:0.5rem">Risk score, uncertainty interval, and SHAP explanation will appear here</div></div>', unsafe_allow_html=True)
