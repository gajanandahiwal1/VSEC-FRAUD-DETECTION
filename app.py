from __future__ import annotations

from datetime import date, time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import FEATURE_COLUMNS, generate_transactions, normalize_transactions
from src.modeling import train_models


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "credit_card_fraud_dataset.csv"
MODEL_PATH = ROOT / "models" / "model_bundle.joblib"
PAGES = ["Dashboard", "Predict Transaction", "Model Comparison", "Analytics", "Data & Models", "About"]

st.set_page_config(page_title="Fraud Detection | Operations", page_icon="◈", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#e8f1f2; --muted:#91a8b0; --paper:#0a141b; --line:#263b45; --teal:#43d6c4; --panel:#12232c; --panel-raised:#172a34; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:linear-gradient(145deg,rgba(33,75,78,.18),transparent 34%),linear-gradient(0deg,#0a141b,#0c1921); color:var(--ink); }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:0 !important; }
    .block-container { padding-top:1.45rem; padding-bottom:2rem; max-width:1500px; }
    [data-testid="stSidebar"] { background:#081219; border-right:1px solid #20343d; }
    [data-testid="stSidebar"] * { color:#d9e7e9; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius:5px; padding:.35rem .45rem; }
    [data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:14px 16px; }
    [data-testid="stMetricLabel"] p { color:var(--muted); }
    [data-testid="stMetricValue"] { color:#f2f8f8; }
    div[data-testid="stPlotlyChart"] { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:5px; }
    .eyebrow { color:var(--teal); text-transform:uppercase; font-weight:700; font-size:.76rem; letter-spacing:1.1px; }
    .page-title { font-family:'Space Grotesk',sans-serif; font-size:2rem; font-weight:700; margin:.1rem 0 .3rem; }
    .subtle { color:var(--muted); }
    .notice { border-left:4px solid #f0bc61; background:#2b261b; padding:12px 16px; border-radius:3px; color:#f3dfb5; }
    .result { padding:22px; border-radius:6px; text-align:center; }
    .result.fraud { background:#351e25; border:1px solid #773c46; color:#ffd6d2; }
    .result.legit { background:#14342f; border:1px solid #28695c; color:#c8f1e4; }
    .result strong { display:block; font-family:'Space Grotesk',sans-serif; font-size:1.4rem; }
    [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def page_heading(title: str, subtitle: str) -> None:
    st.markdown('<div class="eyebrow">Transaction Risk Operations</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtle">{subtitle}</div>', unsafe_allow_html=True)
    st.write("")


@st.cache_data(show_spinner=False)
def read_dataset(path_text: str, modified: float) -> pd.DataFrame:
    del modified
    return normalize_transactions(pd.read_csv(path_text, low_memory=False))


@st.cache_resource(show_spinner=False)
def read_bundle(path_text: str, modified: float) -> dict:
    import joblib

    del modified
    return joblib.load(path_text)


def ensure_project_data(rows: int = 20_000) -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = generate_transactions(rows=rows)
    data.to_csv(DATA_PATH, index=False)
    train_models(data, MODEL_PATH)
    read_dataset.clear()
    read_bundle.clear()


def setup_panel() -> None:
    st.markdown(
        '<div class="notice"><b>Setup needed.</b> This project has no transaction data or trained models yet. '
        'Create a reproducible synthetic sample to explore the full app. It is not real customer or bank data.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    if st.button("Create demo data and train models", type="primary", icon=":material/settings:"):
        with st.spinner("Generating synthetic transactions and fitting models…"):
            ensure_project_data()
        st.success("Demo dataset and models are ready.")
        st.rerun()


def data_models_page() -> None:
    page_heading("Data & Models", "Load a labeled transaction CSV or regenerate the reproducible demo sample.")
    if DATA_PATH.exists():
        current_data = read_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
        st.metric("Loaded transactions", f"{len(current_data):,}")
        if MODEL_PATH.exists():
            bundle = read_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
            st.caption(f"Active model data: {bundle.get('data_note', 'Source not recorded.')}")
    else:
        st.caption("No dataset is loaded yet. You can upload the standard dataset below or create a synthetic demo.")

    replace_active_data = st.checkbox("Replace the current CSV with synthetic demo data")
    if st.button("Regenerate synthetic demo", icon=":material/refresh:", disabled=not replace_active_data):
        with st.spinner("Generating the reproducible demo and retraining models…"):
            ensure_project_data()
        st.success("Synthetic demo data and models are ready.")
        st.rerun()

    st.markdown(
        "Upload either the standard `Time`, `Amount`, `V1`–`V28`, `Class` CSV or a transaction-record CSV "
        "with `TransactionDate`, `Amount`, `MerchantID`, `TransactionType`, `Location`, and `IsFraud`. "
        "The common 284,807-row dataset is available from the "
        "[MLG-ULB Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)."
    )
    uploaded_file = st.file_uploader("Transaction dataset CSV", type=["csv"], help="Supports the standard PCA schema or transaction-record schema shown above.")
    st.caption("The full file is used for training and the stratified holdout evaluation; larger datasets need more time and memory.")
    if st.button("Validate CSV and train all models", type="primary", icon=":material/upload_file:", disabled=uploaded_file is None):
        try:
            raw_transactions = pd.read_csv(uploaded_file, low_memory=False)
            transactions = normalize_transactions(raw_transactions)
            if transactions["Class"].value_counts().min() < 2:
                raise ValueError("The dataset needs at least two examples of each class for a stratified evaluation split.")
            temporary_model = MODEL_PATH.with_name("model_bundle.upload.tmp.joblib")
            with st.spinner(f"Training on {len(transactions):,} uploaded transactions. This may take a few minutes…"):
                bundle = train_models(
                    raw_transactions,
                    temporary_model,
                    data_note=f"Uploaded dataset: {uploaded_file.name} ({len(transactions):,} rows).",
                )
                DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                DATA_PATH.write_bytes(uploaded_file.getvalue())
                temporary_model.replace(MODEL_PATH)
            read_dataset.clear()
            read_bundle.clear()
            st.success(f"Loaded {len(transactions):,} transactions and trained four models.")
            st.rerun()
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError, OSError) as error:
            st.error(f"Could not load this dataset: {error}")


def chart_layout(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=12, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#12232c",
        font=dict(family="DM Sans, sans-serif", color="#d5e4e7"),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(color="#b9cbd0")),
        hoverlabel=dict(bgcolor="#20343e", font_color="#eff7f7"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#354b55", tickfont=dict(color="#a9bdc2"), title_font=dict(color="#c4d4d7"))
    fig.update_yaxes(gridcolor="#263a43", zeroline=False, tickfont=dict(color="#a9bdc2"), title_font=dict(color="#c4d4d7"))
    return fig


def sample_transactions_by_class(data: pd.DataFrame, examples_per_class: int = 5) -> pd.DataFrame:
    fraud_examples = data.loc[data["Class"] == 1, ["Time", "Amount", "Class"]].head(examples_per_class)
    legitimate_examples = data.loc[data["Class"] == 0, ["Time", "Amount", "Class"]].tail(examples_per_class)
    return pd.concat([fraud_examples, legitimate_examples]).sort_values("Class", ascending=False)


def dashboard(data: pd.DataFrame, bundle: dict) -> None:
    page_heading("Fraud Detection Overview", "Monitor transaction volume, model signals, and the current risk mix.")
    total = len(data)
    fraud_count = int(data["Class"].sum())
    legit_count = total - fraud_count
    fraud_rate = 100 * fraud_count / max(total, 1)
    best = bundle["best_model"]
    metrics = bundle["metrics"][best]
    columns = st.columns(4)
    columns[0].metric("Transactions", f"{total:,}", "Loaded dataset")
    columns[1].metric("Flagged fraud", f"{fraud_count:,}", f"{fraud_rate:.2f}% of sample")
    columns[2].metric("Legitimate", f"{legit_count:,}", f"{100 - fraud_rate:.2f}% of sample")
    columns[3].metric("Best model · PR-AUC", f"{metrics['pr_auc']:.3f}", best)

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Transaction amount")
        amount_view = data.assign(Label=data["Class"].map({0: "Legitimate", 1: "Fraud"}))
        fig = px.histogram(
            amount_view,
            x="Amount",
            color="Label",
            color_discrete_map={"Legitimate": "#43d6c4", "Fraud": "#ff796f"},
            nbins=45,
            barmode="overlay",
            opacity=0.78,
        )
        st.plotly_chart(chart_layout(fig), width="stretch")
    with right:
        st.subheader("Class balance")
        counts = pd.DataFrame({"Class": ["Legitimate", "Fraud"], "Transactions": [legit_count, fraud_count]})
        fig = px.bar(counts, x="Class", y="Transactions", color="Class", text_auto=".2s",
                     color_discrete_map={"Legitimate": "#43d6c4", "Fraud": "#ff796f"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_layout(fig), width="stretch")
    st.subheader("Sample transactions · both classes")
    sample = sample_transactions_by_class(data)
    st.dataframe(sample.rename(columns={"Class": "Class (0=legitimate, 1=fraud)"}),
                 width="stretch", hide_index=True)
    st.caption(bundle.get("data_note", "Dataset source not recorded."))


def prediction_page(bundle: dict) -> None:
    record_schema = bundle.get("input_schema") == "transaction_records"
    subtitle = (
        "Enter transaction date, amount, merchant, type, and location to score this transaction."
        if record_schema
        else "Score one transaction with the selected model. Features V1–V28 are anonymized numeric inputs."
    )
    page_heading("Predict a Transaction", subtitle)
    model_name = bundle["best_model"]
    st.caption(f"Active model: {model_name} · prediction threshold: 0.50")
    with st.form("transaction_form"):
        if record_schema:
            date_column, time_column, amount_column = st.columns(3)
            transaction_date = date_column.date_input("Transaction date", value=date(2024, 7, 1))
            transaction_time = time_column.time_input("Transaction time", value=time(12, 0))
            amount = amount_column.number_input("Transaction amount", min_value=0.0, value=120.50, step=1.0)
            merchant_column, type_column, location_column = st.columns(3)
            merchant_id = merchant_column.number_input("Merchant ID", min_value=0, value=100, step=1)
            transaction_type = type_column.text_input("Transaction type", value="purchase")
            location = location_column.text_input("Location", value="New York")
        else:
            base_left, base_right = st.columns(2)
            time_value = base_left.number_input("Time (seconds from reference)", min_value=0.0, value=45_000.0, step=60.0)
            amount = base_right.number_input("Transaction amount", min_value=0.0, value=120.50, step=1.0)
            st.markdown("**Anonymized transaction features**")
            feature_columns = st.columns(4)
            features = {}
            for index, column in enumerate(FEATURE_COLUMNS[2:]):
                features[column] = feature_columns[index % 4].number_input(column, value=0.0, step=0.1, key=f"predict_{column}")
        submitted = st.form_submit_button("Score transaction", type="primary", icon=":material/search:")

    if submitted:
        if record_schema:
            row = normalize_transactions(pd.DataFrame([{
                "TransactionDate": f"{transaction_date.isoformat()} {transaction_time.isoformat()}",
                "Amount": amount,
                "MerchantID": merchant_id,
                "TransactionType": transaction_type,
                "Location": location,
                "IsFraud": 0,
            }]))[FEATURE_COLUMNS]
        else:
            row = pd.DataFrame([{"Time": time_value, "Amount": amount, **features}])
        probability = float(bundle["models"][model_name].predict_proba(row)[0, 1])
        is_fraud = probability >= 0.5
        label = "Potential Fraud" if is_fraud else "Likely Legitimate"
        style = "fraud" if is_fraud else "legit"
        result, detail = st.columns([1, 2])
        with result:
            st.markdown(f'<div class="result {style}"><strong>{label}</strong><br>Fraud probability<br><b>{probability:.1%}</b></div>', unsafe_allow_html=True)
        with detail:
            st.progress(probability, text=f"Fraud probability: {probability:.1%}")
            st.warning("Model output is a demonstration signal, not a financial decision or guarantee.")


def comparison_page(bundle: dict) -> None:
    page_heading("Model Comparison", "Evaluation on a held-out test split. Precision-recall AUC is emphasized for the imbalanced target.")
    metrics = pd.DataFrame(bundle["metrics"]).T.reset_index(names="Model")
    metrics["Model"] = metrics["Model"].str.replace("_", " ")
    st.dataframe(metrics.style.format({column: "{:.3f}" for column in metrics.columns if column != "Model"}),
                 width="stretch", hide_index=True)
    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        st.subheader("Evaluation metrics")
        plot_data = metrics.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(plot_data, x="Metric", y="Score", color="Model", barmode="group",
                     color_discrete_sequence=["#43d6c4", "#ff796f", "#f0bc61", "#91b8ff"])
        st.plotly_chart(chart_layout(fig), width="stretch")
    with col_b:
        st.subheader(f"{bundle['best_model'].replace('_', ' ')} confusion matrix")
        matrix = bundle["confusion_matrices"][bundle["best_model"]]
        fig = px.imshow(matrix, text_auto=True, x=["Legit", "Fraud"],
                y=["Legit", "Fraud"], color_continuous_scale="Teal")
        st.plotly_chart(chart_layout(fig), width="stretch")
    importance = bundle.get("feature_importance", [])
    if importance:
        record_schema = bundle.get("input_schema") == "transaction_records"
        title = "Importance grouped by source CSV column" if record_schema else "Most influential features · selected model"
        st.subheader(title)
        if record_schema:
            st.caption("Engineered model signals are grouped back to the original CSV fields; importance is association, not causation.")
        important = pd.DataFrame(importance, columns=["Feature", "Importance"]).head(12).sort_values("Importance")
        fig = px.bar(important, x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#43d6c4"])
        st.plotly_chart(chart_layout(fig, 370), width="stretch")


def analytics_page(data: pd.DataFrame) -> None:
    page_heading("Transaction Analytics", "Explore transaction amounts, time, and fraud labels in the active dataset.")
    maximum = float(data["Amount"].quantile(0.99))
    selected = st.slider("Maximum amount shown (99th percentile cap)", min_value=0.0,
                         max_value=max(maximum, 1.0), value=max(maximum, 1.0))
    view = data[data["Amount"] <= selected].copy()
    view["Label"] = view["Class"].map({0: "Legitimate", 1: "Fraud"})
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Amount distribution")
        fig = px.box(view, x="Label", y="Amount", color="Label", points=False,
                     color_discrete_map={"Legitimate": "#43d6c4", "Fraud": "#ff796f"})
        st.plotly_chart(chart_layout(fig), width="stretch")
    with c2:
        st.subheader("Amount vs. time")
        fig = px.scatter(view.sample(min(2500, len(view)), random_state=7), x="Time", y="Amount", color="Label",
                         opacity=.55, color_discrete_map={"Legitimate": "#43d6c4", "Fraud": "#ff796f"})
        st.plotly_chart(chart_layout(fig), width="stretch")
    st.download_button("Download filtered transactions", view.drop(columns="Label").to_csv(index=False),
                       file_name="filtered_transactions.csv", mime="text/csv", icon=":material/download:")


def about_page(bundle: dict | None = None) -> None:
    page_heading("Project Notes", "A reproducible machine-learning demo for transaction risk exploration.")
    st.markdown(f"""
    The app starts with **synthetic** data, and the Data & Models page can load the standard labeled CSV. No real
    cardholder dataset is bundled with this project. Results from synthetic data are not real-world performance.

    **Pipeline**
    - Generate an imbalanced sample with `Time`, `Amount`, anonymized `V1`–`V28`, and `Class`.
    - Compare class-weighted Logistic Regression, calibrated linear SVM, Random Forest, and XGBoost on a stratified holdout split.
    - Compare precision, recall, F1, ROC-AUC, and PR-AUC; select the model with the strongest PR-AUC.
    - Score manually entered transactions and inspect sample-level analytics.

    **Current dataset:** {bundle.get("data_note", "Synthetic demo data is generated locally.") if bundle else "Synthetic demo data is generated locally."}

    **Important:** This is an educational prototype, not a payment authorization system. Synthetic features and labels
    do not reproduce real cardholder behavior, regulatory controls, or production decision thresholds.
    """)


def main() -> None:
    with st.sidebar:
        st.markdown("## ◈ Risk Desk")
        st.caption("CREDIT & ATM · FRAUD MONITOR")
        page = st.radio("Workspace", PAGES, label_visibility="collapsed")
        st.divider()
        st.caption("DEMO ENVIRONMENT")
        st.caption("Dataset source shown on dashboard")
        if DATA_PATH.exists():
            st.caption(f"{DATA_PATH.stat().st_size / (1024 * 1024):.1f} MB dataset")

    if page == "Data & Models":
        data_models_page()
        return

    if not DATA_PATH.exists() or not MODEL_PATH.exists():
        page_heading("Credit Card / ATM Fraud Detection", "Machine learning · transaction monitoring · model evaluation")
        setup_panel()
        if page == "About":
            about_page()
        return

    data = read_dataset(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    bundle = read_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
    if page == "Dashboard":
        dashboard(data, bundle)
    elif page == "Predict Transaction":
        prediction_page(bundle)
    elif page == "Model Comparison":
        comparison_page(bundle)
    elif page == "Analytics":
        analytics_page(data)
    else:
        about_page(bundle)


if __name__ == "__main__":
    main()