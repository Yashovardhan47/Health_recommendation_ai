"""pages/reports.py — Performance reports & data export"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from utils.ui_components import section_title, PLOTLY_LAYOUT, PALETTE


def render(mm, user):
    st.markdown("## 📈 Reports & Performance Analysis")
    loader = mm.loader

    tab1, tab2, tab3 = st.tabs(["📊 Algorithm Comparison","📈 Engagement Metrics","📤 Export Data"])

    with tab1:
        section_title("Model Performance Comparison", "🤖")
        trained = st.session_state.get("models_trained", False)
        metrics = []
        if trained:
            metrics = [
                {"Model": "Random Forest (Disease)",        "Accuracy": mm.disease_model.accuracy,   "RMSE": None},
                {"Model": "SVD Collaborative Filter",       "Accuracy": None, "RMSE": mm.collab_cf.rmse},
                {"Model": "Content-Based (TF-IDF)",         "Accuracy": None, "RMSE": None},
                {"Model": "Hybrid Recommender",             "Accuracy": None, "RMSE": None},
                {"Model": "Neural Collaborative Filter",    "Accuracy": None, "RMSE": None},
            ]
        else:
            st.info("Train models first to see performance metrics.")
            metrics = [
                {"Model": m, "Accuracy": None, "RMSE": None}
                for m in ["Random Forest","SVD CF","TF-IDF","Hybrid","Neural CF"]
            ]

        df_m = pd.DataFrame(metrics)
        st.dataframe(df_m, use_container_width=True)

        if trained and mm.disease_model.accuracy > 0:
            acc_val = mm.disease_model.accuracy
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="metric-card"><div class="metric-val">{acc_val:.1%}</div>'
                        f'<div class="metric-label">Disease Accuracy</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="metric-val">{mm.collab_cf.rmse:.4f}</div>'
                        f'<div class="metric-label">Collab RMSE</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="metric-val">5</div>'
                        f'<div class="metric-label">Models Active</div></div>', unsafe_allow_html=True)

    with tab2:
        section_title("User Engagement Over Time", "📈")
        act = loader.activity.copy()
        if not act.empty:
            act["date"] = pd.to_datetime(act["timestamp"]).dt.date
            daily = act.groupby(["date","action"]).size().reset_index(name="count")
            fig = px.area(daily.sort_values("date"), x="date", y="count", color="action",
                          template="plotly_dark", color_discrete_sequence=PALETTE)
            fig.update_layout(**PLOTLY_LAYOUT, height=300)
            st.plotly_chart(fig, use_container_width=True)

            section_title("Action Type Breakdown", "🔢")
            action_cnt = act["action"].value_counts().reset_index()
            action_cnt.columns = ["Action","Count"]
            fig2 = px.bar(action_cnt, x="Action", y="Count", color="Count",
                          color_continuous_scale=["#0064ff","#00c8ff"],
                          template="plotly_dark")
            fig2.update_layout(**PLOTLY_LAYOUT, height=280, coloraxis_showscale=False)
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        section_title("Export Datasets", "📤")
        dataset_map = {
            "Users":          loader.users.drop(columns=["password_hash"], errors="ignore"),
            "Diseases":       loader.diseases,
            "Medicines":      loader.medicines,
            "Medical Records":loader.records,
            "Interactions":   loader.interactions,
            "Activity Log":   loader.activity,
            "Reviews":        loader.reviews,
        }
        choice = st.selectbox("Select dataset:", list(dataset_map.keys()))
        df_exp = dataset_map[choice]
        fmt    = st.radio("Format:", ["CSV","JSON"], horizontal=True)
        if fmt == "CSV":
            data = df_exp.to_csv(index=False).encode()
            mime = "text/csv"
            fname= f"{choice.lower().replace(' ','_')}.csv"
        else:
            data = df_exp.to_json(orient="records", indent=2).encode()
            mime = "application/json"
            fname= f"{choice.lower().replace(' ','_')}.json"
        st.download_button(f"⬇️ Download {choice} ({fmt})", data, fname, mime)
        st.dataframe(df_exp.head(20), use_container_width=True)
