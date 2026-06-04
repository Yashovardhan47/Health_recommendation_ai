"""
HealthAI — Personalized Healthcare & Medicine Recommendation System
Main Streamlit Application (Fixed)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(__file__))

from utils.auth          import authenticate_user, register_user, has_permission
from utils.ui_components import (metric_card, section_title, render_medicine_grid,
                                  model_not_trained_warning, PLOTLY_LAYOUT, PALETTE)
from models.ml_models    import ModelManager, DataLoader

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HealthAI — Personalized Medicine",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
  html,body,[class*="css"]{ font-family:'DM Sans',sans-serif; }
  .stApp{ background:linear-gradient(135deg,#0a0f1e 0%,#0d1b2a 40%,#0a1628 100%); }
  [data-testid="stSidebar"]{ background:linear-gradient(180deg,#0d1f35 0%,#091525 100%)!important;
    border-right:1px solid rgba(0,200,255,.1); }
  .health-card{ background:rgba(13,31,53,.8);border:1px solid rgba(0,200,255,.15);
    border-radius:16px;padding:1.5rem;backdrop-filter:blur(10px);margin-bottom:1rem; }
  .metric-card{ background:linear-gradient(135deg,rgba(0,200,255,.08),rgba(0,100,255,.05));
    border:1px solid rgba(0,200,255,.2);border-radius:12px;padding:1.2rem;text-align:center; }
  .metric-val{ font-size:2.2rem;font-weight:700;background:linear-gradient(90deg,#00c8ff,#0064ff);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent; }
  .metric-label{ font-size:.8rem;color:rgba(255,255,255,.5);text-transform:uppercase;letter-spacing:.08em; }
  .badge-green{ background:#16a34a22;color:#4ade80;border:1px solid #16a34a44;border-radius:20px;padding:2px 10px;font-size:.75rem; }
  .badge-blue{  background:#1d4ed822;color:#60a5fa;border:1px solid #1d4ed844;border-radius:20px;padding:2px 10px;font-size:.75rem; }
  .badge-orange{background:#ea580c22;color:#fb923c;border:1px solid #ea580c44;border-radius:20px;padding:2px 10px;font-size:.75rem; }
  .badge-red{   background:#dc262622;color:#f87171;border:1px solid #dc262644;border-radius:20px;padding:2px 10px;font-size:.75rem; }
  h1,h2,h3{ color:#e2f0ff!important; }
  .section-title{ font-size:1.25rem;font-weight:700;color:#00c8ff;border-left:3px solid #00c8ff;
    padding-left:.75rem;margin:1.5rem 0 1rem; }
  .stTabs [data-baseweb="tab"]{ color:rgba(255,255,255,.5); }
  .stTabs [aria-selected="true"]{ color:#00c8ff!important;border-bottom-color:#00c8ff!important; }
  .stButton>button{ background:linear-gradient(135deg,#00c8ff,#0064ff)!important;color:white!important;
    border:none!important;border-radius:10px!important;font-weight:600!important;padding:.5rem 1.5rem!important; }
  .stButton>button:hover{ transform:translateY(-1px);box-shadow:0 4px 20px rgba(0,200,255,.3)!important; }
  .stTextInput input,.stSelectbox select,.stNumberInput input{
    background:rgba(13,31,53,.9)!important;border:1px solid rgba(0,200,255,.2)!important;
    color:white!important;border-radius:8px!important; }
  .logo-header{ text-align:center;padding:1rem 0 .3rem;font-size:1.5rem;font-weight:800;
    background:linear-gradient(90deg,#00c8ff,#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent; }
  .logo-sub{ text-align:center;font-size:.65rem;color:rgba(255,255,255,.3);letter-spacing:.15em;margin-bottom:1.5rem; }
  div[data-testid="stDataFrame"]{ border:1px solid rgba(0,200,255,.15);border-radius:8px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
def _init():
    for k, v in {"logged_in": False, "user": None, "model_manager": None,
                  "models_trained": False, "active_page": "dashboard"}.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# ══════════════════════════════════════════════════════════════════════════════
# AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════════
def auth_page():
    _, c2, _ = st.columns([1, 1.2, 1])
    with c2:
        st.markdown('<div class="logo-header">⚕️ HealthAI</div>', unsafe_allow_html=True)
        st.markdown('<div class="logo-sub">PERSONALIZED MEDICINE RECOMMENDATION SYSTEM</div>',
                    unsafe_allow_html=True)
        tab_l, tab_r = st.tabs(["🔐 Sign In", "📝 Register"])
        with tab_l:
            st.markdown('<div class="health-card">', unsafe_allow_html=True)
            st.markdown("**Demo credentials:**")
            st.markdown("| Username | Password | Role |\n|---|---|---|\n"
                        "| `admin` | `admin123` | Admin |\n"
                        "| `analyst` | `analyst123` | Analyst |\n"
                        "| `user1` | `user123` | User |\n"
                        "| `user2` | `user123` | User |")
            u = st.text_input("Username", key="li_u")
            p = st.text_input("Password", type="password", key="li_p")
            if st.button("Sign In →", use_container_width=True):
                ok, info = authenticate_user(u, p)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.user = info
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
            st.markdown('</div>', unsafe_allow_html=True)
        with tab_r:
            st.markdown('<div class="health-card">', unsafe_allow_html=True)
            ru = st.text_input("Username", key="reg_u")
            re = st.text_input("Email",    key="reg_e")
            rp = st.text_input("Password", type="password", key="reg_p")
            c1r, c2r = st.columns(2)
            ra = c1r.number_input("Age", 1, 120, 30)
            rg = c2r.selectbox("Gender", ["Male", "Female", "Other"])
            if st.button("Create Account", use_container_width=True):
                if ru and rp:
                    ok, result = register_user(ru, rp, re, ra, rg)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.user = result
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
def sidebar():
    user = st.session_state.user
    with st.sidebar:
        st.markdown('<div class="logo-header">⚕️ HealthAI</div>', unsafe_allow_html=True)
        st.markdown('<div class="logo-sub">RECOMMENDATION SYSTEM</div>', unsafe_allow_html=True)
        rc = {"Admin": "badge-red", "Analyst": "badge-orange", "User": "badge-blue"}.get(
            user["role"], "badge-blue")
        st.markdown(
            f'<div class="health-card" style="padding:.9rem">'
            f'<div style="font-weight:600;color:#e2f0ff">👤 {user["username"]}</div>'
            f'<div style="font-size:.75rem;color:rgba(255,255,255,.35)">ID: {user.get("user_id","")}</div>'
            f'<span class="{rc}">{user["role"]}</span></div>',
            unsafe_allow_html=True,
        )

        st.markdown("### Navigation")
        pages = [
            ("🏠",  "Dashboard",          "dashboard",       True),
            ("🔬",  "Disease Predictor",  "disease_predict", True),
            ("💊",  "Medicine Recommender","medicine_rec",   True),
            ("🧠",  "Advanced Engines",   "advanced_rec",    True),
            ("🤖",  "ML Models",          "ml_models",       has_permission(user["role"], "view_analytics")),
            ("📊",  "Analytics",          "analytics",       has_permission(user["role"], "view_analytics")),
            ("📈",  "Reports",            "reports",         has_permission(user["role"], "view_reports")),
            ("📁",  "Data Management",    "data_mgmt",       has_permission(user["role"], "view_dashboard")),
            ("👥",  "User Management",    "user_mgmt",       has_permission(user["role"], "manage_users")),
            ("👤",  "My Profile",         "profile",         True),
        ]
        for icon, label, pid, allowed in pages:
            if allowed:
                if st.sidebar.button(f"{icon} {label}", key=f"nav_{pid}",
                                     use_container_width=True):
                    st.session_state.active_page = pid
                    st.rerun()

        st.markdown("---")
        if st.button("⚡ Train All Models", use_container_width=True):
            with st.spinner("Loading data…"):
                mm = ModelManager()
            bar = st.progress(0)
            txt = st.empty()
            def cb(msg, pct):
                bar.progress(int(pct) / 100)
                txt.text(msg)
            metrics = mm.train_all(progress_callback=cb)
            st.session_state.model_manager  = mm
            st.session_state.models_trained = True
            st.success(
                f"✅ Done!  Acc: {metrics['disease_accuracy']:.2%}  "
                f"RMSE: {metrics['collab_rmse']:.4f}"
            )

        if st.session_state.models_trained:
            st.markdown('<span class="badge-green">● Models Active</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-red">● Models Not Trained</span>',
                        unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.logged_in   = False
            st.session_state.user        = None
            st.session_state.active_page = "dashboard"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    loader = DataLoader()
    st.markdown("## 🏠 Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        ("👥", len(loader.users),     "Users"),
        ("🦠", len(loader.diseases),  "Diseases"),
        ("💊", len(loader.medicines), "Medicines"),
        ("📋", len(loader.records),   "Records"),
        ("⭐", len(loader.reviews),   "Reviews"),
    ]
    for col, (icon, val, label) in zip([c1, c2, c3, c4, c5], kpis):
        col.markdown(
            f'<div class="metric-card">'
            f'<div style="font-size:1.6rem">{icon}</div>'
            f'<div class="metric-val">{val:,}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    l, r = st.columns(2)
    with l:
        section_title("Disease Distribution", "🦠")
        dc = loader.records["diagnosis"].value_counts().reset_index()
        dc.columns = ["disease_id", "count"]
        dn = loader.diseases.set_index("disease_id")["name"].to_dict()
        dc["name"] = dc["disease_id"].map(dn).fillna(dc["disease_id"])
        fig = px.bar(dc, x="count", y="name", orientation="h", color="count",
                     color_continuous_scale=["#0064ff","#00c8ff"], template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=320, margin=dict(l=0,r=0,t=10,b=0),
                          coloraxis_showscale=False, yaxis_title="", xaxis_title="Records")
        st.plotly_chart(fig, use_container_width=True)
    with r:
        section_title("Medicine Categories", "💊")
        cats = loader.medicines["category"].value_counts()
        fig2 = px.pie(values=cats.values, names=cats.index,
                      color_discrete_sequence=px.colors.sequential.Blues_r,
                      template="plotly_dark", hole=.45)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=320,
                           margin=dict(l=0,r=0,t=10,b=0),
                           legend=dict(font=dict(color="white")))
        st.plotly_chart(fig2, use_container_width=True)

    section_title("User Activity Timeline", "📅")
    act = loader.activity.copy()
    act["date"] = pd.to_datetime(act["timestamp"]).dt.date
    daily = act.groupby(["date", "action"]).size().reset_index(name="count")
    fig3 = px.line(daily.sort_values("date"), x="date", y="count", color="action",
                   template="plotly_dark",
                   color_discrete_sequence=px.colors.qualitative.Set2)
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       height=260, margin=dict(l=0,r=0,t=10,b=30),
                       xaxis_title="", yaxis_title="Actions",
                       legend=dict(font=dict(color="white", size=10)))
    st.plotly_chart(fig3, use_container_width=True)

    # ── Trending Medicines (Last 30 Days) ─────────────────────────────────────
    section_title("🔥 Trending Medicines (Last 30 Days)", "📈")
    try:
        act2 = loader.activity.copy()
        act2["date"] = pd.to_datetime(act2["timestamp"])
        recent = act2[act2["date"] > act2["date"].max() - pd.Timedelta(days=30)]
        view_act = recent[recent["action"] == "view_medicine"]
        if not view_act.empty:
            trend = (view_act["reference_id"]
                     .value_counts().head(8).reset_index())
            trend.columns = ["med_id", "views"]
            med_name_map = loader.medicines.set_index("med_id")["name"].to_dict()
            trend["medicine"] = trend["med_id"].map(med_name_map).fillna(trend["med_id"])
            fig_t = px.bar(trend, x="views", y="medicine", orientation="h",
                           color="views", color_continuous_scale=["#0064ff","#00c8ff"],
                           template="plotly_dark",
                           title="Most Viewed Medicines (Rolling 30 Days)")
            fig_t.update_layout(**PLOTLY_LAYOUT, height=280,
                                coloraxis_showscale=False,
                                yaxis_title="", xaxis_title="Views")
            st.plotly_chart(fig_t, use_container_width=True)
        else:
            st.info("No medicine view data available yet.")
    except Exception as e:
        st.warning(f"Could not render trending chart: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DISEASE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
def page_disease_predict():
    st.markdown("## 🔬 Disease Predictor")
    if not st.session_state.models_trained:
        model_not_trained_warning(); return
    mm = st.session_state.model_manager
    loader = mm.loader

    st.info("Enter patient vitals and lab results to get AI-powered disease predictions.")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="health-card">', unsafe_allow_html=True)
        st.markdown("#### 🧑‍⚕️ Patient Vitals")
        age      = st.slider("Age", 5, 90, 45)
        bp       = st.slider("Blood Pressure Systolic (mmHg)", 80, 200, 130)
        glucose  = st.slider("Glucose Level (mg/dL)", 60, 350, 100)
        hr       = st.slider("Heart Rate (bpm)", 50, 150, 80)
        bmi      = st.slider("BMI", 15.0, 50.0, 25.0, 0.1)
        chol     = st.slider("Cholesterol (mg/dL)", 100, 350, 190)
        smoking  = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x else "No")
        exercise = st.slider("Exercise Days/Week", 0, 7, 3)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="health-card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Top-3 Predicted Diagnoses")
        features = {
            "age": age, "blood_pressure_systolic": bp,
            "glucose_level": glucose, "heart_rate": hr,
            "bmi": bmi, "cholesterol": chol,
            "smoking": smoking, "exercise_frequency": exercise,
        }
        preds = mm.disease_model.predict(features)
        dmap  = loader.diseases.set_index("disease_id")
        for i, pred in enumerate(preds):
            did  = pred["disease_id"]
            prob = pred["probability"]
            dname = dmap.loc[did, "name"] if did in dmap.index else did
            dcat  = dmap.loc[did, "category"] if did in dmap.index else ""
            col   = ["#00c8ff", "#60a5fa", "#1d4ed8"][i]
            st.markdown(f"""
            <div style="margin-bottom:1rem">
              <div style="display:flex;justify-content:space-between">
                <span style="font-weight:700;color:#e2f0ff">#{i+1} {dname}</span>
                <span style="color:{col};font-weight:700">{prob:.1%}</span>
              </div>
              <div style="font-size:.75rem;color:rgba(255,255,255,.4)">{dcat}</div>
              <div style="background:rgba(255,255,255,.1);border-radius:4px;height:8px;margin-top:.4rem">
                <div style="width:{int(prob*100)}%;background:{col};height:100%;border-radius:4px"></div>
              </div>
            </div>""", unsafe_allow_html=True)

        fi_df = pd.DataFrame(
            list(mm.disease_model.feature_importance().items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance")
        fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale=["#0064ff","#00c8ff"],
                     template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=220, margin=dict(l=0,r=0,t=0,b=0),
                          coloraxis_showscale=False, yaxis_title="", xaxis_title="",
                          title="Feature Importances")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if preds:
        top_d = preds[0]["disease_id"]
        dname = dmap.loc[top_d, "name"] if top_d in dmap.index else top_d
        section_title(f"Recommended Medicines for {dname}", "💊")
        uid   = st.session_state.user.get("user_id", "U0001")
        alpha = st.slider("Hybrid Weight α (content ← → collaborative)",
                           0.0, 1.0, 0.5, 0.05, key="dp_alpha")
        recs  = mm.hybrid.recommend(uid, top_d, loader.medicines, alpha=alpha, n=6)
        render_medicine_grid(recs, loader.interactions, cols=3)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MEDICINE RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
def page_medicine_rec():
    st.markdown("## 💊 Medicine Recommendation Engine")
    if not st.session_state.models_trained:
        model_not_trained_warning(); return
    mm     = st.session_state.model_manager
    loader = mm.loader
    uid    = st.session_state.user.get("user_id", "U0001")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🧬 Content-Based", "👥 Collaborative", "🔀 Hybrid", "🤖 Deep Learning"])
    med_names     = loader.medicines.set_index("med_id")["name"].to_dict()
    disease_names = loader.diseases.set_index("disease_id")["name"].to_dict()

    with tab1:
        st.markdown("**TF-IDF + Cosine Similarity** — finds medicines with similar descriptions and disease associations.")
        sel_med = st.selectbox("Select medicine:", list(med_names.keys()),
                               format_func=lambda x: med_names.get(x, x), key="cb_med")
        n_cb  = st.slider("Top-N", 3, 10, 5, key="cb_n")
        recs  = mm.content_cb.recommend(sel_med, n=n_cb)
        section_title(f"Medicines Similar to {med_names.get(sel_med, sel_med)}", "🧬")
        render_medicine_grid(recs, loader.interactions)

    with tab2:
        st.markdown("**SVD Matrix Factorisation** — learns latent patterns from user-medicine interactions.")
        n_cf = st.slider("Top-N", 3, 10, 5, key="cf_n")
        recs = mm.collab_cf.recommend(uid, loader.medicines, n=n_cf)
        section_title(f"Collaborative Picks for {uid}", "👥")
        render_medicine_grid(recs, loader.interactions)
        st.caption(f"Model CV RMSE: {mm.collab_cf.rmse:.4f}")

    with tab3:
        st.markdown("**Hybrid (α · Content + (1-α) · Collaborative)** — combines both signals.")
        c1, c2 = st.columns(2)
        sel_dis = c1.selectbox("Disease context:", list(disease_names.keys()),
                               format_func=lambda x: disease_names.get(x, x), key="hyb_d")
        alpha   = c2.slider("α weight", 0.0, 1.0, 0.5, 0.05, key="hyb_a")
        n_hyb   = st.slider("Top-N", 3, 10, 5, key="hyb_n")
        recs    = mm.hybrid.recommend(uid, sel_dis, loader.medicines, alpha=alpha, n=n_hyb)
        section_title(f"Hybrid Recommendations for {disease_names.get(sel_dis, sel_dis)}", "🔀")
        render_medicine_grid(recs, loader.interactions)

    with tab4:
        st.markdown("**Neural Collaborative Filtering** — dual embedding network with BatchNorm + Dropout.")
        n_dl = st.slider("Top-N", 3, 10, 5, key="dl_n")
        recs = mm.deep_model.recommend(uid, loader.medicines, n=n_dl)
        section_title("Deep Learning Recommendations", "🤖")
        render_medicine_grid(recs, loader.interactions)
        losses, val_losses = mm.deep_model.training_loss()
        if losses:
            section_title("Training Loss Curve", "📉")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=losses, name="Train Loss",
                                     line=dict(color="#00c8ff", width=2)))
            fig.add_trace(go.Scatter(y=val_losses, name="Val Loss",
                                     line=dict(color="#fb923c", width=2, dash="dash")))
            fig.update_layout(**PLOTLY_LAYOUT, height=220,
                              xaxis_title="Epoch", yaxis_title="MSE")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
def page_ml_models():
    st.markdown("## 🤖 ML Model Architecture & Metrics")
    if not st.session_state.models_trained:
        model_not_trained_warning(); return
    mm     = st.session_state.model_manager
    loader = mm.loader

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🌲 Disease RF", "🔢 TF-IDF", "🔄 SVD", "🧠 Neural CF", "🔑 Feature Analysis"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="metric-val">{mm.disease_model.accuracy:.2%}</div>'
                    f'<div class="metric-label">Accuracy</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-val">RF</div>'
                    f'<div class="metric-label">Algorithm</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-val">150</div>'
                    f'<div class="metric-label">Estimators</div></div>', unsafe_allow_html=True)
        st.markdown("")
        imp   = mm.disease_model.feature_importance()
        fi_df = pd.DataFrame(list(imp.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)
        fig   = px.bar(fi_df, x="Feature", y="Importance", color="Importance",
                       color_continuous_scale=["#0064ff","#00c8ff"],
                       template="plotly_dark", title="Feature Importances")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=300, coloraxis_showscale=False,
                          margin=dict(t=40,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)
        if mm.disease_model.report:
            section_title("Classification Report", "📋")
            rep_df = pd.DataFrame(mm.disease_model.report).transpose().round(3)
            dn = loader.diseases.set_index("disease_id")["name"].to_dict()
            rep_df.index = [dn.get(i, i) for i in rep_df.index]
            st.dataframe(rep_df, use_container_width=True)

    with tab2:
        vocab_size = len(mm.content_cb.vectorizer.vocabulary_) if mm.content_cb.item_matrix is not None else 0
        st.markdown(f"**Vocabulary:** {vocab_size:,} terms · **N-gram:** (1,2) · "
                    f"**Matrix shape:** {mm.content_cb.item_matrix.shape if mm.content_cb.item_matrix is not None else 'N/A'}")
        if mm.content_cb.sim_matrix is not None:
            n_show      = min(12, len(loader.medicines))
            med_names_list = loader.medicines["name"].tolist()[:n_show]
            sim_sub     = mm.content_cb.sim_matrix[:n_show, :n_show]
            fig = px.imshow(sim_sub, x=med_names_list, y=med_names_list,
                            color_continuous_scale="Blues", template="plotly_dark",
                            title="Medicine Similarity Matrix")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=420,
                              margin=dict(t=40,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        n_u = mm.collab_cf.interaction_df["user_id"].nunique() if not mm.collab_cf.interaction_df.empty else 0
        n_m = mm.collab_cf.interaction_df["med_id"].nunique()  if not mm.collab_cf.interaction_df.empty else 0
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-card"><div class="metric-val">{mm.collab_cf.rmse:.4f}</div>'
                    f'<div class="metric-label">RMSE</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-val">{n_u}</div>'
                    f'<div class="metric-label">Users</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-val">{n_m}</div>'
                    f'<div class="metric-label">Medicines</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-val">50</div>'
                    f'<div class="metric-label">Factors</div></div>', unsafe_allow_html=True)
        if not mm.collab_cf.interaction_df.empty:
            fig = px.histogram(mm.collab_cf.interaction_df["rating"], nbins=20,
                               template="plotly_dark", color_discrete_sequence=["#00c8ff"],
                               title="Rating Distribution")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=260, margin=dict(t=40,b=0,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.code("""
Input(user_id) ──► Embedding(16) ──► Flatten ─┐
                                                ├──► Concat ──► Dense(128,relu) ──► BN ──► Dropout(0.3)
Input(med_id)  ──► Embedding(16) ──► Flatten ─┘        ──► Dense(64,relu) ──► Dropout(0.2)
                                                         ──► Dense(32,relu) ──► Dense(1,sigmoid)
Optimizer: Adam | Loss: MSE | Metric: MAE""", language="text")
        losses, val_losses = mm.deep_model.training_loss()
        if losses:
            epochs = list(range(1, len(losses)+1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=epochs, y=losses, name="Train Loss",
                                     line=dict(color="#00c8ff", width=3)))
            fig.add_trace(go.Scatter(x=epochs, y=val_losses, name="Val Loss",
                                     line=dict(color="#fb923c", width=3, dash="dash")))
            fig.update_layout(**PLOTLY_LAYOUT, height=280,
                              xaxis_title="Epoch", yaxis_title="MSE Loss")
            st.plotly_chart(fig, use_container_width=True)

    with tab5:
        section_title("Feature Correlation Matrix", "🔑")
        numeric_cols = ["age","blood_pressure_systolic","glucose_level",
                        "heart_rate","bmi","cholesterol"]
        corr = loader.records[numeric_cols].corr()
        fig  = px.imshow(corr, color_continuous_scale="RdBu_r", template="plotly_dark",
                         title="Feature Correlation Matrix", zmin=-1, zmax=1)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=400,
                          margin=dict(t=40,b=0,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown("## 📊 Analytics & Insights")
    loader = DataLoader()

    tab1, tab2, tab3 = st.tabs(
        ["👥 User Analytics", "💊 Medicine Analytics", "😊 Sentiment Analysis"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            section_title("Age Distribution", "👶")
            fig = px.histogram(loader.users, x="age", nbins=20,
                               template="plotly_dark", color_discrete_sequence=["#00c8ff"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=260, margin=dict(t=10,b=30,l=0,r=0))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            section_title("Gender & Role", "🔢")
            gr = loader.users.groupby(["gender","role"]).size().reset_index(name="count")
            fig = px.bar(gr, x="gender", y="count", color="role",
                         template="plotly_dark", barmode="group",
                         color_discrete_sequence=["#00c8ff","#0064ff","#60a5fa"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=260, margin=dict(t=10,b=30,l=0,r=0),
                              legend=dict(font=dict(color="white")))
            st.plotly_chart(fig, use_container_width=True)

        section_title("Top 15 Most Active Users", "🏆")
        top_u = (loader.activity.groupby("user_id").size()
                 .reset_index(name="actions")
                 .sort_values("actions", ascending=False).head(15))
        fig = px.bar(top_u, x="user_id", y="actions", color="actions",
                     color_continuous_scale=["#0064ff","#00c8ff"], template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=250, margin=dict(t=10,b=30,l=0,r=0),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        mn = loader.medicines.set_index("med_id")["name"].to_dict()
        c1, c2 = st.columns(2)
        with c1:
            section_title("Top Rated Medicines", "⭐")
            mr = loader.interactions.groupby("med_id")["rating"].mean().reset_index()
            mr["name"] = mr["med_id"].map(mn)
            mr = mr.sort_values("rating", ascending=False).head(10)
            fig = px.bar(mr, x="rating", y="name", orientation="h", color="rating",
                         color_continuous_scale=["#0064ff","#00c8ff"], template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=300, margin=dict(t=10,b=0,l=0,r=0),
                              coloraxis_showscale=False, yaxis_title="", xaxis_title="Avg Rating")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            section_title("Interaction Distribution", "📊")
            mc = loader.interactions["med_id"].value_counts().head(10).reset_index()
            mc.columns = ["med_id", "interactions"]
            mc["name"] = mc["med_id"].map(mn)
            fig = px.pie(mc, values="interactions", names="name", hole=.45,
                         template="plotly_dark",
                         color_discrete_sequence=px.colors.sequential.Blues_r)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300,
                              margin=dict(t=0,b=0,l=0,r=0),
                              legend=dict(font=dict(color="white", size=10)))
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if not st.session_state.models_trained:
            model_not_trained_warning(); return
        mm    = st.session_state.model_manager
        mn2   = mm.loader.medicines.set_index("med_id")["name"].to_dict()
        section_title("Per-Medicine Sentiment Scores", "😊")
        sentiments = []
        for mid in mm.loader.medicines["med_id"]:
            r = mm.sentiment.medicine_sentiment_summary(mm.loader.reviews, mid)
            if r:
                sentiments.append({
                    "name": mn2.get(mid, mid),
                    "avg_sentiment": r["avg_sentiment"],
                    "n_reviews": r["n_reviews"],
                })
        if sentiments:
            sent_df = pd.DataFrame(sentiments).sort_values("avg_sentiment", ascending=False)
            fig = px.bar(sent_df, x="name", y="avg_sentiment", color="avg_sentiment",
                         color_continuous_scale=["#dc2626","#16a34a"], template="plotly_dark",
                         title="Sentiment Score per Medicine")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              height=320, coloraxis_showscale=False,
                              margin=dict(t=40,b=60,l=0,r=0),
                              xaxis_tickangle=-35, xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        section_title("🧪 Live Sentiment Tester", "")
        txt = st.text_area("Enter a patient review:",
                            "This medication worked extremely well. My symptoms improved significantly within a week.")
        if st.button("Analyze"):
            res = mm.sentiment.analyze(txt)
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:1.3rem">'
                        f'{res["label"]}</div><div class="metric-label">Sentiment</div></div>',
                        unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="metric-val">{res["compound"]:.2f}</div>'
                        f'<div class="metric-label">Compound</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="metric-val">{res["pos"]:.2f}</div>'
                        f'<div class="metric-label">Positive</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric-card"><div class="metric-val">{res["neg"]:.2f}</div>'
                        f'<div class="metric-label">Negative</div></div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def page_data_mgmt():
    st.markdown("## 📁 Data Management")
    loader = DataLoader()
    tab1, tab2, tab3 = st.tabs(["📊 View Datasets", "📤 Upload Data", "🔗 Public Datasets"])

    with tab1:
        choice = st.selectbox("Dataset:", ["Users","Diseases","Medicines",
                                            "Medical Records","Interactions","Reviews","Activity Log"])
        ds_map = {
            "Users":          loader.users.drop(columns=["password_hash"], errors="ignore"),
            "Diseases":       loader.diseases,
            "Medicines":      loader.medicines,
            "Medical Records":loader.records,
            "Interactions":   loader.interactions,
            "Reviews":        loader.reviews,
            "Activity Log":   loader.activity,
        }
        df = ds_map[choice]
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows",    f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Memory",  f"{df.memory_usage(deep=True).sum()/1024:.1f} KB")
        st.dataframe(df.head(50), use_container_width=True)
        st.download_button(f"⬇️ Download CSV",
                           df.to_csv(index=False).encode(),
                           f"{choice.lower().replace(' ','_')}.csv", "text/csv")

    with tab2:
        st.markdown("### Upload Custom Datasets")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Medical Records** (disease prediction):")
            st.code("age, blood_pressure_systolic, glucose_level,\nheart_rate, bmi, cholesterol, smoking,\nexercise_frequency, diagnosis")
            up = st.file_uploader("Upload medical_records.csv", type="csv", key="up_rec")
            if up:
                df_up = pd.read_csv(up)
                data_path = os.path.join(os.path.dirname(__file__), "data", "medical_records.csv")
                df_up.to_csv(data_path, index=False)
                st.success(f"✅ Saved {len(df_up)} rows. Retrain to apply.")
        with c2:
            st.markdown("**Interactions** (collaborative filtering):")
            st.code("user_id, med_id, rating (1-5)")
            up2 = st.file_uploader("Upload interactions.csv", type="csv", key="up_int")
            if up2:
                df_up2 = pd.read_csv(up2)
                data_path2 = os.path.join(os.path.dirname(__file__), "data", "interactions.csv")
                df_up2.to_csv(data_path2, index=False)
                st.success(f"✅ Saved {len(df_up2)} rows. Retrain to apply.")

    with tab3:
        st.markdown("### Recommended Public Datasets")
        for name, url, desc in [
            ("🏥 Heart Disease UCI",     "https://archive.ics.uci.edu/ml/datasets/Heart+Disease", "303 records, 14 features"),
            ("🩺 Pima Indians Diabetes", "https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database", "768 records"),
            ("💊 Drug200",               "https://www.kaggle.com/datasets/prathamtripathi/drug-classification", "200 patient records"),
            ("🤒 Disease Symptom",       "https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset", "41 diseases, 120+ symptoms"),
        ]:
            st.markdown(
                f'<div class="health-card" style="padding:.8rem 1.2rem">'
                f'<a href="{url}" target="_blank" style="color:#00c8ff;font-weight:600">{name} ↗</a>'
                f'<div style="color:rgba(255,255,255,.5);font-size:.82rem;margin-top:.3rem">{desc}</div>'
                f'</div>', unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: USER MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
def page_user_mgmt():
    st.markdown("## 👥 User Management")
    loader = DataLoader()
    roles  = loader.users["role"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-val">{len(loader.users)}</div>'
                f'<div class="metric-label">Total Users</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-val">{roles.get("Admin",0)}</div>'
                f'<div class="metric-label">Admins</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-val">{roles.get("User",0)}</div>'
                f'<div class="metric-label">Regular Users</div></div>', unsafe_allow_html=True)
    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        section_title("Role Distribution", "🔢")
        fig = px.pie(values=roles.values, names=roles.index,
                     template="plotly_dark",
                     color_discrete_sequence=["#00c8ff","#0064ff","#60a5fa"], hole=.5)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=260,
                          margin=dict(t=0,b=0,l=0,r=0),
                          legend=dict(font=dict(color="white")))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        section_title("Chronic Conditions Breakdown", "🏥")
        conds = loader.users["chronic_conditions"].value_counts()
        fig2  = px.bar(x=conds.values, y=conds.index, orientation="h",
                       template="plotly_dark", color=conds.values,
                       color_continuous_scale=["#0064ff","#00c8ff"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           height=260, margin=dict(t=0,b=0,l=0,r=0),
                           coloraxis_showscale=False, yaxis_title="", xaxis_title="Count")
        st.plotly_chart(fig2, use_container_width=True)

    section_title("User Table", "📋")
    search  = st.text_input("🔍 Search:", placeholder="username, blood type, condition…")
    df_show = loader.users.drop(columns=["password_hash"], errors="ignore")
    if search:
        mask = df_show.astype(str).apply(lambda c: c.str.contains(search, case=False)).any(axis=1)
        df_show = df_show[mask]
    st.dataframe(df_show, use_container_width=True, height=300)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if not st.session_state.logged_in:
        auth_page(); return
    sidebar()
    mm   = st.session_state.model_manager
    user = st.session_state.user
    page = st.session_state.active_page

    if   page == "dashboard":       page_dashboard()
    elif page == "disease_predict": page_disease_predict()
    elif page == "medicine_rec":    page_medicine_rec()
    elif page == "advanced_rec":
        from pages.advanced_recommenders import render as adv_render
        adv_render(mm, user)
    elif page == "ml_models":       page_ml_models()
    elif page == "analytics":       page_analytics()
    elif page == "reports":
        from pages.reports import render as rep_render
        rep_render(mm, user)
    elif page == "data_mgmt":       page_data_mgmt()
    elif page == "user_mgmt":       page_user_mgmt()
    elif page == "profile":
        from pages.profile import render as prof_render
        prof_render(mm, user)


if __name__ == "__main__":
    main()
