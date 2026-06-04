"""utils/ui_components.py — Shared Streamlit UI helpers"""
import streamlit as st
import pandas as pd

PALETTE = ["#00c8ff","#0064ff","#60a5fa","#4ade80","#fb923c","#f87171"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", size=12),
    legend=dict(font=dict(color="white", size=10)),
    margin=dict(l=0, r=0, t=40, b=0),
)

def metric_card(value, label, icon="", color="#00c8ff"):
    st.markdown(
        f'<div class="metric-card">'
        f'<div style="font-size:1.6rem">{icon}</div>'
        f'<div class="metric-val" style="color:{color}">{value}</div>'
        f'<div class="metric-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

def section_title(title: str, icon: str = ""):
    label = f"{icon} {title}" if icon else title
    st.markdown(f'<div class="section-title">{label}</div>', unsafe_allow_html=True)

def model_not_trained_warning():
    st.warning("⚠️ Models not trained yet. Click **⚡ Train All Models** in the sidebar first.")

def render_medicine_grid(recs: list, interactions: pd.DataFrame, cols: int = 3):
    if not recs:
        st.info("No recommendations available.")
        return
    avg_ratings = {}
    if not interactions.empty and "med_id" in interactions.columns:
        avg_ratings = interactions.groupby("med_id")["rating"].mean().to_dict()

    for row_start in range(0, len(recs), cols):
        row_recs = recs[row_start: row_start + cols]
        columns = st.columns(cols)
        for col, rec in zip(columns, row_recs):
            with col:
                mid  = rec.get("med_id", "")
                name = rec.get("name", mid)
                cat  = rec.get("category", "")
                dos  = rec.get("dosage", "")
                eff  = rec.get("effectiveness", 0)
                price= rec.get("price_inr", 0)
                score= rec.get("score", eff)
                rating = avg_ratings.get(mid, 0)
                stars = "⭐" * round(rating) if rating else ""
                badge_cls = {"Antibiotic":"badge-orange","SSRI":"badge-blue",
                             "Antihistamine":"badge-green"}.get(cat, "badge-blue")
                st.markdown(
                    f'<div class="health-card">'
                    f'<div style="font-weight:700;color:#e2f0ff;font-size:.95rem">{name}</div>'
                    f'<span class="{badge_cls}" style="font-size:.7rem">{cat}</span>'
                    f'<div style="color:rgba(255,255,255,.5);font-size:.75rem;margin:.4rem 0">{dos}</div>'
                    f'<div style="display:flex;justify-content:space-between;margin-top:.6rem">'
                    f'<span style="color:#4ade80;font-size:.8rem">Score: {score:.2f}</span>'
                    f'<span style="color:#fb923c;font-size:.8rem">₹{price}</span></div>'
                    f'<div style="color:rgba(255,255,255,.4);font-size:.75rem">{stars} {f"({rating:.1f})" if rating else ""}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
