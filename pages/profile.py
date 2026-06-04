"""pages/profile.py — User profile & preference editor"""
import streamlit as st
import pandas as pd
from pathlib import Path
from utils.ui_components import section_title

DATA_DIR = Path(__file__).parent.parent / "data"


def render(mm, user):
    st.markdown("## 👤 My Profile")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(
            f'<div class="health-card" style="text-align:center">'
            f'<div style="font-size:4rem">👤</div>'
            f'<div style="font-size:1.3rem;font-weight:700;color:#e2f0ff">{user.get("username","")}</div>'
            f'<div style="font-size:.8rem;color:rgba(255,255,255,.4)">{user.get("email","")}</div>'
            f'<div style="margin-top:.5rem"><span class="badge-blue">{user.get("role","User")}</span></div>'
            f'</div>', unsafe_allow_html=True,
        )

    with c2:
        section_title("Personal Information", "ℹ️")
        c_a, c_b = st.columns(2)
        age    = c_a.number_input("Age",       1, 120, int(user.get("age", 30) or 30))
        gender = c_b.selectbox("Gender",       ["Male","Female","Other"],
                               index=["Male","Female","Other"].index(user.get("gender","Male"))
                               if user.get("gender") in ["Male","Female","Other"] else 0)
        blood  = st.selectbox("Blood Type",    ["A+","A-","B+","B-","O+","O-","AB+","AB-"],
                              index=["A+","A-","B+","B-","O+","O-","AB+","AB-"].index(user.get("blood_type","O+"))
                              if user.get("blood_type") in ["A+","A-","B+","B-","O+","O-","AB+","AB-"] else 4)
        conds  = st.text_input("Chronic Conditions", value=str(user.get("chronic_conditions","None") or "None"))
        allerg = st.text_input("Allergies",           value=str(user.get("allergies","None") or "None"))

        if st.button("💾 Save Profile", use_container_width=True):
            try:
                p = DATA_DIR / "users.csv"
                users = pd.read_csv(p)
                idx = users[users["user_id"] == user["user_id"]].index
                if not idx.empty:
                    users.loc[idx, "age"]               = age
                    users.loc[idx, "gender"]            = gender
                    users.loc[idx, "blood_type"]        = blood
                    users.loc[idx, "chronic_conditions"]= conds
                    users.loc[idx, "allergies"]         = allerg
                    users.to_csv(p, index=False)
                    st.session_state.user["age"]               = age
                    st.session_state.user["gender"]            = gender
                    st.session_state.user["blood_type"]        = blood
                    st.session_state.user["chronic_conditions"]= conds
                    st.session_state.user["allergies"]         = allerg
                    st.success("✅ Profile updated!")
                else:
                    st.warning("User not found in database.")
            except Exception as e:
                st.error(f"Could not save: {e}")

    section_title("Activity Summary", "📊")
    loader = mm.loader
    act    = loader.activity
    if not act.empty:
        uid    = user.get("user_id", "")
        my_act = act[act["user_id"] == uid] if uid else act.head(0)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Actions",    len(my_act))
        c2.metric("Medicine Views",   len(my_act[my_act["action"] == "view_medicine"]))
        c3.metric("Disease Searches", len(my_act[my_act["action"] == "search_disease"]))
