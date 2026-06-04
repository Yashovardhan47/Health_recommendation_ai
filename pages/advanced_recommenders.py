"""pages/advanced_recommenders.py — Context / Graph / RL page"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random, math
from utils.ui_components import section_title, render_medicine_grid, PLOTLY_LAYOUT


# ── Context-Aware Engine ──────────────────────────────────────────────────────
def _context_recommend(medicines: pd.DataFrame, season: str, time_of_day: str,
                        user_profile: dict, n: int = 5) -> list:
    season_boost = {
        "Winter": {"Antihistamine": 0.2, "Analgesic": 0.15},
        "Summer": {"Decongestant": 0.1},
        "Monsoon": {"Antibiotic": 0.2, "Antiviral": 0.15},
        "Spring": {"Antihistamine": 0.25},
    }
    time_boost = {
        "Morning": {"SSRI": 0.1, "Antidiabetic": 0.1},
        "Evening": {"Analgesic": 0.1, "CCB": 0.05},
        "Night":   {"Benzodiazepine": 0.15, "Anxiolytic": 0.1},
    }
    results = []
    for _, row in medicines.iterrows():
        score = float(row.get("effectiveness", 0.7))
        cat   = row.get("category", "")
        score += season_boost.get(season, {}).get(cat, 0)
        score += time_boost.get(time_of_day, {}).get(cat, 0)
        # Age-based adjustment
        age = user_profile.get("age", 40)
        if age > 60 and cat in ("Benzodiazepine",):
            score -= 0.15
        results.append({
            "med_id": row["med_id"], "name": row["name"],
            "category": cat, "dosage": row.get("dosage", ""),
            "effectiveness": row.get("effectiveness", 0),
            "price_inr": row.get("price_inr", 0), "score": round(min(score, 1.0), 4),
        })
    return sorted(results, key=lambda x: -x["score"])[:n]


# ── Graph-Based Engine (NetworkX + PageRank) ──────────────────────────────────
def _graph_recommend(medicines: pd.DataFrame, interactions: pd.DataFrame,
                     seed_med: str, n: int = 5) -> list:
    try:
        import networkx as nx
        G = nx.Graph()
        for _, row in medicines.iterrows():
            G.add_node(row["med_id"], name=row["name"],
                       category=row.get("category",""), eff=row.get("effectiveness",0.7))
        # Add edges based on shared disease associations
        if "disease_ids" in medicines.columns:
            for i, r1 in medicines.iterrows():
                for j, r2 in medicines.iterrows():
                    if i >= j: continue
                    d1 = set(str(r1.get("disease_ids","")).split(","))
                    d2 = set(str(r2.get("disease_ids","")).split(","))
                    shared = len(d1 & d2)
                    if shared > 0:
                        G.add_edge(r1["med_id"], r2["med_id"], weight=shared)
        # Add edges from interactions
        if not interactions.empty:
            user_meds = interactions.groupby("user_id")["med_id"].apply(list)
            for meds in user_meds:
                for i in range(len(meds)):
                    for j in range(i+1, min(i+4, len(meds))):
                        if G.has_node(meds[i]) and G.has_node(meds[j]):
                            if G.has_edge(meds[i], meds[j]):
                                G[meds[i]][meds[j]]["weight"] = G[meds[i]][meds[j]].get("weight",0) + 0.5
                            else:
                                G.add_edge(meds[i], meds[j], weight=0.5)

        personalization = {node: (2.0 if node == seed_med else 1.0) for node in G.nodes()}
        pr = nx.pagerank(G, alpha=0.85, personalization=personalization, weight="weight")
        ranked = sorted(pr.items(), key=lambda x: -x[1])
        results = []
        for mid, score in ranked:
            if mid == seed_med: continue
            data = G.nodes[mid]
            row  = medicines[medicines["med_id"] == mid]
            if row.empty: continue
            row  = row.iloc[0]
            results.append({
                "med_id": mid, "name": data.get("name", mid),
                "category": data.get("category",""), "dosage": row.get("dosage",""),
                "effectiveness": row.get("effectiveness",0),
                "price_inr": row.get("price_inr",0), "score": round(score, 6),
            })
            if len(results) >= n: break
        return results
    except Exception as e:
        st.error(f"Graph error: {e}")
        return []


# ── Q-Learning RL Agent ───────────────────────────────────────────────────────
class RLAgent:
    def __init__(self, n_states=10, n_actions=30, lr=0.1, gamma=0.9, eps=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = lr; self.gamma = gamma; self.eps = eps

    def act(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.Q.shape[1]-1)
        return int(np.argmax(self.Q[state % self.Q.shape[0]]))

    def update(self, s, a, r, s2):
        s  = s  % self.Q.shape[0]
        s2 = s2 % self.Q.shape[0]
        a  = a  % self.Q.shape[1]
        td = r + self.gamma * np.max(self.Q[s2]) - self.Q[s, a]
        self.Q[s, a] += self.lr * td

    def train(self, medicines, interactions, episodes=100):
        mids = medicines["med_id"].tolist()
        n    = len(mids)
        avg_ratings = {}
        if not interactions.empty:
            avg_ratings = interactions.groupby("med_id")["rating"].mean().to_dict()
        rewards_log = []
        for ep in range(episodes):
            state   = random.randint(0, self.Q.shape[0]-1)
            ep_rew  = 0
            for _ in range(10):
                action = self.act(state)
                mid    = mids[action % n]
                reward = avg_ratings.get(mid, 3.0) / 5.0
                next_s = (state + 1) % self.Q.shape[0]
                self.update(state, action % self.Q.shape[1], reward, next_s)
                state   = next_s
                ep_rew += reward
            rewards_log.append(ep_rew / 10)
        return rewards_log

    def recommend(self, state, medicines, n=5):
        mids    = medicines["med_id"].tolist()
        n_meds  = len(mids)
        s       = state % self.Q.shape[0]
        scores  = self.Q[s]
        ranked  = np.argsort(-scores)
        seen    = set()
        results = []
        for idx in ranked:
            mid_idx = idx % n_meds
            mid     = mids[mid_idx]
            if mid in seen: continue
            seen.add(mid)
            row = medicines[medicines["med_id"]==mid].iloc[0]
            results.append({
                "med_id": mid, "name": row["name"],
                "category": row.get("category",""), "dosage": row.get("dosage",""),
                "effectiveness": row.get("effectiveness",0),
                "price_inr": row.get("price_inr",0), "score": round(float(scores[idx % self.Q.shape[1]]), 4),
            })
            if len(results) >= n: break
        return results


# ── Page render ───────────────────────────────────────────────────────────────
def render(mm, user):
    st.markdown("## 🧠 Advanced Recommendation Engines")
    loader = mm.loader

    tab1, tab2, tab3 = st.tabs(["🌍 Context-Aware", "🕸️ Graph-Based", "🎮 RL Agent"])

    with tab1:
        st.markdown("**Context-Aware Engine** adapts recommendations based on season, time of day, and user profile.")
        c1, c2 = st.columns(2)
        season      = c1.selectbox("Season:", ["Winter","Spring","Summer","Monsoon"])
        time_of_day = c2.selectbox("Time of Day:", ["Morning","Afternoon","Evening","Night"])
        n_ctx       = st.slider("Top-N:", 3, 10, 5, key="ctx_n")
        profile     = {"age": user.get("age", 35), "gender": user.get("gender","M")}
        recs = _context_recommend(loader.medicines, season, time_of_day, profile, n=n_ctx)
        section_title(f"Context-Aware Picks — {season} / {time_of_day}", "🌍")
        render_medicine_grid(recs, loader.interactions)

    with tab2:
        st.markdown("**Knowledge Graph + PageRank** traverses medicine-disease associations.")
        med_names = loader.medicines.set_index("med_id")["name"].to_dict()
        seed = st.selectbox("Seed medicine:", list(med_names.keys()),
                            format_func=lambda x: med_names.get(x, x), key="gr_seed")
        n_gr = st.slider("Top-N:", 3, 10, 5, key="gr_n")
        recs = _graph_recommend(loader.medicines, loader.interactions, seed, n=n_gr)
        section_title("Graph-Based Recommendations", "🕸️")
        render_medicine_grid(recs, loader.interactions)

    with tab3:
        st.markdown("**Q-Learning Agent** learns which medicines to surface from interaction rewards.")
        n_rl   = st.slider("Top-N:", 3, 10, 5, key="rl_n")
        ep_rl  = st.slider("Training Episodes:", 50, 300, 100, key="rl_ep")

        if st.button("🎮 Train RL Agent", key="rl_train"):
            agent = RLAgent(n_states=15, n_actions=max(30, len(loader.medicines)))
            with st.spinner("Training Q-Learning agent…"):
                rewards = agent.train(loader.medicines, loader.interactions, episodes=ep_rl)
            st.session_state["rl_agent"]   = agent
            st.session_state["rl_rewards"] = rewards
            st.success(f"✅ Agent trained! Final avg reward: {rewards[-1]:.3f}")

        if "rl_rewards" in st.session_state:
            section_title("Reward Curve", "📈")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=st.session_state["rl_rewards"],
                                     line=dict(color="#00c8ff", width=2), name="Avg Reward"))
            fig.update_layout(**PLOTLY_LAYOUT, height=220, xaxis_title="Episode", yaxis_title="Reward")
            st.plotly_chart(fig, use_container_width=True)

        if "rl_agent" in st.session_state:
            state = hash(user.get("user_id", "U0001")) % 15
            recs  = st.session_state["rl_agent"].recommend(state, loader.medicines, n=n_rl)
            section_title("RL Agent Recommendations", "🎮")
            render_medicine_grid(recs, loader.interactions)
        else:
            st.info("Click **Train RL Agent** above to generate recommendations.")
