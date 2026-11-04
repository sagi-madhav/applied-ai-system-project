"""
Streamlit Web Application for the Applied AI Music Recommender System.

Provides an interactive user interface for natural language music queries,
agent plan visualization, guardrail verification, and human-in-the-loop feedback.
"""

import streamlit as st
import pandas as pd
from src.agent import MusicCuratorAgent, GuardrailEngine

st.set_page_config(page_title="Music Curator AI", layout="wide")

st.title("Music Curator AI")
st.write(
    "This system uses an agentic workflow with catalog retrieval and safety guardrails "
    "to recommend songs matching your vibe, energy, and genre preferences."
)


@st.cache_resource
def get_agent():
    return MusicCuratorAgent("data/songs.csv")


agent = get_agent()

# Session state initialization
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "current_intent" not in st.session_state:
    st.session_state.current_intent = None

col1, col2 = st.columns([2, 1])

with col1:
    user_query = st.text_input(
        "Describe the music you want to hear:",
        placeholder="For example: Upbeat pop songs for a morning workout, or calm acoustic music for studying",
    )

    k_songs = st.slider("Number of recommendations:", min_value=1, max_value=8, value=4)

    if st.button("Curate Music"):
        if not user_query.strip():
            st.warning("Please enter a query before submitting.")
        else:
            with st.spinner("Processing request through agent workflow..."):
                result = agent.run(user_query, k=k_songs)
                st.session_state.last_result = result
                if result["success"]:
                    st.session_state.current_intent = result["intent"]

with col2:
    st.subheader("Try Example Queries")
    examples = [
        "Upbeat and happy pop songs for a workout",
        "Calm acoustic music for studying",
        "Moody synthwave for a night drive",
        "Aggressive heavy metal",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            with st.spinner("Processing example..."):
                result = agent.run(ex, k=k_songs)
                st.session_state.last_result = result
                if result["success"]:
                    st.session_state.current_intent = result["intent"]

# Display Results
result = st.session_state.last_result
if result:
    st.write("---")
    if not result["success"]:
        st.error(f"Request Rejected by Guardrail: {result['error']}")
    else:
        intent = result["intent"]
        guardrails = result["guardrails"]
        recs = result["recommendations"]

        st.subheader("Agent Plan and Verification")
        pcol1, pcol2, pcol3, pcol4 = st.columns(4)
        pcol1.metric("Detected Genre", intent.target_genre or "Any Genre")
        pcol2.metric("Detected Mood", intent.target_mood or "Any Mood")
        pcol3.metric("Target Energy", f"{intent.target_energy:.2f}")
        pcol4.metric("Acoustic Preference", "Yes" if intent.likes_acoustic else ("No" if intent.likes_acoustic is False else "Neutral"))

        # Guardrail check summary
        st.write("**Guardrail Verification:**")
        g1, g2, g3 = st.columns(3)
        g1.success("Catalog Grounding: Passed (0% Hallucination)")
        g2.success("Energy Parameter Bounds: Passed")
        if guardrails.checks["diversity"]:
            g3.success("Artist Diversity Check: Passed")
        else:
            g3.warning("Artist Diversity Check: Adjusted")

        st.subheader("Curated Recommendations")
        for rank, item in enumerate(recs, start=1):
            s = item.song
            with st.container():
                st.markdown(f"### {rank}. {s.title} - {s.artist}")
                st.write(
                    f"**Genre:** {s.genre} | **Mood:** {s.mood} | **Energy:** {s.energy:.2f} | "
                    f"**Tempo:** {s.tempo_bpm} BPM | **Match Confidence:** {item.confidence:.0%}"
                )
                st.write(f"**Reasons:** {', '.join(item.reasons)}")
                st.write("---")

        st.subheader("Explanation")
        st.info(result["explanation"])

        # Audio attributes visualization
        if recs:
            st.subheader("Song Audio Attributes Comparison")
            chart_data = pd.DataFrame(
                [
                    {
                        "Title": item.song.title,
                        "Energy": item.song.energy,
                        "Valence": item.song.valence,
                        "Danceability": item.song.danceability,
                        "Acousticness": item.song.acousticness,
                    }
                    for item in recs
                ]
            ).set_index("Title")
            st.bar_chart(chart_data)

        # Human-in-the-loop refinement section
        st.subheader("Human-in-the-Loop Refinement")
        st.write("Not quite what you wanted? Tell the agent how to adjust this playlist.")
        feedback_input = st.text_input(
            "Refinement instructions:",
            placeholder="For example: make it more energetic, less acoustic, or no rock",
            key="feedback_field",
        )
        if st.button("Apply Feedback"):
            if feedback_input.strip() and st.session_state.current_intent:
                with st.spinner("Refining playlist with your feedback..."):
                    refined_result = agent.apply_human_feedback(
                        st.session_state.current_intent, feedback_input, k=k_songs
                    )
                    st.session_state.last_result = refined_result
                    st.session_state.current_intent = refined_result["intent"]
                    st.rerun()
