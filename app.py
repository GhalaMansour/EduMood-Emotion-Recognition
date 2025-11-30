import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import pandas as pd
import altair as alt
import logging
from edumood_recognizer import EduMoodRecognizer, EduMoodSessionStats

# ========================= Cute Kids UI Theme ========================= #

page_bg = """
<style>
.block-container {
    background-color: #ffffff;
    background-image:
        radial-gradient(circle at 20% 20%, rgba(255,222,235,0.08) 12px, transparent 40px),
        radial-gradient(circle at 80% 30%, rgba(215,243,255,0.08) 12px, transparent 40px),
        radial-gradient(circle at 50% 80%, rgba(255,245,204,0.08) 12px, transparent 40px),
        radial-gradient(circle at 30% 60%, rgba(229,255,214,0.08) 12px, transparent 40px);
    background-size: 200px 200px;
    padding-top: 30px;
    padding-bottom: 30px;
}

/* Main Title Yellow Accent */
h1 {
    position: relative;
    padding-left: 25px !important;
    font-weight: 700;
    color: #333;
}

h1:before {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    height: 100%;
    width: 12px;
    background-color: #FFD966;
    border-radius: 6px;
}

/* Section Titles */
h2, h3 {
    padding-left: 20px;
    border-left: 8px solid #FFD966;
    color: #444;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ========================= Streamlit Setup ========================= #

st.set_page_config(layout="wide")
st.sidebar.title("EduMood – Students Emotion Recognition")

st.title("Emotion Capture")
st.markdown("A clear real-time view showing all detected emotions moment-by-moment.")

# ========================= Session State ========================= #

if "session_stats" not in st.session_state:
    st.session_state["session_stats"] = EduMoodSessionStats()

session_stats: EduMoodSessionStats = st.session_state["session_stats"]

# ========================= WebRTC Video Stream ========================= #

# 👇 IMPORTANT: create recognizer first so endVideo can use it
recognizer = EduMoodRecognizer(session_stats, analyze_every_n=5)

def endVideo():
    recognizer.print_latency_summary()
    logging.info("Video has ended!")


webrtc_streamer(
    key="edumood-example",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True},
    video_frame_callback=recognizer.recognize,
    on_video_ended=endVideo,
)

# ========================= Emotion Table (Main First Table) ========================= #

df = session_stats.to_dataframe()

st.title("Emotion Capture")
st.markdown("This table shows every emotion detected at each moment.")

if df.empty:
    st.info("No emotion data yet. Start the camera to begin detecting emotions.")
    st.stop()

df_ordered = df[
    ["recorded_at", "happy", "sad", "angry", "surprise",
     "neutral", "disgusted", "fearful"]
]

st.dataframe(df_ordered, use_container_width=True, hide_index=True)

# ========================= Dashboard Data Prep ========================= #

df["recorded_at"] = pd.to_datetime(df["recorded_at"])
df_indexed = df.set_index("recorded_at")

emotion_cols = ["happy", "sad", "angry", "surprise", "neutral", "disgusted", "fearful"]
df_grouped = df_indexed[emotion_cols].sum().sort_values(ascending=False)

emotion_display = {
    "happy": "Happy 😀",
    "sad": "Sad 😢",
    "angry": "Angry 😡",
    "surprise": "Surprised 😮",
    "neutral": "Neutral 😐",
    "disgusted": "Disgusted 🤢",
    "fearful": "Fearful 😨",
}

total_all = float(df_grouped.sum()) if df_grouped.sum() > 0 else 1.0

# ========================= Overall Emotion Summary ========================= #

st.title("Overall Emotion Summary")
st.markdown("A simple view showing how students felt during the class.")

for key in emotion_cols:
    count = int(df_grouped.get(key, 0))
    pct = round((count / total_all) * 100, 1)
    label = emotion_display.get(key, key)

    st.markdown(
        f"""
        <div style="padding:10px; margin:5px 0; border-radius:12px;
                    background:rgba(255,255,255,0.7); border:1px solid #e0e0e0;">
            <strong>{label}</strong><br>
            <span style="font-size:18px; font-weight:600;">{pct}%</span>
            <span style="color:#777;"> ({count})</span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ========================= Top 3 Emotions ========================= #

st.title("Top 3 Emotions")
st.markdown("These are the three most common emotions detected in class.")

top3 = df_grouped.head(3)

top3_df = pd.DataFrame({
    "emotion_key": top3.index,
    "emotion": [emotion_display.get(e, e) for e in top3.index],
    "count": [int(v) for v in top3.values],
})

total_top3 = sum(top3.values) if sum(top3.values) > 0 else 1
top3_df["percentage"] = round((top3_df["count"] / total_top3) * 100, 1)

# Pie chart
top3_pie = (
    alt.Chart(top3_df)
    .mark_arc(innerRadius=40)
    .encode(
        theta="count",
        color="emotion",
        tooltip=["emotion", "count", "percentage"],
    )
)
st.altair_chart(top3_pie, use_container_width=True)

# Highlight blocks
st.subheader("Top Emotional Highlights")
ranks = ["1st", "2nd", "3rd"]

for rank, (_, row) in zip(ranks, top3_df.iterrows()):
    st.markdown(
        f"""
        <div style="
            padding:12px;
            margin:8px 0;
            border-radius:12px;
            background:rgba(255, 255, 255, 0.7);
            border:1px solid #e0e0e0;
        ">
            <strong style="font-size:16px;">{rank} – {row['emotion']}</strong><br>
            <span style="color:#444; font-size:15px;">
                {row['percentage']}% of the top emotions.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ========================= Full Pie Chart ========================= #

st.title("Emotion Distribution")
st.markdown("A general look at how emotions were spread through the whole class.")

pie_df = pd.DataFrame(
    {
        "emotion": [emotion_display.get(k, k) for k in df_grouped.index],
        "count": [int(v) for v in df_grouped.values],
    }
)
pie_df = pie_df[pie_df["count"] > 0]

pie_chart = (
    alt.Chart(pie_df)
    .mark_arc(innerRadius=40)
    .encode(
        theta="count",
        color="emotion",
        tooltip=["emotion", "count"],
    )
)
st.altair_chart(pie_chart, use_container_width=True)

# ========================= Line Chart (Over Time) ========================= #

st.title("Emotions Over Time")
st.markdown("This chart shows how students' emotions changed throughout the class.")

line_df = df_indexed[emotion_cols]

if line_df.sum().sum() == 0:
    st.info("Not enough data yet to draw the time-line chart.")
else:
    st.line_chart(line_df)

# ========================= Histogram ========================= #

st.title("Emotion Histogram")
st.markdown("A simple chart showing how many times each emotion appeared.")

hist_df = df_grouped.reset_index()
hist_df.columns = ["emotion", "count"]

vertical_hist = (
    alt.Chart(hist_df)
    .mark_bar(size=40)
    .encode(
        x=alt.X("emotion:N", title="Emotion"),
        y=alt.Y("count:Q", title="Frequency"),
        color=alt.Color("emotion:N", legend=None),
        tooltip=["emotion", "count"],
    )
)

st.altair_chart(vertical_hist, use_container_width=True)
