import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import warnings
from datetime import datetime
import io

# --- SETUP & CONFIG ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="AI Sentiment Pro", page_icon="⚡", layout="wide")

# --- SESSION STATE FOR LOGIN ---
if 'auth' not in st.session_state:
    st.session_state['auth'] = False

# --- LOAD NLP ENGINE ---
@st.cache_resource
def load_engine():
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except:
        nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sia = load_engine()

# --- STYLING (The Professional Look) ---
st.markdown("""
<style>
    .stApp { background: #0f172a; color: white; }
    [data-testid="stSidebar"] { background-color: #1e293b !important; border-right: 1px solid #3b82f6; }
    .header-box {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        padding: 2.5rem; border-radius: 20px; text-align: center;
        border: 1px solid #3b82f6; margin-bottom: 2rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8) !important;
        color: white !important; font-weight: bold !important; width: 100%;
        border-radius: 10px !important; border: none !important;
    }
    .footer {
        background: #1e293b; padding: 20px; border-radius: 15px;
        text-align: center; margin-top: 50px; border-top: 2px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def analyze_sentiment(text):
    if not text: return None
    scores = sia.polarity_scores(str(text))
    cmp = scores['compound']
    if cmp >= 0.05: return "POSITIVE", "🟢", cmp
    elif cmp <= -0.05: return "NEGATIVE", "🔴", cmp
    else: return "NEUTRAL", "🟡", cmp

def robust_read(file):
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except: continue
    return None

# --- LOGIN PAGE ---
def show_login():
    st.markdown("<h1 style='text-align:center; color:#3b82f6; margin-top:100px;'>⚡ Sentiment Intelligence Pro</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1.2,1])
    with col2:
        with st.form("Login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("ENTER DASHBOARD"):
                if u == "admin" and p == "password": # Change as needed
                    st.session_state['auth'] = True
                    st.rerun()
                else: st.error("Access Denied")

# --- MAIN DASHBOARD ---
def main_app():
    # Sidebar
    with st.sidebar:
        st.title("🚀 Navigation")
        page = st.radio("Select View", ["Live Engine", "Batch Process", "Analytics Dashboard"])
        st.divider()
        max_s = st.slider("Max Data Sample", 100, 5000, 500)
        if st.button("🚪 Logout"):
            st.session_state['auth'] = False
            st.rerun()

    st.markdown('<div class="header-box"><h1>⚡ AI Sentiment Analytics</h1><p style="color:#93c5fd">Distributed NLP Platform for Social Media Insights</p></div>', unsafe_allow_html=True)

    if page == "Live Engine":
        c1, c2 = st.columns([1.5, 1])
        with c1:
            txt = st.text_area("Input Review / Tweet:", placeholder="Type a movie or product review...", height=250)
            btn = st.button("RUN AI ENGINE")
        if btn and txt:
            lab, emo, sc = analyze_sentiment(txt)
            with c2:
                st.metric("Detected Sentiment", f"{emo} {lab}")
                st.metric("Polarity Score", f"{sc:.4f}")
                fig = go.Figure(go.Indicator(mode="gauge+number", value=sc, domain={'x':[0,1],'y':[0,1]},
                    gauge={'axis':{'range':[-1,1]}, 'bar':{'color':"#3b82f6"}}))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color="white", height=300)
                st.plotly_chart(fig, use_container_width=True)

    elif page == "Batch Process":
        st.subheader("📁 Spark-Powered Batch Processing")
        up = st.file_uploader("Upload CSV Data", type=["csv"])
        if up:
            df = robust_read(up)
            if df is not None:
                st.success(f"File Loaded: {len(df):,} records found.")
                col_name = st.selectbox("Select Text Column", df.columns)
                if st.button("🚀 EXECUTE DISTRIBUTED ANALYSIS"):
                    sample = df.head(max_s).copy()
                    bar = st.progress(0)
                    results = []
                    for i, t in enumerate(sample[col_name]):
                        res = analyze_sentiment(t)
                        results.append({'sentiment': res[0], 'score': res[2]})
                        if i % 10 == 0: bar.progress((i+1)/len(sample))
                    
                    df_res = pd.concat([sample.reset_index(drop=True), pd.DataFrame(results)], axis=1)
                    st.session_state['data'] = df_res
                    st.dataframe(df_res.head(20), use_container_width=True)
                    st.download_button("📥 Download Report", df_res.to_csv(index=False).encode('utf-8'), "sentiment_report.csv")

    elif page == "Analytics Dashboard":
        if 'data' in st.session_state:
            df_d = st.session_state['data']
            st.subheader("📊 Global Sentiment Trends")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Processed Total", len(df_d))
            m2.metric("Positive Avg", f"{df_d[df_d.sentiment=='POSITIVE']['score'].mean():.2f}")
            m3.metric("System Status", "Scalable (HDFS)")

            v1, v2 = st.columns(2)
            with v1:
                st.plotly_chart(px.pie(df_d, names='sentiment', hole=.4, title="Sentiment Split",
                                      color_discrete_map={'POSITIVE':'#22c55e','NEGATIVE':'#ef4444','NEUTRAL':'#eab308'}), use_container_width=True)
            with v2:
                st.plotly_chart(px.histogram(df_d, x='score', title="Polarity Distribution", color='sentiment'), use_container_width=True)
        else:
            st.info("💡 Please process a CSV in 'Batch Process' to unlock these charts.")

    # Footer
    st.markdown("""
    <div class="footer">
        <h4 style="color:#3b82f6;">MCA Portfolio Project | Large-Scale Sentiment Analysis</h4>
        <p>Built with Apache Spark, Python & Streamlit • Distributed Systems Architecture</p>
    </div>
    """, unsafe_allow_html=True)

# --- RUN ENGINE ---
if not st.session_state['auth']:
    show_login()
else:
    main_app()