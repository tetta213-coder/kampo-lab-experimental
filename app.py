import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# 1. ページ基本設定
st.set_page_config(page_title="漢方マッピング・ラボ", layout="wide")

# --- 【ライトモード固定CSS】 ---
st.markdown("""
    <style>
    .stApp { background-color: white !important; color: #31333F !important; }
    [data-testid="stHeader"] { background-color: white !important; }
    [data-testid="stSidebar"] { background-color: #f0f2f6 !important; }
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp span { color: #31333F !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌿 証空間の地図：148処方の宇宙")

# 2. データの読み込み
@st.cache_data
def load_data():
    df = pd.read_csv("kampo_yakuno_integrated.csv")
    df = df[df['No'] <= 148].copy()
    return df

df_full = load_data()

yakuno_cols = [
    "補気", "理気", "降気", "補血", "駆瘀血", "止血", "利水", "潤水", "温", "清", 
    "安心鎮静", "認知知能", "鎮痙", "眼精疲労", "清頭目", "排膿", "解毒", "疣贅", 
    "制吐", "鎮嘔", "瀉下", "黄疸", "安胎", "通乳"
]

# --- 3. サイドバー：入力インターフェース ---
st.sidebar.header("👤 患者の病態入力")

# 【追加】年齢入力（腎虚に直結）
age = st.sidebar.number_input("患者の年齢", min_value=0, max_value=120, value=40, step=1)

st.sidebar.subheader("1. 基本の10指標 (証)")
sho_input = {}
sho_names = ['虚実', '寒', '熱', '気虚', '気鬱', '気逆', '血虚', '瘀血', '水毒', '腎虚']
for name in sho_names:
    sho_input[name] = st.sidebar.slider(f"{name}", 0.0, 1.0, 0.1, key=f"slider_{name}")

st.sidebar.subheader("2. 特定の随伴症状")
raw_input = {}
symptom_labels = [
    "安心鎮静 (不眠・不安)", "認知知能 (物忘れ)", "鎮痙 (足のつり)", "眼精疲労", 
    "清頭目 (のぼせ・頭痛)", "排膿 (にきび)", "解毒 (かゆみ)", "疣贅 (いぼ)", 
    "制吐・鎮嘔", "瀉下 (便秘)", "黄疸", "安胎", "通乳"
]
for label in symptom_labels:
    raw_input[label] = st.sidebar.radio(f"{label}", ["なし", "あり"], index=0, horizontal=True, key=f"radio_{label}")

# --- 4. 24次元患者ベクトルの生成ロジック（年齢ブースト版） ---
def create_patient_vec(sho, raw, age):
    p = {k: 0.0 for k in yakuno_cols}
    
    # 【臨床ロジック】年齢による腎虚の自動計算
    # 40歳を起点に、1歳ごとに 0.02 ずつ加算（80歳で+0.8）
    age_jinkyo_bonus = max(0, (age - 40) * 0.02)
    total_jk = min(1.0, sho['腎虚'] + age_jinkyo_bonus)
    
    # 腎虚（Jinkyo）の各軸への配分
    p["補気"] = sho['気虚'] + (total_jk * 0.8)
    p["補血"] = sho['血虚'] + (total_jk * 0.8)
    p["潤水"] = (total_jk * 0.8)
    p["利水"] = sho['水毒'] + (total_jk * 0.5)
    
    p["理気"] = sho['気鬱']
    p["降気"] = sho['気逆']
    p["駆瘀血"] = sho['瘀血']
    p["温"] = sho['寒']
    p["清"] = sho['熱']

    mapping = {
        "安心鎮静 (不眠・不安)": ["安心鎮静"], "認知知能 (物忘れ)": ["認知知能"],
        "鎮痙 (足のつり)": ["鎮痙"], "眼精疲労": ["眼精疲労"],
        "清頭目 (のぼせ・頭痛)": ["清頭目"], "排膿 (にきび)": ["排膿"],
        "解毒 (かゆみ)": ["解毒"], "疣贅 (いぼ)": ["疣贅"],
        "制吐・鎮嘔": ["制吐", "鎮嘔"], "瀉下 (便秘)": ["瀉下"],
        "黄疸": ["黄疸"], "安胎": ["安胎"], "通乳": ["通乳"]
    }
    for label, target_keys in mapping.items():
        if raw.get(label) == "あり":
            for k in target_keys: p[k] = 0.8

    if p["降気"] < 0.3 and sho['気鬱'] > 0.5 and sho['熱'] > 0.5:
        p["降気"] = (sho['気鬱'] + sho['熱']) / 2
    return np.array([p[k] for k in yakuno_cols])

patient_vec = create_patient_vec(sho_input, raw_input, age)

# --- 5. 地図の計算 ---
yakuno_data = df_full[yakuno_cols].fillna(0)
yakuno_data_jittered = yakuno_data + np.random.normal(0, 1e-6, yakuno_data.shape)

tsne = TSNE(n_components=2, perplexity=25, random_state=42, init='pca', learning_rate='auto')
coords = tsne.fit_transform(yakuno_data_jittered)
df_full['x'], df_full['y'] = coords[:, 0], coords[:, 1]

# --- 6. マッチングと★の描画 ---
similarities = cosine_similarity([patient_vec], yakuno_data.values)[0]
df_full['一致度'] = similarities
top_3 = df_full.sort_values('一致度', ascending=False).head(3)
star_x, star_y = top_3['x'].mean(), top_3['y'].mean()

fig = px.scatter(
    df_full, x='x', y='y', text='formula',
    color='一致度', color_continuous_scale='Viridis',
    hover_name='formula', height=800
)

# 特大赤スター
fig.add_trace(go.Scatter(
    x=[star_x], y=[star_y], mode='markers+text',
    marker=dict(symbol='star', size=75, color='red', line=dict(width=3, color='red')),
    text=["あなたの現在地"], 
    textposition="top center", 
    textfont=dict(size=24, color='red', family="HiraKakuPro-W6"), 
    name="現在の証"
))

fig.update_traces(textposition='top center', marker=dict(size=12))
fig.update_layout(plot_bgcolor='white', xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)

st.plotly_chart(fig, use_container_width=True)

# 推薦処方
st.subheader("🌟 推奨処方")
cols = st.columns(3)
for i, (idx, row) in enumerate(top_3.iterrows()):
    cols[i].metric(f"{i+1}. {row['formula']}", f"{row['一致度']:.1%}")
