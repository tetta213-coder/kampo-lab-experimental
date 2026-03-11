import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import plotly.express as px
import plotly.graph_objects as go

# 1. ページ基本設定
st.set_page_config(page_title="漢方マッピング・ラボ", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: white !important; color: #31333F !important; }
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp span { color: #31333F !important; }
    [data-testid="stMetricValue"] { font-size: 1.6rem !important; color: #1f77b4 !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌿 証空間の地図")

# --- 2. 固定パラメータの設定 ---
SENSITIVITY = 250.0
ZOOM_SCALE = 4.0

# --- 3. データの読み込み ---
@st.cache_data
def load_base_data():
    df = pd.read_csv("kampo_yakuno_integrated.csv")
    df = df[df['No'] <= 148].copy()
    yakuno_cols = [
        "補気", "理気", "降気", "補血", "駆瘀血", "止血", "利水", "潤水", "温", "清", 
        "安心鎮静", "認知知能", "鎮痙", "眼精疲労", "清頭目", "排膿", "解毒", "疣贅", 
        "制吐", "鎮嘔", "瀉下", "黄疸", "安胎", "通乳"
    ]
    return df, yakuno_cols

df_base, yakuno_cols = load_base_data()

# --- 4. サイドバー：患者入力 ---
st.sidebar.header("👤 患者の病態入力")
age = st.sidebar.number_input("患者の年齢", min_value=0, max_value=120, value=40, step=1)

st.sidebar.subheader("1. 基本の10指標 (証)")
defaults = {
    '虚実': 0.5, '寒': 0.5, '熱': 0.5,
    '気虚': 0.3, '気鬱': 0.3, '気逆': 0.3,
    '血虚': 0.3, '瘀血': 0.3, '水毒': 0.3, '腎虚': 0.3
}

sho_input = {}
sho_names = ['虚実', '寒', '熱', '気虚', '気鬱', '気逆', '血虚', '瘀血', '水毒', '腎虚']
for name in sho_names:
    label_text = "虚実 (0:虚 ←→ 1:実)" if name == '虚実' else name
    sho_input[name] = st.sidebar.slider(label_text, 0.0, 1.0, defaults.get(name, 0.0), key=f"slider_{name}")

st.sidebar.subheader("2. 特定の随伴症状")
raw_input = {}
symptom_labels = ["安心鎮静 (不眠・不安)", "認知知能 (物忘れ)", "鎮痙 (足のつり)", "眼精疲労", "清頭目 (のぼせ・頭痛)", "排膿 (にきび)", "解毒 (かゆみ)", "疣贅 (いぼ)", "制吐・鎮嘔", "瀉下 (便秘)", "黄疸", "安胎", "通乳"]
for label in symptom_labels:
    raw_input[label] = st.sidebar.radio(f"{label}", ["なし", "あり"], index=0, horizontal=True, key=f"radio_{label}")

# --- 5. 計算ロジック ---
def create_patient_vec(sho, raw, age):
    p = {k: 0.01 for k in yakuno_cols} 
    kyo, jitsu = max(0, 0.5-sho['虚実'])*2.0, max(0, sho['虚実']-0.5)*2.0
    p["補気"] += (sho['気虚']*3.0) + (kyo*0.5)
    p["補血"] += (sho['血虚']*3.0) + (kyo*0.5)
    p["利水"] += (sho['水毒']*3.0)
    p["駆瘀血"] += (sho['瘀血']*3.0) + (jitsu*0.5)
    p["理気"] += sho['気鬱']*3.0; p["降気"] += sho['気逆']*3.0
    p["温"] += sho['寒']*3.0; p["清"] += sho['熱']*3.0
    jk = min(1.0, sho['腎虚'] + max(0, (age-40)*0.02))
    p["補気"] += jk; p["補血"] += jk; p["潤水"] += jk; p["利水"] += jk
    mapping = {"安心鎮静 (不眠・不安)": ["安心鎮静"], "認知知能 (物忘れ)": ["認知知能"], "鎮痙 (足のつり)": ["鎮痙"], "眼精疲労": ["眼精疲労"], "清頭目 (のぼせ・頭痛)": ["清頭目"], "排膿 (にきび)": ["排膿"], "解毒 (かゆみ)": ["解毒"], "疣贅 (いぼ)": ["疣贅"], "制吐・鎮嘔": ["制吐", "鎮嘔"], "瀉下 (便秘)": ["瀉下"], "黄疸": ["黄疸"], "安胎": ["安胎"], "通乳": ["通乳"]}
    for label, target_keys in mapping.items():
        if raw.get(label) == "あり":
            for k in target_keys: p[k] = 5.0
    vec = np.array([p[k] for k in yakuno_cols])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

patient_vec = create_patient_vec(sho_input, raw_input, age)
yakuno_data_raw = df_base[yakuno_cols].fillna(0).values
yakuno_norms = np.linalg.norm(yakuno_data_raw, axis=1, keepdims=True)
yakuno_data_norm = np.divide(yakuno_data_raw, yakuno_norms, out=np.zeros_like(yakuno_data_raw), where=yakuno_norms!=0)

@st.cache_data
def get_fixed_coords(data):
    tsne = TSNE(n_components=2, perplexity=25, random_state=42, init='pca', learning_rate='auto')
    return tsne.fit_transform(data)

coords = get_fixed_coords(yakuno_data_raw)
df_base['x'], df_base['y'] = coords[:, 0], coords[:, 1]

dists = euclidean_distances([patient_vec], yakuno_data_norm)[0]
near_indices = dists.argsort()[:3]
near_dists = dists[near_indices]
near_coords = coords[near_indices]
d_min = near_dists.min()
weights = np.exp(-SENSITIVITY * (near_dists - d_min) / (near_dists.max() - d_min + 1e-9))
star_x = (near_coords[:, 0] * weights).sum() / weights.sum()
star_y = (near_coords[:, 1] * weights).sum() / weights.sum()

# --- 6. 推奨ランキング表示（並列レイアウト） ---
df_base['cos_sim'] = cosine_similarity([patient_vec], yakuno_data_norm)[0]
df_base['dist_2d'] = np.sqrt((df_base['x'] - star_x)**2 + (df_base['y'] - star_y)**2)
df_base['prox_2d'] = (1 - (df_base['dist_2d'] / (df_base['dist_2d'].max() + 1e-9)))

st.subheader("🌟 推奨処方")
col_main_left, col_main_right = st.columns(2)

with col_main_left:
    st.write("**24次元でのコサイン類似度**")
    top_cos = df_base.sort_values('cos_sim', ascending=False).head(3)
    sub_cols = st.columns(3)
    for i, (idx, row) in enumerate(top_cos.iterrows()):
        sub_cols[i].metric(f"{i+1}. {row['formula']}", f"{row['cos_sim']:.1%}")

with col_main_right:
    st.write("**2D地図上の近接 (位置)**")
    top_dist = df_base.sort_values('dist_2d', ascending=True).head(3)
    sub_cols = st.columns(3)
    for i, (idx, row) in enumerate(top_dist.iterrows()):
        sub_cols[i].metric(f"{i+1}. {row['formula']}", f"{row['prox_2d']:.1%}")

st.write("---")

# 地図描画
fig = px.scatter(df_base, x='x', y='y', text='formula', color='cos_sim', color_continuous_scale='Viridis', hover_name='formula', height=800)
fig.add_trace(go.Scatter(x=[star_x], y=[star_y], mode='markers+text', marker=dict(symbol='star', size=80, color='red', line=dict(width=3, color='black')), text=["149: 患者"], textposition="top center", textfont=dict(size=22, color='red', family="HiraKakuPro-W6")))
fig.update_traces(textposition='top center', marker=dict(size=12))
fig.update_layout(plot_bgcolor='white', xaxis=dict(visible=False, range=[star_x - ZOOM_SCALE, star_x + ZOOM_SCALE]), yaxis=dict(visible=False, range=[star_y - ZOOM_SCALE, star_y + ZOOM_SCALE]), showlegend=False, uirevision='constant', margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)
