import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
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

# --- 2. 固定パラメータ ---
SENSITIVITY = 250.0
ZOOM_SCALE = 4.0

# --- 3. データの読み込み ---
@st.cache_data
def load_and_normalize_data():
    # update_integrated_data.py が出力する最新ファイルを参照
    df = pd.read_csv("kampo_yakuno_integrated_24dims_final.csv")
    df = df[df['No'] <= 148].copy()
    
    # 24次元カテゴリー（解熱・消炎を統合、排膿を解毒へ統合）
    yakuno_cols = [
        "補気", "理気", "降気", "補血", "駆瘀血", "利水", "補腎", "温", "清", "瀉下",
        "鎮痛", "健胃・整腸", "鎮咳", "安心鎮静", "去痰", "清頭目", "止瀉", 
        "潤燥", "発表", "鎮痙", "制吐・鎮嘔", "解毒", "解熱・消炎", "止血"
    ]
    
    raw_values = df[yakuno_cols].fillna(0).values
    normalized_values = normalize(raw_values, norm='l2')
    return df, yakuno_cols, raw_values, normalized_values

df_base, yakuno_cols, yakuno_data_raw, yakuno_data_norm = load_and_normalize_data()

# --- 4. サイドバー：患者入力 ---
st.sidebar.header("👤 患者の病態入力")

st.sidebar.subheader("1. 証の10指標")
sho_input = {}
sho_names = ['虚実', '寒', '熱', '気虚', '気鬱', '気逆', '血虚', '瘀血', '水毒', '腎虚']
for name in sho_names:
    val = 0.5 if name in ['虚実', '寒', '熱'] else 0.1
    sho_input[name] = st.sidebar.slider(name, 0.0, 1.0, val, key=f"slider_{name}")

st.sidebar.subheader("2. 随伴症状 (14項目)")
raw_input = {}
symptom_mapping = {
    "痛みの改善 (鎮痛・痺・疝)": "鎮痛",
    "胃腸の調整 (健胃・整腸)": "健胃・整腸",
    "咳を鎮める (鎮咳)": "鎮咳",
    "不眠・不安 (安心鎮静)": "安心鎮静",
    "痰の改善 (去痰)": "去痰",
    "のぼせ・目の充血 (清頭目)": "清頭目",
    "下痢を止める (止瀉)": "止瀉",
    "乾燥を潤す (潤燥)": "潤燥",
    "風邪の初期 (発表)": "発表",
    "筋肉の痙攣 (鎮痙)": "鎮痙",
    "吐き気 (制吐・鎮嘔)": "制吐・鎮嘔",
    "化膿・皮膚炎 (解毒)": "解毒",
    "発熱・炎症 (解熱・消炎)": "解熱・消炎",
    "出血 (止血)": "止血"
}

for label in symptom_mapping.keys():
    raw_input[label] = st.sidebar.radio(label, ["なし", "あり"], index=0, horizontal=True)

# --- 5. 計算ロジック ---
def create_patient_vec(sho, raw):
    p = {k: 0.001 for k in yakuno_cols}
    kyo_weight = max(0, 0.5 - sho['虚実']) * 4.0
    jitsu_weight = max(0, sho['虚実'] - 0.5) * 4.0
    
    # 証の射影
    p["補気"] += (sho['気虚'] * 3.0) + (kyo_weight * 1.0)
    p["補血"] += (sho['血虚'] * 3.0) + (kyo_weight * 1.0)
    p["補腎"] += (sho['腎虚'] * 3.0) + (kyo_weight * 1.0)
    p["瀉下"] += (jitsu_weight * 2.0)
    p["駆瘀血"] += (sho['瘀血'] * 3.0) + (jitsu_weight * 0.5)
    p["理気"] += sho['気鬱'] * 3.0
    p["降気"] += sho['気逆'] * 3.0
    p["利水"] += sho['水毒'] * 3.0
    p["温"] += sho['寒'] * 3.0
    p["清"] += sho['熱'] * 3.0

    # 症状の反映
    for ui_label, col_name in symptom_mapping.items():
        if raw.get(ui_label) == "あり":
            p[col_name] = 5.0

    vec = np.array([p[k]**2 for k in yakuno_cols])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

patient_vec = create_patient_vec(sho_input, raw_input)

# --- 6. 可視化 ---
@st.cache_data
def get_fixed_coords(norm_data):
    tsne = TSNE(n_components=2, perplexity=25, random_state=42, init='pca', learning_rate='auto')
    return tsne.fit_transform(norm_data)

coords = get_fixed_coords(yakuno_data_norm)
df_base['x'], df_base['y'] = coords[:, 0], coords[:, 1]

dists = euclidean_distances([patient_vec], yakuno_data_norm)[0]
near_indices = dists.argsort()[:3]
near_dists = dists[near_indices]
near_coords = coords[near_indices]
d_min = near_dists.min()
weights = np.exp(-SENSITIVITY * (near_dists - d_min) / (near_dists.max() - d_min + 1e-9))
star_x = (near_coords[:, 0] * weights).sum() / weights.sum()
star_y = (near_coords[:, 1] * weights).sum() / weights.sum()

df_base['cos_sim'] = cosine_similarity([patient_vec], yakuno_data_norm)[0]
df_base['dist_2d'] = np.sqrt((df_base['x'] - star_x)**2 + (df_base['y'] - star_y)**2)
df_base['prox_2d'] = (1 - (df_base['dist_2d'] / (df_base['dist_2d'].max() + 1e-9)))

st.subheader("🌟 推奨処方")
c_left, c_right = st.columns(2)
with c_left:
    st.write("**24次元類似度**")
    for i, (idx, row) in enumerate(df_base.sort_values('cos_sim', ascending=False).head(3).iterrows()):
        st.write(f"{i+1}. {row['formula']} ({row['cos_sim']:.1%})")
with c_right:
    st.write("**地図上の近接**")
    for i, (idx, row) in enumerate(df_base.sort_values('dist_2d').head(3).iterrows()):
        st.write(f"{i+1}. {row['formula']} ({row['prox_2d']:.1%})")

fig = px.scatter(df_base, x='x', y='y', text='formula', color='cos_sim', color_continuous_scale='Viridis', height=800)
fig.add_trace(go.Scatter(x=[star_x], y=[star_y], mode='markers+text', marker=dict(symbol='star', size=30, color='red'), text=["患者"], textposition="top center"))
fig.update_layout(xaxis=dict(range=[star_x - ZOOM_SCALE, star_x + ZOOM_SCALE]), yaxis=dict(range=[star_y - ZOOM_SCALE, star_y + ZOOM_SCALE]))
st.plotly_chart(fig, use_container_width=True)
