import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
import plotly.express as px
import plotly.graph_objects as go
import os

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

# 最新の24次元カテゴリー定義（証10 + 症状14）
YAKUNO_COLS = [
    "補気", "理気", "降気", "補血", "駆瘀血", "利水", "補腎", "温", "清", "瀉下",
    "鎮痛", "健胃・整腸", "鎮咳", "安心鎮静", "去痰", "清頭目", "止瀉", 
    "潤燥", "発表", "鎮痙", "制吐・鎮嘔", "解毒", "解熱・消炎", "止血"
]

# --- 3. データの読み込み ---
@st.cache_data
def load_and_normalize_data():
    # 吉野先生がアップロードされた統合済みファイル名
    source_file = "kampo_yakuno_integrated.csv"
    
    if not os.path.exists(source_file):
        st.error(f"ファイル '{source_file}' が見つかりません。")
        st.stop()
        
    df = pd.read_csv(source_file)
    # 148処方に限定
    df = df[df['No'] <= 148].copy()
    
    # データの正規化
    raw_values = df[YAKUNO_COLS].fillna(0).values
    normalized_values = normalize(raw_values, norm='l2')
    return df, YAKUNO_COLS, raw_values, normalized_values

df_base, yakuno_cols, yakuno_data_raw, yakuno_data_norm = load_and_normalize_data()

# --- 4. サイドバー：患者入力 ---
st.sidebar.header("👤 患者の病態入力")

st.sidebar.subheader("1. 証の10指標")
sho_input = {}
# 証の10項目をスライダーで設定
sho_names = ['虚実', '寒', '熱', '気虚', '気鬱', '気逆', '血虚', '瘀血', '水毒', '腎虚']
for name in sho_names:
    # 寒熱・虚実は0.5（中庸）をデフォルトに
    default_val = 0.5 if name in ['虚実', '寒', '熱'] else 0.1
    sho_input[name] = st.sidebar.slider(name, 0.0, 1.0, default_val, key=f"slider_{name}")

st.sidebar.subheader("2. 随伴症状 (14項目)")
raw_input = {}
# 症状14項目とCSVのカラムをマッピング
symptom_mapping = {
    "痛みの改善 (鎮痛・痺・疝)": "鎮痛",
    "胃腸の調整 (健胃・整腸)": "健胃・整腸",
    "咳を鎮める (鎮咳)": "鎮咳",
    "不眠・不安の改善 (安心鎮静)": "安心鎮静",
    "痰の改善 (去痰)": "去痰",
    "のぼせ・目の充血 (清頭目)": "清頭目",
    "下痢を止める (止瀉)": "止瀉",
    "粘膜の乾燥を潤す (潤燥)": "潤燥",
    "風邪の初期・袪風 (発表)": "発表",
    "足のつり・痙攣 (鎮痙)": "鎮痙",
    "吐き気 (制吐・鎮嘔)": "制吐・鎮嘔",
    "皮膚炎・かゆみ (解毒)": "解毒",
    "発熱・急性炎症 (解熱・消炎)": "解熱・消炎",
    "出血を止める (止血)": "止血"
}

for label in symptom_mapping.keys():
    raw_input[label] = st.sidebar.radio(label, ["なし", "あり"], index=0, horizontal=True)

# --- 5. 計算ロジック ---
def create_patient_vec(sho, raw):
    # ベースの小さな値（ノイズ床）
    p = {k: 0.001 for k in YAKUNO_COLS}
    
    # 虚実バランスの計算
    kyo_weight = max(0, 0.5 - sho['虚実']) * 4.0
    jitsu_weight = max(0, sho['虚実'] - 0.5) * 4.0
    
    # 証（10次元）の射影
    p["補気"] += (sho['気虚'] * 3.0) + (kyo_weight * 1.0)
    p["補血"] += (sho['血虚'] * 3.0) + (kyo_weight * 1.0)
    p["補腎"] += (sho['腎虚'] * 3.0) + (kyo_weight * 1.0)
    
    p["瀉下"] += (jitsu_weight * 2.5)
    p["駆瘀血"] += (sho['瘀血'] * 3.0) + (jitsu_weight * 0.5)
    
    p["理気"] += sho['気鬱'] * 3.0
    p["降気"] += sho['気逆'] * 3.0
    p["利水"] += sho['水毒'] * 3.0
    p["温"] += sho['寒'] * 3.0
    p["清"] += sho['熱'] * 3.0

    # 症状（14次元）の射影
    for ui_label, col_name in symptom_mapping.items():
        if raw.get(ui_label) == "あり":
            p[col_name] = 5.0 # 確信的なフラグとして強力に設定

    # 特徴を際立たせるため二乗処理後にL2正規化
    vec = np.array([p[k]**2 for k in YAKUNO_COLS])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

patient_vec = create_patient_vec(sho_input, raw_input)

# --- 6. 座標計算と可視化 ---
@st.cache_data
def get_fixed_coords(norm_data):
    tsne = TSNE(n_components=2, perplexity=25, random_state=42, init='pca', learning_rate='auto')
    return tsne.fit_transform(norm_data)

coords = get_fixed_coords(yakuno_data_norm)
df_base['x'], df_base['y'] = coords[:, 0], coords[:, 1]

# 最近傍3処方の座標を用いた患者位置の算出
dists = euclidean_distances([patient_vec], yakuno_data_norm)[0]
near_indices = dists.argsort()[:3]
near_dists = dists[near_indices]
near_coords = coords[near_indices]
d_min = near_dists.min()
weights = np.exp(-SENSITIVITY * (near_dists - d_min) / (near_dists.max() - d_min + 1e-9))
star_x = (near_coords[:, 0] * weights).sum() / weights.sum()
star_y = (near_coords[:, 1] * weights).sum() / weights.sum()

# 類似度の算出
df_base['cos_sim'] = cosine_similarity([patient_vec], yakuno_data_norm)[0]
df_base['dist_2d'] = np.sqrt((df_base['x'] - star_x)**2 + (df_base['y'] - star_y)**2)
df_base['prox_2d'] = (1 - (df_base['dist_2d'] / (df_base['dist_2d'].max() + 1e-9)))

# 結果の表示
st.subheader("🌟 推奨処方の解析")
c_left, c_right = st.columns(2)
with c_left:
    st.write("**24次元の薬理類似度**")
    top_cos = df_base.sort_values('cos_sim', ascending=False).head(3)
    for i, (idx, row) in enumerate(top_cos.iterrows()):
        st.write(f"{i+1}. {row['formula']} ({row['cos_sim']:.1%})")
with c_right:
    st.write("**地図上の空間的近接**")
    top_dist = df_base.sort_values('dist_2d', ascending=True).head(3)
    for i, (idx, row) in enumerate(top_dist.iterrows()):
        st.write(f"{i+1}. {row['formula']} ({row['prox_2d']:.1%})")

st.write("---")

# 地図の描画
fig = px.scatter(df_base, x='x', y='y', text='formula', color='cos_sim', 
                 color_continuous_scale='Viridis', height=800,
                 labels={'cos_sim': '類似度'})

fig.add_trace(go.Scatter(x=[star_x], y=[star_y], mode='markers+text', 
                         marker=dict(symbol='star', size=35, color='red', line=dict(width=2, color='white')),
                         text=["患者"], textposition="top center", 
                         textfont=dict(size=18, color='red')))

fig.update_traces(textposition='top center', marker=dict(size=10))
fig.update_layout(plot_bgcolor='white', 
                  xaxis=dict(visible=False, range=[star_x - ZOOM_SCALE, star_x + ZOOM_SCALE]), 
                  yaxis=dict(visible=False, range=[star_y - ZOOM_SCALE, star_y + ZOOM_SCALE]),
                  margin=dict(l=0, r=0, t=0, b=0))

st.plotly_chart(fig, use_container_width=True)
