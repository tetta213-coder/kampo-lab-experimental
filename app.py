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
    </style>
    """, unsafe_allow_html=True)

st.title("🌿 証空間の地図：デュアル・メトリクス・エディション")

# --- 2. データの読み込み ---
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

# --- 3. サイドバー設定 ---
st.sidebar.header("⚙️ 計算エンジン設定")
engine = st.sidebar.selectbox(
    "149番目のプロット手法",
    ["幾何学的射影 (地図固定)", "149番目として全計算 (地図動的)"]
)

st.sidebar.header("👤 患者の病態入力")
age = st.sidebar.number_input("患者の年齢", min_value=0, max_value=120, value=40, step=1)

# 【ここを 10 に修正！】初期値を小さくして最初から寄った状態にします
zoom_scale = st.sidebar.slider("表示範囲", 5, 100, 10)

st.sidebar.subheader("1. 基本の10指標 (証)")
defaults = {
    '虚実': 0.5, '寒': 0.5, '熱': 0.5,
    '気虚': 0.3, '気鬱': 0.3, '気逆': 0.3,
    '血虚': 0.3, '瘀血': 0.3, '水毒': 0.3, '腎虚': 0.3
}

sho_input = {}
sho_names = ['虚実', '寒', '熱', '気虚', '気鬱', '気逆', '血虚', '瘀血', '水毒', '腎虚']
for name in sho_names:
    val = defaults.get(name, 0.0)
    sho_input[name] = st.sidebar.slider(f"{name}", 0.0, 1.0, val, key=f"slider_{name}")

st.sidebar.subheader("2. 特定の随伴症状")
raw_input = {}
symptom_labels = ["安心鎮静 (不眠・不安)", "認知知能 (物忘れ)", "鎮痙 (足のつり)", "眼精疲労", "清頭目 (のぼせ・頭痛)", "排膿 (にきび)", "解毒 (かゆみ)", "疣贅 (いぼ)", "制吐・鎮嘔", "瀉下 (便秘)", "黄疸", "安胎", "通乳"]
for label in symptom_labels:
    raw_input[label] = st.sidebar.radio(f"{label}", ["なし", "あり"], index=0, horizontal=True, key=f"radio_{label}")

# --- 4. ベクトル生成 ---
def create_patient_vec(sho, raw, age):
    p = {k: 0.05 for k in yakuno_cols} 
    kyo, jitsu = max(0, 0.5-sho['虚実'])*2.0, max(0, sho['虚実']-0.5)*2.0
    p["補気"] += (sho['気虚']*1.5) + (kyo*0.3)
    p["補血"] += (sho['血虚']*1.5) + (kyo*0.3)
    p["利水"] += (sho['水毒']*1.5)
    p["潤水"] += (sho['血虚']*0.5) + (kyo*0.2)
    p["駆瘀血"] += (sho['瘀血']*1.5) + (jitsu*0.3)
    p["瀉下"] += (jitsu*0.5)
    p["理気"] += sho['気鬱']*1.5; p["降気"] += sho['気逆']*1.5; p["温"] += sho['寒']*1.5; p["清"] += sho['熱']*1.5
    jk = min(1.0, sho['腎虚'] + max(0, (age-40)*0.02))
    p["補気"] += jk*0.5; p["補血"] += jk*0.5; p["潤水"] += jk*0.8; p["利水"] += jk*0.4
    mapping = {"安心鎮静 (不眠・不安)": ["安心鎮静"], "認知知能 (物忘れ)": ["認知知能"], "鎮痙 (足のつり)": ["鎮痙"], "眼精疲労": ["眼精疲労"], "清頭目 (のぼせ・頭痛)": ["清頭目"], "排膿 (にきび)": ["排膿"], "解毒 (かゆみ)": ["解毒"], "疣贅 (いぼ)": ["疣贅"], "制吐・鎮嘔": ["制吐", "鎮嘔"], "瀉下 (便秘)": ["瀉下"], "黄疸": ["黄疸"], "安胎": ["安胎"], "通乳": ["通乳"]}
    for label, target_keys in mapping.items():
        if raw.get(label) == "あり":
            for k in target_keys: p[k] = 1.2
    return np.array([p[k]**2 for k in yakuno_cols])

patient_vec = create_patient_vec(sho_input, raw_input, age)
yakuno_data = df_base[yakuno_cols].fillna(0).values

# --- 5. 座標計算 ---
if engine == "幾何学的射影 (地図固定)":
    @st.cache_data
    def get_fixed_coords(data):
        tsne = TSNE(n_components=2, perplexity=25, random_state=42, init='pca', learning_rate='auto')
        return tsne.fit_transform(data + np.random.normal(0, 1e-6, data.shape))
    coords = get_fixed_coords(yakuno_data)
    df_base['x'], df_base['y'] = coords[:, 0], coords[:, 1]
    dists = euclidean_distances([patient_vec], yakuno_data)[0]
    sigma = np.percentile(dists, 10) + 1e-9
    geo_weights = np.exp(- (dists**2) / (2 * sigma**2))
    star_x = (df_base['x'] * geo_weights).sum() / geo_weights.sum()
    star_y = (df_base['y'] * geo_weights).sum() / geo_weights.sum()
    df_calc = df_base.copy()
else:
    full_data = np.vstack([yakuno_data, patient_vec])
    tsne = TSNE(n_components=2, perplexity=25, random_state=42, init='pca', learning_rate='auto')
    full_coords = tsne.fit_transform(full_data + np.random.normal(0, 1e-6, full_data.shape))
    df_calc = df_base.copy()
    df_calc['x'], df_calc['y'] = full_coords[:-1, 0], full_coords[:-1, 1]
    star_x, star_y = full_coords[-1, 0], full_coords[-1, 1]

# --- 6. ランキング計算 ---
df_calc['cos_sim'] = cosine_similarity([patient_vec], yakuno_data)[0]
df_calc['dist_2d'] = np.sqrt((df_calc['x'] - star_x)**2 + (df_calc['y'] - star_y)**2)
max_d = df_calc['dist_2d'].max()
df_calc['prox_2d'] = (1 - (df_calc['dist_2d'] / (max_d + 1e-9)))

# --- 7. UI表示 ---
st.subheader("🌟 推奨処方ランキング（比較表示）")
tab_cos, tab_dist = st.tabs(["24Dパターンの合致 (Cosine Sim)", "2D地図上の近接度 (Euclidean Dist)"])

with tab_cos:
    top_cos = df_calc.sort_values('cos_sim', ascending=False).head(3)
    cols = st.columns(3)
    for i, (idx, row) in enumerate(top_cos.iterrows()):
        cols[i].metric(f"{i+1}. {row['formula']}", f"{row['cos_sim']:.1%}", "Cosine Similarity")

with tab_dist:
    top_dist = df_calc.sort_values('dist_2d', ascending=True).head(3)
    cols = st.columns(3)
    for i, (idx, row) in enumerate(top_dist.iterrows()):
        cols[i].metric(f"{i+1}. {row['formula']}", f"{row['prox_2d']:.1%}", f"Dist: {row['dist_2d']:.2f}")

st.write("---")

sim_min, sim_max = df_calc['cos_sim'].min(), df_calc['cos_sim'].max()
df_calc['表示色'] = (df_calc['cos_sim'] - sim_min) / (sim_max - sim_min + 1e-9)

fig = px.scatter(df_calc, x='x', y='y', text='formula', color='表示色', color_continuous_scale='Viridis', hover_name='formula', height=700)
fig.add_trace(go.Scatter(x=[star_x], y=[star_y], mode='markers+text', marker=dict(symbol='star', size=75, color='red', line=dict(width=3, color='red')), text=["149: 患者"], textposition="top center", textfont=dict(size=22, color='red', family="HiraKakuPro-W6"), name="現在の証"))
fig.update_traces(textposition='top center', marker=dict(size=11))
fig.update_layout(plot_bgcolor='white', xaxis=dict(visible=False, range=[star_x - zoom_scale, star_x + zoom_scale]), yaxis=dict(visible=False, range=[star_y - zoom_scale, star_y + zoom_scale]), showlegend=False, uirevision='constant', margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)
