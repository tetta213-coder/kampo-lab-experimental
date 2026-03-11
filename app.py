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

st.title("🌿 証空間の地図：ワープ加速エディション")

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
# 地図を固定しないと座標が飛び跳ねるため、デフォルトは固定にします
engine = st.sidebar.selectbox("149番目のプロット手法", ["幾何学的射影 (地図固定)", "149番目として全計算 (地図動的)"])

# 感度のスケールを 100〜500 に大幅強化
sensitivity = st.sidebar.slider("星のワープ感度", 10.0, 500.0, 100.0)

st.sidebar.header("👤 患者の病態入力")
age = st.sidebar.number_input("患者の年齢", min_value=0, max_value=120, value=40, step=1)
zoom_scale = st.sidebar.slider("表示範囲", 2, 50, 8) # より寄り気味に

st.sidebar.subheader("1. 基本の10指標 (証)")
defaults = {'虚実': 0.5, '寒': 0.5, '熱': 0.5, '気虚': 0.3, '気鬱': 0.3, '気逆': 0.3, '血虚': 0.3, '瘀血': 0.3, '水毒': 0.3, '腎虚': 0.3}
sho_input = {}
for name in ['虚実', '寒', '熱', '気虚', '気鬱', '気逆', '血虚', '瘀血', '水毒', '腎虚']:
    sho_input[name] = st.sidebar.slider(f"{name}", 0.0, 1.0, defaults.get(name, 0.0), key=f"slider_{name}")

st.sidebar.subheader("2. 特定の随伴症状")
raw_input = {}
for label in ["安心鎮静 (不眠・不安)", "認知知能 (物忘れ)", "鎮痙 (足のつり)", "眼精疲労", "清頭目 (のぼせ・頭痛)", "排膿 (にきび)", "解毒 (かゆみ)", "疣贅 (いぼ)", "制吐・鎮嘔", "瀉下 (便秘)", "黄疸", "安胎", "通乳"]:
    raw_input[label] = st.sidebar.radio(f"{label}", ["なし", "あり"], index=0, horizontal=True, key=f"radio_{label}")

# --- 4. ベクトル生成 ---
def create_patient_vec(sho, raw, age):
    p = {k: 0.05 for k in yakuno_cols} 
    kyo, jitsu = max(0, 0.5-sho['虚実'])*2.0, max(0, sho['虚実']-0.5)*2.0
    p["補気"] += (sho['気虚']*2.0) + (kyo*0.5) # 重みをさらに強化
    p["補血"] += (sho['血虚']*2.0) + (kyo*0.5)
    p["利水"] += (sho['水毒']*2.0)
    p["駆瘀血"] += (sho['瘀血']*2.0) + (jitsu*0.5)
    p["理気"] += sho['気鬱']*2.0; p["降気"] += sho['気逆']*2.0; p["温"] += sho['寒']*2.0; p["清"] += sho['熱']*2.0
    jk = min(1.0, sho['腎虚'] + max(0, (age-40)*0.02))
    p["補気"] += jk; p["補血"] += jk; p["潤水"] += jk; p["利水"] += jk
    return np.array([p[k]**2 for k in yakuno_cols])

patient_vec = create_patient_vec(sho_input, raw_input, age)
yakuno_data = df_base[yakuno_cols].fillna(0).values

# --- 5. 座標計算 (ワープ実装) ---
if engine == "幾何学的射影 (地図固定)":
    @st.cache_data
    def get_fixed_coords(data):
        tsne = TSNE(n_components=2, perplexity=25, random_state=42, init='pca', learning_rate='auto')
        return tsne.fit_transform(data + np.random.normal(0, 1e-6, data.shape))
    coords = get_fixed_coords(yakuno_data)
    df_base['x'], df_base['y'] = coords[:, 0], coords[:, 1]
    
    # 24次元のユークリッド距離
    dists = euclidean_distances([patient_vec], yakuno_data)[0]
    
    # 【ワープの核心】超近傍3つだけに絞り、重みを指数関数で爆発させる
    near_indices = dists.argsort()[:3]
    near_dists = dists[near_indices]
    near_coords = coords[near_indices]
    
    # 最小距離を引いてから指数計算することで、1位への集中度を高める
    d_min = near_dists.min()
    weights = np.exp(-sensitivity * (near_dists - d_min) / (near_dists.max() - d_min + 1e-9))
    
    star_x = (near_coords[:, 0] * weights).sum() / weights.sum()
    star_y = (near_coords[:, 1] * weights).sum() / weights.sum()
    df_calc = df_base.copy()
else:
    # 動的モード（全再計算）
    full_data = np.vstack([yakuno_data, patient_vec])
    tsne = TSNE(n_components=2, perplexity=25, random_state=42, init='pca', learning_rate='auto')
    full_coords = tsne.fit_transform(full_data + np.random.normal(0, 1e-6, full_data.shape))
    df_calc = df_base.copy()
    df_calc['x'], df_calc['y'] = full_coords[:-1, 0], full_coords[:-1, 1]
    star_x, star_y = full_coords[-1, 0], full_coords[-1, 1]

# --- 6. ランキング ---
df_calc['cos_sim'] = cosine_similarity([patient_vec], yakuno_data)[0]
df_calc['dist_2d'] = np.sqrt((df_calc['x'] - star_x)**2 + (df_calc['y'] - star_y)**2)
df_calc['prox_2d'] = (1 - (df_calc['dist_2d'] / (df_calc['dist_2d'].max() + 1e-9)))

# --- 7. 表示 ---
st.subheader("🌟 推奨処方ランキング")
c1, c2 = st.columns(2)
top_cos = df_calc.sort_values('cos_sim', ascending=False).head(3)
top_dist = df_calc.sort_values('dist_2d', ascending=True).head(3)

with c1:
    st.write("**24Dパターン合致**")
    for i, (idx, row) in enumerate(top_cos.iterrows()):
        st.write(f"{i+1}. {row['formula']} ({row['cos_sim']:.1%})")
with c2:
    st.write("**2D地図上の近接**")
    for i, (idx, row) in enumerate(top_dist.iterrows()):
        st.write(f"{i+1}. {row['formula']} ({row['prox_2d']:.1%})")

fig = px.scatter(df_calc, x='x', y='y', text='formula', color='cos_sim', color_continuous_scale='Viridis', hover_name='formula', height=750)
fig.add_trace(go.Scatter(x=[star_x], y=[star_y], mode='markers+text', marker=dict(symbol='star', size=80, color='red', line=dict(width=3, color='black')), text=["149: 患者"], textposition="top center", textfont=dict(size=22, color='red', family="HiraKakuPro-W6")))
fig.update_traces(textposition='top center', marker=dict(size=12))
fig.update_layout(plot_bgcolor='white', xaxis=dict(visible=False, range=[star_x - zoom_scale, star_x + zoom_scale]), yaxis=dict(visible=False, range=[star_y - zoom_scale, star_y + zoom_scale]), showlegend=False, uirevision='constant', margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)
