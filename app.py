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

st.title("🌿 証空間の地図：中心部重力脱出エディション")

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
engine = st.sidebar.selectbox("149番目のプロット手法", ["幾何学的射影 (地図固定)", "149番目として全計算 (地図動的)"])

# 重力脱出のための感度（デフォルトを強めに設定）
sensitivity = st.sidebar.slider("重力脱出感度 (周辺への飛びやすさ)", 10.0, 500.0, 200.0)

st.sidebar.header("👤 患者の病態入力")
age = st.sidebar.number_input("患者の年齢", min_value=0, max_value=120, value=40, step=1)
zoom_scale = st.sidebar.slider("表示範囲", 2, 50, 8)

st.sidebar.subheader("1. 基本の10指標 (証)")
defaults = {'虚実': 0.5, '寒': 0.5, '熱': 0.5, '気虚': 0.3, '気鬱': 0.3, '気逆': 0.3, '血虚': 0.3, '瘀血': 0.3, '水毒': 0.3, '腎虚': 0.3}
sho_input = {}
for name in ['虚実', '寒', '熱', '気虚', '気鬱', '気逆', '血虚', '瘀血', '水毒', '腎虚']:
    sho_input[name] = st.sidebar.slider(f"{name}", 0.0, 1.0, defaults.get(name, 0.0), key=f"slider_{name}")

st.sidebar.subheader("2. 特定の随伴症状")
raw_input = {}
for label in ["安心鎮静 (不眠・不安)", "認知知能 (物忘れ)", "鎮痙 (足のつり)", "眼精疲労", "清頭目 (のぼせ・頭痛)", "排膿 (にきび)", "解毒 (かゆみ)", "疣贅 (いぼ)", "制吐・鎮嘔", "瀉下 (便秘)", "黄疸", "安胎", "通乳"]:
    raw_input[label] = st.sidebar.radio(f"{label}", ["なし", "あり"], index=0, horizontal=True, key=f"radio_{label}")

# --- 4. ベクトル生成（正規化処理付き） ---
def create_patient_vec(sho, raw, age):
    p = {k: 0.01 for k in yakuno_cols} # ノイズを極限までカット
    kyo, jitsu = max(0, 0.5-sho['虚実'])*2.0, max(0, sho['虚実']-0.5)*2.0
    
    # 重みを大幅に強化（特化型処方へ届かせるため）
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
            for k in target_keys: p[k] = 5.0 # 随伴症状は最優先

    vec = np.array([p[k] for k in yakuno_cols])
    # 【最重要】正規化：ベクトルの長さを1にする（これで中央の重力から脱出）
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

patient_vec = create_patient_vec(sho_input, raw_input, age)

# 処方データも正規化して比較準備
yakuno_data_raw = df_base[yakuno_cols].fillna(0).values
yakuno_norms = np.linalg.norm(yakuno_data_raw, axis=1, keepdims=True)
yakuno_data_norm = np.divide(yakuno_data_raw, yakuno_norms, out=np.zeros_like(yakuno_data_raw), where=yakuno_norms!=0)

# --- 5. 座標計算 ---
if engine == "幾何学的射影 (地図固定)":
    @st.cache_data
    def get_fixed_coords(data):
        tsne = TSNE(n_components=2, perplexity=25, random_state=42, init='pca', learning_rate='auto')
        return tsne.fit_transform(data)
    
    coords = get_fixed_coords(yakuno_data_raw) # 地図の形はオリジナルの重力バランスを維持
    df_base['x'], df_base['y'] = coords[:, 0], coords[:, 1]
    
    # 正規化された空間での距離を計算（これでランキングとリンクする）
    dists = euclidean_distances([patient_vec], yakuno_data_norm)[0]
    near_indices = dists.argsort()[:3]
    near_dists = dists[near_indices]
    near_coords = coords[near_indices]
    
    d_min = near_dists.min()
    weights = np.exp(-sensitivity * (near_dists - d_min) / (near_dists.max() - d_min + 1e-9))
    
    star_x = (near_coords[:, 0] * weights).sum() / weights.sum()
    star_y = (near_coords[:, 1] * weights).sum() / weights.sum()
    df_calc = df_base.copy()
else:
    full_data = np.vstack([yakuno_data_raw, patient_vec * 10]) # 患者をあえて巨大化させて再計算
    tsne = TSNE(n_components=2, perplexity=25, random_state=42, init='pca', learning_rate='auto')
    full_coords = tsne.fit_transform(full_data)
    df_calc = df_base.copy()
    df_calc['x'], df_calc['y'] = full_coords[:-1, 0], full_coords[:-1, 1]
    star_x, star_y = full_coords[-1, 0], full_coords[-1, 1]

# --- 6. ランキング ---
df_calc['cos_sim'] = cosine_similarity([patient_vec], yakuno_data_norm)[0]
df_calc['dist_2d'] = np.sqrt((df_calc['x'] - star_x)**2 + (df_calc['y'] - star_y)**2)
df_calc['prox_2d'] = (1 - (df_calc['dist_2d'] / (df_calc['dist_2d'].max() + 1e-9)))

# --- 7. UI表示 ---
st.subheader("🌟 推奨処方ランキング")
c1, c2 = st.columns(2)
top_cos = df_calc.sort_values('cos_sim', ascending=False).head(3)
top_dist = df_calc.sort_values('dist_2d', ascending=True).head(3)

with c1:
    st.write("**24Dパターン合致（質重視）**")
    for i, (idx, row) in enumerate(top_cos.iterrows()):
        st.write(f"{i+1}. {row['formula']} ({row['cos_sim']:.1%})")
with c2:
    st.write("**2D地図上の近接（位置重視）**")
    for i, (idx, row) in enumerate(top_dist.iterrows()):
        st.write(f"{i+1}. {row['formula']} ({row['prox_2d']:.1%})")

fig = px.scatter(df_calc, x='x', y='y', text='formula', color='cos_sim', color_continuous_scale='Viridis', hover_name='formula', height=750)
fig.add_trace(go.Scatter(x=[star_x], y=[star_y], mode='markers+text', marker=dict(symbol='star', size=80, color='red', line=dict(width=3, color='black')), text=["149: 患者"], textposition="top center", textfont=dict(size=22, color='red', family="HiraKakuPro-W6")))
fig.update_traces(textposition='top center', marker=dict(size=12))
fig.update_layout(plot_bgcolor='white', xaxis=dict(visible=False, range=[star_x - zoom_scale, star_x + zoom_scale]), yaxis=dict(visible=False, range=[star_y - zoom_scale, star_y + zoom_scale]), showlegend=False, uirevision='constant', margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(fig, use_container_width=True)
