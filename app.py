import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
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

st.title("🌿 証空間の地図：プロフェッショナル・エディション")

# --- データの読み込みと座標の固定計算 ---
@st.cache_data
def load_fixed_data():
    df = pd.read_csv("kampo_yakuno_integrated.csv")
    df = df[df['No'] <= 148].copy()
    
    yakuno_cols = [
        "補気", "理気", "降気", "補血", "駆瘀血", "止血", "利水", "潤水", "温", "清", 
        "安心鎮静", "認知知能", "鎮痙", "眼精疲労", "清頭目", "排膿", "解毒", "疣贅", 
        "制吐", "鎮嘔", "瀉下", "黄疸", "安胎", "通乳"
    ]
    
    yakuno_data = df[yakuno_cols].fillna(0).values
    # 地図の座標は常に固定（キャッシュ）
    tsne = TSNE(n_components=2, perplexity=25, random_state=42, init='pca', learning_rate='auto')
    coords = tsne.fit_transform(yakuno_data + np.random.normal(0, 1e-6, yakuno_data.shape))
    df['x'], df['y'] = coords[:, 0], coords[:, 1]
    return df, yakuno_cols

df_full, yakuno_cols = load_fixed_data()

# --- サイドバー：入力インターフェース ---
st.sidebar.header("👤 患者の病態入力")
age = st.sidebar.number_input("患者の年齢", min_value=0, max_value=120, value=40, step=1)
zoom_scale = st.sidebar.slider("表示範囲 (小さいほど拡大)", 5, 100, 20)

st.sidebar.subheader("1. 基本の10指標 (証)")
sho_input = {}
sho_names = ['虚実', '寒', '熱', '気虚', '気鬱', '気逆', '血虚', '瘀血', '水毒', '腎虚']
for name in sho_names:
    # 初期値は 0 で良いですが、計算上は微小な値を持たせます
    sho_input[name] = st.sidebar.slider(f"{name}", 0.0, 1.0, 0.0, key=f"slider_{name}")

st.sidebar.subheader("2. 特定の随伴症状")
raw_input = {}
symptom_labels = ["安心鎮静 (不眠・不安)", "認知知能 (物忘れ)", "鎮痙 (足のつり)", "眼精疲労", "清頭目 (のぼせ・頭痛)", "排膿 (にきび)", "解毒 (かゆみ)", "疣贅 (いぼ)", "制吐・鎮嘔", "瀉下 (便秘)", "黄疸", "安胎", "通乳"]
for label in symptom_labels:
    raw_input[label] = st.sidebar.radio(f"{label}", ["なし", "あり"], index=0, horizontal=True, key=f"radio_{label}")

# --- 4. 計算ロジック ---
def create_patient_vec(sho, raw, age):
    # 【重要】ベースラインを 0.1 に設定し、何も入力しなくても「平均的な位置」からスタート
    p = {k: 0.1 for k in yakuno_cols} 
    
    age_jinkyo_bonus = max(0, (age - 40) * 0.02)
    total_jk = min(1.0, sho['腎虚'] + age_jinkyo_bonus)
    
    p["補気"] = sho['気虚'] + (total_jk * 0.8)
    p["補血"] = sho['血虚'] + (total_jk * 0.8)
    p["潤水"] = (total_jk * 0.9)
    p["利水"] = sho['水毒'] + (total_jk * 0.6)
    p.update({"理気": sho['気鬱'], "降気": sho['気逆'], "駆瘀血": sho['瘀血'], "温": sho['寒'], "清": sho['熱']})

    mapping = {
        "安心鎮静 (不眠・不安)": ["安心鎮静"], "認知知能 (物忘れ)": ["認知知能"], "鎮痙 (足のつり)": ["鎮痙"], 
        "眼精疲労": ["眼精疲労"], "清頭目 (のぼせ・頭痛)": ["清頭目"], "排膿 (にきび)": ["排膿"], 
        "解毒 (かゆみ)": ["解毒"], "疣贅 (いぼ)": ["疣贅"], "制吐・鎮嘔": ["制吐", "鎮嘔"], 
        "瀉下 (便秘)": ["瀉下"], "黄疸": ["黄疸"], "安胎": ["安胎"], "通乳": ["通乳"]
    }
    for label, target_keys in mapping.items():
        if raw.get(label) == "あり":
            for k in target_keys: p[k] = 1.0

    return np.array([p[k]**2 for k in yakuno_cols])

patient_vec = create_patient_vec(sho_input, raw_input, age)
yakuno_data = df_full[yakuno_cols].fillna(0).values

# マッチング計算
df_full['raw_sim'] = cosine_similarity([patient_vec], yakuno_data)[0]
spec_bonus = 1.0 / (np.sum(yakuno_data, axis=1) + 1.0)
df_full['一致度'] = df_full['raw_sim'] * (1.0 + spec_bonus * 0.2)

# 【重要】色の正規化：今の入力に対して「相対的」に色を付ける
# これにより、数値が小さくても「一番近いもの」が黄色くなります
sim_min = df_full['一致度'].min()
sim_max = df_full['一致度'].max()
df_full['表示色'] = (df_full['一致度'] - sim_min) / (sim_max - sim_min + 1e-9)

top_3 = df_full.sort_values('一致度', ascending=False).head(3)
star_x, star_y = top_3['x'].mean(), top_3['y'].mean()

# --- 5. 表示 ---
st.subheader("🌟 推奨処方（特化型優先）")
cols = st.columns(3)
for i, (idx, row) in enumerate(top_3.iterrows()):
    # 実際の一致度（コサイン類似度）をパーセント表示
    cols[i].metric(f"{i+1}. {row['formula']}", f"{row['raw_sim']:.1%}")

st.write("---")

# 地図描画
fig = px.scatter(
    df_full, x='x', y='y', text='formula', 
    color='表示色', color_continuous_scale='Viridis', # 相対的な色付け
    hover_name='formula', height=700
)

# あなたの現在地（★）
fig.add_trace(go.Scatter(
    x=[star_x], y=[star_y], mode='markers+text',
    marker=dict(symbol='star', size=70, color='red', line=dict(width=3, color='red')),
    text=["あなたの現在地"], textposition="top center",
    textfont=dict(size=22, color='red', family="HiraKakuPro-W6"),
    name="現在の証"
))

fig.update_traces(textposition='top center', marker=dict(size=10))
fig.update_layout(
    plot_bgcolor='white',
    xaxis=dict(visible=False, range=[star_x - zoom_scale, star_x + zoom_scale]),
    yaxis=dict(visible=False, range=[star_y - zoom_scale, star_y + zoom_scale]),
    showlegend=False, uirevision='constant', margin=dict(l=10, r=10, t=10, b=10)
)

st.plotly_chart(fig, use_container_width=True)
