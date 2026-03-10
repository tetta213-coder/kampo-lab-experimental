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
    [data-testid="stHeader"] { background-color: white !important; }
    [data-testid="stSidebar"] { background-color: #f0f2f6 !important; }
    .stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp span { color: #31333F !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌿 証空間の地図：148処方の宇宙（特化型チューニング）")

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

# --- サイドバー：入力インターフェース ---
st.sidebar.header("👤 患者の病態入力")
age = st.sidebar.number_input("患者の年齢", min_value=0, max_value=120, value=40, step=1)
zoom_scale = st.sidebar.slider("ズーム倍率 (小さいほど拡大)", 5, 100, 15)

st.sidebar.subheader("1. 基本の10指標 (証)")
sho_input = {}
sho_names = ['虚実', '寒', '熱', '気虚', '気鬱', '気逆', '血虚', '瘀血', '水毒', '腎虚']
for name in sho_names:
    # 初期値を 0.0 にして、動かしたところだけが光るように変更
    sho_input[name] = st.sidebar.slider(f"{name}", 0.0, 1.0, 0.0, key=f"slider_{name}")

st.sidebar.subheader("2. 特定の随伴症状")
raw_input = {}
symptom_labels = ["安心鎮静 (不眠・不安)", "認知知能 (物忘れ)", "鎮痙 (足のつり)", "眼精疲労", "清頭目 (のぼせ・頭痛)", "排膿 (にきび)", "解毒 (かゆみ)", "疣贅 (いぼ)", "制吐・鎮嘔", "瀉下 (便秘)", "黄疸", "安胎", "通乳"]
for label in symptom_labels:
    raw_input[label] = st.sidebar.radio(f"{label}", ["なし", "あり"], index=0, horizontal=True, key=f"radio_{label}")

# --- 計算ロジック：患者ベクトル生成（シャープ化） ---
def create_patient_vec(sho, raw, age):
    p = {k: 0.0 for k in yakuno_cols}
    # 腎虚ブースト（より強力に）
    age_jinkyo_bonus = max(0, (age - 40) * 0.02)
    total_jk = min(1.0, sho['腎虚'] + age_jinkyo_bonus)
    
    # 薬能への配分
    p["補気"] = sho['気虚'] + (total_jk * 0.8)
    p["補血"] = sho['血虚'] + (total_jk * 0.8)
    p["潤水"] = (total_jk * 0.9) # 腎虚＝乾燥のイメージを強化
    p["利水"] = sho['水毒'] + (total_jk * 0.6)
    p.update({"理気": sho['気鬱'], "降気": sho['気逆'], "駆瘀血": sho['瘀血'], "温": sho['寒'], "清": sho['熱']})

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
            for k in target_keys: p[k] = 1.0 # 随伴症状は 1.0 で最大強調

    # 【重要】二乗によるシャープ化（0.1は0.01に、1.0は1.0に）
    sharp_vec = np.array([p[k]**2 for k in yakuno_cols])
    return sharp_vec

patient_vec = create_patient_vec(sho_input, raw_input, age)

# --- 地図とマッチングの計算 ---
yakuno_data = df_full[yakuno_cols].fillna(0).values
# 処方側の専門性スコア（全薬能の合計が少ないほど、特定分野のスペシャリストとみなす）
# 広く浅い処方にペナルティ、狭く深い処方にボーナスを与える
specialist_bonus = 1.0 / (np.sum(yakuno_data, axis=1) + 1.0)

# コサイン類似度に専門性ボーナスを乗算
raw_similarities = cosine_similarity([patient_vec], yakuno_data)[0]
# 専門性ボーナスを少しだけ加味（0.2の重み）
df_full['一致度'] = raw_similarities * (1.0 + specialist_bonus * 0.2)

# 正規化（0-100%に収める）
df_full['一致度'] = df_full['一致度'] / df_full['一致度'].max()

top_3 = df_full.sort_values('一致度', ascending=False).head(3)
tsne = TSNE(n_components=2, perplexity=25, random_state=42, init='pca', learning_rate='auto')
# 団子防止ジッター
coords = tsne.fit_transform(yakuno_data + np.random.normal(0, 1e-6, yakuno_data.shape))
df_full['x'], df_full['y'] = coords[:, 0], coords[:, 1]
star_x, star_y = top_3['x'].mean(), top_3['y'].mean()

# --- 表示 ---
st.subheader("🌟 推奨処方（特化型優先）")
cols = st.columns(3)
for i, (idx, row) in enumerate(top_3.iterrows()):
    cols[i].metric(f"{i+1}. {row['formula']}", f"{row['一致度']:.1%}")

st.write("---")

fig = px.scatter(df_full, x='x', y='y', text='formula', color='一致度', color_continuous_scale='Viridis', hover_name='formula', height=750)
fig.add_trace(go.Scatter(x=[star_x], y=[star_y], mode='markers+text', marker=dict(symbol='star', size=75, color='red', line=dict(width=3, color='red')), text=["あなたの現在地"], textposition="top center", textfont=dict(size=24, color='red', family="HiraKakuPro-W6"), name="現在の証"))
fig.update_traces(textposition='top center', marker=dict(size=12))
fig.update_layout(plot_bgcolor='white', xaxis=dict(visible=False, range=[star_x - zoom_scale, star_x + zoom_scale]), yaxis=dict(visible=False, range=[star_y - zoom_scale, star_y + zoom_scale]), showlegend=False, uirevision='constant', margin=dict(l=0, r=0, t=0, b=0))

st.plotly_chart(fig, use_container_width=True)
