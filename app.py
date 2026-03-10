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

st.title("🌿 証空間の地図：患者と処方の出会い")
st.write("スライダーを動かすと、あなたの証（ベクトル）が地図上の最適な処方へと吸い寄せられます。")

# 2. データの読み込み
@st.cache_data
def load_data():
    # EQ列（24次元スコア）まで含まれる統合CSVを読み込む
    return pd.read_csv("kampo_yakuno_integrated.csv")

df_full = load_data()

# --- 3. サイドバー：入力インターフェース ---
st.sidebar.header("👤 患者の病態入力")

# A. 主要10指標 (CHOWスコア相当)
st.sidebar.subheader("1. 基本の10指標 (証)")
sho_input = {}
sho_names = ['虚実', '寒', '熱', '気虚', '気鬱', '気逆', '血虚', '瘀血', '水毒', '腎虚']
for name in sho_names:
    # デフォルトは0.3（中等度）に設定
    sho_input[name] = st.sidebar.slider(f"{name}", 0.0, 1.0, 0.3)

# B. 個別13症状 (ラジオボタン)
st.sidebar.subheader("2. 特定の随伴症状")
raw_input = {}
symptom_labels = [
    "安心鎮静 (不眠・不安)", "認知知能 (物忘れ)", "鎮痙 (足のつり)", "眼精疲労", 
    "清頭目 (のぼせ・頭痛)", "排膿 (にきび)", "解毒 (かゆみ)", "疣贅 (いぼ)", 
    "制吐・鎮嘔", "瀉下 (便秘)", "黄疸", "安胎", "通乳"
]
for label in symptom_labels:
    raw_input[label] = st.sidebar.radio(f"{label}", ["なし", "あり"], index=0, horizontal=True)

# --- 4. 24次元患者ベクトルの生成ロジック ---
def create_patient_vec(sho, raw):
    # 器を定義（CSVの右端24列の順序に合わせる）
    p = {
        "補気": sho['気虚'] + (sho['腎虚'] * 0.3),
        "理気": sho['気鬱'],
        "降気": sho['気逆'],
        "補血": sho['血虚'] + (sho['腎虚'] * 0.4),
        "駆瘀血": sho['瘀血'],
        "止血": 0.0,
        "利水": sho['水毒'],
        "潤水": (sho['腎虚'] * 0.3),
        "温": sho['寒'],
        "清": sho['熱'],
        "安心鎮静": 0.0, "認知知能": 0.0, "鎮痙": 0.0, "眼精疲労": 0.0, 
        "清頭目": 0.0, "排膿": 0.0, "解毒": 0.0, "疣贅": 0.0, 
        "制吐": 0.0, "鎮嘔": 0.0, "瀉下": 0.0, "黄疸": 0.0, "安胎": 0.0, "通乳": 0.0
    }

    # ラジオボタンの「あり」を重み0.8でマッピング
    mapping = {
        "安心鎮静 (不眠・不安)": ["安心鎮静"],
        "認知知能 (物忘れ)": ["認知知能"],
        "鎮痙 (足のつり)": ["鎮痙"],
        "眼精疲労": ["眼精疲労"],
        "清頭目 (のぼせ・頭痛)": ["清頭目"],
        "排膿 (にきび)": ["排膿"],
        "解毒 (かゆみ)": ["解毒"],
        "疣贅 (いぼ)": ["疣贅"],
        "制吐・鎮嘔": ["制吐", "鎮嘔"],
        "瀉下 (便秘)": ["瀉下"],
        "黄疸": ["黄疸"],
        "安胎": ["安胎"],
        "通乳": ["通乳"]
    }

    for label, target_keys in mapping.items():
        if raw.get(label) == "あり":
            for k in target_keys:
                p[k] = 0.8

    # 代理ロジック：気逆の補正
    if p["降気"] < 0.3 and sho['気鬱'] > 0.5 and sho['熱'] > 0.5:
        p["降気"] = (sho['気鬱'] + sho['熱']) / 2
        
    return np.array(list(p.values()))

# 患者ベクトルの計算
patient_vec = create_patient_vec(sho_input, raw_input)

# --- 5. 地図（証空間）の計算 ---
# 右端24列（薬能スコア）を抽出
yakuno_cols = [
    "補気", "理気", "降気", "補血", "駆瘀血", "止血", "利水", "潤水", "温", "清", 
    "安心鎮静", "認知知能", "鎮痙", "眼精疲労", "清頭目", "排膿", "解毒", "疣贅", 
    "制吐", "鎮嘔", "瀉下", "黄疸", "安胎", "通乳"
]
yakuno_data = df_full[yakuno_cols].fillna(0)

# t-SNEによる次元圧縮（薬能空間での配置）
# 角度を固定するため random_state を 42 に固定
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
coords = tsne.fit_transform(yakuno_data)
df_full['x'], df_full['y'] = coords[:, 0], coords[:, 1]

# --- 6. マッチング（★の位置と推薦） ---
# 患者ベクトルと全処方のコサイン類似度
similarities = cosine_similarity([patient_vec], yakuno_data.values)[0]
df_full['一致度'] = similarities

# 上位3処方の座標から「★」の現在地を推定（重心計算）
top_3 = df_full.sort_values('一致度', ascending=False).head(3)
star_x = top_3['x'].mean()
star_y = top_3['y'].mean()

# --- 7. Plotlyによる地図描画 ---
fig = px.scatter(
    df_full, x='x', y='y', text='formula',
    color='一致度', color_continuous_scale='Viridis',
    hover_name='formula', height=800,
    labels={'一致度': 'マッチ度'}
)

# 患者の現在地（★）をレイヤーとして追加
fig.add_trace(go.Scatter(
    x=[star_x], y=[star_y],
    mode='markers+text',
    marker=dict(symbol='star', size=25, color='red', line=dict(width=2, color='white')),
    text=["★ あなたの現在地"],
    textposition="top center",
    name="現在の証"
))

fig.update_traces(
    textposition='top center', 
    marker=dict(size=10),
    textfont=dict(family="HiraKakuPro-W3", color="black")
)

fig.update_layout(
    plot_bgcolor='white',
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    title=dict(text="証空間マップ（薬能の類似性による配置）", font=dict(size=24))
)

st.plotly_chart(fig, use_container_width=True)

# --- 8. 結果表示 ---
st.subheader("🌟 マッチ度が高い推奨処方")
cols = st.columns(3)
for i, (idx, row) in enumerate(top_3.iterrows()):
    with cols[i]:
        st.metric(f"{i+1}. {row['formula']}", f"{row['一致度']:.1%}")
        # その処方の特徴的な薬能を上位3つ表示
        top_yakuno = row[yakuno_cols].sort_values(ascending=False).head(3).index.tolist()
        st.write(f"主な薬能: {', '.join(top_yakuno)}")
