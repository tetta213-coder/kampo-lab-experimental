import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
import plotly.express as px
import plotly.graph_objects as go
import os

# 1. ページ基本設定（オリジナルUI維持）
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

# 24次元カテゴリー定義（CSVの列名と完全一致）
YAKUNO_COLS = [
    "補気", "理気", "降気", "補血", "駆瘀血", "利水", "補腎", "温", "清", "瀉下",
    "鎮痛", "健胃・整腸", "鎮咳", "安心鎮静", "去痰", "清頭目", "止瀉", "潤燥", "発表", "鎮痙", "制吐・鎮嘔", "解毒", "解熱・消炎", "止血"
]

@st.cache_data
def load_data():
    if os.path.exists("kampo_yakuno_integrated.csv"):
        df = pd.read_csv("kampo_yakuno_integrated.csv")
    else:
        st.warning("kampo_yakuno_integrated.csv が見つかりません。ダミーデータを使用します。")
        np.random.seed(42)
        dummy_data = np.random.randint(0, 3, size=(148, 24))
        df = pd.DataFrame(dummy_data, columns=YAKUNO_COLS)
        df.insert(0, 'formula', [f"処方{i}" for i in range(1, 149)])
    
    vecs = df[YAKUNO_COLS].values
    vecs_norm = normalize(vecs, norm='l2')
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    coords = tsne.fit_transform(vecs_norm)
    df['x'] = coords[:, 0]
    df['y'] = coords[:, 1]
    
    return df, vecs_norm

df_base, formula_vecs_norm = load_data()

# ---------------------------------------------------------
# 3. サイドバー：問診UI構築
# ---------------------------------------------------------
with st.sidebar:
    st.header("📝 漢方問診入力")
    
    likert_opts = {0: "0: なし", 1: "1: 軽度", 2: "2: 中等度", 3: "3: 高度"}
    def likert_radio(label):
        return st.radio(label, options=list(likert_opts.keys()), format_func=lambda x: likert_opts[x], horizontal=True)

    with st.expander("1. 基本情報", expanded=True):
        age = st.number_input("年齢", min_value=0, max_value=120, value=40)
        sex = st.radio("性別", options=["男性", "女性"], index=1, horizontal=True)
        # 修正箇所1：身長・体重を整数(int)に変更
        height = st.number_input("身長 (cm)", min_value=50, max_value=250, value=160)
        weight = st.number_input("体重 (kg)", min_value=10, max_value=200, value=55)
        
        # BMI自動計算
        height_m = height / 100.0
        bmi = weight / (height_m ** 2) if height_m > 0 else 22.0
        st.info(f"算出BMI: **{bmi:.1f}**")
        
        # 血圧も復活させておきます
        sbp = st.number_input("収縮期血圧", value=120)
        dbp = st.number_input("拡張期血圧", value=80)

    with st.expander("2. 寒・熱に関する症状"):
        cold_part = likert_radio("体の冷え（手足など）")
        cold_sens = likert_radio("寒がり")
        hot_flush = likert_radio("のぼせや顔のほてり")
        heat_sens = likert_radio("暑がり")

    with st.expander("3. 気・血・水に関する症状"):
        depressed = likert_radio("気分が憂うつになる")
        no_energy = likert_radio("気力がない")
        tired = likert_radio("疲れやすい")
        irritated = likert_radio("イライラする")
        throat_jam = likert_radio("のどがつかえる")
        chest_jam = likert_radio("胸のつまり")
        stomach_jam = likert_radio("腹が張る")
        
        appetite_opts = {0: "ない", 1: "普通", 2: "旺盛"}
        # 修正箇所2：食欲のデフォルトを「普通」にするため index=1 を追加
        appetite_raw = st.radio("食欲", options=[0, 1, 2], index=1, format_func=lambda x: appetite_opts[x], horizontal=True)
        appetite_inv = 2 - appetite_raw 

        acne = likert_radio("にきび")
        dry_skin = likert_radio("皮膚がカサカサする")
        cramps = likert_radio("足がつる")
        mens_pain = likert_radio("月経痛")
        abd_pain = likert_radio("月経痛以外の腹痛")
        blur = likert_radio("目がかすむ")
        
        leg_pain = likert_radio("足腰膝など下半身の痛み")
        
        # 修正箇所3：夜間にトイレに立つ回数の専用ラベルを作成
        urine_opts = {0: "0回", 1: "1回", 2: "2回", 3: "3回以上"}
        night_urine = st.radio("夜間にトイレに立つ回数", options=list(urine_opts.keys()), format_func=lambda x: urine_opts[x], horizontal=True)
        
        stomach_rumble = likert_radio("腹がゴロゴロ鳴る")
        dizzy = likert_radio("めまい")

    with st.expander("4. その他の特記事項"):
        diarrhea = likert_radio("大便が軟らかい・下痢")
        blood_stool = likert_radio("大便に血が混じる")
        sub_bleed = likert_radio("皮下出血")
        hemorrhoid = likert_radio("痔がある")
        cough = likert_radio("咳")
        sputum = likert_radio("痰")
        throat_pain = likert_radio("のどが痛む")
        nausea = likert_radio("吐き気、嘔吐")
        insomnia = likert_radio("眠れない")
        palpitation = likert_radio("動悸")
        headache = likert_radio("頭痛")
        stiff_shoulder = likert_radio("肩こり")

    calc_button = st.button("ベクトルを正規化してマップにプロット", type="primary", use_container_width=True)

# ---------------------------------------------------------
# 4. メインエリア：解析処理と可視化（オリジナルUI完全維持）
# ---------------------------------------------------------
if calc_button:
    vec = np.zeros(24)

    # 10の証と14の症状の計算ロジック
    vec[0] = (no_energy * 3.02) + (tired * 2.19) + (appetite_inv * 1.25) # 補気
    vec[1] = (depressed * 5.92) + (throat_jam * 1.77) + (chest_jam * 1.70) + (stomach_jam * 1.12) # 理気
    vec[2] = (irritated * 2.62) # 降気
    vec[3] = (dry_skin * 2.90) + (blur * 1.18) # 補血
    vec[4] = (acne * 4.77) + (mens_pain * 2.32) + (abd_pain * 2.10) # 駆瘀血
    vec[5] = (stomach_rumble * 2.15) + (dizzy * 1.12) # 利水
    vec[6] = (leg_pain * 5.42) + (night_urine * 4.72) # 補腎
    vec[7] = (cold_part * 7.92) + (cold_sens * 4.15) # 温
    vec[8] = (hot_flush * 2.16) + (heat_sens * 2.23) # 清
    vec[9] = 0 # 瀉下（必要に応じて後から追加可能）

    vec[10] = max(headache, stiff_shoulder, mens_pain, abd_pain, leg_pain) # 鎮痛
    vec[11] = max(appetite_inv, stomach_jam) # 健胃・整腸
    vec[12] = cough # 鎮咳
    vec[13] = max(insomnia, palpitation) # 安心鎮静
    vec[14] = sputum # 去痰
    vec[15] = headache # 清頭目
    vec[16] = diarrhea # 止瀉
    vec[17] = dry_skin # 潤燥
    vec[18] = cold_sens # 発表
    vec[19] = cramps # 鎮痙
    vec[20] = nausea # 制吐・鎮嘔
    vec[21] = acne # 解毒
    vec[22] = throat_pain # 解熱・消炎
    vec[23] = max(blood_stool, sub_bleed, hemorrhoid) # 止血

    # L2正規化
    norm = np.linalg.norm(vec)
    vec_normalized = vec / norm if norm > 0 else vec

    # コサイン類似度の計算
    cos_sim = cosine_similarity([vec_normalized], formula_vecs_norm)[0]
    df_base['cos_sim'] = cos_sim

    # マップ上の座標計算（ガウスカーネル加重平均）
    top3_idx = np.argsort(cos_sim)[::-1][:3]
    top3_similarities = cos_sim[top3_idx]
    top3_coords = df_base.iloc[top3_idx][['x', 'y']].values

    weights = np.exp(SENSITIVITY * top3_similarities)
    weights /= weights.sum()

    star_x = np.sum(top3_coords[:, 0] * weights)
    star_y = np.sum(top3_coords[:, 1] * weights)

    df_base['dist_2d'] = np.sqrt((df_base['x'] - star_x)**2 + (df_base['y'] - star_y)**2)
    max_dist = df_base['dist_2d'].max()
    df_base['prox_2d'] = 1.0 - (df_base['dist_2d'] / max_dist)

    # オリジナルのアウトプットUI
    st.write("---")
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

    # オリジナルの地図描画UI
    fig = px.scatter(df_base, x='x', y='y', text='formula', color='cos_sim', 
                     color_continuous_scale='Viridis', height=800,
                     labels={'cos_sim': '類似度'})

    # 処方データのラベルと点サイズを先に固定
    fig.update_traces(textposition='top center', marker=dict(size=10))

    # 患者を示す星を巨大に設定
    fig.add_trace(go.Scatter(x=[star_x], y=[star_y], mode='markers+text', 
                             marker=dict(symbol='star', size=30, color='red', line=dict(width=2, color='DarkSlateGrey')),
                             text=["★ 患者最適位置"], textposition="bottom center",
                             name='Patient'))

    x_range = [star_x - ZOOM_SCALE, star_x + ZOOM_SCALE]
    y_range = [star_y - ZOOM_SCALE, star_y + ZOOM_SCALE]
    fig.update_layout(xaxis_range=x_range, yaxis_range=y_range, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("👈 左のサイドバーから患者の症状を入力し、「ベクトルを正規化してマップにプロット」を押してください。")
