import streamlit as st
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import plotly.express as px
import plotly.graph_objects as go
import os

# ---------------------------------------------------------
# 1. 初期設定とデータ定義
# ---------------------------------------------------------
st.set_page_config(page_title="漢方24次元マッピング", layout="wide")

SENSITIVITY = 250.0
ZOOM_SCALE = 4.0

# ユーザー提供のCSVヘッダーと完全に一致させる
YAKUNO_COLS = [
    "補気", "理気", "降気", "補血", "駆瘀血", "利水", "補腎", "温", "清", "瀉下",
    "鎮痛", "健胃・整腸", "鎮咳", "安心鎮静", "去痰", "清頭目", "止瀉", "潤燥", "発表", "鎮痙", "制吐・鎮嘔", "解毒", "解熱・消炎", "止血"
]

@st.cache_data
def load_data():
    if os.path.exists("kampo_yakuno_integrated.csv"):
        df = pd.read_csv("kampo_yakuno_integrated.csv")
        
        # カラムの存在チェック（デバッグ用）
        missing_cols = [col for col in YAKUNO_COLS if col not in df.columns]
        if missing_cols:
            st.error(f"CSVに存在しない列名があります: {missing_cols}")
            st.stop()
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
# 2. 問診UI構築
# ---------------------------------------------------------
st.title("🌿 漢方24次元マッピング 問診・解析システム")
st.write("患者の症状を入力し、最適な漢方処方を24次元空間マップ上で特定します。")

st.header("1. 症状の入力 (0:なし 〜 3:高度)")
likert_opts = {0: "0: なし", 1: "1: 軽度", 2: "2: 中等度", 3: "3: 高度"}
def likert_radio(label):
    return st.radio(label, options=list(likert_opts.keys()), format_func=lambda x: likert_opts[x], horizontal=True)

with st.expander("【寒・熱に関する症状】", expanded=True):
    cold_part = likert_radio("最も冷えを感じるところ（足など）")
    cold_sens = likert_radio("寒がり")
    hot_flush = likert_radio("のぼせや顔のほてり")
    heat_sens = likert_radio("暑がり")

with st.expander("【気・血・水に関する症状】", expanded=True):
    depressed = likert_radio("気分が憂うつになる")
    no_energy = likert_radio("気力がない")
    tired = likert_radio("疲れやすい")
    irritated = likert_radio("イライラする")
    throat_jam = likert_radio("のどがつかえる")
    chest_jam = likert_radio("胸のつまり")
    stomach_jam = likert_radio("腹が張る")
    
    appetite_opts = {0: "ない", 1: "普通", 2: "旺盛"}
    appetite_raw = st.radio("食欲", options=[0, 1, 2], format_func=lambda x: appetite_opts[x], horizontal=True)
    appetite_inv = 2 - appetite_raw 

    acne = likert_radio("にきび")
    dry_skin = likert_radio("皮膚がカサカサする")
    cramps = likert_radio("足がつる")
    mens_pain = likert_radio("月経痛")
    abd_pain = likert_radio("月経痛以外の腹痛")
    blur = likert_radio("目がかすむ")
    
    leg_pain = likert_radio("足腰膝など下半身の痛み")
    night_urine = likert_radio("夜間にトイレに立つ回数")
    stomach_rumble = likert_radio("腹がゴロゴロ鳴る")
    dizzy = likert_radio("めまい")

with st.expander("【その他の特記事項】", expanded=True):
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

# ---------------------------------------------------------
# 3. 解析処理と可視化
# ---------------------------------------------------------
if st.button("24次元ベクトルを算出してマップにプロット", type="primary"):
    vec = np.zeros(24)

    # 順序を YAKUNO_COLS (CSVの列順) に完全に一致させる
    vec[0] = (no_energy * 3.02) + (tired * 2.19) + (appetite_inv * 1.25) # 補気
    vec[1] = (depressed * 5.92) + (throat_jam * 1.77) + (chest_jam * 1.70) + (stomach_jam * 1.12) # 理気
    vec[2] = (irritated * 2.62) # 降気
    vec[3] = (dry_skin * 2.90) + (blur * 1.18) # 補血
    vec[4] = (acne * 4.77) + (mens_pain * 2.32) + (abd_pain * 2.10) # 駆瘀血
    vec[5] = (stomach_rumble * 2.15) + (dizzy * 1.12) # 利水
    vec[6] = (leg_pain * 5.42) + (night_urine * 4.72) # 補腎
    vec[7] = (cold_part * 7.92) + (cold_sens * 4.15) # 温
    vec[8] = (hot_flush * 2.16) + (heat_sens * 2.23) # 清
    vec[9] = 0 # 瀉下

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

    st.header("解析結果")
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("24次元の薬理類似度 上位3処方")
        top_cos = df_base.sort_values('cos_sim', ascending=False).head(3)
        for i, (_, row) in enumerate(top_cos.iterrows()):
            st.write(f"{i+1}. {row['formula']} ({row['cos_sim']:.1%})")
            
    with col_r:
        st.subheader("患者の24次元病態プロファイル")
        result_dict = {name: round(val, 3) for name, val in zip(YAKUNO_COLS, vec_normalized)}
        st.bar_chart(result_dict)

    st.subheader("証空間の地図（患者の最適位置）")
    fig = px.scatter(df_base, x='x', y='y', text='formula', color='cos_sim', 
                     color_continuous_scale='Viridis', height=800,
                     labels={'cos_sim': '類似度'})

    fig.update_traces(textposition='top center', marker=dict(size=10))

    fig.add_trace(go.Scatter(x=[star_x], y=[star_y], mode='markers+text', 
                             marker=dict(symbol='star', size=30, color='red', line=dict(width=2, color='DarkSlateGrey')),
                             text=["★ 患者最適位置"], textposition="bottom center",
                             name='Patient'))

    x_range = [star_x - ZOOM_SCALE, star_x + ZOOM_SCALE]
    y_range = [star_y - ZOOM_SCALE, star_y + ZOOM_SCALE]
    fig.update_layout(xaxis_range=x_range, yaxis_range=y_range, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)
