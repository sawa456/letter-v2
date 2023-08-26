import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# モデルの読み込み
model = load_model("/Users/sawadashuichi/Desktop/Letter app/Letter app/model.h5")

st.title("あなたの字、本当に綺麗ですか？")

uploaded_file = st.file_uploader("自身の書いた文字の画像をアップロードしてください...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # 画像を配列に変換し、モデルの入力形式に合わせる
    image = np.array(image.resize((128, 128))) / 255.0  # ピクセル値をスケール
    image = image.reshape((1, 128, 128, 3))

    # 予測
    prediction = model.predict(image)

    # 予測結果の確率をパーセンテージで表示
    cleanliness_percentage = prediction[0][0] * 100

    # 予測結果に基づいて表示するメッセージと画像を設定
    if prediction < 0.3:
        st.markdown(f"# 汚い文字です！綺麗な字になれるよう練習しましょう！\n### きれい度：{cleanliness_percentage:.2f}%")    
    else:
        st.markdown(f"# とてもきれい！素晴らしい！\n### きれい度：{cleanliness_percentage:.2f}%")