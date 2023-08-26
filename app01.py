# 訓練データとして使用する画像を読み込み、適切な形式に前処理します。
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# 汚い文字と美文字の画像を格納したディレクトリ
dir_messy = r"DL/messy"
dir_clean = r"DL/clean"

# 画像データとラベルデータを格納するリスト
images = []
labels = []

# 散らかった文字の画像を読み込み、ラベルと共にリストに追加
for file in os.listdir(dir_messy):
    if not file.endswith(('.jpg', '.jpeg', '.png')):
        continue
    image = cv2.imread(os.path.join(dir_messy, file))
    if image is None:
        print(f"Could not read image: {file}")
        continue
    image = cv2.resize(image, (128, 128))
    images.append(image)
    labels.append(0)

# 整頓された文字の画像を読み込み、ラベルと共にリストに追加
for file in os.listdir(dir_clean):
    if not file.endswith(('.jpg', '.jpeg', '.png')):
        continue
    image = cv2.imread(os.path.join(dir_clean, file))
    if image is None:
        print(f"Could not read image: {file}")
        continue
    image = cv2.resize(image, (128, 128))
    images.append(image)
    labels.append(1)

# リストをNumPy配列に変換
images = np.array(images)
labels = np.array(labels)

# データセットを訓練用とテスト用に分割
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2)


# Kerasを使って画像分類モデルを設定します。
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # 二値分類なので、出力層は1ユニットで活性化関数はsigmoidを使用

# 設定したモデルをコンパイルします。この際、損失関数、オプティマイザ、評価指標を設定します。
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# コンパイルしたモデルを使って、読み込んだ画像データを訓練します。
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 作成したモデルはmodel.save("room_model.h5")というコードで保存することができます。保存したモデルは、後でkeras.models.load_model("room_model.h5")というコードで読み込むことができます。
# そして、Streamlitを用いてWebアプリケーションを作成し、ユーザーがアップロードした画像をモデルに入力し、その結果を表示します。

import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# モデルの読み込み
model = load_model("model.h5")

st.title("Letter Cleanliness Classifier")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # 画像を配列に変換し、モデルの入力形式に合わせる
    image = np.array(image.resize((128, 128))).reshape((1, 128, 128, 3))

    # 予測
    prediction = model.predict(image)

    # 予測結果に基づいて表示するメッセージを設定
    if prediction < 0.5:
        st.write("The letter is messy!")
    else:
        st.write("The letter is clean!")

