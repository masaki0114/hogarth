from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np

# 保存したモデルの読み込み
model = model_from_json(open("model/hogarth_model.json").read())
# 保存した重みの読み込み
model.load_weights("model/hogarth_weight.hdf5")

categories = ["hogarth-curve","another"]

# 画像を読み込み
print("判断する画像を指定してください")
number = str(input())
img_path = "unknown-deta/" + number + ".jpeg" # 未知データの画像を指定
img = image.load_img(img_path,target_size=(250,250,3))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# ここでのxは画像の行列(255. 255. 255)のようなRGB配列
# 予測する
features = model.predict(x)
print(features) #onehot表現になっている

# 条件分岐
if features[0,0] == 1 :
	print("ホガースカーブです")

else:
	print("ホガースカーブではありません")