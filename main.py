from keras import models
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np
import os
import glob
import pathlib

# 保存したモデルの読み込み
model = model_from_json(open("model/hogarth_model.json").read())
# 保存した重みの読み込み
model.load_weights("model/hogarth_weight.hdf5")

categories = ["hula","india","jawa","myan","nichibu"]

#出力ディレクトリの設定
for x in range(len(categories)):
	output_dir = os.path.join(".","R","ans",categories[x])
	if not(os.path.exists(output_dir)):
		os.mkdir(output_dir)
	#画像の読み込み
	images = glob.glob(os.path.join(".","R",categories[x],"*.png"))
	# print(images)

	with open(os.path.join(output_dir,"ans.txt"),mode="w") as f:
		for y in range(len(images)):
			l = []
			img_path = images[y]
			img = image.load_img(img_path,target_size=(250,250,3))
			img = image.img_to_array(img)
			img = np.expand_dims(img, axis=0)
			features = model.predict(img)	#featuresは２次元配列
			for z in range(7):
				l.append(features[0][z])
			f.write(img_path+'\n')
			l = [str(i) for i in l]
			f.write('\n'.join(l) + '\n')
			message = '最も近いのは'+str(np.argmax(features)+1)+'番目のホガースカーブ'
			f.write(message + '\n')

