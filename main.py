import os
import random
from tkinter import filedialog as tk

import cv2 as cv
import kaggle as kg
import numpy as np
import seaborn as sns
from keras import Sequential
from keras.applications import MobileNetV2
from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.losses import CategoricalCrossentropy
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import image as mlt, pyplot as plt

def kaggle_downloader():
	"""
	Tải dữ liệu từ thư viện Kaggle
	:return:
	"""
	if os.path.exists('datasets'):
		return
	# Xác thực thông tin đăng nhập
	kg.api.authenticate()
	# Tải tập dữ liệu ảnh
	kg.api.dataset_download_files(
		dataset='ashishjangra27/face-mask-12k-images-dataset',
		path='datasets/',
		unzip=True,
		quiet=False
	)
	# Tải mô hình phát hiện khuôn mặt
	kg.api.dataset_download_files(
		dataset='gpreda/haar-cascades-for-face-detection',
		path='datasets/face_detection/',
		unzip=True,
		quiet=False
	)

kaggle_downloader()
# Thiết lập đường dẫn chung
train_dir = 'datasets/Face Mask Dataset/Train/'
test_dir = 'datasets/Face Mask Dataset/Test/'
face_detect_model = 'datasets/face_detection/haarcascade_frontalface_default.xml'
save_dir = 'model/'

def check_files(dir_path):
	"""
	Kiểm tra tệp trong thư mục
	:param dir_path: Đường dẫn thư mục cha
	:return: Đếm số lượng tệp & thư mục con trong thư mục cha
	"""
	for dir_root, dir_sub, file_names in os.walk(dir_path):
		print(f'Có {len(file_names)} tệp và {len(dir_sub)} thư mục con trong {dir_root}')

print('Kiểm tra tập ảnh trong thư mục Train')
check_files(train_dir)
# Tạo nhãn cho bộ dữ liệu
class_names = sorted(os.listdir(train_dir))
print('Nhãn của tập dữ liệu\n', class_names)

def check_imbalance(dir_path):
	"""
	Kiểm tra độ cân bằng dữ liệu
	:param dir_path: Đường dẫn thư mục
	:return: Đồ thị thống kê & tỉ lệ giữa các nhãn
	"""
	x = class_names
	y = []
	for i in range(len(class_names)):
		path = dir_path + class_names[i]
		count = len(os.listdir(path))
		y.append(count)
	plt.figure()
	sns.barplot(x=x, y=y)
	plt.xlabel('Nhãn')
	plt.ylabel('Số lượng ảnh')
	plt.savefig(save_dir + 'check_imbalance.jpg')
	plt.show()
	v_max = max(y)
	for i in range(len(y)):
		print('=> Nhãn ' + x[i] + ':', np.round((y[i] / v_max) * 100, 4))

print('Kiểm tra độ cân bằng tập Train')
check_imbalance(train_dir)

def view_image(dir_path):
	"""
	Hàm in ảnh ngẫu nhiên
	:param dir_path:
	:return:
	"""
	a = random.randint(0, len(class_names) - 1)
	path = dir_path + class_names[a]
	images_list = random.sample(os.listdir(path), 1)
	show_image = images_list[0]
	image = mlt.imread(path + '/' + show_image)
	plt.figure()
	plt.imshow(image)
	plt.xlabel(class_names[a])
	plt.ylabel(show_image)
	plt.colorbar()
	plt.show()

print('In 1 ảnh ngẫu nhiên')
view_image(train_dir)

def face_detection(image):
	"""
	Mô hình phát hiện khuôn mặt
	:param image: Dữ liệu ảnh ở dạng Array (s/d hàm cv.imread())
	:return: Tuple/Array tọa độ bboxs của các khuôn mặt đã cắt
	"""
	# Xử lý ảnh (Chuyển ảnh sang thang độ xám)
	gray_image = cv.cvtColor(image, cv.IMREAD_GRAYSCALE)
	face_model = cv.CascadeClassifier(face_detect_model)
	cropped_faces = face_model.detectMultiScale(gray_image)
	return cropped_faces

def training_model():
	"""
	Xây dựng mô hình đào tạo
	:return:
	"""
	# Nạp & tăng cường ảnh
	img_model = ImageDataGenerator(
		horizontal_flip=True,
		zoom_range=.2,
		rescale=1. / 255.

	)
	train_ds = img_model.flow_from_directory(
		train_dir,
		target_size=(128, 128),
		classes=class_names,
		class_mode='categorical'
	)
	test_ds = img_model.flow_from_directory(
		test_dir,
		target_size=(128, 128),
		classes=class_names,
		class_mode='categorical'
	)

	# Xây dựng mô hình
	def my_model():
		model = Sequential(
			[
				MobileNetV2(
					input_shape=(128, 128, 3),
					include_top=False
				),
				GlobalAveragePooling2D(),

				Flatten(),
				Dense(128, activation='relu', kernel_regularizer='l2'),
				Dense(64, activation='relu'),
				Dense(len(class_names), activation='softmax')
			]
		)
		# Biên dịch mô hình
		model.compile(
			optimizer='adam',
			loss=CategoricalCrossentropy(),
			metrics=['accuracy']
		)
		return model

	# Chọn mô hình
	model = my_model()
	# Khái quát mô hình
	model.summary()
	# Đào tạo mô hình
	history = model.fit(
		train_ds,
		epochs=10,
		steps_per_epoch=train_ds.samples // train_ds.batch_size,
		validation_data=test_ds,
		validation_steps=test_ds.samples // test_ds.batch_size,
		callbacks=[
			EarlyStopping(
				monitor='val_accuracy',
				patience=8,
				verbose=1
			),
			CSVLogger(
				save_dir + 'training_logger.csv',
				append=True
			),
			ReduceLROnPlateau(
				monitor='val_accuracy',
				patience=3
			)]
	)
	# Lưu mô hình
	model.save(save_dir + 'mask_detection_model.h5')
	# Đánh giá quá trình đào tạo
	plt.title('Loss')
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig(save_dir + 'loss.jpg')
	plt.show()

	plt.title('Accuracy')
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_accuracy'])
	plt.legend(['train', 'val'], loc='upper left')
	plt.savefig(save_dir + 'accuracy.jpg')
	plt.show()

# training_model()

# Nhận diện khẩu trang qua ảnh
def predict_over_image():
	"""
	Nhận diện khẩu trang qua ảnh
	:return:
	"""
	class_names_color = [(0, 255, 0), (255, 0, 0)]  # 0- With Mask; 1- WithoutMask
	# Nạp mô hình đã qua đào tạo
	model = load_model(save_dir + 'mask_detection_model.h5')
	# Nạp ảnh
	file_path = tk.askopenfilename(title='Chọn tệp')
	image = cv.imread(file_path)
	cropped_faces = face_detection(image)
	image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
	for i in range(len(cropped_faces)):
		# Xử lý phần ảnh có khuôn mặt đã cắt
		(x_min, y_min, x_max, y_max) = cropped_faces[i]
		x_scale = x_min + x_max
		y_scale = y_min + y_max
		cropped_face = image[y_min: y_scale, x_min: x_scale]
		cropped_face = cv.resize(cropped_face, (128, 128))
		cropped_face = np.reshape(cropped_face, (1, 128, 128, 3)) / 255.0
		# Dự đoán giá trị
		du_doan = model.predict(cropped_face)
		label = class_names[np.argmax(du_doan)]
		color = class_names_color[np.argmax(du_doan)]
		acc = str(np.round(np.max(du_doan) * 100, 2)) + '%'
		# Vẽ bbox nơi có khuôn mặt
		## Vẽ viền quanh khuôn mặt
		cv.rectangle(
			image,
			(x_min, y_min),
			(x_scale, y_scale),
			color=color,
			thickness=2
		)
		## Đặt text
		cv.putText(
			image,
			label + ' ' + acc,
			(x_min, y_min - 6),
			fontFace=cv.FONT_HERSHEY_COMPLEX,
			fontScale=0.4,
			color=(255, 255, 0),
			thickness=1
		)
	plt.figure()
	plt.imshow(image)
	plt.show()

# while True:
#	predict_over_image()

def predict_over_cam():
	"""
	Nhận diện khẩu trang qua Webcam
	:return:
	"""
	# 0- With Mask; 1- WithoutMask
	# RGB: (Blue, Green, Red)
	class_names_color = [(0, 255, 0), (0, 0, 255)]
	# Nạp mô hình đã qua đào tạo
	model = load_model(
		save_dir + 'mask_detection_model.h5'
	)
	# Khai báo đối tượng Webcam
	cam = cv.VideoCapture(0)
	while True:
		try:
			status, frame = cam.read()
			# Kiểm tra trạng thái cam
			if not status:
				break
			# Nhấn ESC để thoát
			if cv.waitKey(33) == 27:
				break
			# Mỗi frame chính là một ảnh tĩnh riêng biệt
			# Nhận diện & vẽ bbox lên ALL khuôn mặt được phát hiện
			cropped_faces = face_detection(frame)
			for (x_min, y_min, x_max, y_max) in cropped_faces:
				x_scale = x_min + x_max
				y_scale = y_min + y_max
				cropped_face = frame[y_min: y_scale, x_min: x_scale]  # Rescale lại kích thước ảnh đã cắt
				cropped_face = cv.resize(cropped_face, (128, 128))
				cropped_face = np.reshape(cropped_face, (1, 128, 128, 3)) / 255.0
				# Dự đoán giá trị
				du_doan = model.predict(cropped_face)
				label = class_names[np.argmax(du_doan)]
				color = class_names_color[np.argmax(du_doan)]
				acc = str(np.round(np.max(du_doan) * 100, 2)) + '%'
				# Đặt Text
				cv.putText(
					frame,
					label + ' ' + acc,
					(x_min, y_min - 6),
					fontFace=cv.FONT_HERSHEY_COMPLEX,
					fontScale=0.4,
					color=(255, 255, 0),
					thickness=1
				)
				# Vẽ viền quanh khuôn mặt
				cv.rectangle(
					frame,
					(x_min, y_min),
					(x_scale, y_scale),
					color=color,
					thickness=2
				)
			cv.imshow("Mask Detection", frame)
		except Exception as ex:
			print('Lỗi rồi !!!\n', str(ex))
	cam.release()
	cv.destroyAllWindows()

predict_over_cam()
