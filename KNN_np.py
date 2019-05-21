from Pre_Processing_Data import *
import numpy as np 
import operator
from sklearn.metrics import accuracy_score

# Đọc dữ liệu
X_train, y_train, X_test, y_test = Load_Data('F:/HK4/ML/Case_Study_#1/Source_Code_Titanic_Dataset/My_Data.txt')

# Hàm Định nghĩa và tính khoảng cách giữ hai điểm dữ liệu
def distance(point_1, point_2):
	distance = 0
	length = point_1.shape[0]
	for i in range(length):
		#distance += (point_1[i] - point_2[i])**2
		# Vì các giá trị ở mỗi thuộc tính là độc lập và không phân thứ tự nên không thể dùng công thức euclid
		if point_1[i] != point_2[i]: 
			distance += 1
	return distance

# Hàm tìm k điểm trong tập train có khoảng cách gần nhất với điểm mới 
def find_Neighbor(point, X_train, k):
	all_distance = []
	for i in range(X_train.shape[0]):
		new_distance = distance(point, X_train[i])
		all_distance.append([i, new_distance])
	
	all_distance.sort(key = operator.itemgetter(1))
	
	all_neighbor = []
	for i in range(k):
		all_neighbor.append(all_distance[i])

	return all_neighbor

# Hàm dự đoán nhãn của điểm mới từ tập k điểm lân cận
def Predict(point, X_train, y_train, k):
	class_0 = 0
	class_1 = 0
	all_neighbor = find_Neighbor(point, X_train, k)

	for i in all_neighbor:
		if y_train[i[0]] == 0:
			class_0 += 1
		elif y_train[i[0]] == 1:
			class_1 += 1

	if class_1 < class_0:
		return 0
	else:
		return 1

# Hàm dự đoán nhãn của tập test với k cho trước
def GetPredict(X_test, X_train, y_train, k):
	all_predict = []
	for test_case in X_test:
		label = Predict(test_case, X_train, y_train, k)
		all_predict.append(label)
	return np.array(all_predict)
# Dư đoán tập X_test
pre = GetPredict(X_test, X_train, y_train, 1)
# Hàm so khớp kết quả nhãn dự đoán và nhãn thật
accuracy = accuracy_score(y_test, pre)


# # Ghi dữ liệu y predict ra file để so sánh
# result_knn_np = open("F:/HK4/ML/Case_Study_#1/Source_Code_Titanic_Dataset/result_knn_np.txt", "a+")
# for i in range (0, len(pre)):
# 	result_knn_np.write(str(pre[i]) + "\n")
# result_knn_np.close()

#Tính F1_Score - Độ đo hiệu quả của mô hình
def F1_Score(pre, y_test):
	precision = 0 # = TP / (TP + FP)
	recall = 0 # = TP / P
	TP = 0
	FP = 0
	P = 0
	F1 = 0

	for i in range (0, len(y_test)):
		if pre[i] == 1 and y_test[i] == 1:
			TP += 1
		elif pre[i] == 1 and y_test[i] == 0:
			FP += 1
	precision = TP / (TP + FP)
	print("Precision = ", precision)

	for i in range (0, len(y_test)):
		if y_test[i] == 1:
			P += 1
	recall = TP / P
	print("Recall = ", recall)

	F1 = 2*precision*recall/(precision + recall)
	return F1
print("F1_Score = ", F1_Score(pre, y_test))

# in ra giá trị accuracy
print("Accuracy resulted by sklearn = ", accuracy)