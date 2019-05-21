from Pre_Processing_Data import *
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Đọc dữ liệu
X_train, y_train, X_test, y_test = Load_Data('F:/HK4/ML/Case_Study_#1/Source_Code_Titanic_Dataset/My_Data.txt')
#Phân loại bằng cách thực hiện Gaussian Naive Bayes
gnb = GaussianNB()
# fix Gaussian Naive Bayes bằng cách sử dụng X_train làm dữ liệu huấn luyện
# y_train làm giá trị đích (label)
gnb = gnb.fit(X_train, y_train)
# hàm ước tính trả về xác xuất cho dữ liệu huấn luyện X_test
pre = gnb.predict(X_test)
# hàm tính toán độ chính xác y_test/pre
accuracy = accuracy_score(y_test, pre)

# # Ghi dữ liệu y predict ra file để so sánh
# result_naivebayes_sk = open("F:/HK4/ML/Case_Study_#1/Source_Code_Titanic_Dataset/result_naivebayes_sk.txt", "a+")
# for i in range (0, len(y_test)):
# 	result_naivebayes_sk.write(str(pre[i]) + "\n")
# result_naivebayes_sk.close()

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