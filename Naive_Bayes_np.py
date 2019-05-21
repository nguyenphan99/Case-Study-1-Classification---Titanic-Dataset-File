from Pre_Processing_Data import *
import numpy as np 

# Đọc dữ liệu
X_train, y_train, X_test, y_test = Load_Data('F:/HK4/ML/Case_Study_#1/Source_Code_Titanic_Dataset/My_Data.txt')

# # Ghi dữ liệu y predict ra file để so sánh
# result_naivebayes_np = open("F:/HK4/ML/Case_Study_#1/Source_Code_Titanic_Dataset/result_naivebayes_np.txt", "a+")

# hàm tính xác xuất các feature tương ứng với yes và no của label
def Probability(X_train,y_train):
	m = len(y_train)
	#tổng số trường hợp alive
	alive = 0
	#tổng số trường hợp died
	death = 0
	for i in range(m):
		if y_train[i] == 1:
			alive += 1
		else:
			death += 1
	
	live = alive/m
	die = death/m
	#tỉ lệ giữa sống và chết
	R=[live,die]

	Class_1_alive = 0
	Class_1_died = 0
	Class_2_alive = 0
	Class_2_died = 0
	Class_3_alive = 0
	Class_3_died = 0
	Class_4_alive = 0
	Class_4_died = 0

	Age_Adult_alive = 0
	Age_Adult_died = 0
	Age_Child_alive = 0
	Age_Child_died = 0

	Male_alive = 0
	Male_died = 0
	Female_alive = 0
	Female_died = 0

	for i in range(m):
		if(X_train[i][0] == 0):
			
			if(y_train[i] == 1):
				Class_1_alive+=1
			else:
				Class_1_died+=1
		elif(X_train[i][0] == 1):
			
			if(y_train[i] == 1):
				Class_2_alive+=1
			else:
				Class_2_died+=1
		elif(X_train[i][0] == 2):
			
			if(y_train[i] == 1):
				Class_3_alive+=1
			else:
				Class_3_died+=1
		elif(X_train[i][0] == 3):
			
			if(y_train[i] == 1):
				Class_4_alive+=1
			else:
				Class_4_died+=1
		Accuracy_H1_alive = Class_1_alive/alive
		Accuracy_H2_alive = Class_2_alive/alive
		Accuracy_H3_alive = Class_3_alive/alive
		Accuracy_H4_alive = Class_4_alive/alive

		Accuracy_H1_death = Class_1_died/death
		Accuracy_H2_death = Class_2_died/death
		Accuracy_H3_death = Class_3_died/death
		Accuracy_H4_death = Class_4_died/death
		# tỉ lệ sống và chết của loại khách hàng
		H=[[Accuracy_H1_alive,Accuracy_H1_death],
			[Accuracy_H2_alive,Accuracy_H2_death],
			[Accuracy_H3_alive,Accuracy_H3_death],
			[Accuracy_H4_alive,Accuracy_H4_death]]

		if X_train[i][1] == 0:
			
			if(y_train[i] == 1):
				Age_Adult_alive+=1
			else:
				Age_Adult_died+=1
		elif X_train[i][1] == 1:
			
			if(y_train[i] == 1):
				Age_Child_alive+=1
			else:
				Age_Child_died+=1
		Accuracy_AgeA_alive = Age_Adult_alive/alive
		Accuracy_AgeC_alive = Age_Child_alive/alive

		Accuracy_AgeC_death = Age_Child_died/death
		Accuracy_AgeA_death = Age_Adult_died/death
		# tỉ lệ sống và chết của tuổi khách hàng
		A=[[Accuracy_AgeA_alive,Accuracy_AgeA_death],
			[Accuracy_AgeC_alive,Accuracy_AgeC_death]]

		
		if X_train[i][2] == 0:
			
			if(y_train[i] == 1):
				Male_alive+=1
			else:
				Male_died+=1
		elif X_train[i][2] == 1:
			
			if(y_train[i] == 1):
				Female_alive+=1
			else:
				Female_died+=1

		Accuracy_Sex_Male_alive = Male_alive/alive
		Accuracy_Sex_Female_alive = Female_alive/alive

		Accuracy_Sex_Male_death = Male_died/death
		Accuracy_Sex_Female_death = Female_died/death
		# tỉ lệ sống và chết của giới tính khách hàng
		S = [[Accuracy_Sex_Male_alive,Accuracy_Sex_Male_death],
		[Accuracy_Sex_Female_alive,Accuracy_Sex_Female_death]]
		
	return H, S, A, R

pre = []

# hàm tính độ chính xác của tập test
def Predict(X_test,y_test,X_train,y_train):
	global pre
	H,S,A,R = Probability(X_train,y_train)
	#hàm lấy số lượng y test
	n = len(y_test)
	flag=0
	for i in range(n):
		temp_no = [0,0,0]
		temp_yes = [0,0,0]
		if(X_test[i][0] == 0):
			temp_no[0] = H[0][1]
			temp_yes[0] = H[0][0]
		elif(X_test[i][0] == 1):
			temp_no[0] = H[1][1]
			temp_yes[0] = H[1][0]
		elif(X_test[i][0] == 2):
			temp_no[0] = H[2][1]
			temp_yes[0] = H[2][0]
		elif(X_test[i][0] == 3):
			temp_no[0] = H[3][1]
			temp_yes[0] = H[3][0]

		if X_test[i][1] == 0:
			temp_no[1] =A[0][1]
			temp_yes[1] = A[0][0]
		elif X_test[i][1] == 1:
			temp_no[1] =A[1][1]
			temp_yes[1] = A[1][0]

		if X_test[i][2] == 0:
			temp_no[2] = S[0][1]
			temp_yes[2] = S[0][0]
		elif X_test[i][2] == 1:
			temp_no[2] = S[1][1]
			temp_yes[2] = S[1][0]

		ac = temp_yes[0]*temp_yes[1]*temp_yes[2]*R[0]
		bc = temp_no[0]*temp_no[1]*temp_no[2]*R[1]
		Px = ac + bc
		Pyes = ac/Px
		Pno = bc/Px
		temp_ytest = -1
		
		if(Pyes > Pno):
			temp_ytest = 1
		else:
			temp_ytest = 0

		pre.append(temp_ytest)
		# #Ghi dữ liệu y predict ra file để so sánh
		# result_naivebayes_np.write(str(temp_ytest) + "\n")
		if temp_ytest == y_test[i]:
			flag += 1
	return flag/n


# result_naivebayes_np.close()

acc = Predict(X_test,y_test,X_train,y_train)

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
print("Accuracy resulted by Predict = ", acc)
