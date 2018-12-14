# coding=utf-8
'''
2018/7/3
作用：针对单分类预测问题，使用DNN网络进行预测----二分类问题----阿西吧，其他多分类自己改去吧--主要改data_deal(不用rashape了)、和score(得把onehot解码)
源数据：需要特征矩阵xdata和标签矩阵ydata
xdata：特征矩阵，类型np.array ---- [[1,2],[3,3],[2,7],[1,3]]
ydata: 标签矩阵, 类型np.array ---- [1,0,1,1] 
'''

import numpy as np
#import pandas as pd
import tensorflow as tf	   
import sklearn.preprocessing as pps					# 数据处理
from sklearn.model_selection import	train_test_split # 划分数据集
from sklearn.cross_validation import StratifiedKFold		# K折交叉验证
from sklearn.metrics import roc_auc_score			# AUC值
#import scipy as	sp  


class BPNeuralNetwork:
	def	__init__ (self):	
		self.sess =	tf.Session()	# 运行环境
		self.loss = None			# 损失函数-- 交叉熵
		self.optimizer = None		# 优化算法-- 梯度下降
		self.y_predict = []				# 需要预测的结果
	
	'''
	------------------  输入一系列参数  ----------------
	输入：
		self.input_n：		输入层神经元数目
		self.hidden_n：		每层隐藏层神经元数目
		self.output_n：		输出层数目
		self.limit：		训练次数
		self.learn_rate ：	学习率
	其他参数：
		self.hidden_ceng	隐藏层的层数
		self.input_layer	输入
		self.label_layer	标签Z【正确】
		self.hidden_layers	隐藏层输出Z
		self.output_layer	输出Z【预测】
		self.weights = {}	所有权重参数
		self.basis = {}		所有偏置参数
	'''
	def parameter (self, input_n, hidden_n, output_n, limit = 1000, learn_rate = 0.001):
		self.input_n = input_n
		self.hidden_ceng = len(hidden_n)  # 隐藏层的层数
		self.hidden_size = hidden_n  
		self.output_n =	output_n
		self.limit = limit
		self.learn_rate = learn_rate

		self.input_layer = tf.placeholder(tf.float32, [None, self.input_n])	#维度为输入的神经元数目
		self.label_layer = tf.placeholder(tf.float32, [None, self.output_n]) #维度为数出的数目
		self.hidden_layers = []		#隐藏层输出Z
		self.output_layer =	None	#输出Z
		self.weights = {}
		self.basis = {}
			

	'''
	----------- 划分训练集与测试集、对训练集输出进行独热编码 ---------------
	输入: 
		xdata		特征矩阵 m*n 
		ydata		标签矩阵 1*m
		x_predict	需要预测的特征矩阵
	输出：训练集特征、训练集标签【onthot】、测试集输入、测试集标签【一维】、需要预测的特征矩阵
	'''
	def	data_deal(self, xdata, ydata, x_predict=[]):		
		#lab_en = pps.LabelEncoder()
		#ydata = lab_en.fit_transform(ydata)  # 当标签不是0,1....时，将n个标签改为数字0-n
		#print('对应标签意义为：{}  ---  {}'.format(lab_en.classes_, lab_en.fit_transform(lab_en.classes_)))
		x_train,x_test,y_train_one,y_test =	train_test_split(xdata,ydata,test_size=0.25,random_state=0)
		x_train = pps.scale(x_train)
		x_test	= pps.scale(x_test)
		if len(x_predict)==0:
			x_predict = x_test	# 默认为测试集
		else:
			x_predict = pps.scale(x_predict)
		y_train = y_train_one.reshape((-1,1))
		# 独热编码
		onehot = pps.OneHotEncoder()
		onehot.fit(y_train)
		y_train = onehot.transform(y_train).toarray()
		print('---- 训练集：\n 输入数据：{}, 输出数据:{} '.format(x_train.shape, y_train.shape))
		print('---- 测试集：\n 输入数据：{}, 输出数据:{} \n'.format(x_test.shape, y_test.shape))
		return x_train, y_train, x_test, y_test, x_predict
		#return x_train, y_train, x_train, y_train_one, x_predict  # 在训练集上测试
	 
	'''
	-----------  定义一层计算 ----------------
	输入：
		ww: 初始化权重维度in*out，初始值全为0.1 (矩阵可选为tf.zeros或者tf.random_normal)
		bb: 初始化偏置维度1*out，一列，初始值全为0.1
		zz: Z函数运算=（w*x）+b = 输入*权重矩阵 + 偏置
		activate：激励函数，默认为无
	输出：输出值、权重矩阵、偏置矩阵
	'''
	def	make_layer(self, in_data, in_size, out_size, activate =	None): 
		ww = tf.Variable(tf.zeros([in_size, out_size]) + 0.1)	 
		bb = tf.Variable(tf.zeros([1,out_size]) + 0.1)   
		zz = tf.matmul(in_data,ww) + bb 
		if activate	is None:  
			ans	= zz		
		else:  
			ans	= activate(zz)	
		return ans,ww,bb  	
	
	'''---------------- 定义预测函数 ---------------
	predict(test_data)：输入特征矩阵，输出预测标签【独热编码】
	'''
	def	predict(self, xdata): 
		result = self.sess.run(self.output_layer,	feed_dict={self.input_layer:xdata})
		return result
	
	
	'''
	---------------- 在训练好模型之后---将softmax输出的独热编码转化为类别 ---------------
	oneDecoder(x_test) :输入特征矩阵x_test ；输出预测结果x_test_predict【一维】

	'''
	def	oneDecoder(self, x_test):
		x_test_predict = self.predict(x_test)
		predict_y = []
		for i in range(len(x_test_predict)):
			k = list(map(round, x_test_predict[i]))
			flag = k.index(1)   # 逆独热编码
			predict_y.append(flag)
		return predict_y

	'''
	---------------- 计算各种率（对于测试集） ---------------
	score_5(truth_y, predict_y)：输入实际标签truth_y【一维】和预测标签predict【一维的】；输出TPR---FPR ---P---A---AUC

	'''
	def score_5(self, truth_y, predict_y):
		#print("实际标签：{}    预测标签：{}".format(truth_y,predict_y))
		flag = 1  # 假设1为异常
		TP = sum([1 for k in range(len(predict_y)) if predict_y[k]==flag and truth_y[k]==flag]) #预正实正
		FP = sum([1 for k in range(len(predict_y)) if predict_y[k]==flag and truth_y[k]!=flag]) #预正实负
		FN = sum([1 for k in range(len(predict_y)) if predict_y[k]!=flag and truth_y[k]==flag]) #预负实正
		TN = sum([1 for k in range(len(predict_y)) if predict_y[k]!=flag and truth_y[k]!=flag]) #预负实负
		if TP==0:
			print("==================辣鸡")
			print("---【TP: {} ---TN: {} ---FP: {} ---FN: {} 】---".format(TP,TN,FP,FN)) 
			FPR = round(float(FP) / (FP + TN),2)
			A = round(np.mean(predict_y == truth_y),2)  #float(TP+TN)/(TP+FP+TN+FN)
			AUC = round(roc_auc_score(truth_y,predict_y),2)
			return [0, 0, 0, 0, 0]
		else:			
			TPR = round(float(TP)/(TP+FN),2)		# TPR=预正实正/（所有实正）=预正实正/（预正实正+预负实正）= 召回率			
			FPR = round(float(FP)/(FP+TN),2)		# FPR=预正实负/（所有实负）=预正实负/（预负实负+预正实负）=			
			P = round(float(TP)/(TP+FP),2)			# 精确率 = 预正实正/所有预正			
			A = round(np.mean(predict_y==truth_y),2)  # 正确率=所有匹配/所有样本   float(TP+TN)/(TP+FP+TN+FN)  
			AUC = round(roc_auc_score(truth_y,predict_y),2)
			#print("---【TP: {} ---TN: {} ---FP: {} ---FN: {} 】---".format(TP,TN,FP,FN))  
			#print("---【TPR: {} ---FPR: {} ---P: {} ---A: {} ---AUC:{}】---".format(TPR, FPR, P, A, AUC)) 
			return [TPR, FPR, P, A, AUC]
	
	'''
	---------------- 构建DNN神经网络【要改成其他网络就改这个函数】，构建流-flow-----
	分类问题--一般前面使用sigmod或者rule作为激励函数，输出用softmax函数
	'''
	def	DNN(self): 
		# 输入层
		in_size	= self.input_n 
		out_size = self.hidden_size[0]
		inputs = self.input_layer 
		ans,ww,bb =	self.make_layer(inputs,	in_size, out_size, activate=tf.nn.tanh)
		self.weights['w1'] = ww
		self.basis['b1'] = bb

		# 隐藏层
		self.hidden_layers.append(ans) 
		for	i in range(self.hidden_ceng-1):
			in_size	= out_size 
			out_size = self.hidden_size[i+1]
			inputs = self.hidden_layers[-1]	
			ans,ww,bb =	self.make_layer(inputs,	in_size, out_size, activate=tf.nn.tanh)
			self.weights['w'+str(i+2)] = ww
			self.basis['b'+str(i+2)] = bb
			self.hidden_layers.append(ans)
	
		# 加个dropout层
		#ans = tf.nn.dropout(self.hidden_layers[-1], 0.5)
		#self.hidden_layers.append(ans)

		# 输出层
		ans,ww,bb =	self.make_layer(self.hidden_layers[-1],	self.hidden_size[-1], self.output_n, activate =	tf.nn.softmax)
		self.weights['w'+str(self.hidden_ceng+1)] = ww
		self.basis['b'+str(self.hidden_ceng+1)] = bb
		self.output_layer =	ans
		
	'''
	---------------- 运行神经网络 -----
	1、得到原数据 data_deal
	2、设定参数 parameter
	3、按参数构建网络 DNN
	4、定义loss【self.loss】和寻优算法【self.optimizer】
	5、初始化网络、开始训练
	6、查看模型在测试集上的效果 score
	'''
	def runDNN (self, xdata, ydata, hidden_n, x_predict=[], limit = 1000, learn_rate = 0.001):
		x_train, y_train, x_test, y_test, x_predict = self.data_deal(xdata, ydata, x_predict)
		input_n = x_train.shape[1]
		output_n = y_train.shape[1]
		self.parameter(input_n, hidden_n, output_n, limit, learn_rate)
		self.DNN()
		self.loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_layer,logits=self.output_layer))
		self.optimizer = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)
		
		# 初始化
		initer = tf.initialize_all_variables()
	
		# 在Session环境中进行训练
		self.sess.run(initer) 
		for	i in range(limit):
			#self.run_opt(x_train, y_train)  # 梯度下降训练网络参数
			self.sess.run(self.optimizer, feed_dict={self.input_layer: x_train, self.label_layer: y_train})
			if (i+1)%(500)==0:
				loss = self.sess.run(self.loss,feed_dict={self.input_layer: x_train, self.label_layer: y_train})
				predict_y = self.oneDecoder(x_test)		# 测试集预测结果
				result = self.score_5(y_test, predict_y)# 测试集预测效果
				#print('第{}次loss:{}, 预测标签：{}'.format(i+1, loss, predict_y))
				print('第{}次loss:{}'.format(i+1, loss))

		saver =	tf.train.Saver()
		saver.save(self.sess, './model_1/model.ckpt')	# 存储模型(需要自己在目录下新建model_1文件夹)
		
		# 最后的结果、对x_predict特征集的预测
		loss = self.sess.run(self.loss,feed_dict={self.input_layer: x_train, self.label_layer: y_train})
		predict_y = self.oneDecoder(x_test)		# 最后的测试集预测效果
		result = self.score_5(y_test, predict_y)
		print('\n\n--- 对测试集------ \n ------------- 最终结果输出：权重值、偏置值矩阵 ---------：')
		#for	j in range(len(self.weights)):
		#	print('w'+str(j+1)+':\n	',self.sess.run(self.weights['w'+str(j+1)]))
		#	print('b'+str(j+1)+':\n	',self.sess.run(self.basis['b'+str(j+1)]))
		print('---- 预测结果为：{}\n-----正确结果为：{}'.format(predict_y, list(map(int,y_test))))
		print("\n\n---【TPR: {} ---FPR: {} ---P: {} ---A: {} ---AUC:{}】---".format(result[0], result[1], result[2], result[3], result[4])) 

		self.y_predict = self.oneDecoder(x_predict)  # 需要预测的结果
		print('---- \n\n需要预测的数据集的最终预测结果为\n', self.y_predict )
	
	'''
	---------------- 数据处理--K次交叉检验 -----
	输入（划分好的训练集和测试集）：x_train, y_train, x_test, y_test, x_predict
	输出：转化好的x_train, y_train, x_test, y_test, x_predict
	'''
	def	data_deal_K(self, x_train, y_train_one, x_test, y_test, x_predict=[]):		
		x_train = pps.scale(x_train)
		x_test	= pps.scale(x_test)
		if len(x_predict)==0:
			x_predict = x_test	# 默认为测试集
		else:
			x_predict = pps.scale(x_predict)
		y_train = y_train_one.reshape((-1,1))
		# 独热编码
		onehot = pps.OneHotEncoder()
		onehot.fit(y_train)
		y_train = onehot.transform(y_train).toarray()
		print('---- 训练集：\n 输入数据：{}, 输出数据:{} '.format(x_train.shape, y_train.shape))
		print('---- 测试集：\n 输入数据：{}, 输出数据:{} \n'.format(x_test.shape, y_test.shape))
		return x_train, y_train, x_test, y_test, x_predict
		#return x_train, y_train, x_train, y_train_one, x_predict  # 在训练集上测试
	'''
	输入list，输出数组的 [最高 1/4位 中位数 均值 最低]
	'''
	def sort5(self,list_1): 
		list_2 = sorted(list_1,reverse=True)
		m1 = list_2[0]
		m2 = list_2[int(len(list_2)/4)]
		m3 = list_2[int(len(list_2)/2)]
		m4 = round(sum(list_2)/len(list_2),2)
		m5 = list_2[-1]
		return [m1,m2,m3,m4,m5]
	'''
	---------------- 运行神经网络--K次交叉检验 -----
	'''
	def runDNN_K (self, xdata, ydata, hidden_n, K=10, x_predict=[], limit = 1000, learn_rate = 0.001):
		score_K = []
		data_index = StratifiedKFold(ydata, n_folds=K, shuffle=True)
		for train,test in data_index:
			x_train = xdata[train]
			y_train = ydata[train]
			x_test = xdata[test]
			y_test = ydata[test]
			x_train, y_train, x_test, y_test, x_predict = self.data_deal_K(x_train, y_train, x_test, y_test, x_predict)

			input_n = x_train.shape[1]
			output_n = y_train.shape[1]
			self.parameter(input_n, hidden_n, output_n, limit, learn_rate)
			self.DNN()
			self.loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_layer,logits=self.output_layer))
			self.optimizer = tf.train.AdamOptimizer(learn_rate).minimize(self.loss)
			
			# 初始化
			initer = tf.initialize_all_variables()
		
			# 在Session环境中进行训练
			self.sess.run(initer) 
			for	i in range(limit):
				self.sess.run(self.optimizer, feed_dict={self.input_layer: x_train, self.label_layer: y_train})
				if (i+1)%(500)==0:
					loss = self.sess.run(self.loss,feed_dict={self.input_layer: x_train, self.label_layer: y_train})
					predict_y = self.oneDecoder(x_test)		# 测试集预测结果
					result = self.score_5(y_test, predict_y)# 测试集预测效果
					#print('第{}次loss:{}, 预测标签：{}'.format(i+1, loss, predict_y))
					#print('第{}次loss:{}'.format(i+1, loss))

			#saver =	tf.train.Saver()
			#saver.save(self.sess, './model_1/model.ckpt')	# 存储模型(需要自己在目录下新建model_1文件夹)
			
			# 最后的结果、对x_predict特征集的预测
			loss = self.sess.run(self.loss,feed_dict={self.input_layer: x_train, self.label_layer: y_train})
			predict_y = self.oneDecoder(x_test)		# 最后的测试集预测效果
			result = self.score_5(y_test, predict_y)			
			print('---- 预测结果为：{}\n-----正确结果为：{}'.format(predict_y, list(map(int,y_test))))
			print("\n\n---【TPR: {} ---FPR: {} ---P: {} ---A: {} ---AUC:{}】---".format(result[0], result[1], result[2], result[3], result[4])) 
			score_K.append(list(result))

		result = []
		for i in range(len(score_K[0])):
			list_1 = [j[i] for j in score_K]
			result.append(self.sort5(list_1))
		good = [[i[j] for i in result] for j in range(5)] # 5 个参数，5种计数
		print('最大值： {}\n1/4分位值：{}\n中位数：{}\n均值：{}\n最小值：{}\n'.format(good[0],good[1],good[2],good[3],good[4]))
		return good

if __name__	== '__main__':
	import GetData
	import GetData_cp
	GD = GetData_cp.GetData(10,0)
	xdata, ydata = GD.xdata, GD.ydata

	#D = GetData.Data(index = 1, flag = 'imbalance', num = 40)
	#xdata, ydata = D.xdata, D.ydata
	
	print(xdata.shape)
	print(ydata.shape)


	'''
	# 	自带的特征选择LinearSVC
	print("=======================自带的特征选择2===================")
	import sklearn.feature_selection as SELECT			# 特征选择一系列函数
	from sklearn import svm								# 支持向量机 SVM
	c = 0.004
	print('\n\n惩罚系数c= ',c)
	lsvc = svm.LinearSVC(C=c,penalty="l1",dual=False).fit(xdata,ydata)
	model = SELECT.SelectFromModel(lsvc, prefit=True)
	xdata = model.transform(xdata)
	print(xdata.shape)

	'''
	BP	= BPNeuralNetwork()
	hidden_n = [20]	
	#BP.runDNN(xdata, ydata, hidden_n, x_predict=[], limit = 50000, learn_rate = 0.0001)
	BP.runDNN_K(xdata, ydata, hidden_n, K=5, x_predict=[], limit = 1000, learn_rate = 0.01)



