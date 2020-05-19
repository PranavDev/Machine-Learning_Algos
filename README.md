# MachineLearningAlgos
### Author: Pranav H. Deo ### 116763752 ### ENTS669D Project ###

•	I have 8 codes named:
ML_Bayesian.m
LDA_Bayesian.m
PCA_Bayesian.m
KNN.m
LDA_KNN.m
PCA_KNN.m
Final1.m (All classifiers are integrated)
Final2.m (data.mat with C=2 integrated for all classifiers)

•	All codes are independent. For ease of running and testing, I have integrated all the datasets 
  and classifiers under Final1.m and Final2.m code-file.
  
  You can run Final1.m/Final2.m or run each code individually.
  
  Final1.m: 
  o	data.mat with C=200 for all classifiers.
  o	illumination.mat for all classifiers.
  o	pose.mat for all the classifiers.
  
  Final2.m:
  o	Designed specifically to meet the first type of experiment with C=2 for data.mat.
  o	All classifiers are designed for it.

•	You will be prompted to test on any of the datasets (data.mat, illumination.mat and pose.mat). 
  Enter ‘1’ for data.mat, ‘2’ for illumination.mat and ‘3’ for pose.mat.

•	In case you plan to manually assign dataset index, you can do so to the Dt_test but make sure 
  you change the value of the variables, train_set and test_set, since it keeps a count of how 
  many samples are being used for training and testing.
  
  
  E.g. Say you want to test the pose.mat
  You want to assign 4 values to testing, j = 1, 3, 7, 9 to Dt_test, and rest 9 to training, 
  Dt_train, you can do so. Just change the variable: test_set = (4 values) * 68 = 272 and 
  train_set = (9 values) * 68 = 612.
  When you change the number of testing indices, you will have to change the variable named 
  test_set and train_set too. Else accuracy won’t be calculated correctly/accurately.
