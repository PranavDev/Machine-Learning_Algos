% PCA Followed by KNN Classification

clc;
close all;

disp("********************************** PCA KNN **********************************");
disp(" ");
disp("Choose one dataset: 1.data.mat; 2.illumination.mat; 3.pose.mat; 4.EXIT\n");
choice = input("Enter your choice: ");

if choice == 1
    s = "data.mat";
    disp("** You Chose: "+choice+" -> data.mat **");
    % data.mat has 3 images per subject(c:200) = 600 images
    load('/Users/pranavdeo/Desktop/MLProject/Data/data.mat');

    train_set = 400;                    %use 400 to train
    test_set = 200;                     %use 200 to test

    dim = 24 * 21;                      %dimensions of one image
    C = 200;                            %classes/subjects
    img = 600;                          %total images
    n = train_set / C;                  %no. of images being trained per class
    
    P_of_Wi = train_set / img;          %P(Wi)

    Dt_train = zeros(dim,train_set);    %training dataset
    Dt_test = zeros(dim,test_set);      %testing dataset

    Label_train = zeros(train_set,1);   %labels-> training data
    Label_test = zeros(test_set,1);     %labels-> testing data

    % We need to segregate training and testing data from entire dataset
    for i = 0:C-1
        val = 1;
        for j = 1:3
            %segregation of data -> test
            if j == 1
                Dt_test(:,i+1) = reshape(face(:,:,3*i+j), [dim,1]);
                Label_test(i+1) = i+1;
            %segregation of data -> train
            else
                Dt_train(:,2*i+val) = reshape(face(:,:,3*i+j), [dim,1]);
                Label_train(2*i+val) = i+1;
                val = val+1;

            end
        end
    end
    
    disp("Train Index: "+(train_set)+" and Test Index: "+(test_set));


elseif choice == 2
    s = "illumination.mat";
    disp("** You Chose: "+choice+" -> illumination.mat **");
    load('/Users/pranavdeo/Desktop/MLProject/Data/illumination.mat');
    % illumination.mat has 21 images per subject(c:68) = 1428 images
    
    train_set = 1020;                   %use 1360 images for training
    test_set = 408;                     %use 68 images for testing
    
    C = 68;                             %68 classes/subjects
    dim = 48 * 40;                      %single image dimensions
    img = 1428;                         %total no. of images in file
    n = train_set / C;                  %no. of images being trained per class
    
    P_of_Wi = train_set / img;          %P(Wi)
    
    Dt_train = zeros(dim, train_set);   %training dataset
    Dt_test = zeros(dim, test_set);     %testing dataset
    
    Label_train = zeros(train_set, 1);  %labels-> training data
    Label_test = zeros(test_set,1);     %labels-> testing data
    
    
    %segregating training and testing datasets
    train_index = 1;
    test_index = 1;
    
    for i=1:C
        for j=1:21
            if j > 15
                Dt_test(:,test_index) = reshape(illum(:,j,i), [dim,1]);
                Label_test(test_index) = i;
                test_index = test_index + 1;
            else
                Dt_train(:,train_index) = reshape(illum(:,j,i), [dim,1]);
                Label_train(train_index) = i;
                train_index = train_index + 1;
            end
        end
    end
    
    disp("Train Index: "+(train_index-1)+" and Test Index: "+(test_index-1));
    
    
elseif choice == 3
    s = "pose.mat";
    disp("** You Chose: "+choice+" -> pose.mat **");
    load('/Users/pranavdeo/Desktop/MLProject/Data/pose.mat');
    %pose.mat has c:68, 13 images per class/suject = 884
    
    train_set = 612;
    test_set = 272;
    
    C = 68;
    dim = 48 * 40;
    img = 884;
    n = train_set / C;
    
    P_of_Wi = train_set / img;          
    
    Dt_train = zeros(dim, train_set);
    Dt_test = zeros(dim, test_set);
    
    Label_train = zeros(train_set, 1);
    Label_test = zeros(test_set, 1);
    
    %segregating training and testing datasets
    train_index = 1;
    test_index = 1;
    
    for i=1:C
        for j=1:13
            if j == 1 || j == 3 || j == 9 || j == 11
                Dt_test(:,test_index) = reshape(pose(:,:,j,i), [dim,1]);
                Label_test(test_index) = i;
                test_index = test_index + 1;
            else
                Dt_train(:,train_index) = reshape(pose(:,:,j,i), [dim,1]);
                Label_train(train_index) = i;
                train_index = train_index + 1;
            end
        end
    end
    
    disp("Train Index: "+(train_index-1)+" and Test Index: "+(test_index-1));
    
    
else
    disp("** You Chose: "+choice+" -> EXIT !! **");
end



%############################################################################
if choice == 1 || choice == 2 || choice == 3
    
    % Use the SVD to obtain U, S, V matrices (A = USV')
    [W,S,V] = svds(Dt_train,C-1);
    
    X_train = zeros(C-1,train_set);
    X_test = zeros(C-1,test_set);
    
    for i=1:train_set
        X_train(:,i) = W.' * Dt_train(:,i);
    end
    
    for i=1:test_set
        X_test(:,i) = W.' * Dt_test(:,i);
    end
    
    
%##########################################################################
%#########################     CLASSIFICATION     #########################
%##########################################################################
    
                         % KNN CLASSIFICATION
  
  test_set = size(X_test,2);
  dim = size(X_train,1);          
  K = input("-> Enter the value for K: ");
  
  % Calculating the Euclidean distance
  distances = zeros(train_set,1);
  Labels = zeros(train_set,1);
  final_labels = zeros(test_set,1);
  sum = zeros(test_set,1);
  indx = 0;
  
  for i=1:test_set
      for j=1:train_set
          d = sqrt((X_test(:,i) - X_train(:,j)).' * (X_test(:,i) - X_train(:,j)));
          indx = indx + 1;
          distances(indx) = d;
          Labels(indx) = Label_train(j);
      end
      
      [Dk,I] = mink(distances,K);
      
      for a=1:K
          sum(a) = Labels(I(a));
      end
      %sum = floor(sum/K);
      final_labels(i) = mode(sum);
      
      sum = 0;
      indx = 0;
  end
  
  
  
  % Accuracy Calculation
  match = 0;
  for i=1:test_set
      if final_labels(i) == Label_test(i)
          match = match + 1;
      end
  end
  
  accuracy = (match / test_set);
  disp(" ");
  disp("***** PCA_KNN Classification Accuracy for Dataset ~ "+s+" ~ is: "+accuracy+" *****");
                         
%############################################################################

else
    disp("SESSION TERMINATED...");
end