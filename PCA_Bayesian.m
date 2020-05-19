% PCA Followed by Bayesian Classification

clc;
close all;

disp("********************************** PCA BAYESIAN **********************************");
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
            if j > 9
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
    
    % Using SVD to obtain U, S, V matrices (A = USV')
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
    
                         % BAYESIAN CLASSIFICATION
                         
    test_set = size(X_test,2);
    dim = size(X_train,1);
    
    % Mean Calculation
    mean = zeros(dim,C);
    val = 0;
    for i=1:C
        for j=1:n
            mean(:,i) = mean(:,i) + X_train(:,i+val);
            val = val + 1;
        end
        val = val - 1;
        mean(:,i) = mean(:,i) / n;
    end
    
    
    
    % Variance calculation
    variance = zeros(dim,dim,C);
    variance_inv = zeros(dim,dim,C);
    val = 0;
    theta = 1;
    for i=0:C-1
        for j=1:n
            val = val + 1;
            variance(:,:,i+1) = variance(:,:,i+1) + ((X_train(:,i+val) - (mean(:,i+1))) * (X_train(:,i+val) - (mean(:,i+1))).');
        end
        val = val - 1;
        
        variance(:,:,i+1) = variance(:,:,i+1) + (theta * eye(dim));
        variance_inv(:,:,i+1) = inv(variance(:,:,i+1));
    end
    
    
    
    % Calculate the Wi, wi, wio
    Wi = zeros(dim,dim,C);
    wi = zeros(dim,C);
    wio = zeros(C,1);
    for i=1:C
        Wi(:,:,i) = (-1/2) * variance_inv(:,:,i);
        wi(:,i) = variance_inv(:,:,i) * mean(:,i);
        wio(i) = (-1/2 * ( mean(:,i).' * variance_inv(:,:,i) * mean(:,i))) + (-1/2 * log(det(variance(:,:,i)))) + log(P_of_Wi);
    end
    
    
    % Solve the equation: g(x) = (x' * Wi * x) + (wi' * x) + (wio)
    final_labels = zeros(test_set,1);
    for i=1:test_set
        gmax = (X_test(:,i).' * Wi(:,:,1) * X_test(:,i)) + (wi(:,1).' * X_test(:,i)) + wio(1);
        
        for j=1:C
            gval = (X_test(:,i).' * Wi(:,:,j) * X_test(:,i)) + (wi(:,j).' * X_test(:,i)) + wio(j);
            
            if(gval >= gmax)
                gmax = gval;
                final_labels(i) = j;
            end
        end
    end
    
    
    
    % Accuracy calculation
    match = 0;
    for i=1:test_set
        if final_labels(i) == Label_test(i)
            match = match + 1;
        end
    end
    
    accuracy = (match / test_set);
    disp(" ");
    disp("***** PCA_Bayesian Classification Accuracy for Dataset ~ "+s+" ~ is: "+accuracy+" *****");
    
%############################################################################

else
    disp("SESSION TERMINATED...");
end