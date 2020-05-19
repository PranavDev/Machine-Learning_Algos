% LDA Followed by KNN Classification

clc;
close all;

disp("********************************** LDA KNN **********************************");
disp(" ");
disp("Choose one dataset: 1.data.mat; 2.illumination.mat; 3.pose.mat; 4.EXIT\n");
choice = input("Enter your choice: ");

if choice == 1
    s = "data.mat";
    disp("** You Chose: "+choice+" -> data.mat **");
    load('/Users/pranavdeo/Desktop/MLProject/Data/data.mat');
    % data.mat has 3 images per subject(c:200) = 600 images
    
    train_set = 400;
    test_set = 200;
    
    C = 200;
    dim = 24 * 21;
    n = train_set / C;
    img = 600;

    Dt_train = zeros(dim,train_set);
    Dt_test = zeros(dim,test_set);
    
    Label_train = zeros(train_set,1);
    Label_test = zeros(test_set,1);
    
    %segregating training and testing datasets
    train_index = 1;
    test_index = 1;
    val = 1;
    
    for i=1:C
        for j=1:3
            if j == 1
                Dt_test(:,test_index) = reshape(face(:,:,val), [dim,1]);
                Label_test(test_index) = i;
                test_index = test_index + 1;
                val = val + 1;
            else
                Dt_train(:,train_index) = reshape(face(:,:,val), [dim,1]);
                Label_train(train_index) = i;
                train_index = train_index + 1;
                val = val + 1;
            end
        end
    end
    
    disp("Train Index: "+(train_set)+" and Test Index: "+(test_set));
    
    
elseif choice == 2
    s = "illumination.mat";
    disp("** You Chose: "+choice+" -> illumination.mat **");
    load('/Users/pranavdeo/Desktop/MLProject/Data/illumination.mat');
    % illumination.mat has 21 images per subject(c:68) = 1428 images
    
    train_set = 1020;
    test_set = 408;
    
    C = 68;
    dim = 48 * 40;
    n = train_set / C;
    img = 1428;
        
    Dt_train = zeros(dim,train_set);
    Dt_test = zeros(dim,test_set);
    
    Label_train = zeros(train_set,1);
    Label_test = zeros(test_set,1);
    
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
    
    train_set = 544;
    test_set = 340;
    
    C = 68;
    dim = 48 * 40;
    n = train_set / C;
    img = 884;
        
    Dt_train = zeros(dim,train_set);
    Dt_test = zeros(dim,test_set);
    
    Label_train = zeros(train_set,1);
    Label_test = zeros(test_set,1);
    
    %segregating training and testing datasets
    train_index = 1;
    test_index = 1;
    
    for i=1:C
        for j=1:13
            if j > 8
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



%##########################################################################
if choice ~= 4
    
    % Mean per class calculation
    mean_per_class = zeros(dim,C);
    val = 0;
    for i=1:C
        for j=1:n
            mean_per_class(:,i) = mean_per_class(:,i) + Dt_train(:,i+val);
            val = val + 1;
        end
        val = val - 1;
        mean_per_class(:,i) = mean_per_class(:,i) / n;
    end
    
    
    
    
    % Total mean over all classes calculation
    Mean_all = zeros(dim,1);
    for i=1:train_set
        Mean_all = Mean_all + Dt_train(:,i);
    end
    Mean_all = Mean_all / train_set;
    
    
    
    
    % Within Class scatter matrix (Sw)
    Sw = zeros(dim,dim);
    alp = 0.05;
    index = 0;
    for i=1:C
        for j=1:n
            index = index + 1;
            temp = (Dt_train(:,index) - mean_per_class(:,i)) * (Dt_train(:,index) - mean_per_class(:,i)).';
        end
        
        temp = temp + alp * eye(dim);       % Avoiding a singular matrix
        Sw = Sw + temp;     
    end
    
    
    
    
    % Between Class scatter matrix (Sb)
    Sb = zeros(dim,dim);
    for i=1:C
        Sb = Sb + n * ( (mean_per_class(:,i) - Mean_all) * (mean_per_class(:,i) - Mean_all).' );
    end
    
    
    
    
    % Finding Eigen Values
    [Wi,EigVal] = eigs(Sb,Sw,C-1);
    
    
    % Transformation of Matrices
    X_train = zeros(C-1,train_set);
    X_test = zeros(C-1,test_set);
    for i=1:train_set
        X_train(:,i) = Wi.' * Dt_train(:,i);
    end
    for i=1:test_set
        X_test(:,i) = Wi.' * Dt_test(:,i);
    end
    
    
%##########################################################################
%#########################     CLASSIFICATION     #########################
%##########################################################################
    
                         % KNN CLASSIFICATION
    
    K = input("-> Enter the value of K: ");
    
    test_set = size(X_test,2);
    dim = size(X_train,1);
    
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
    
    
    
    %Accuracy calculation
    count = 0;
    
    for i=1:test_set
        if final_labels(i) == Label_test(i)
            count = count + 1;
        end
    end
    
    accuracy = (count/test_set);
    disp(" ");
    disp("***** LDA_KNN Classification Accuracy for Dataset ~ "+s+" ~ is: "+accuracy+" *****");
    
    
%##########################################################################


else
    disp("SESSION TERMINATED...");
end