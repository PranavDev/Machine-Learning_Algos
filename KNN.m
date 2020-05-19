% KNN Classifier

clc;
close all;

disp("********************************** KNN **********************************");
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
            if j == 3
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
    
    
elseif choice == 3
    s = "pose.mat";
    disp("** You Chose: "+choice+" -> pose.mat **");
    load('/Users/pranavdeo/Desktop/MLProject/Data/pose.mat');
    %pose.mat has c:68, 13 images per class/suject = 884
    
    train_set = 680;
    test_set = 204;
    
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
            if j == 1 || j == 5 || j == 9
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
    
    
else
    disp("** You Chose: "+choice+" -> EXIT !! **");
end



%##########################################################################
if choice == 1 || choice == 2 || choice == 3
    
    K = input("-> Enter the value of K (odd) : ");
    
    % Euclidean Distance calculation
    final_Label = zeros(test_set,1);
    Label = zeros(train_set,1);
    distances = zeros(train_set,1);
    sum = zeros(K,1);
    indx = 0;
    
    for i=1:test_set
        for j=1:train_set
            d = (sqrt(Dt_test(:,i) - Dt_train(:,j)).' * (Dt_test(:,i) - Dt_train(:,j)));
            indx = indx + 1;
            distances(indx) = d;
            Label(indx) = Label_train(j);
            
        end
        
        %getting K smallest elements array
        [Dk, I] = mink(distances,K); 
        
        for a=1:K
            sum(a) = Label(I(a));
        end
        
        %sum = floor(sum/K);
        final_Label(i) = mode(sum);

        sum = 0;
        indx = 0;
    end
    
    
    
    
    %Accuracy calculation
    count = 0;
    
    for i=1:test_set
        if final_Label(i) == Label_test(i)
            count = count + 1;
        end
    end
    
    accuracy = (count/test_set);
    disp(" ");
    disp("***** KNN Classification Accuracy for Dataset ~ "+s+" ~ is: "+accuracy+" *****");
    
    
%##########################################################################

else
    disp("SESSION TERMINATED...");
end
    