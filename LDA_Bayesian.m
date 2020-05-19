% LDA Followed by Bayesian Classification

clc;
close all;

disp("********************************** LDA BAYESIAN **********************************");
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
    
    P_of_Wi = train_set / img;

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
    
    P_of_Wi = train_set / img;
    
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
    
    train_set = 612;
    test_set = 272;
    
    C = 68;
    dim = 48 * 40;
    n = train_set / C;
    img = 884;
    
    P_of_Wi = train_set / img;
    
    Dt_train = zeros(dim,train_set);
    Dt_test = zeros(dim,test_set);
    
    Label_train = zeros(train_set,1);
    Label_test = zeros(test_set,1);
    
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
    
    
else
    disp("** You Chose: "+choice+" -> EXIT !! **");
end



%##########################################################################
if choice == 1 || choice == 2 || choice == 3
    
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
    alp = 1;
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
    [W,EigVal] = eigs(Sb,Sw,C-1);
    
    
    % Transformation of Matrices
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
    dim = size(X_train,1);         % Updated dimension after transformation
    
    % recalculate the mean
    new_mean = zeros(dim,C);
    val = 0;
    for i=1:C
        for j=1:n
            new_mean(:,i) = new_mean(:,i) + X_train(:,i+val);
            val = val + 1;
        end
        val = val - 1;
        new_mean(:,i) = new_mean(:,i) / n;
    end
    
    
    
    % recalculate the variance
    new_variance = zeros(dim,dim,C);
    new_variance_inv = zeros(dim,dim,C);
    beta = 1;
    val = 0;
    for i=0:C-1
        for j=1:n
            val = val + 1;
            new_variance(:,:,i+1) = new_variance(:,:,i+1) + ( ( X_train(:,i+val) - new_mean(:,i+1) ) * ( X_train(:,i+val) - new_mean(:,i+1) ).' );
        end
        val = val - 1;
        
        new_variance(:,:,i+1) = new_variance(:,:,i+1) + beta * eye(dim);
        new_variance_inv(:,:,i+1) = inv(new_variance(:,:,i+1));
    end
    
    
    
    
    % Calculate the Wi, wi, wio
    % We consider that the covariance matrices are arbitrary.
    % g(x) = (x' * Wi * x) + (wi' * x) + (wio)
    Wi = zeros(dim,dim,C);
    wi = zeros(dim,C);
    wio = zeros(C,1);
    
    for i=1:C
        Wi(:,:,i) = (-1/2) * (new_variance_inv(:,:,i));
        wi(:,i) = new_variance_inv(:,:,i) * new_mean(:,i);
        wio(i) = (-1/2 * new_mean(:,i).' * new_variance_inv(:,:,i) * new_mean(:,i)) + (-1/2 * log(det(new_variance(:,:,i)))) + log(P_of_Wi);
    end
    
    
    
    
    % Solve the equation: g(x) = (x' * Wi * x) + (wi' * x) + (wio)
    final_labels = zeros(test_set,1);
    for i = 1:test_set
        gmax = (X_test(:,i).' * Wi(:,:,1) * X_test(:,i)) + (wi(:,1).' * X_test(:,i)) + wio(1);
        
        for j=1:C
            gval = (X_test(:,i).' * Wi(:,:,j) * X_test(:,i)) + (wi(:,j).' * X_test(:,i)) + wio(j);
            
            if(gval >= gmax)
                gmax = gval;
                final_labels(i) = j;       %We will be checking the labels for accuracy
            end
            
        end
    end
    
    
    
    % Accuracy Calculations
    match = 0;
    for i=1:test_set
        if final_labels(i) == Label_test(i)
            match = match + 1;
        end
    end
    
    accuracy = match / test_set;
    disp(" ");
    disp("***** LDA_Bayesian Classification Accuracy for Dataset ~ "+s+" ~ is: "+accuracy+" *****");
    
%##########################################################################


else
    disp("SESSION TERMINATED...");
end