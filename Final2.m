
% --------- Author: PRANAV H. DEO -- (116763752) -- ENTS669D -------- %

% Specifically made for dataset: data.mat when C=2 (Neutral and
% Expressions).

% Implemented Classifiers: ML-Bayesian, PCA-Bayesian, LDA-Bayesian, KNN,
% PCA-KNN, LDA-KNN.

% This code contains the classification done on data.mat when C=2 (Neutral
% & Expression). We can randomly allocate the training and testing data.

clc;
close all;
load('/Users/pranavdeo/Desktop/MLProject/Data/data.mat');
disp("******* Dataset: data.mat considering C = 2 *******");
disp(" ");

C = 2;
img = 400;

train_set = 280;
test_set = 120;
alpha = 1;
flag = 0;

dim = 24 * 21;
P_of_Wi = train_set / img;

Dt_train = zeros(dim,train_set);
Dt_test = zeros(dim,test_set);

Label_train = zeros(train_set,1);
Label_test = zeros(test_set,1);

train_index = 0;
test_index = 0;

for i=0:199
    for j=1:C
        if i >= 140
            test_index = test_index + 1;
            Dt_test(:,test_index) = reshape(face(:,:,3*i+j), [dim,1]);
            Label_test(test_index) = j;
        else
            train_index = train_index + 1;
            Dt_train(:,train_index) = reshape(face(:,:,3*i+j), [dim,1]);
            Label_train(train_index) = j;
        end
    end
end


disp("Train Index: "+(train_set)+" and Test Index: "+(test_set));
disp(" ");



disp("Classification 0.Exit  1.Bayesian  2.PCA-Bayesian  3.LDA-Bayesian  4.KNN  5.PCA-KNN  6.LDA-KNN");
choice = input("Enter your choice: ");


%##############################################################################################
if choice == 0
    disp(" ");
    disp("SESSION TERMINATED...");
    flag = 1;
    
%##############################################################################################

elseif choice == 1

    disp("*********** ML-Bayesian Classification ***********");
    st = "ML-Bayesian";

    % calculate the mean per class in the training dataset
    mean = zeros(dim,train_set);
    for i=1:train_set
        mean(:,i) = mean(:,i) + Dt_train(:,i);
    end
    mean = mean / 200;

    % calculating the variance
    variance = zeros(dim,dim,train_set);
    variance_inv = zeros(dim,dim,train_set);

    for i=1:train_set
        variance(:,:,i) = variance(:,:,i) + ((Dt_train(:,i) - mean(:,i)) * (Dt_train(:,i) - mean(:,i)).');
    end

    for i=1:train_set
        variance(:,:,i) = variance(:,:,i) + (alpha * eye(dim));
    end

    for i=1:train_set
        variance_inv(:,:,i) = inv(variance(:,:,i));
    end

    % We consider that the covariance matrices are arbitrary.
    % g(x) = (x' * Wi * x) + (wi' * x) + (wio)
    Wi = zeros(dim,dim,train_set);
    wi = zeros(dim,train_set);
    wio = zeros(train_set,1);

    for i = 1:train_set
        Wi(:,:,i) = -(1/2) * variance_inv(:,:,i);
        wi(:,i) = variance_inv(:,:,i) * mean(:,i);
        wio(i) = -(1/2) * ( (mean(:,i).' * variance_inv(:,:,i) * mean(:,i)) + (log(det(variance(:,:,i)))) ) + log(P_of_Wi);
    end


    % Now we use the testdata values as x to solve the equation: g(x) = (x' * Wi * x) + (wi' * x) + (wio)
    final_labels = zeros(test_set,1);

    for i = 1:test_set
        gmax = (Dt_test(:,i).'* Wi(:,:,1) * Dt_test(:,i)) + (wi(:,1).' * Dt_test(:,i)) + wio(1);

        for j=1:C
            gval = (Dt_test(:,i).'* Wi(:,:,j) * Dt_test(:,i)) + (wi(:,j).' * Dt_test(:,i)) + wio(j);

            if(gval >= gmax)
                gmax = gval;
                final_labels(i) = j;       %We will be checking the labels for accuracy
            end

        end
    end

    
%##############################################################################################

elseif choice == 2
    
    disp("*********** PCA-Bayesian Classification ***********");
    st = "PCA-Bayesian";

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

    mean = zeros(dim,train_set);
    for i=1:train_set
        mean(:,i) = mean(:,i) + X_train(:,i);
    end
    mean = mean / 200;

    % calculating the variance
    variance = zeros(dim,dim,train_set);
    variance_inv = zeros(dim,dim,train_set);

    for i=1:train_set
        variance(:,:,i) = variance(:,:,i) + ((X_train(:,i) - mean(:,i)) * (X_train(:,i) - mean(:,i)).');
    end

    for i=1:train_set
        variance(:,:,i) = variance(:,:,i) + (alpha * eye(dim));
    end

    for i=1:train_set
        variance_inv(:,:,i) = inv(variance(:,:,i));
    end

    % We consider that the covariance matrices are arbitrary.
    % g(x) = (x' * Wi * x) + (wi' * x) + (wio)
    Wi = zeros(dim,dim,train_set);
    wi = zeros(dim,train_set);
    wio = zeros(train_set,1);

    for i = 1:train_set
        Wi(:,:,i) = -(1/2) * variance_inv(:,:,i);
        wi(:,i) = variance_inv(:,:,i) * mean(:,i);
        wio(i) = -(1/2) * ( (mean(:,i).' * variance_inv(:,:,i) * mean(:,i)) + (log(det(variance(:,:,i)))) ) + log(P_of_Wi);
    end


    % Now we use the testdata values as x to solve the equation: g(x) = (x' * Wi * x) + (wi' * x) + (wio)
    final_labels = zeros(test_set,1);

    for i = 1:test_set
        gmax = (X_test(:,i).'* Wi(:,:,1) * X_test(:,i)) + (wi(:,1).' * X_test(:,i)) + wio(1);

        for j=1:C
            gval = (X_test(:,i).'* Wi(:,:,j) * X_test(:,i)) + (wi(:,j).' * X_test(:,i)) + wio(j);

            if(gval >= gmax)
                gmax = gval;
                final_labels(i) = j;       %We will be checking the labels for accuracy
            end

        end
    end

    
    
%##############################################################################################

elseif choice == 3
    
    disp(" ");
    disp("*********** LDA-Bayesian Classification ***********");
    st = "LDA-Bayesian";
    alpha = 0.05;
    beta = 0.05;

    % Mean per class calculation
    mean = zeros(dim,1);
    for i=1:train_set
        mean = mean + Dt_train(:,i);
    end
    mean = mean / train_set;


    % Total mean over all classes calculation
    Mean_all = zeros(dim,1);
    for i=1:train_set
        Mean_all = Mean_all + Dt_train(:,i);
    end
    Mean_all = Mean_all / train_set;


    % Within Class scatter matrix (Sw)
    Sw = zeros(dim,dim);
    
    for i=1:train_set
        temp = (Dt_train(:,i) - mean) * (Dt_train(:,i) - mean).';
    end

    temp = temp + alpha * eye(dim);       % Avoiding a singular matrix
    Sw = Sw + temp;


    % Between Class scatter matrix (Sb)
    Sb = zeros(dim,dim);
    for i=1:C
        Sb = Sb + train_set * ( (mean - Mean_all) * (mean - Mean_all).' );
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
    new_mean = zeros(dim,1);
    for i=1:200
        new_mean = new_mean + X_train(:,i);
    end
    new_mean = new_mean / train_set;


    % recalculate the variance
    new_variance = zeros(dim,dim,train_set);
    new_variance_inv = zeros(dim,dim,train_set);
    for i=1:train_set
        new_variance(:,:,i) = new_variance(:,:,i) + ((X_train(:,i) - new_mean) * (X_train(:,i) - new_mean).');
    end
    for i=1:train_set
        new_variance(:,:,i) = new_variance(:,:,i) + (beta * eye(dim));
    end
    for i=1:train_set
        new_variance_inv(:,:,i) = inv(new_variance(:,:,i));
    end



    % Calculate the Wi, wi, wio
    % We consider that the covariance matrices are arbitrary.
    % g(x) = (x' * Wi * x) + (wi' * x) + (wio)
    Wi = zeros(dim,dim,C);
    wi = zeros(dim,C);
    wio = zeros(C,1);

    for i=1:C
        Wi(:,:,i) = (-1/2) * (new_variance_inv(:,:,i));
        wi(:,i) = new_variance_inv(:,:,i) * new_mean;
        wio(i) = (-1/2 * new_mean.' * new_variance_inv(:,:,i) * new_mean) + (-1/2 * log(det(new_variance(:,:,i)))) + log(P_of_Wi);
    end


    % Solve the equation: g(x) = (x' * Wi * x) + (wi' * x) + (wio)
    final_labels = zeros(test_set,1);
    for i = 1:test_set
        gmax = (X_test(:,i).'* Wi(:,:,1) * X_test(:,i)) + (wi(:,1).' * X_test(:,i)) + wio(1);

        for j=1:C
            gval = (X_test(:,i).'* Wi(:,:,j) * X_test(:,i)) + (wi(:,j).' * X_test(:,i)) + wio(j);

            if(gval >= gmax)
                gmax = gval;
                final_labels(i) = j;       %We will be checking the labels for accuracy
            end

        end
    end
    
      
    
%##############################################################################################

elseif choice == 4
    
    disp(" ");
    disp("*********** KNN Classification ***********");
    st = "KNN";

    K = input("-> Enter the value of K (odd) : ");

    % Euclidean Distance calculation
    final_labels = zeros(test_set,1);
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
        final_labels(i) = mode(sum);

        sum = 0;
        indx = 0;
    end


    
%##############################################################################################

elseif choice == 5
    
    disp(" ");
    disp("*********** PCA-KNN Classification ***********");
    st = "PCA-KNN";

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

      
      
%##############################################################################################

elseif choice == 6
    disp(" ");
    disp("*********** LDA-KNN Classification ***********");
    st = "LDA-KNN";

    % Mean per class calculation
    mean = zeros(dim,1);
    for i=1:train_set
        mean = mean + Dt_train(:,i);
    end
    mean = mean / train_set;


    % Total mean over all classes calculation
    Mean_all = zeros(dim,1);
    for i=1:train_set
        Mean_all = Mean_all + Dt_train(:,i);
    end
    Mean_all = Mean_all / train_set;


    % Within Class scatter matrix (Sw)
    Sw = zeros(dim,dim);
    alpha = 0.05;
    index = 0;

    for i=1:train_set
        temp = (Dt_train(:,i) - mean) * (Dt_train(:,i) - mean).';
    end

    temp = temp + alpha * eye(dim);       % Avoiding a singular matrix
    Sw = Sw + temp;


    % Between Class scatter matrix (Sb)
    Sb = zeros(dim,dim);
    for i=1:train_set
        Sb = Sb + train_set * ( (mean - Mean_all) * (mean - Mean_all).' );
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
    


%##############################################################################################    
else
    disp(" ");
    disp("SESSION TERMINATED...");
end
%##############################################################################################    
    

if flag == 0
    match = 0;
    for i=1:test_set
        if final_labels(i) == Label_test(i)
            match = match + 1;
        end
    end

    accuracy = (match / test_set);
    disp(" ");
    disp("***** Accuracy by using "+st+" Classification is: "+accuracy+" *****");
    
else
    disp(" ");
    disp("SESSION TERMINATED...");
end

%##############################################################################################