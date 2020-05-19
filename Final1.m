
% --------- Author: PRANAV H. DEO -- (116763752) -- ENTS669D -------- %

% All the Classifiers are integrated into this Code.

%You can run it for every mentioned classifier and train/test it on every 
% sample dataset as you wish.

% When changing the training and testing datasets indices, also change the
% train_set, test_set variable value as per the number of training and
% testing sample points.

% Used all the datasets. data.mat has two operations 1.When C=200 and C=2.
% Final1.m file has code for C=200. Visit Final2.m file specifically for
% data.mat when C=2 (Neutral and Expression).

% Implemented Classifiers: ML-Bayesian, PCA-Bayesian, LDA-Bayesian, KNN,
% PCA-KNN, LDA-KNN.

% You are interrupted for a choice of dataset and a classifier.

clc;
close all;

disp("Choose one dataset: 1.data.mat; 2.illumination.mat; 3.pose.mat; 4.EXIT\n");
ch = input("Enter your choice: ");

if ch == 1
    s = "data.mat";
    disp(" ");
    disp(" ");
    disp("** You Chose: "+ch+" -> data.mat **");
    load('/Users/pranavdeo/Desktop/MLProject/Data/data.mat');

        
    train_set = 400;                    %use 400 to train
    test_set = 200;                     %use 200 to test

    dim = 24 * 21;                      %dimensions of one image
    C = 200;                            %classes/subjects
    img = 600;                          %total images
    n = train_set / C;                  %no. of images being trained per class
    alpha = 1;

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
    


elseif ch == 2
    
    s = "illumination.mat";
    disp(" ");
    disp(" ");
    disp("** You Chose: "+ch+" -> illumination.mat **");
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
    

    
    
elseif ch == 3
    
    s = "pose.mat";
    disp(" ");
    disp(" ");
    disp("** You Chose: "+ch+" -> pose.mat **");
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
    disp("** You Chose: "+ch+" -> EXIT !! **");
end






%-------------------------------Start-Program------------------------------%

if ch ~= 4

    disp(" ");
    disp("Classification 0.Exit 1.Bayesian 2.PCA-Bayesian 3.LDA-Bayesian 4.KNN 5.PCA-KNN 6.LDA-KNN");
    choice = input("Enter your choice: ");
    
    %##############################################################################################

    if choice == 0
        disp(" ");
        disp("SESSION TERMINATED...");

    %##############################################################################################
    
    
    %##############################################################################################
    elseif choice == 1

        disp(" ");
        disp("*********** ML-Bayesian Classification ***********");
        st = "ML-Bayesian";
        
        % calculate the mean per class in the training dataset
        mean = zeros(dim,C);
        val = 0;
        for i = 1:C
            for j=1:n
                mean(:,i) = mean(:,i) + Dt_train(:,i+val);
                val = val+1;
            end
            val = val - 1;
            mean(:,i) = mean(:,i) / n;
        end

        
        % calculating the variance
        variance = zeros(dim,dim,C);
        variance_inv = zeros(dim,dim,C);
        val = 0;

        for i = 0:C-1
            for j = 1:n
                variance(:,:,i+1) = variance(:,:,i+1) + ((Dt_train(:,i+val+1) - mean(:,i+1)) * (Dt_train(:,i+val+1) - mean(:,i+1)).');
                val = val + 1;
            end
            val = val - 1;
            variance(:,:,i+1) = variance(:,:,i+1) / n;

            variance(:,:,i+1) = variance(:,:,i+1) + (alpha * eye(dim));
            variance_inv(:,:,i+1) = inv(variance(:,:,i+1));
        end

        
        % We consider that the covariance matrices are arbitrary.
        % g(x) = (x' * Wi * x) + (wi' * x) + (wio)
        Wi = zeros(dim,dim,C);
        wi = zeros(dim,C);
        wio = zeros(C,1);

        for i = 1:C
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


    
    %##############################################################################################
    elseif choice == 2

        disp(" ");
        disp("*********** PCA-Bayesian Classification ***********");
        st = "PCA-Bayesian";
        
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
        alpha = 1;
        for i=0:C-1
            for j=1:n
                val = val + 1;
                variance(:,:,i+1) = variance(:,:,i+1) + ((X_train(:,i+val) - (mean(:,i+1))) * (X_train(:,i+val) - (mean(:,i+1))).');
            end
            val = val - 1;

            variance(:,:,i+1) = variance(:,:,i+1) + (alpha * eye(dim));
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


    %##############################################################################################

    

    %##############################################################################################
    elseif choice == 3

        disp(" ");
        disp("*********** LDA-Bayesian Classification ***********");
        st = "LDA-Bayesian";
        
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
        alpha = 0.05;
        index = 0;
        for i=1:C
            for j=1:n
                index = index + 1;
                temp = (Dt_train(:,index) - mean_per_class(:,i)) * (Dt_train(:,index) - mean_per_class(:,i)).';
            end

            temp = temp + alpha * eye(dim);       % Avoiding a singular matrix
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
        beta = 0.05;
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


    %##############################################################################################

    
    
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

    
    
    %##############################################################################################
    else
        
        disp(" ");
        disp("*********** LDA-KNN Classification ***********");
        st = "LDA-KNN";
        
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
        alpha = 0.05;
        index = 0;
        for i=1:C
            for j=1:n
                index = index + 1;
                temp = (Dt_train(:,index) - mean_per_class(:,i)) * (Dt_train(:,index) - mean_per_class(:,i)).';
            end

            temp = temp + alpha * eye(dim);       % Avoiding a singular matrix
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
    end
    
    
%##############################################################################################

    % Now check for accuracy
        match = 0;
        for i=1:test_set
            if final_labels(i) == Label_test(i)
                match = match + 1;
            end
        end

        accuracy = (match / test_set);
        disp(" ");
        disp("***** Accuracy for Dataset ~ "+s+" ~ by using "+st+" Classification is: "+accuracy+" *****");

        
%##############################################################################################
%----------------------------------------END-Program------------------------------------------%
else
    
    if ch == 0
        disp(" ");
        disp("SESSION TERMINATED...");
    end
end

%##############################################################################################