clear all, close all,

%Set up

C_priors = [0.25, 0.25, 0.25, 0.25]; %Class Priors all 0.25
N_classes = 4;
N_samples = 10000;

%Guassian Parameters
    %Mean Values
mu = cell(1, N_classes);
    %Arbitrary Values
mu{1} = [1,2];
mu{2} = [6,7];
mu{3} = [10,2];
mu{4} = [3,8];

    %Cov Matricies
Sigma = cell(1, N_classes);
Sigma{1} = [2, 1; 1, 2];
Sigma{2} = [1, 0; 0, 2.5];
Sigma{3} = [0.9, 0; 0, 0.9];
Sigma{4} = [2, -1; -1, 2.5];

%Define X
X = zeros(2, N_samples);
True_Labels = zeros(1, N_samples);
Predicted_Labels = zeros(1,N_samples);

rng(45); %Keep same samples for debugging

prior_cum = cumsum(C_priors);


%Gathering random data from given classes
for k = 1:N_samples
    r = rand(); %Random number 0-1 to determine which gaussian data set will be chosen
    %Chosing which class will be chosen based on 2 and CDS
    if r <= prior_cum(1)
        True_Label = 1;
    elseif (prior_cum(1) < r) && (r <= prior_cum(2))
        True_Label = 2;
    elseif (prior_cum(2) < r) && (r <= prior_cum(3))
        True_Label = 3;
    elseif (prior_cum(3) < r) && (r <= prior_cum(4))
        True_Label = 4;
    end

    True_Labels(k) = True_Label;
    
    %Assign a vector withing the chosen Gaussian Class
    X(:,k) = mvnrnd(mu{True_Label}' , Sigma{True_Label}, 1)';
end

%MAP Classification method
Likelihood = zeros(N_classes, N_samples);

%Determine the probability density for each class
for i = 1:N_classes
    Likelihood(i,:) = mvnpdf(X', mu{i},Sigma{i})*C_priors(i);
end

%Assigns the Label from the class with the highest density value to predicted labels
%Tracking which class each random vector is most likely from
[~, Predicted_Labels] = max(Likelihood, [], 1);

%Confusion Matrix
C_matrix = zeros(N_classes, N_classes);

%True Class
for j = 1:N_classes
    True_Class_j = (True_Labels == j); %Track when value in True_Labels vector is equal to class index
    Num_j = sum(True_Class_j);

    for i = 1:N_classes
        Num_ij = sum(Predicted_Labels(True_Class_j) == i); %When the True_labels vector matched the class index
        C_matrix(i,j)= Num_ij / Num_j;
    end
end

%Plot
figure;
hold on
title('MAP Classification');
xlabel('x_1');
ylabel('x_2');
grid on;

marker_shapes = {'o','d','s','^'};
correct_color = [0,0.6,0]; %Green
incorrect_color = [1,0,0];%Red

Correct = (True_Labels == Predicted_Labels);

for n = 1:N_samples
    if Correct(n)
        marker = correct_color;
    else
        marker = incorrect_color;
    end

    marker_style = marker_shapes{True_Labels(n)}; %Chose marker shape based on class number
    scatter(X(1,n), X(2,n), 15, marker, marker_style,"filled", 'MarkerEdgeColor', 'k', 'LineWidth', 0.2);

end
legend('True Label 1 (\circ)', 'True Label 2 (diamond)', 'True Label 3 (square)', 'True Label 4 (triangle)');
hold off;

%Part B

Lambda = [0, 10, 10, 100;
          1, 0, 10, 100;
          1, 1, 0, 100;
          1, 1, 1, 0];


p_x = sum(Likelihood, 1); %Sum likelyhood values for each index
P_Lx = Likelihood./p_x; %Posterior prob

Con_risk = Lambda * P_Lx;
[Min_risk, ERM_Pred_Lables] = min(Con_risk,[],1);

Min_Expected_Risk = mean(Min_risk);

disp('ERM Results: ');
disp(Min_Expected_Risk);
