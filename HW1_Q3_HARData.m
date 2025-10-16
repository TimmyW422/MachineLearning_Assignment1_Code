clear all, close all;



x_train = load('X_train.txt');
x_test = load('X_test.txt');

y_train = load('y_train.txt');
y_test = load('y_test.txt');


Alpha = 0.0001;

Features_data = [x_train; x_test];% Features
quality_data = [y_train; y_test]; %class labels

feature_mean = mean(Features_data);
feature_std = std(Features_data);
feature_std(feature_std == 0) = 1;
feature_norm = (Features_data-feature_mean)./ feature_std;


class_labels = unique(quality_data); %only present classes
num_classes = length(class_labels);
[N_samples, N_features] = size(feature_norm);

priors = zeros(num_classes,1);
means = zeros(num_classes, N_features);
covar = cell(num_classes,1);

for i = 1:num_classes
    Ci = class_labels(i); %Quality score/class label
    features_i = feature_norm(quality_data == Ci, :); %Grab data that belongs to class i
    Num_samples_i = size(features_i,1); %number of samples in class i
    
    %Fix error when a class has no samples
    if Num_samples_i <= 1
        priors(i) = Num_samples_i / N_samples; % Priors can still be 0
        
        %for classes with 0 or 1 sample
        covar{i} = eye(N_features);
        if Num_samples_i == 1
             means(i,:) = features_i; % Use the single sample as the mean
        else
             means(i,:) = zeros(1, N_features);
        end
        continue; 
    end

    priors(i) = Num_samples_i / N_samples;
    means(i,:) = mean(features_i, 1);
    cov_sample_avg = cov(features_i, 1);

    %Calc lambda
    trace_c = trace(cov_sample_avg);
    D = N_features;
    R = min(Num_samples_i - 1, D);

    %for cases with 0 occurence
    if R<=0
        R=D;
    end

    lambda = Alpha * (trace_c / R);

    cov_reg = cov_sample_avg + lambda * eye(D);

    
    covar{i} = cov_reg;

end

Y_predict = zeros(N_samples, 1);
scores = zeros(1, num_classes);

for n =1:N_samples
    sample_n = feature_norm(n, :);

    for i =1:num_classes
        P_Ci = priors(i);
        Mu_i = means(i,:);
        C_reg_i = covar{i};

        if P_Ci == 0 
            scores(i) = -inf;
            continue;
        end

        %Utilized AI to help with this part a lot
        %Due to the large features number the origional log method used for the Wine data was not working
        try
            % 1. Cholesky Factorization: C = L*L'
            L = chol(C_reg_i, 'lower'); 
            
            % 2. Log-Determinant Term: 2 * sum(log(diag(L)))
            log_C = 2 * sum(log(diag(L))); 
            
            % 3. Mahalanobis Distance: || L^-1 * (x - Mu) ||^2
            diff = (sample_n - Mu_i)';
            y = L \ diff; % Solve L*y = diff
            mahal_dist = sum(y.^2); % Calculate ||y||^2
            
            % 4. Discriminant Score: g_i(x)
            g_ix = log(P_Ci) - 0.5 * log_C - 0.5 * mahal_dist;
            scores(i) = g_ix;
            
        catch
            % Catch numerical instability errors during Cholesky
            scores(i) = -inf;
        end
    % diff = (sample_n - Mu_i)';
    % mahal_dist = diff' * (C_reg_i\diff);
    % 
    % log_C = log(det(C_reg_i));
    % 
    % g_ix = log(P_Ci) - 0.5*log_C - 0.5*mahal_dist;
    % scores(i) = g_ix;

    end

    [~, max_val] = max(scores);
    Y_predict(n) = class_labels(max_val);
end

fprintf('HAR RESULTS');

num_errors = sum(Y_predict ~= quality_data);
error_rate = num_errors / N_samples;

fprintf('   - Error Probability Estimate (Pe): %.4f\n', error_rate);
fprintf('   - Total Misclassifications: %d / %d\n', num_errors, N_samples);

conf_matrix = confusionmat(quality_data, Y_predict, 'Order', class_labels);
fprintf('   - Confusion Matrix:\n');
disp(conf_matrix);

C_overall = cov(feature_norm);


%PLOTS

[V, D_diag] = eig(C_overall); 
eigenvalues = diag(D_diag);

% Sort eigenvalues and eigenvectors 
[eigenvalues, sort_idx] = sort(eigenvalues, 'descend');
V_sorted = V(:, sort_idx);

% Use first two pricipal components
Phi_2D = V_sorted(:, 1:2); 

% 4. Project the scaled data onto the 2D subspace
% Projected_data (N_samples x 2)
Projected_data = feature_norm * Phi_2D; 

%Scatter plot by label
figure;
gscatter(Projected_data(:, 1), Projected_data(:, 2), quality_data);

title('PCA Projection of White Wine Data (PC1 vs. PC2)');
xlabel(sprintf('First Principal Component (Variance: %.2f%%)', ...
    100 * eigenvalues(1) / sum(eigenvalues)));
ylabel(sprintf('Second Principal Component (Variance: %.2f%%)', ...
    100 * eigenvalues(2) / sum(eigenvalues)));