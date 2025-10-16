clear all, close all,
N = 10000; p0 = 0.65; p1 = 0.35;
u = rand(1,N)>=p0; N0 = length(find(u==0)); N1 = length(find(u==1));

mu = [-1/2;-1/2;-1/2]; Sigma = [1,-0.5,0.3;-0.5,1,-0.5;0.3,-0.5,1];
r0 = mvnrnd(mu, Sigma, N0);
figure(1), plot3(r0(:,1),r0(:,2),r0(:,3),'b.'); axis equal, hold on,
mu = [1;1;1]; Sigma = [1,0.3,-0.2;0.3,1,0.3;-0.2,0.3,1];
r1 = mvnrnd(mu, Sigma, N1);
figure(1), plot3(r1(:,1),r1(:,2),r1(:,3),'r.'); axis equal, hold on,

%Create data matrix and Label vector
X = zeros(N,3);
L_true = u';%[zeros(N0, 1); ones(N1, 1)];

%Changing Sigmas to Identity matrix for Part B
mu0 =  [-1/2;-1/2;-1/2];
Sigma0_NB = [1,0,0;0,1,0;0,0,1];
mu1 = [1;1;1];
Sigma1_NB = [1,0,0;0,1,0;0,0,1];


%L=0
Class0 = find(L_true == 0);
%L=1
Class1 = find(L_true == 1);

X(Class0, :) = r0;
X(Class1, :) = r1;

%Find mean vector and cov
P0 = mvnpdf(X, mu0', Sigma0_NB); 
P1 = mvnpdf(X, mu1', Sigma1_NB);
Lambda = P1 ./ P0;
%Thresholds
gamma_threshold = unique(sort(Lambda));
%For 0 --> infinity
gamma_threshold = [0; gamma_threshold; Inf];

TP = zeros(length(gamma_threshold), 1); %P(D=1|L=1)
FP = zeros(length(gamma_threshold), 1); %P(D=1|L=0)
%FN = zeros(length(gamma_thresholds), 1); %P(D=0|L=1)

%Iterate and find each true and false positive
for i = 1:length(gamma_threshold)
    gamma = gamma_threshold(i);
    D = (Lambda>gamma);

    True_Pos = sum(D(Class1));
    False_Pos = sum(D(Class0));

    TP(i) = True_Pos/N1;
    FP(i) = False_Pos/N0;
end

%Plot ROC
figure(2);
plot(FP, TP, 'b-', 'LineWidth',1.5); hold on;

%Plot Min Expected Risk
MER_point = p0/p1;
[~, MER] = min(abs(gamma_threshold - MER_point));
plot(FP(MER), TP(MER), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'g'); hold on;

%Miss
P_miss = 1 - TP;

%Error Prob for every threshold value
P_error = FP .* p0 + P_miss .* p1;
[Min_error_value, min_error_index] = min(P_error);

Gamma_min_error = gamma_threshold(min_error_index);
TP_min_error = TP(min_error_index);
FP_min_error = FP(min_error_index);


plot(FP_min_error, TP_min_error, 'rx', 'MarkerSize',8, 'LineWidth',2); 
legend('ROC Curve', 'MER', 'Min Error Point');
hold off;

display(Gamma_min_error/(p0/p1));
display(Min_error_value);