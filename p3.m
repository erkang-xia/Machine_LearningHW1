%% P3
clear all;
wine = readtable('winequality-white.csv');
wine = table2array(wine);
[N, ~] = size(wine);
cSize = 11; 

priors =[1 1 1 1 1 1 1 1 1 1 1];

for i = 1:N
    label(i) = wine(i,12);
end


meanVectors = zeros(11,12);
for i = 1:cSize
    meanVectors(i,:) = mean(wine(wine(:,12)==i-1,:)); 
end 
meanVectors(isnan(meanVectors))=0; 


classVar = zeros(11,11);
for i = 1:cSize
    classVar = cov(wine(wine(:,12)==i-1,1:11));
    classVar(isnan(classVar)) = 0;
    rCov(:,:,i) = classVar + 0.000001*(trace(classVar)/rank(classVar))*eye(11); %λ = 0.000001
end 
rCov(isnan(rCov))=0; 


for i=1:cSize
    classPDF(i,:) = evalgaussian(wine(:,1:11)',meanVectors(i,1:11)',rCov(:,:,i),cSize);
end
classPDF(isnan(classPDF))=0;

lossMatrix = ones(cSize,cSize) - eye(cSize);

Px = priors*classPDF;
Px(isnan(Px)) = 0; 
posteriors = classPDF.*repmat(priors',1,N)./repmat(Px,cSize,1);% P(L=l|x)
posteriors(isnan(posteriors))=0;
risKMat = lossMatrix * posteriors;
[~, Decision] = min(risKMat, [], 1);
falseDeci = (Decision~=label);
confusionMatrix = zeros(cSize, cSize); 
for ind1=1:cSize
    for ind2=1:cSize
        confusionMatrix(ind1,ind2) = sum(Decision==ind1 & label==ind2)
    end
end
probError = sum(falseDeci == 1)/N;
fprintf('Probability of Error: Error=%1.2f%%\n',100*probError);
%Accuracy 
accuracy = sum(Decision==label)/N;  
fprintf('Accuracy:%1.2f%%\n',100*accuracy);

graphname = ["fixed acidity";"volatile acidity";"citric acid";"residual sugar";...
    "chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";...
    "sulphates";"alcohol";"quality"];

%Plot
%Feature Plot
figure,
for i=1:12
    subplot(3,4,i)
    hist(wine(:,i))
    title(graphname(i))
end
%Frequncy plot
figure,
hist(label)
title("Wine quality Frequency")
%Estimate Frequncy plot
figure,
hist(Decision)
title("Esitmated Wine quality Frequency")
figure,


clear all;
Human = table2array(readtable('X_train.txt'));
labels = (table2array(readtable('y_train.txt')))';
[N, n] = size(Human);
cSize = 6; 

priors =[0.5 0.5 0.5 0.5 0.5 0.5];


meanVectors = zeros(cSize,n);
for i = 1:cSize
    meanVectors(i,:) = mean(Human(labels(1:N)==i,:)); 
end 


classVar = zeros(cSize,n);
for i = 1:cSize
    classVar = cov(Human(labels(1:N)==i,:));

    rCov(:,:,i) = classVar + 8.2*(trace(classVar)/rank(classVar))*eye(n); %%λ = 8.2
end 

%Gauss Conditional Probability
for i=1:cSize
    classPDF(i,:) = evalgaussian(Human',meanVectors(i,:)',rCov(:,:,i),cSize);
end


lossMatrix = ones(cSize,cSize) - eye(cSize);

Px = priors*classPDF;
Px(isnan(Px)) = 0; 
posteriors = classPDF.*repmat(priors',1,N)./repmat(Px,cSize,1);% P(L=l|x)
posteriors(isnan(posteriors))=0;
risKMat = lossMatrix * posteriors;
[~, Decision] = min(risKMat, [], 1);
falseDeci = (Decision~=labels);
confusionMatrix = zeros(cSize, cSize); 
for ind1=1:cSize
    for ind2=1:cSize
        confusionMatrix(ind1,ind2) = sum(Decision==ind1 & labels==ind2,'all')
    end
end
probError = sum(falseDeci == 1)/N;
fprintf('Probability of Error: Error=%1.2f%%\n',100*probError);
%Accuracy 
accuracy = sum(Decision==labels)/N;  
fprintf('Accuracy:%1.2f%%\n',100*accuracy);

%% Function 
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
function g = evalgaussian(x,mu,Sigma,n)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[~,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
