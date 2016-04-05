
clear all;
close all;
clc;

%% Neural Networks Assignment 1
% Joel Cappelli
% 12137384

%% Qu1.1 

x1 = [0.8;0.5;0.0;0.1];
x2 = [0.2;0.1;1.3;0.9];
x3 = [0.9;0.7;0.3;0.3];
x4 = [0.2;0.7;0.8;0.2];
x5 = [1.0;0.8;0.5;0.7];
x6 = [0.0;0.2;0.3;0.6];

setX = [x1 x2 x3 x4 x5 x6];

augInput = -1;
x = [setX; augInput*ones(1,size(setX,2))];

trainingSize = size(x,2);

d1 = 1;
d2 = -1;
d3 = 1;
d4 = -1;
d5 = 1;
d6 = -1;

d = [d1 d2 d3 d4 d5 d6];

eta = 0.05;

wts = [0.0976;0.8632;-0.3296;0.3111;-0.2162];

sse = Inf;
tol = 1e-2;

maxEpochs = 10;
patternErrors = zeros(maxEpochs,trainingSize);
wtsArray = zeros(size(wts,1),maxEpochs*trainingSize + 1);
epochs = 1;

wtsArray(:,1) = wts;

while(epochs <= maxEpochs)
    for i = 1:trainingSize
        v = wts'*x(:,i);
        z = TLU(v);    
        error = d(i)-z;
        r = error;
        patternErrors(epochs,i) = 0.5*error*error;
        wts = wts + eta*x(:,i)*r;
        wtsArray(:,(epochs-1)*trainingSize+i) = wts;
    end
    epochs = epochs +1;
end

cycleError = sum(patternErrors,2);

figure;
plot(cycleError);
title('Qu1.1 - Cycle Error');
grid on;
xlabel('Epoch');

figure;
for i = 1:trainingSize
    subplot(trainingSize,1,i);plot(patternErrors(:,i));
    grid on;
    title(strcat('Pattern ',num2str(i)));
end
xlabel('Epoch');

%% Qu1.2

%re-init
setX = [x1 x2 x3 x4 x5 x6];
augInput = 1;
x = [setX; augInput*ones(1,size(setX,2))];

trainingSize = size(x,2);

eta = 0.5;

wts = [0.0976;0.8632;-0.3296;0.3111;-0.2162];

maxEpochs = 50;
patternErrors = zeros(maxEpochs,trainingSize);
wtsArray = zeros(size(wts,1),maxEpochs*trainingSize + 1);
epochs = 1;

wtsArray(:,1) = wts;

while(epochs <= maxEpochs)
    for i = 1:trainingSize
        v = wts'*x(:,i);
        z = bipolarLog(v);    
        error = d(i)-z;
        r = error*bipolarLog(z,'deriv');
        patternErrors(epochs,i) = 0.5*error*error;
        wts = wts + eta*x(:,i)*r;
        wtsArray(:,(epochs-1)*trainingSize+i + 1) = wts;
    end
    epochs = epochs +1;
end

cycleError = sum(patternErrors,2);

figure;
plot(cycleError);
title('Qu1.2 - Cycle Error');
grid on;
xlabel('Epoch');

figure;
for i = 1:trainingSize
    subplot(trainingSize,1,i);plot(patternErrors(:,i));
    grid on;
    title(strcat('Pattern ',num2str(i)));
end
xlabel('Epoch');



    



