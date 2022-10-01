
%%========================================================%
%                                                              NARX network
%=========================================================%

clear
close all
clc

addpath = 'R:\Github\Matlab\Data';                  % Data path

%%========================================================%
%                     Data pre-processing
%========================================================%

% Load training and test data

Training_data = xlsread('Training_data.xlsx');
Test_data        = xlsread('Test_data.xlsx');

fprintf('Training and Test data are load successfully \n') 

% Separet inputs and targets of training data

Train_inputs = Training_data(:,1:3);
Train_targets = Training_data(:,4:6);

%========================================================%
%                     NARX network
%========================================================%

% Prepare Inputs and outputs data to NN data set
X_narxinputs  = tonndata(Train_inputs, false, false);     % Narx network training inputs
T_narxtargets = tonndata(Train_targets, false, false);    % Narx network training targets

% NARX network
InputDelays        = 0:3;                  % Input delays (can be greater than or equal to 0)
FeedbackDelays = 1:4;                  % Feedback delays (must be greater than  0)
HiddenNeurons  = 10;               % Number of Hidden neurons per layer
trainFcn = 'trainlm';                    % Levenberg-Marquardt backpropagation.
net = narxnet(InputDelays, FeedbackDelays, HiddenNeurons, 'open', trainFcn);
view (net);

% NARX network Training parameters

net.divideFcn  = 'divideblock';
net.divideParam.trainRatio = 0.70;
net.divideParam.valRatio   = 0.15;
net.divideParam.testRatio  = 0.15;
net.trainparam.epochs = 1000;
net.performFcn = 'mse';                     % Mean Squared Error;
net.trainParam.min_grad = 1e-7;
net.trainParam.showCommandLine = 1;
%net.layers{2}.transferFcn = 'tansig';      % Hyperbolic(Tanh) activation function

% Input and Feedback Post-Processing Functions: for productivity
net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.inputs{2}.processFcns = {'removeconstantrows','mapminmax'};

% Prepare the Data for Training
[X_inputs, Xi, Ti, T_targets] = preparets(net, X_narxinputs, {}, T_narxtargets);
 
	  % X_inputs: training inputs
	  % T_targets: training targets
	  % Xi: initial input states
	  % Ti: initial target state

% Train NARX Network 

tic
% If GPU is available 'useParallel','yes' to use the parallel computing
[net, training_record] = train(net, X_inputs, T_targets, Xi, Ti, 'useParallel','no','showResources','no'); 
Trainingtime = toc

fprintf('The network is trained successfully \n') 
% Test the Network

y = net(X_inputs, Xi, Ti);
etrain = cell2mat(gsubtract(T_targets,y)); % Open-Loop (training) error
OpenLoopPerformance = perform(net,T_targets,y)

% Recalculate Training, Validation and Test Performance
trainTargets = gmultiply(T_targets, training_record.trainMask);
valTargets = gmultiply(T_targets, training_record.valMask);
testTargets = gmultiply(T_targets, training_record.testMask);
trainPerformance = perform(net, trainTargets,y)
valPerformance = perform(net, valTargets,y)
testPerformance1 = perform(net, testTargets,y)

% View the Network (Open-loop)
view(net);

% Closed Loop Network for recursive and multistep prediction
netc = closeloop(net);
[Xclosed_inputs, Xic, Tic, Tclosed_targets] = preparets(netc,  X_narxinputs, {}, T_narxtargets);
yclosed = netc(Xclosed_inputs, Xic, Tic);                           % predict with closed loop
epredict = cell2mat(gsubtract(Tclosed_targets, yclosed)); % Closed-Loop error
closedLoopPerformance = perform(netc, Tclosed_targets, yclosed)

% View the Network (Closed lose-loop)
view(netc);


%delay configuration: because the same length of Input and Output data needed 

if max(InputDelays)  < max(FeedbackDelays)
   delay = max(FeedbackDelays);
else
   delay = max(InputDelays);
end

% ============================================================================%
% Test the trained network with unseen test data set to evaluate the generalization ability of the model

% Separet inputs and targets of training data

Test_inputs = Test_data(:,1:3);
Test_targets = Test_data(:,4:6);

X_narxtest = tonndata(Test_inputs,false,false);
T_narxtest = tonndata(Test_targets,false,false);


% prediction on test data using Open-Loop Network
[X_test, Xti, Tti, T_test] = preparets(net,X_narxtest,{},T_narxtest);
predictop = net(X_test, Xti, Tti);
etest = cell2mat(gsubtract(T_test, predictop));
OpenTestPerformance_unseen_testdata = perform(net, T_test, predictop)
netc = closeloop(net);
% prediction on test data using Closed-Loop Network
[Xclosed_testinputs, Xtic, Ttic, Tclosed_testtargets] = preparets(netc,  X_narxtest, {}, T_narxtest);
predict = netc(Xclosed_testinputs, Xtic, Ttic);          % predict with closed loop
epredi_test = cell2mat(gsubtract(Tclosed_testtargets, predict)); % Closed-Loop error
closedLoopPerformance_unseen_testdata = perform(netc ,Tclosed_testtargets, predict)

%%
%%==========================================================================%%
%                         Visualization
%============================================================================%
%plot real target data and training target data % do not forget to consider
%the delay
actual_data = (cell2mat(T_narxtargets))'; 
training = (cell2mat(y))';

figure
subplot(3,1,1)
plot(actual_data(delay:300,1), 'b-','LineWidth',1.4)
hold on
plot(training(delay:300,1),'r-.','LineWidth',1.4)
ylabel('Input1','Interpreter','latex')
legend('Actual data','Predicted outputs','Interpreter','latex','Location','NorthOutside','Orientation','Horizontal')
xt = get(gca, 'XTick');
xticklabels([ ]);
set(gca, 'XTick',xt, 'XTickLabel',xt,FontSize=12)
grid on

subplot(3,1,2)
plot(actual_data(delay:300,2), 'b-','LineWidth',1.4)
hold on
plot(training(delay:300,2),'r-.','LineWidth',1.4)
% xlabel('$k$','Interpreter','latex')
ylabel('Input2','Interpreter','latex')
%legend('Actual data','Training','Interpreter','latex','Location','NorthOutside','Orientation','Horizontal')
xt = get(gca, 'XTick');
set(gca, 'XTick',xt, 'XTickLabel',xt,FontSize=12)
grid on

subplot(3,1,3)
plot(actual_data(delay:300,3), 'b-','LineWidth',1.4)
hold on
plot(training(delay:300,3),'r-.','LineWidth',1.4)
ylabel('Input3','Interpreter','latex')
%legend('Actual data','Training','Interpreter','latex','Location','NorthOutside','Orientation','Horizontal')
xt = get(gca, 'XTick');
set(gca, 'XTick',xt, 'XTickLabel',xt,FontSize=12)
grid on

print -dsvg Actual_vs_prediction
% plot Ac data and test data
test_data = (cell2mat(T_narxtest))'; % test data
training = (cell2mat(predictop))';
figure
subplot(3,1,1)
plot(test_data(delay:300,1), 'b-','LineWidth',1.4)
hold on
plot(training(delay:300,1),'r-.','LineWidth',1.4)
ylabel('Input1','Interpreter','latex')
legend('Test data','Predicted outputs','Interpreter','latex','Location','NorthOutside','Orientation','Horizontal')

xt = get(gca, 'XTick');
set(gca, 'XTick',xt, 'XTickLabel',xt,FontSize=12)
grid on

subplot(3,1,2)
plot(test_data(delay:300,2), 'b-','LineWidth',1.4)
hold on
plot(training(delay:300,2),'r-.','LineWidth',1.4)

ylabel('Input2','Interpreter','latex')

xt = get(gca, 'XTick');
set(gca, 'XTick',xt, 'XTickLabel',xt,FontSize=12)
grid on

subplot(3,1,3)
plot(test_data(delay:300,3), 'b-','LineWidth',1.4)
hold on
plot(training(delay:300,3),'r-.','LineWidth',1.4)

ylabel('Input3','Interpreter','latex')

xt = get(gca, 'XTick');
set(gca, 'XTick',xt, 'XTickLabel',xt,FontSize=12)
grid on
print -dsvg Test_vs_prediction

%% Save the trained model
save('Trained_model.mat')
