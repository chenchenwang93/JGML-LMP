function [TestingAccuracy, pred_label] = elm_kernel(train_data, test_data, number_class, C, Kernel_type, Kernel_para)
%%%%%%%%%%% Load training dataset
T=train_data(:,1)';
P=train_data(:,2:size(train_data,2))';
clear train_data;

%%%%%%%%%%% Load testing dataset
TV.T=test_data(:,1)';
TV.P=test_data(:,2:size(test_data,2))';
clear test_data;

NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);
NumberofOutputNeurons=number_class;

%%%%%%%%%% Processing the targets of training
temp = T;
T = -1*ones(NumberofOutputNeurons, NumberofTrainingData);
for i = 1: number_class
    T(i,temp==i)=1;
end

%%%%%%%%%%% Training Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = size(T,2);
Omega_train = kernel_matrix(P',Kernel_type, Kernel_para);
OutputWeight=((Omega_train+speye(n)/C)\(T'));

%%%%%%%%%%% Calculate the output of testing input
Omega_test = kernel_matrix(P',Kernel_type, Kernel_para,TV.P');
TY=(Omega_test' * OutputWeight)';                                  % TY: the actual output of the testing data

%%%%%%%%%% Calculate testing classification accuracy
[~, pred_label]=sort(TY,1,'descend');
pred_label = pred_label(1,:);
Classification=sum(pred_label==TV.T);
TestingAccuracy=Classification/NumberofTestingData;


%%%%%%%%%%%%%%%%%% Kernel Matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function omega = kernel_matrix(Xtrain,kernel_type, kernel_pars,Xt)

nb_data = size(Xtrain,1);

if strcmp(kernel_type,'RBF_kernel'),
    if nargin<4,
        XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
        omega = XXh+XXh'-2*(Xtrain*Xtrain');
        omega = exp(-omega./kernel_pars(1));
    else
        XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
        XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
        omega = XXh1+XXh2' - 2*Xtrain*Xt';
        omega = exp(-omega./kernel_pars(1));
    end
end
