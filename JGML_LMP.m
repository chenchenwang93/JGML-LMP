%2022Joint global metric learning and local manifold preservation for scene recognition
clear
clc
options = [];
options.PCARatio = 0.99;
%
load('E:\MIT-67\places365_resnet152_67');
resnet152_67 = normcols(double(resnet152_67));
[~,~,~,F] = PCA(resnet152_67',options);
F = F';
F = normcols(F);
clear resnet152_67
%
load('E:\MIT-67\gnd_67');
gnd = gnd_67;
clear gnd_67
nnClass = length(unique(gnd));  % The number of classes;
nntr = 80;
Ftr = [];
Fte = [];
tr_label = [];
te_label = [];
for j = 1:nnClass
    idx = find(gnd == j);
    randIdx = randperm(length(idx)); %randIdx create m random number, m is the size of idx.
    Ftr = [Ftr,F(:,idx(randIdx(1:nntr)))];
    tr_label = [tr_label; gnd(idx(randIdx(1:nntr)))];
    Fte = [Fte,F(:,idx(randIdx(nntr+1:100)))];
    te_label = [te_label; gnd(idx(randIdx(nntr+1:100)))];
end
%clear F gnd idx randIdx nnClass nntr
%% 
run ('D:/program files/matlab/toolbox/vlfeat-0.9.21/toolbox/vl_setup.m')
n = 5360;
d = size(Ftr,1);
c = 67;
K = 10;
ntr = 80;
X = Ftr;
anchors = [];
for i = 1:c
    data = X(:, ntr*(i-1)+1:ntr*i);
    data = vl_kmeans(data,K);
    anchors = [anchors, data];
end
%
Rw = zeros(d,d);
Rb = zeros(d,d);
for i = 1 : n
    j = ceil(i / ntr);
    X1 = anchors(:, K*(j-1)+1 : K*j); %
    X2 = anchors;
    X2(:, K*(j-1)+1 : K*j) = []; %
    Rw = Rw + (repmat(X(:, i), 1, K) - X1)*(repmat(X(:, i), 1, K) - X1)';
    Rb = Rb + (repmat(X(:, i), 1, K*c - K) - X2)*(repmat(X(:, i), 1, K*c - K) - X2)';
end
R = Rw - Rb;
R = (R + R') / 2;
% initializ W
dr = 400;
[eigvector, eigvalue] = eig(R);
[eigvalue, ind] = sort(diag(eigvalue));
W = eigvector(:,ind(1:dr));
k1 = 10;
beta = 100;
addind1 = zeros(ntr,n);
for i = 1:c
    addind1(:, ntr*(i-1)+1:ntr*i) = (i-1)*ntr*ones(ntr,ntr);
end
% update W
for t = 1 : 3
    %ind1
    XX = W' * X;
    dis = zeros(ntr, n);
    for i = 1:ntr
        X1 = reshape(XX(:, i:ntr:(n-ntr+i)), [1, c*dr]);
        X1 = repmat(X1, ntr, 1);
        X1 = reshape(X1, [ntr, dr, c]);
        X1 = reshape(permute(X1, [2, 1, 3]), dr, n);
        dis(i, :) = sum((XX - X1).^2, 1);
    end
    [~, ind1] = sort(dis);
    ind1 = ind1 + addind1;
    %
    C1 = zeros(d, d);
    for j = 1 : k1
        X1 = X(:, ind1(j+1, :));
        C1 = C1 + (X - X1)*(X - X1)';
    end
    C1 = C1/2;
    [eigvector, eigvalue] = eig(R + beta * C1);
    [eigvalue, ind] = sort(diag(eigvalue));
    W = eigvector(:,ind(1:dr));
end
% ELM classification
tr_fea = W' * Ftr;
te_fea = W' * Fte;
tr3_fea = [tr_label, tr_fea'];
te3_fea  = [te_label, te_fea'];
GA = [0.00000001:0.00000002:0.00000009, 0.0000001:0.0000002:0.0000009, 0.000001:0.000002:0.000009, 0.00001:0.00002:0.00009, 0.0001:0.0002:0.0009, 0.001:0.002:0.009, 0.01:0.02:0.09, 0.1:0.2:0.9, 1:0.3:20];
GA = GA(30:80);
CC = 1;
for i = 1:length(CC)
    para_C = CC(i);
    for j = 1:length(GA)
        gamma = GA(j);
        [~, ~, ~, TestAC, TY, pred_label] = elm_kernel(tr3_fea, te3_fea, 1, para_C, 'RBF_kernel',gamma);
        accy(j) = TestAC;
    end
    ACC(i) = max(accy);
    disp(ACC(i));
end

