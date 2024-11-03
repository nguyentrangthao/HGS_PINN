%---------------------------------------------------------------------------------------------------------------------------
% HGS-ANN
%  Nguyen Trang Thao  
% Institute for computational science and artificial intelligence, Van Lang University, Vietnam
%  e-Mail: thao.nguyentrang@vlu.edu.vn,

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [train_net, mse] =HGSANN(inputs,targets,n,act_func1,act_func2,N,FEs)
dim = n*size(inputs,1)+n  +  n*size(targets,1)  +  size(targets,1); %dimension of the problem
lb = -10*ones(1,dim);
ub = ones(1,dim)*10;
net.layers{1}.transferFcn=act_func1;
net.layers{2}.transferFcn=act_func2;
% config the net
net = feedforwardnet([n]);
%net = train(net,inputs,targets);
net= configure(net,inputs,targets);
save('netinform.mat','net','n','inputs','targets')
tic
[Destination_fitness,bestPositions,besthist,Convergence_curve]=HGS(N,FEs,lb,ub,dim,@objfunc);
time=toc
plot(Convergence_curve,'Color','b','LineWidth',4);
title('Convergence curve')
xlabel('Iteration');
ylabel('Best fitness obtained so far');
legend('HGS')
save('besthist.mat','besthist');
%% net best
load('netinform.mat');
train_net=net;
mse=Destination_fitness;
v=bestPositions;
% w0=v(1:n*size(inputs,1));
% w0=reshape(w0,n,size(inputs,1));
% train_net.IW{1,1}=w0;
% b0=v(n*size(inputs,1)+1:n*size(inputs,1)+n);
% train_net.b{1}=b0';
% current=n*size(inputs,1)+n;
% w1=v(current+1:current+n);
% train_net.LW{2,1}=w1;
% train_net.b{2}=v(end);
w0=v(1:n*size(inputs,1));
w0=reshape(w0,n,size(inputs,1));
train_net.IW{1,1}=w0;
b0=v(n*size(inputs,1)+1:n*size(inputs,1)+n);
train_net.b{1}=b0';
current=n*size(inputs,1)+n;
w1=v(current+1:current+n*size(targets,1));
w1=reshape(w1,size(targets,1),n);
train_net.LW{2,1}=w1;
current=current+n*size(targets,1);
train_net.b{2}=v(current+1:end)';
end

