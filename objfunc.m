function [fval]=objfunc(v)
%alp=.01;
load('netinform.mat');
load('PSinput.mat');
load('PStarget.mat');
w0=v(1:n*size(inputs,1));
w0=reshape(w0,n,size(inputs,1));
net.IW{1,1}=w0;
b0=v(n*size(inputs,1)+1:n*size(inputs,1)+n);
net.b{1}=b0';
current=n*size(inputs,1)+n;
w1=v(current+1:current+n*size(targets,1));
w1=reshape(w1,size(targets,1),n);
net.LW{2,1}=w1;
current=current+n*size(targets,1);
net.b{2}=v(current+1:end)';
y_predicted =net(inputs);
fval=perform(net,y_predicted,targets);


end