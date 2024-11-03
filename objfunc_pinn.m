function [fval]=objfunc_pinn(v)
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
fval1=perform(net,y_predicted,targets);
% Phyphical information

y_predicted_rev=mapminmax('reverse',y_predicted,PStarget);
input_rev=mapminmax('reverse',inputs,PSinput);

dy_dx = gradient(y_predicted_rev, input_rev);

fval2=mean((2*(44-y_predicted_rev)-dy_dx).^2+(y_predicted_rev(1)-400).^2);
fval=fval1+fval2/10;
end