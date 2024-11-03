clc;
clear all; close all
load('data.mat')
rng(7)
k=1;
N=100; %KICH CO DAN SO
n=5;
FEs=20;
% activation function;
act_func1='tansig';
act_func2='purelin';

errt_ann0=zeros(k,1);
errt_ann=zeros(k,FEs);
errt_pinn=zeros(k,FEs);

mae_ann0=zeros(k,1);
mae_ann=zeros(k,FEs);
mae_pinn=zeros(k,FEs);

r2_ann0=zeros(k,1);
r2_ann=zeros(k,FEs);
r2_pinn=zeros(k,FEs);


inputs = traindata(:,1)';
targets =traindata(:,2)';
[inputs_nm,PSinput] = mapminmax(inputs);
[targets_nm,PStarget] = mapminmax(targets);
save('PSinput.mat','PSinput')
save('PStarget.mat','PStarget')
inputs_t = testdata(:,1)';
targets_t =testdata(:,2)';
inputs_t_nm = mapminmax('apply',inputs_t,PSinput)
targets_t_nm = mapminmax('apply',targets_t,PStarget)
for j=1:k
    %% ann
    net0 = feedforwardnet([n]);
    net0.trainParam.epochs = 20;
    net0.trainParam.goal = 0;
    net0.trainParam.min_grad=0;
    net0.trainParam.mu_max=10^100;
    net0.trainParam.showWindow = 1;
    net0.divideParam.trainRatio = 100/100;
    net0.divideParam.valRatio = 0/100;
    net0.divideParam.testRatio = 0/100;
    [net0,tr0] = train(net0,inputs_nm,targets_nm);
    % test
    y_predicted_t =net0(inputs_t_nm);
    errt=perform(net0,y_predicted_t,targets_t_nm);
    y_predicted_t_ANN=mapminmax('reverse',y_predicted_t,PStarget);
    %fprintf('Sai so phan loai tren tap train = %f\n',mse);
    fprintf('Sai so phan loai tren tap test = %f\n',errt);
    errt_ann0(j,1)=errt;
   
    for ii=1:size(y_predicted_t,1)
        MAE(ii) = mean(abs(y_predicted_t(ii,:) - targets_t_nm(ii,:)));
        % Tính R-squared (R²)
        SS_total = sum((targets_t_nm(ii,:) - mean(targets_t_nm(ii,:))).^2); % Tổng bình phương sai
        SS_residual = sum((targets_t_nm(ii,:) - y_predicted_t(ii,:)).^2);   % Tổng bình phương sai còn lại
        R_squared(ii) = 1 - (SS_residual / SS_total);
    end
    mae_ann0(j,1)=mean(MAE);
    r2_ann0(j,1)=mean(R_squared);   
    
    %% HGSANN
    [train_net, mse]=HGSANN(inputs_nm,targets_nm,n,act_func1,act_func2,N,FEs)
    % test
    y_predicted_t =train_net(inputs_t_nm);
    errt=perform(train_net,y_predicted_t,targets_t_nm);
    fprintf('Sai so phan loai tren tap train = %f\n',mse);
    fprintf('Sai so phan loai tren tap test = %f\n',errt);
    % de ve hinh
    load('besthist.mat');
    load('netinform.mat');
    train_net=net;
    for i=1:size(besthist,1)
        v=besthist(i,:);
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
        % validate
        y_predicted_t =train_net(inputs_t_nm);
        errt_ann(j,i)=perform(train_net,y_predicted_t,targets_t_nm);
        
        for ii=1:size(y_predicted_t,1)
            MAE(ii) = mean(abs(y_predicted_t(ii,:) - targets_t_nm(ii,:)));
        % Tính R-squared (R²)
            SS_total = sum((targets_t_nm(ii,:) - mean(targets_t_nm(ii,:))).^2); % Tổng bình phương sai
            SS_residual = sum((targets_t_nm(ii,:) - y_predicted_t(ii,:)).^2);   % Tổng bình phương sai còn lại
            R_squared(ii) = 1 - (SS_residual / SS_total);
        end
        mae_ann(j,i)=mean(MAE);
        r2_ann(j,i)=mean(R_squared);     
    end
    y_predicted_t_HGSANN=mapminmax('reverse',y_predicted_t,PStarget); 
    
    
    %% hgs-pinn
    [train_net, mse]=HGSANN_pinn(inputs_nm,targets_nm,n,act_func1,act_func2,N,FEs)
    % test
    y_predicted_t =train_net(inputs_t_nm);
    errt=perform(train_net,y_predicted_t,targets_t_nm);
    fprintf('Sai so phan loai tren tap train = %f\n',mse);
    fprintf('Sai so phan loai tren tap test = %f\n',errt);
    load('besthist_pinn.mat');
    load('netinform.mat');
    train_net=net;
    for i=1:size(besthist,1)
        v=besthist_pinn(i,:);
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
        % validate
        y_predicted_t =train_net(inputs_t_nm);
        errt_pinn(j,i)=perform(train_net,y_predicted_t,targets_t_nm);
        
         for ii=1:size(y_predicted_t,1)
            MAE(ii) = mean(abs(y_predicted_t(ii,:) - targets_t_nm(ii,:)));
        % R-squared (R²)
            SS_total = sum((targets_t_nm(ii,:) - mean(targets_t_nm(ii,:))).^2); % Tổng bình phương sai
            SS_residual = sum((targets_t_nm(ii,:) - y_predicted_t(ii,:)).^2);   % Tổng bình phương sai còn lại
            R_squared(ii) = 1 - (SS_residual / SS_total);
        end
        mae_pinn(j,i)=mean(MAE);
        r2_pinn(j,i)=mean(R_squared);  
    end
    y_predicted_t_HGSPINN=mapminmax('reverse',y_predicted_t,PStarget);
end
meanann0=mean(errt_ann0);
meanann=mean(errt_ann,1);
meanpinn=mean(errt_pinn,1);

mean_mae_ann0=mean(mae_ann0);
mean_mae_ann=mean(mae_ann,1);
mean_mae_pinn=mean(mae_pinn,1);

mean_r2_ann0=mean(r2_ann0);
mean_r2_ann=mean(r2_ann,1);
mean_r2_pinn=mean(r2_pinn,1);

[Xsorted,I] = sort(inputs_t)

figure
plot(Xsorted,targets_t(I),'k--','LineWidth',2)
hold on
plot(Xsorted,y_predicted_t_ANN(I),'r','LineWidth',2)
plot(Xsorted,y_predicted_t_HGSANN(I),'g','LineWidth',2)
plot(Xsorted,y_predicted_t_HGSPINN(I),'b','LineWidth',4)
xlabel('Time (Hours)');
ylabel('Temperature (F)');
legend('Actual data','ANN','HGS-ANN','HGS-PINN')
