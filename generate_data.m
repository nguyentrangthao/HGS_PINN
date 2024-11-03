clear all
clc
%tspan = [0 10];
tspan1= linspace(0,.5,100);
tspan2= linspace(0.51,.6,10);
tspan=[tspan1 tspan2];
y0 = 400;
[t,y] = ode45(@(t,y) 2*(44-y), tspan, y0);
data=[t y];

traindata=data(1:length(tspan1),:);
testdata=data(length(tspan1)+1:end,:);
figure
plot(traindata(:,1),traindata(:,2),'k','LineWidth',2)
hold on
plot(testdata(:,1),testdata(:,2),'k--','LineWidth',2)
xlabel('Time (Hours)');
ylabel('Temperature (F)');
legend('traindata','testdata')

save('data.mat','data','traindata','testdata');