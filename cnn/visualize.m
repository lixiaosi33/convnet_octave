vanilla = csvread('master.csv');
nesterov = csvread('nesterov.csv');
rmsprop = csvread('rmsprop.csv');

figure
hold on
plot(vanilla(:,1), vanilla(:,2), 'g')
plot(nesterov(:,1), nesterov(:,2), 'b')
plot(rmsprop(:,1), rmsprop(:,2), 'r')
hold off
keyboard('done>')