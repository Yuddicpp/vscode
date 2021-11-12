function plot_CRB2()
k=1;
x=-10:1:20;
a=ones(1,2);
b=ones(1,2);
%music_azimuth,music_elavation,ml_azimuth,ml_elavation


% a=[0.452 0.7072 0.7425 1.284 1.957 3.838 1.674 2.804];
% b=[-0.07412 -0.07396 -0.08194 -0.0806 -0.1594 -0.1212 -0.1285 -0.1051];%第二正确的一组保存

% a=[0.3372 0.4795 0.8327 0.9805 1.957 3.838 1.658 2.174];
% b=[-0.1151 -0.1151 -0.08571 -0.08523 -0.1594 -0.1212 -0.1141 -0.1029];%最终正确的一组保存

a=[0.3372 0.4795 1.658 2.174 1.957 3.838 1.776 2.97];
b=[-0.1151 -0.1151 -0.1141 -0.1029 -0.1594 -0.1212 -0.1277 -0.1243];%最终正确的一组保存

c=[0 0 0 0 0 0 0 0];
set(0,'defaultfigurecolor','w');
    plot(x,a(5)*exp(b(5)*x)+c(5),'LineWidth',2,'Color',[170 90 154]/255);
    hold on;
    plot(x,a(6)*exp(b(6)*x)+c(6),'--','LineWidth',2,'Color',[170 90 154]/255);
    
    plot(x,a(7)*exp(b(7)*x)+c(7),'LineWidth',2,'Color',[117 186 85]/255);
    plot(x,a(8)*exp(b(8)*x)+c(8),'--','LineWidth',2,'Color',[117 186 85]/255);
    
    plot(x,a(3)*exp(b(3)*x)+c(3),'LineWidth',2,'Color',[34 86 166]/255);
    plot(x,a(4)*exp(b(4)*x)+c(4),'--','LineWidth',2,'Color',[34 86 166]/255);
    
    plot(x,a(1)*exp(b(1)*x),'LineWidth',2,'Color','r');
    plot(x,a(2)*exp(b(2)*x),'--','LineWidth',2,'Color','r');




%     y=a(ii)*exp(b(ii)*x);
set(gca,'FontName','Times','FontSize',16);
xlabel('SNR [dB]');
ylabel('RMSE [deg]');
%title('AoA estimation error');
% legend1=legend('azimuth (RD-MUSIC)','elevation (RD-MUSIC)','azimuth (ML)','elevation (ML)',...
%     'azimuth Root CRLB (scalar array)','elevation Root CRLB (scalar array)',...
%     'azimuth Root CRLB (PSA)','elevation Root CRLB (PSA)');
legend1=legend('RD-MUSIC (\theta)','RD-MUSIC (\phi)','AIS without iter(\theta)','AIS without iter(\phi)',...
    'AIS with iter(\theta)','AIS with iter(\phi)','Root CRLB (\theta, PSA)','Root CRLB (\phi, PSA)');

set(legend1,'FontName','Times','FontSize',14);
axis([-10,20,0,14]);

end