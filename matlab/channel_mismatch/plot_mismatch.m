function plot_mismatch(real_phase,test_phase)
% subplot(1,2,1);
figure();
real_phase(3,find(real_phase(3,:)<-pi*9/10)) = pi;
test1=test_phase(:,1:1215);
test1(:,1216:1518)=test_phase(:,2321:2623);
test1(:,1519:2622)=test_phase(:,1217:2320);
test1(:,2623:5216)=test_phase(:,2623:end);
test1(:,5217:5218)=test_phase(:,5215:5216);
% figure();
set(0,'defaultfigurecolor','w');
% set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);
plot((0:0.069:360)*pi/180,real_phase(3,:),'LineWidth',2,'Color','r');
hold on
plot((0:0.069:360)*pi/180,smooth(test1(3,:)),'LineWidth',2,'Color','b');

set(gca,'FontName','Times','FontSize',14);
xlabel('azimuth[rad]');
ylabel('phase diff[rad]');
% title('phase difference(ant_{target}-ant_{ref})');
legend1=legend('ideal value','measured value');
set(legend1,'FontName','Times','FontSize',12);
axis([0,2*pi,-pi,4]);
% set(gca,'XTick',(1:1:8));

% subplot(1,2,2);
% plot(real_amp(1,:),'LineWidth',2,'Color','r');
% hold on
% plot(real_amp(2,:),'LineWidth',2,'Color','b');
% set(gca,'FontName','Times','FontSize',14);
% xlabel('azimuth');
% ylabel('amplitude');
% title('amplitude alignment');
% legend1=legend('theoretical value','test value');
% set(legend1,'FontName','Times','FontSize',12);

end