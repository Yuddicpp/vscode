function plot_mismatch_amp()

flag=0;
for fw=0:1:360
    flag=flag+1;
[ground_phase,ground_amp]=groundphase(fw,0,0);
gp(1:12,flag)=ground_phase(:,1);
ga(1:12,flag)=ground_amp(:,1);
end
% [ground_phase,ground_amp]=groundphase();
filepath='G:\bluetooth\ant13\data_laoyang\0505\qu1.txt';
[test_phase, test_amp]=testphase(filepath);

% subplot(1,2,2);
figure();
plot(smooth(test_amp(4,512:1018),20),'LineWidth',2,'Color','r');
hold on
plot(smooth(test_amp(7,512:1018),20),'LineWidth',2,'Color','b');
set(gca,'FontName','Times','FontSize',14);
xlabel('sample index');
ylabel('amplitude');
% title('amplitude alignment');
legend1=legend('amp_{ant1}','amp_{ant2}');
set(legend1,'FontName','Times','FontSize',12);


end



