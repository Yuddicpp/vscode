function show_loc()

axis([-6,6,-6,6]);
grid on;
% set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);
set(gcf,'position',[300 0 800 800]);
set(gca,'XTick',(-6:0.5:6));
set(gca,'YTick',(-6:0.5:6));
set(gca,'Color',[0 0 0]);
hold on;
plot(0,0,'^','MarkerFaceColor',[rand rand rand],'MarkerSize',12,'MarkerEdgeColor','k');
hold on
ax = gca;
ax.GridColor = [1, 1, 1];

filepath='loc.txt';
file = fopen(filepath);
fpos=0;
while 1
fseek(file,fpos,-1);
input = textscan(file,'%s','delimiter','\n');
data1=input{1,1};
% data2=str2double(data1);
data2=cellfun(@(x) strsplit(x),data1,'UniformOutput', false);
% data2=strsplit(data1{1,1});
data3=cellfun(@(x) str2double(x),data2,'UniformOutput', false);

for kk=1:size(data3,1)
    if exist('h1','var')
        delete(h1);
        delete(h2);
        % set(h1,'Visible','off')
    end
    h1 = plot(data3{kk,1}(1),data3{kk,1}(2),'.','MarkerSize',30,'MarkerFaceColor','r');

    str1=[num2str(roundn(data3{kk,1}(1),-2)),' ',',',' ',num2str(roundn(data3{kk,1}(2),-2))];
    h2 = text(data3{kk,1}(1)+0.1,data3{kk,1}(2)+0.1,str1,'color','r','FontSize',12);
    drawnow;
end

fpos=ftell(file);
end



end