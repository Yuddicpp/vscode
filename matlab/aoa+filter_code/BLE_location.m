function BLE_location(obj)
fopen(obj);
his_IQ = [];
% track_plot_prepare();
P=gene_DML_P();
fileID = fopen('loc.txt','a+');    % w : 删掉原来文件中的内容。a+：追加写入
% fileID1 = fopen('aoa.txt','a+');    % w : 删掉原来文件中的内容。a+：追加写入

% axis([-3.5,3.5,-3.5,3.5]);
% grid on;
% % set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);
% set(gcf,'position',[300 0 800 800]);
% set(gca,'XTick',(-3.5:0.2:3.5));
% set(gca,'YTick',(-3.5:0.2:3.5));
% set(gca,'Color',[0 0 0]);
% hold on;
% plot(0,0,'^','MarkerFaceColor',[rand rand rand],'MarkerSize',12,'MarkerEdgeColor','k');
% hold on
% ax = gca;
% ax.GridColor = [1, 1, 1];

while 1
    [Idata,Qdata,rssi,his_IQ]=read_serial(obj,his_IQ);
    if size(Idata,2)==0
        continue;
    end
    [data,index]=data_process1(Idata,Qdata);
    if size(data,1)~=8
        continue;
    end
    data = compensate(data,index);
    
    loc1 = DML(data,P,fileID)
    
%     for kk=1: size(loc1,1)
%         if exist('h1','var')
%             delete(h1);
%             delete(h2);
%             % set(h1,'Visible','off')
%         end
%         h1 = plot(loc1(kk,1),loc1(kk,2),'.','MarkerSize',30,'MarkerFaceColor','r');
% 
%         str1=[num2str(roundn(loc1(kk,1),-3)),',',num2str(roundn(loc1(kk,2),-3))];
%         h2 = text(loc1(kk,1)+0.05,loc1(kk,2)+0.05,str1,'color','r');  
%         drawnow;
%     end
    
end
fclose(obj);

end

function [Idata,Qdata,rssi,his_IQ]=read_serial(obj,his_IQ)
while ~exist('Idata')
    data=fread(obj);
    input=dec2hex(data);
    input=[his_IQ;input];
    a=find(input(:,1)=='3'& input(:,2)=='0');
    b = [];
    for n=1:size(a,1)-4
        if a(n+1)==a(n)+1 && a(n+2)==a(n)+2 && a(n+3)==a(n)+3 && a(n+4)~=a(n)+4
            b(n:n+3) = a(n:n+3);
        end
    end
    b = b(b~=0);
    for n=1:4:size(b,2)-8
        if b(n+4) - b(n) ~= 2068
            b(n:n+3) = 0;
        end
    end
    % b = b(b~=0 & b<=18612);

    for n=1:4:size(b,2)-4
        % convert hex 2 dec, ex: if input = 0E FF , what we want is 'FF0E'
        Qdata(:,(n-1)/4+1) = hex2dec([input(4+1+b(n):4:1+b(n)+2048,:) input(4+b(n):4:b(n)+2048,:)]);
        q = find(Qdata(:,(n-1)/4+1) > 4095);
        Qdata(q,(n-1)/4+1) = Qdata(q,(n-1)/4+1) -16^4;
        Idata(:,(n-1)/4+1) = hex2dec([input(4+3+b(n):4:3+b(n)+2048,:) input(4+2+b(n):4:2+b(n)+2048,:)]);
        i = find(Idata(:,(n-1)/4+1) > 4095);
        Idata(i,(n-1)/4+1) = Idata(i,(n-1)/4+1) -16^4;
        rssi(:,(n-1)/4+1) = hex2dec(input(b(n)+2060,:))-255;
    end
end
if size(b,2)>=4
    his_IQ=input(b(end-3):end,:);
else
    his_IQ=[];
end
end



function [Idata,Qdata,rssi,his_IQ]=read_serial1(obj,his_IQ)
    data=fread(obj);
    input=dec2hex(data);
    input=[his_IQ;input];
    if ~isempty(input)
        a=find(input(:,1)=='3'& input(:,2)=='0');
        b = [];
        for n=1:size(a,1)-4
            if a(n+1)==a(n)+1 && a(n+2)==a(n)+2 && a(n+3)==a(n)+3 && a(n+4)~=a(n)+4
                b(n:n+3) = a(n:n+3);
            end
        end
        b = b(b~=0);
        for n=1:4:size(b,2)-8
            if b(n+4) - b(n) ~= 2068
                b(n:n+3) = 0;
            end
        end
        % b = b(b~=0 & b<=18612);

        for n=1:4:size(b,2)-4
            % convert hex 2 dec, ex: if input = 0E FF , what we want is 'FF0E'
            Qdata(:,(n-1)/4+1) = hex2dec([input(4+1+b(n):4:1+b(n)+2048,:) input(4+b(n):4:b(n)+2048,:)]);
            q = find(Qdata(:,(n-1)/4+1) > 4095);
            Qdata(q,(n-1)/4+1) = Qdata(q,(n-1)/4+1) -16^4;
            Idata(:,(n-1)/4+1) = hex2dec([input(4+3+b(n):4:3+b(n)+2048,:) input(4+2+b(n):4:2+b(n)+2048,:)]);
            i = find(Idata(:,(n-1)/4+1) > 4095);
            Idata(i,(n-1)/4+1) = Idata(i,(n-1)/4+1) -16^4;
            rssi(:,(n-1)/4+1) = hex2dec(input(b(n)+2060,:))-255;
        end
    
        if size(b,2)>=4
            his_IQ=input(b(end-3):end,:);
        else
            his_IQ=[];
        end
    elseif isempty(input)
        Idata=[];
        Qdata=[];
        rssi=0;
        
    end

end

function track_plot_prepare()
axis([-7,7,-7,7]);
grid on;
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);
set(gca,'XTick',(-7:0.2:7));
set(gca,'YTick',(-7:0.4:7));
set(gca,'Color',[0 0 0]);
hold on;

N = 2;  % 想要的坐标轴显示，即隔一个网格显示一个刻度，10/5 = 2
% 设置想要的坐标轴刻度
a = get(gca,'XTickLabel');  
b = cell(size(a));
b(mod(1:size(a,1),N)==1,:) = a(mod(1:size(a,1),N)==1,:);
set(gca,'XTickLabel',b);

plot(0,0,'^','MarkerFaceColor',[rand rand rand],'MarkerSize',12,'MarkerEdgeColor','k');
hold on;
ax = gca;
ax.GridColor = [1, 1, 1];
end