function para=DML(R)

para=run_algorithm(R);
end

function para=run_algorithm(R)
% page=size(IQ,3);
phi=(0:1:359)*pi/180; theta=(0:1:60)*pi/180;
for ii=1:length(theta)
   for jj=1:length(phi)
       Q=gene_Q(theta(ii),phi(jj));
       P{ii,jj}=Q*pinv(Q);
   end
end
for ii=1:1
%     x=squeeze(IQ(:,:,ii));
    Pmusic=spectrum(P,R);
    para(ii,1:2) = find_peaks(Pmusic);
%     mesh(Pmusic);
%     loc=location(para);
%     track_plot(loc(end,:));
end

end

function Pmusic=spectrum(P,R)
phi=(0:1:359)*pi/180; theta=(0:1:60)*pi/180;
% R=x*x';
for ii=1:length(theta)
   for jj=1:length(phi)
%        Q=gene_Q(theta(ii),phi(jj));
%        P=Q{ii,jj}*pinv(Q{ii,jj});
       Pmusic(ii,jj)=abs(trace(P{ii,jj}*R));
   end
end
end

function Q=gene_Q(theta,phi)
R=0.059;
c=3e8;
f=2.44e9;
A=2*pi*f/c;
array=zeros(6,3);
for ii=1:6
%    array(ii,:)=R*[cos(pi/3*(ii-1)) sin(pi/3*(ii-1)) 0];
    array(ii,:)=R*[cos(pi-pi/3*(ii-1)) sin(pi-pi/3*(ii-1)) 0];
end

r=[sin(theta)*cos(phi) sin(theta)*sin(phi) cos(theta)];
U=zeros(12,12);
for ii=1:6
    tmp=exp(-1j*array(ii,:)*r.'*A);
    U((ii-1)*2+1,(ii-1)*2+1)=tmp;
    U(ii*2,ii*2)=tmp;
end

beta = ant_direction();

L=[-sin(phi) cos(phi)*cos(theta);cos(phi) sin(phi)*cos(theta)];
% L=[-sin(phi) cos(phi)*sin(theta);cos(phi) sin(phi)*cos(theta)];
% P=[cos(gama);sin(gama)*exp(1j*yita)];
D=U*beta*L;
amp_offset=[342 496 279 387 393 310 304 380 344 430 275 400];
phase_offset=exp(1j.*[0 2.1 -2.4 -2.4 -1.35 0.8 -2.35 -2.4 -1.4 0.8 -1.1 -2.14]);
offset=amp_offset.*phase_offset;
B=diag(offset);

Q=D;
end

function beta = ant_direction()
% dir_ant=[0 90 60 -30 120 210 180 270 240 330 300 210]*pi/180;
% dir_ant=[0 90 60 150 120 210 180 270 240 330 300 30]*pi/180;
dir_ant=[180 270 120 210 60 150 0 90 300 30 240 330]*pi/180;
beta=zeros(12,2);
for ii=1:12
   beta(ii,:)=[cos(dir_ant(ii)) sin(dir_ant(ii))]; 
end
end

function para1 = find_peaks(Pmusic)

bw_music = imregionalmax(Pmusic);
[para(:,1), para(:,2)]=find(bw_music==1);
id=para(:,1)==size(Pmusic,1);
para(id,:)=[];
id=para(:,1)==1;
para(id,:)=[];
if(~isempty(para))
    maxval=0;
    flag=1;
    for ii=1:size(para,1)
        if Pmusic(para(ii,1),para(ii,2))>maxval
           flag=ii; 
           maxval=Pmusic(para(ii,1),para(ii,2));
        end
    end
    azimuth = para(flag,2)*360/size(Pmusic,2);
    elevation = para(flag,1)*60/(size(Pmusic,1)-1);
else
    [maxval,ind] = max(Pmusic(:));
    s=size(Pmusic);
    [i,j]=ind2sub(s,ind);
    para=[i,j];
    subPmusic=Pmusic(:,j);
    diff_subPmusic=diff(subPmusic,2);
    [~,sub_n]=min(diff_subPmusic);
    elevation=sub_n*60/(size(Pmusic,1)-1);
    azimuth=j*360/size(Pmusic,2);
end
para1=[elevation,azimuth];
end

function loc=location(para)
height=2;
para=para*pi/180;
para(:,2)=para(:,2)-pi; %??????????????????????????????pi
aa=tan(para(:,1))*height;
loc=[aa.*cos(para(:,2)),aa.*sin(para(:,2))];

end

% function track_plot(pos)
% 
% axis([-3.5,3.5,-3.5,3.5]);
% set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);
% set(gca,'XTick',(-3.5:0.1:3.5));
% set(gca,'YTick',(-3.5:0.1:3.5));
% set(gca,'Color',[0 0 0]);
% hold on;
% plot(pos(:,1),pos(:,2),'.','MarkerSize',30,'MarkerFaceColor','r');
% hold on;
% plot(0,0,'^','MarkerFaceColor',[rand rand rand],'MarkerSize',12,'MarkerEdgeColor','k');
% grid on;
% ax = gca;
% ax.GridColor = [1, 1, 1];
% pause(0.01);
% clf;
% end

function track_plot(pos)

axis([-3.5,3.5,-3.5,3.5]);
grid on;
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0, 0, 1, 1]);
set(gca,'XTick',(-3.5:0.1:3.5));
set(gca,'YTick',(-3.5:0.2:3.5));
set(gca,'Color',[0 0 0]);
hold on;

N = 2;  % ????????????????????????????????????????????10/5 = 2
% ????????????????????
a = get(gca,'XTickLabel');  
b = cell(size(a));
b(mod(1:size(a,1),N)==1,:) = a(mod(1:size(a,1),N)==1,:);
set(gca,'XTickLabel',b);

plot(pos(:,1),pos(:,2),'.','MarkerSize',30,'MarkerFaceColor','r');
hold on;
str1=[num2str(roundn(pos(:,1),-3)),',',num2str(roundn(pos(:,2),-3))];
text(pos(:,1)+0.05,pos(:,2)+0.05,str1,'color','r');
plot(0,0,'^','MarkerFaceColor',[rand rand rand],'MarkerSize',12,'MarkerEdgeColor','k');

ax = gca;
ax.GridColor = [1, 1, 1];
pause(0.01);
clf;
end


