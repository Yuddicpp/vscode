



function MUSIC_CARPEN(X1,K)
% K 快拍数，X1输入信号
    N = 12;               % 阵元个数        
    M = 1;               % 信源数目
    derad = pi/180;      %角度->弧度
    dd = 0.059;            % 阵元间距 
    d=0:dd:(N-1)*dd;
    
    Rxx=X1*X1'/K;
    % 特征值分解
    [EV,D]=eig(Rxx);                   %特征值分解
    EVA=diag(D)';                      %将特征值矩阵对角线提取并转为一行
    [EVA,I]=sort(EVA);                 %将特征值排序 从小到大
    EV=fliplr(EV(:,I));                % 对应特征矢量排序
                     
     
    % 遍历每个角度，计算空间谱
    for iang = 1:360
%         angle(iang)=(iang-181)/2;
        angle(iang)=iang;
        phim=derad*angle(iang);
        a=exp(-1i*2*pi*d*sin(phim)).'; 
        En=EV(:,M+1:N);                   % 取矩阵的第M+1到N列组成噪声子空间
        Pmusic(iang)=1/(a'*En*En'*a);
    end
    Pmusic=abs(Pmusic);
    Pmmax=max(Pmusic)
    Pmusic=10*log10(Pmusic/Pmmax);            % 归一化处理
    h=plot(angle,Pmusic);
    set(h,'Linewidth',2);
    xlabel('入射角/(degree)');
    ylabel('空间谱/(dB)');
    set(gca, 'XTick',[1:10:360]);
    grid on;

end
