function [Idata,Qdata,rssi]=read_file16(filepath)
%%读取IQ数据
%输入：IQ文件存放路径
%输出：IQ数据矩阵，以及信号强度
file = fopen(filepath);
input = textscan(file,'%s','whitespace',' '); 

fclose( file );
d2= hex2dec(input{1});
input=dec2hex(d2);

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
len=fix(size(b,2)/4)-1;
Idata=zeros(512,len);
Qdata=zeros(512,len);
rssi=zeros(1,len);
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