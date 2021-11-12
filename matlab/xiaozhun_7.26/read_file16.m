function [Idata,Qdata,rssi]=read_file16(filepath)
%%读取IQ数据
%输入：IQ文件存放路径
%输出：IQ数据矩阵，以及信号强度
%数据格式: 4个'30'+512组IQ值(每组4个16进制数)+4个'31'+4个'33'+RSSI值(4个16进制数)+4个'34'


% 打开文件
% filepath='data.txt';
file = fopen(filepath);

% 从文本文件或字符串读取格式化数据
% '%s'代表读取为字符向量元胞数组
%  'whitespace'代表为空白字符,指定为由 'Whitespace' 和一个字符向量或字符串（包含一个或多个字符）组成的逗号分隔对组
% 此语句中采用Whitespace和空格符进行分隔
input = textscan(file,'%s','Whitespace',' '); 
% input为1*1的cell数组


% 关闭文件
fclose( file );

% 将十六进制整数的文本表示转换为双精度值
d2= hex2dec(input{1});

% 将十进制整数转换为其十六进制表示形式
input=dec2hex(d2);


% 寻找input中为'30'的数据,返回索引值
a=find(input(:,1)=='3'& input(:,2)=='0');

% 选取a中索引为四个连在一起的,如1,2,3,4
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