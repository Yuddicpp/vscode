function [Idata,Qdata,rssi]=read_file16(filepath)
%%��ȡIQ����
%���룺IQ�ļ����·��
%�����IQ���ݾ����Լ��ź�ǿ��
%���ݸ�ʽ: 4��'30'+512��IQֵ(ÿ��4��16������)+4��'31'+4��'33'+RSSIֵ(4��16������)+4��'34'


% ���ļ�
% filepath='data.txt';
file = fopen(filepath);

% ���ı��ļ����ַ�����ȡ��ʽ������
% '%s'�����ȡΪ�ַ�����Ԫ������
%  'whitespace'����Ϊ�հ��ַ�,ָ��Ϊ�� 'Whitespace' ��һ���ַ��������ַ���������һ�������ַ�����ɵĶ��ŷָ�����
% ������в���Whitespace�Ϳո�����зָ�
input = textscan(file,'%s','Whitespace',' '); 
% inputΪ1*1��cell����


% �ر��ļ�
fclose( file );

% ��ʮ�������������ı���ʾת��Ϊ˫����ֵ
d2= hex2dec(input{1});

% ��ʮ��������ת��Ϊ��ʮ�����Ʊ�ʾ��ʽ
input=dec2hex(d2);


% Ѱ��input��Ϊ'30'������,��������ֵ
a=find(input(:,1)=='3'& input(:,2)=='0');

% ѡȡa������Ϊ�ĸ�����һ���,��1,2,3,4
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