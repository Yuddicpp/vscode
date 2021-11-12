function P=gene_DML_P()
phi=(0:1:359)*pi/180; theta=(0:1:60)*pi/180;

for ii=1:length(theta)
   for jj=1:length(phi)
       Q=gene_Q(theta(ii),phi(jj));
       P{ii,jj}=Q*pinv(Q);
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
    array(ii,:)=R*[cos(pi-2*pi/3*(ii-1)) sin(pi-2*pi/3*(ii-1)) 0];
end

r=[sin(theta)*cos(phi) sin(theta)*sin(phi) cos(theta)];
U=zeros(6,6);
for ii=1:3
    tmp=exp(-1j*array(ii,:)*r.'*A);
    U((ii-1)*2+1,(ii-1)*2+1)=tmp;
    U(ii*2,ii*2)=tmp;
end

beta = ant_direction();

L=[-sin(phi) cos(phi)*cos(theta);cos(phi) sin(phi)*cos(theta)];
% L=[-sin(phi) cos(phi)*sin(theta);cos(phi) sin(phi)*cos(theta)];
% P=[cos(gama);sin(gama)*exp(1j*yita)];
D=U*beta*L;

amp_offset=[342 496 393 310 344 430];
phase_offset=exp(1j.*[0 2.1 -1.35 0.8 -1.4 0.8]);
offset=amp_offset.*phase_offset;
B=diag(offset);

Q=B*D;
end

function beta = ant_direction()

% dir_ant=[180 270 120 210 60 150 0 90 300 30 240 330]*pi/180;
dir_ant=[180 270 60 150 300 30]*pi/180;
beta=zeros(6,2);
for ii=1:6
   beta(ii,:)=[cos(dir_ant(ii)) sin(dir_ant(ii))]; 
end
end
