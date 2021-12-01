function steering_vector=gene_A(theta, phi,gamma,eta)
steering_vector = compute_steering_vector(theta, phi,gamma,eta);
end

function steering_vector = compute_steering_vector(theta, phi,gamma,eta)
R=0.059;
c=3e8;
f=2.426e9;
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
P=[cos(gamma);sin(gamma)*exp(1j*eta)];
D=U*beta*L*P;
steering_vector=D;
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