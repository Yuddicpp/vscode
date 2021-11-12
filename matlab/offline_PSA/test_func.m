function para=test_func(R)
 [eigenvectors num_signal] = noise_space_eigenvectors(R);
  Pmusic = music_spectrum(eigenvectors);
  para = find_peaks(Pmusic);
end



function [eigenvectors num_signal] = noise_space_eigenvectors(R)
    % Data covarivance matrix
    
    [eigenvectors,eigenvalue_matrix]=eig(R);%%%% 
    EVA=diag(eigenvalue_matrix)';
    [EVA,I]=sort(EVA);
    eigenvectors=eigenvectors(:,I);%锟斤拷锟斤拷
    
    for i=1:length(EVA)-1%-2表示总是有多径
        de_ratios(i)=EVA(i)/EVA(i+1);
    end
%     [max_de_ratio,max_de_index]=min(de_ratios);
    [max_de_ratio,max_de_index1]=min(de_ratios(9:end));
    max_de_index=max_de_index1+8;
    eigenvectors = eigenvectors(:,1:max_de_index-6);
    num_signal=12-max_de_index;
%     eigenvectors = eigenvectors(:,1:25);
%     num_signal=30-25;
end

function Pmusic = music_spectrum(eigenvectors)
    theta=(0:1:60)*pi/180; phi=(0:1:359)*pi/180;gama=(0:5:90)*pi/180; yita=(0:5:359)*pi/180;
    Pmusic = zeros(length(theta), length(phi));
    % Angle of Arrival Loop (AoA)
    for ii = 1:length(theta)
        % Time of Flight Loop (ToF)
        for jj = 1:length(phi)
%             for mm=1:length(gama)
%                 for nn=1:length(yita)
                    steering_vector = compute_steering_vector(theta(ii), phi(jj));
                    PP = det(steering_vector' * (eigenvectors * eigenvectors') * steering_vector);
                    Pmusic(ii, jj) = 1/abs(PP);                 
%                 end
%             end
        end
    end

    % Convert to decibels
    % ToF loop
%     for jj = 1:size(Pmusic, 2)
%         % AoA loop
%         for ii = 1:size(Pmusic, 1)
%             Pmusic(ii, jj) = 10 * log10(Pmusic(ii, jj));% / max(Pmusic(:, jj))); 
%         end
%     end
end

function steering_vector = compute_steering_vector(theta, phi)
R=0.059;
c=3e8;
f=2.426e9;
A=2*pi*f/c;
array=zeros(6,3);
for ii=1:6
   array(ii,:)=R*[cos(pi-pi/3*(ii-1)) sin(pi-pi/3*(ii-1)) 0];
end

r=[sin(theta)*cos(phi) sin(theta)*sin(phi) cos(theta)];
U=zeros(12,12);
for ii=1:6
    tmp=exp(-1j*array(ii,:)*r.'*A);
    U((ii-1)*2+1,(ii-1)*2+1)=tmp;
    U(ii*2,ii*2)=tmp;
end

U1=zeros(6,1);
for ii=1:6
    tmp=exp(-1j*array(ii,:)*r.'*A);
    U1(ii,1)=tmp; 
end
beta = ant_direction();

L=[-sin(phi) cos(phi)*cos(theta);cos(phi) sin(phi)*cos(theta)];
% L=[-sin(phi) cos(phi)*sin(theta);cos(phi) sin(phi)*cos(theta)];
% P=[cos(gama);sin(gama)*exp(1j*yita)];
D=U*beta*L;

% D=kron(U1,L);
steering_vector=D;
end


function beta = ant_direction()
% dir_ant=[0 90 0 90 0 90 0 90 0 90 0 90]*pi/180;
dir_ant=[180 270 120 210 60 150 0 90 300 30 240 330]*pi/180;
beta=zeros(12,2);
for ii=1:12
   beta(ii,:)=[cos(dir_ant(ii)) sin(dir_ant(ii))]; 
end
end

function para = find_peaks(Pmusic)
[maxval,ind] = max(Pmusic(:));
s=size(Pmusic);
[i,j]=ind2sub(s,ind);
para=[i,j];
end