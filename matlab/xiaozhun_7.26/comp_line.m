function data = comp_line(data)
amp=abs(data);
phase=angle(data);
% offset=[0 0.6416 2.6 5.35 -2.6 -0.8 2.39 -0.8 1.616 2.13 5.03 2.09];
phase_offset=-[0 2.1 -2.4 -2.4 -1.35 0.8 -2.35 -2.4 -1.4 0.8 -1.1 -2.14];    %quuppa
% amp_offset=100./[226 318 177 266 255 211 202 230 223 266 190 264];
amp_offset=100./[342 496 279 387 393 310 304 380 344 430 275 400];
% 
% phase_offset=[0 -2.28 2.35 2.54 1.5 -0.9 2.4 2.4 1.4 -0.8 1.3 2.13];            

% for ii=12:-1:1
%     phase(ii,:,:)=wrapToPi(phase(ii,:,:)-phase(1,:,:));
% end
for ii=1:12
%    phase(ii,:,:)=wrapToPi(phase(ii,:,:)+phase_offset(ii)); 
%    amp(ii,:,:)=amp(ii,:,:)+amp_offset(ii);
    data(ii,:,:)=data(ii,:,:).*amp_offset(ii)*exp(1j*phase_offset(ii));
end
% data=amp.*exp(1j*phase);

end