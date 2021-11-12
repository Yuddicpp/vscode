function [v_phi,phi0]=search_test(data)
phase=angle(data(:,2:end));
index=data(:,1);
for ii=1:size(data,2)-1
    [v_phi(ii),phi0(ii)]=sub_search_test(index,phase(:,ii));
end
end

function [v_phi,phi0]=sub_search_test(index,phase)
v_phi_range=0.35:0.001:0.415;
phi_0_res=zeros(1,numel(v_phi_range));
min_v=zeros(1,numel(v_phi_range));
data=[index,phase];
% index=data(:,1);
for n=1:numel(v_phi_range)
    func = f(v_phi_range(n), data);
    phi_0_res(n) = fminbnd(func, -pi, pi);
    min_v(n) = func(phi_0_res(n)); 
end
[~, index] = min(min_v);
v_phi = v_phi_range(index);
phi0 = phi_0_res(index);
end
function func = f(v_phi, data)
    function v = inner_func(phi0)
        v = sum(wrapToPi(data(:, 2) - (v_phi*data(:, 1)+phi0)).^2);
    end
func = @inner_func;
end