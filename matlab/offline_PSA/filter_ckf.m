function [his_loc,new_loc,M,P,ind,MM_CKF,PP_CKF]=filter_ckf(his_loc,loc,M,P,ind,MM_CKF,PP_CKF)
% M = [0 0 1 0 0]';
% P = diag([10.1 10.1 1.1 1.1 1]);
a_func = @f_turn;
h_func = @model_h;
Q=0.05*eye(5);
R=eye(2);
dt=0.1;
params = {dt};

count=size(loc,2);
if size(his_loc,2)<12
    his_loc=[his_loc loc];
    for k = ind:ind+count-1
       [M,P] = ckf_predict(M,P,a_func,Q,{dt});
       [M,P] = ckf_update(M,P,his_loc(:,k),h_func,R);
       MM_CKF(:,k)   = M;
       PP_CKF(:,:,k) = P;    
    end
    new_loc=MM_CKF(1:2,ind:ind+count-1);
    ind=ind+count;
    
elseif size(his_loc,2)>11
    his_loc=[his_loc loc];
    flag=0;
    for k=ind-10:ind+count-10-1
        flag=flag+1;
           [M,P] = ckf_predict(M,P,a_func,Q,{dt}); 
           [M,P] = ckf_update(M,P,his_loc(:,k+10),h_func,R);
           MM_CKF(:,k+10)   = M;
           PP_CKF(:,:,k+10) = P;    
           [MMS_CRTS, PPS_CRTS] = crts_smooth(MM_CKF(:,k:k+10),PP_CKF(:,:,k:k+10),a_func,Q,params,1);
           new_loc(:,flag)= MMS_CRTS(1:2,1); 
    end
    ind=ind+count;
end


end