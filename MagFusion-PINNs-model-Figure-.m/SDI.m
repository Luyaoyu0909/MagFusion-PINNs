clc
clear 
close all
rng(2026)
% 0:No Noise,1:With Noise
flag = 1;

Mag_interp = readmatrix("Mag_Test_A3Remaining.dat");

if flag==0
    Mag_obs = readmatrix("Mag_Air1476_Sea492_x_y_z_Bx_By_Bz.dat");
    name = sprintf('Mag_test_SDI_%dT%d.dat',length(Mag_obs),length(Mag_interp));

elseif flag==1
    Mag_obs = readmatrix("Mag_Air1476_Sea492_x_y_z_Bx_By_Bz_snr30.dat");
    name = sprintf('Mag_test_SDI_%dT%d_snr30.dat',length(Mag_obs),length(Mag_interp));
end

X = Mag_obs(:,1);
Y = Mag_obs(:,2);
Z = Mag_obs(:,3);

Bx = Mag_obs(:,4);
By = Mag_obs(:,5);
Bz = Mag_obs(:,6);

Xq = Mag_interp(:,1);
Yq = Mag_interp(:,2);
Zq = Mag_interp(:,3);

V = [Bx,By,Bz];

Method = 'linear';
ExtrapolationMethod = 'linear';

disp("Building model")
tic
F = scatteredInterpolant(X,Y,Z,V,Method,ExtrapolationMethod);
toc

disp("Interpolation")
tic
Vq = F(Xq,Yq,Zq);
toc

temp = [Xq,Yq,Zq,Vq];

save(name,"temp",'-ascii', '-double', '-tabs');
