clc
clear
close all

Mag_Sea = readmatrix("Mag_z0_Bx_By_Bz_dx1000×dy1000.dat");

x_Sea = Mag_Sea(:,1) / 1000;
y_Sea = Mag_Sea(:,2) / 1000;
z_Sea = Mag_Sea(:,3);
Bx_Sea = Mag_Sea(:,4);
By_Sea = Mag_Sea(:,5);
Bz_Sea = Mag_Sea(:,6);

Min_Sea_Bx = min(Bx_Sea);
Min_Sea_By = min(By_Sea); 
Min_Sea_Bz = min(Bz_Sea);

Max_Sea_Bx = max(Bx_Sea);
Max_Sea_By = max(By_Sea);
Max_Sea_Bz = max(Bz_Sea);

Mean_Sea_Bx = mean(Bx_Sea);
Mean_Sea_By = mean(By_Sea);
Mean_Sea_Bz = mean(Bz_Sea);

Std_Sea_Bx = std(Bx_Sea, 1);
Std_Sea_By = std(By_Sea, 1);
Std_Sea_Bz = std(Bz_Sea, 1);

fprintf('Bx Sea: min = %.3f nT, max = %.3f nT, mean = %.3f nT, std = %.3f nT\n', Min_Sea_Bx, Max_Sea_Bx, Mean_Sea_Bx, Std_Sea_Bx);
fprintf('By Sea: min = %.3f nT, max = %.3f nT, mean = %.3f nT, std = %.3f nT\n', Min_Sea_By, Max_Sea_By, Mean_Sea_By, Std_Sea_By);
fprintf('Bz Sea: min = %.3f nT, max = %.3f nT, mean = %.3f nT, std = %.3f nT\n', Min_Sea_Bz, Max_Sea_Bz, Mean_Sea_Bz, Std_Sea_Bz);

Mag_Air = readmatrix("Mag_z-500_Bx_By_Bz_dx500×dy500.dat");
x_Air = Mag_Air(:,1) / 1000;
y_Air = Mag_Air(:,2) / 1000;
z_Air = Mag_Air(:,3);
Bx_Air = Mag_Air(:,4);
By_Air = Mag_Air(:,5);
Bz_Air = Mag_Air(:,6);

Min_Air_Bx = min(Bx_Air);
Min_Air_By = min(By_Air); 
Min_Air_Bz = min(Bz_Air);

Max_Air_Bx = max(Bx_Air);
Max_Air_By = max(By_Air);
Max_Air_Bz = max(Bz_Air);

Mean_Air_Bx = mean(Bx_Air);
Mean_Air_By = mean(By_Air);
Mean_Air_Bz = mean(Bz_Air);

Std_Air_Bx = std(Bx_Air, 1);
Std_Air_By = std(By_Air, 1);
Std_Air_Bz = std(Bz_Air, 1);

fprintf('Bx Air: min = %.3f nT, max = %.3f nT, mean = %.3f nT, std = %.3f nT\n', Min_Air_Bx, Max_Air_Bx, Mean_Air_Bx, Std_Air_Bx);
fprintf('By Air: min = %.3f nT, max = %.3f nT, mean = %.3f nT, std = %.3f nT\n', Min_Air_By, Max_Air_By, Mean_Air_By, Std_Air_By);
fprintf('Bz Air: min = %.3f nT, max = %.3f nT, mean = %.3f nT, std = %.3f nT\n', Min_Air_Bz, Max_Air_Bz, Mean_Air_Bz, Std_Air_Bz);

x_Sea_unique = unique(x_Sea);
y_Sea_unique = unique(y_Sea);
Nx_Sea = length(x_Sea_unique);
Ny_Sea = length(y_Sea_unique);
Bx_Sea_grid = reshape(Bx_Sea, Ny_Sea, Nx_Sea);
By_Sea_grid = reshape(By_Sea, Ny_Sea, Nx_Sea);
Bz_Sea_grid = reshape(Bz_Sea, Ny_Sea, Nx_Sea);

x_Air_unique = unique(x_Air);
y_Air_unique = unique(y_Air);
Nx_Air = length(x_Air_unique);
Ny_Air = length(y_Air_unique);
Bx_Air_grid = reshape(Bx_Air, Ny_Air, Nx_Air);
By_Air_grid = reshape(By_Air, Ny_Air, Nx_Air);
Bz_Air_grid = reshape(Bz_Air, Ny_Air, Nx_Air);

%% === 3. Plot configuration ===
fields = ["Bx", "By", "Bz"];

labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"];
label_idx = 1;

data_sea = {Bx_Sea_grid, By_Sea_grid, Bz_Sea_grid};
data_air = {Bx_Air_grid, By_Air_grid, Bz_Air_grid};

figure('Position', [100, 60, 1200, 950]);
tl = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

for i = 1:3
    % --- Row 1: Sea ---
    ax = nexttile;
    contourf(ax, x_Sea_unique, y_Sea_unique, data_sea{i}, 20);
    colormap(ax, parula);
    axis(ax,'equal');
    cbar = colorbar(ax);
    cbar.Title.String = 'nT';
    cbar.Title.FontSize = 11;
    
    % Show the ylabel only in the first column
    if i == 1
        ylabel(ax, 'y (km)', 'FontSize', 15, 'FontWeight', 'bold');
    else
        ylabel(ax, '');  % clear explicitly
    end
    xlim(ax,[-20 20])
    ylim(ax,[-20 20])
    ax.FontSize = 10;
    % Add the panel label at the top left (normalized coordinates)
    text(0.02, 0.98, labels(label_idx), ...
        'Units', 'normalized', ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'top', ...
        'FontSize', 15, 'FontWeight', 'bold', ...
        'Color', 'w', ...
        'Parent', ax);
    label_idx = label_idx + 1;
end
for i = 1:3
    % --- Row 2: Air ---
    ax = nexttile;
    contourf(ax, x_Air_unique, y_Air_unique, data_air{i}, 20);
    axis(ax,'equal');
    colormap(ax, parula);
    cbar = colorbar(ax);
    cbar.Title.String = 'nT';
    cbar.Title.FontSize = 11;
    
    xlabel(ax, 'x (km)', 'FontSize', 15, 'FontWeight', 'bold');
    % Show the ylabel only in the first column
    if i == 1
        ylabel(ax, 'y (km)', 'FontSize', 15, 'FontWeight', 'bold');
    else
        ylabel(ax, '');  % clear explicitly
    end
    xlim(ax,[-20 20])
    ylim(ax,[-20 20])
    ax.FontSize = 10;
    % Add the panel label at the top left (normalized coordinates)
    text(0.02, 0.98, labels(label_idx), ...
        'Units', 'normalized', ...
        'HorizontalAlignment', 'left', ...
        'VerticalAlignment', 'top', ...
        'FontSize', 15, 'FontWeight', 'bold', ...
        'Color', 'w', ...   
        'Parent', ax);
    label_idx = label_idx + 1;
end
