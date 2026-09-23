clc
clear
close all

Mag_ref = readmatrix("Mag_z0_Bx_By_Bz_dx500×dy500.dat");
x_ref = Mag_ref(:,1);
y_ref = Mag_ref(:,2);

Bx_ref = Mag_ref(:,4);
By_ref = Mag_ref(:,5);
Bz_ref = Mag_ref(:,6);

Mag_Air_ref_snr = readmatrix("Mag_z-500_Bx_By_Bz_snr30_dx500×dy500.dat");

Bx_Air_ref_snr = Mag_Air_ref_snr(:,4);
By_Air_ref_snr = Mag_Air_ref_snr(:,5);
Bz_Air_ref_snr = Mag_Air_ref_snr(:,6);

Mag_PINN = readmatrix("PredMag_GridTestData1968T6561_NN6_40_swish_noEmb_Adam_RADk1c0n3000d600_nEpo12001_BS3000_Seed12345.dat");
Bx_PINN = Mag_PINN(:,4);
By_PINN = Mag_PINN(:,5);
Bz_PINN = Mag_PINN(:,6);

x_unique = unique(x_ref);
y_unique = unique(y_ref);
Nx = length(x_unique);
Ny = length(y_unique);

Bx_ref_grid_snr = reshape(Bx_Air_ref_snr, Ny, Nx)';
By_ref_grid_snr = reshape(By_Air_ref_snr, Ny, Nx)';
Bz_ref_grid_snr = reshape(Bz_Air_ref_snr, Ny, Nx)';

Bx_ref_grid = reshape(Bx_ref, Ny, Nx)';
By_ref_grid = reshape(By_ref, Ny, Nx)';
Bz_ref_grid = reshape(Bz_ref, Ny, Nx)';

Bx_PINN_grid = reshape(Bx_PINN, Ny, Nx)';
By_PINN_grid = reshape(By_PINN, Ny, Nx)';
Bz_PINN_grid = reshape(Bz_PINN, Ny, Nx)';

x_unique = x_unique / 1000;
y_unique = y_unique / 1000;

if 1
figure('Position', [100, 50, 1130, 950]);

tl = tiledlayout(3, 3, 'TileSpacing', 'none', 'Padding', 'compact');

% ========== Row 1: Bx component ==========
ax1 = nexttile;
contourf(ax1, x_unique, y_unique, Bx_ref_grid_snr');

ax2 = nexttile;
contourf(ax2, x_unique, y_unique, Bx_PINN_grid');

ax3 = nexttile;
contourf(ax3, x_unique, y_unique, abs(Bx_PINN_grid' - Bx_ref_grid'));

% ========== Row 2: By component ==========
ax4 = nexttile;
contourf(ax4, x_unique, y_unique, By_ref_grid_snr');

ax5 = nexttile;
contourf(ax5, x_unique, y_unique, By_PINN_grid');

ax6 = nexttile;
contourf(ax6, x_unique, y_unique, abs(By_PINN_grid' - By_ref_grid'));

% ========== Row 3: Bz component ==========
ax7 = nexttile;
contourf(ax7, x_unique, y_unique, Bz_ref_grid_snr');

ax8 = nexttile;
contourf(ax8, x_unique, y_unique, Bz_PINN_grid');

ax9 = nexttile;
contourf(ax9, x_unique, y_unique, abs(Bz_PINN_grid' - Bz_ref_grid'));

tl.TileSpacing = 'compact';

labels =   {'(i)', '(h)', '(g)', '(f)', '(e)', '(d)', '(c)', '(b)', '(a)'};

% Apply uniform axis limits and ticks to all subplots
allAxes = findobj(tl, 'Type', 'axes');
allAxes = reshape(allAxes,3,3);
label_idx = 1;

for col = 1:3 % column
    for row = 1:3 % row
        ax = allAxes(col, row); % column-major order; origin at the bottom right
        
        ax1 = allAxes(col, 1);
        xlabel(ax1, 'x (km)', 'FontSize', 15, 'FontWeight', 'bold');
        ax2 = allAxes(3, row);
        ylabel(ax2, 'y (km)', 'FontSize', 15, 'FontWeight', 'bold');
        cbar = colorbar(ax);
        cbar.Title.String = 'nT';
        cbar.Title.FontSize = 12;
        set(ax, 'FontSize', 12, 'FontWeight', 'bold');
         
        text(ax, 0.02, 0.98, labels{label_idx}, 'Units', 'normalized', ...
             'FontSize', 15, 'FontWeight', 'bold', ...
             'VerticalAlignment', 'top', 'Color', 'w');
        
        axis(ax, "equal");
        xlim(ax, [-20 20]);
        ylim(ax, [-20 20]);
        xticks(ax, -20:10:20);
        yticks(ax, -20:10:20);
        
        label_idx = label_idx + 1;
    end
    
end
end