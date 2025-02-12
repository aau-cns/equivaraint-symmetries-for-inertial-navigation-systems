clear all
close all
clc

%% Load options

dir_path = "result_path";
folder = "result_folder";
CommonFigureSettings
data_path = dir_path + folder;
save_dir = dir_path + folder +  "Plots/";

filter_files = ["result_MEKF.mat", ...
                "result_IEKF_equi.mat", ...
                "result_LIEKF.mat", ...
                "result_TF_IEKF_equi.mat", ...
                "result_SE23_se23_EqF_equi.mat", ...
                "result_SE3_se3_R3_EqF", ...
                "result_SE23_se3_EqF_equi.mat"];

filter_labels = ["MEKF", ...
                 "R-IEKF", ...
                 "L-IEKF", ...
                 "TFG-IEKF", ...
                 "EqF $SE_2(3)_{se_2(3)}$", ...
                 "EqF $SE(3)_{se(3)} \times R^{3}$", ...
                 "EqF $SE_2(3)_{se(3)}$"];

% If gt data is available
exist_gt = true;

%% Preallocation

if plots

    % ---------------------------------------------------------------------
    % Preallocate 3D Trajectory (position) plot
    % ---------------------------------------------------------------------

    % Figure params
    figure_name = "Trajectory";

    % Figure
    fh = figure('Name',figure_name);
    clf(fh);
    set(fh, savefigimages.figOptions);

    % ---------------------------------------------------------------------
    % Preallocate position (error) plot
    % ---------------------------------------------------------------------

    % Figure params
    figure_name = "Position";

    % Figure
    fh = figure('Name',figure_name);
    clf(fh);
    set(fh, savefigimages.figOptions);

    % ---------------------------------------------------------------------
    % Preallocate orientation (error) plot
    % ---------------------------------------------------------------------

    % Figure params
    figure_name = "Orientation";

    % Figure
    fh = figure('Name',figure_name);
    clf(fh);
    set(fh, savefigimages.figOptions);


    % ---------------------------------------------------------------------
    % Preallocate velocity (error) plot
    % ---------------------------------------------------------------------

    % Figure params
    figure_name = "Velocity";

    % Figure
    fh = figure('Name',figure_name);
    clf(fh);
    set(fh, savefigimages.figOptions);

    % ---------------------------------------------------------------------
    % Preallocate position calibration (error) plot
    % ---------------------------------------------------------------------
    
    % Figure params
    figure_name = "Position_Calibration";
    
    % Figure
    fh = figure('Name',figure_name);
    clf(fh);
    set(fh, savefigimages.figOptions);

    % ---------------------------------------------------------------------
    % Preallocate ba (error) plot
    % ---------------------------------------------------------------------

    % Figure params
    figure_name = "Bias_a";

    % Figure
    fh = figure('Name',figure_name);
    clf(fh);
    set(fh, savefigimages.figOptions);

    % ---------------------------------------------------------------------
    % Preallocate bw (error) plot
    % ---------------------------------------------------------------------

    % Figure params
    figure_name = "Bias_w";

    % Figure
    fh = figure('Name',figure_name);
    clf(fh);
    set(fh, savefigimages.figOptions);

    % ---------------------------------------------------------------------
    % Preallocate Error plots
    % ---------------------------------------------------------------------
   
    if exist_gt
        
        % Figure params
        figure_name = "Position Error";
        
        % Figure
        fh = figure('Name',figure_name);
        clf(fh);
        set(fh, savefigimages.figOptions);
        
        % Figure params
        figure_name = "Orientation Error";
        
        % Figure
        fh = figure('Name',figure_name);
        clf(fh);
        set(fh, savefigimages.figOptions);
        
        % Figure params
        figure_name = "Velocity Error";
        
        % Figure
        fh = figure('Name',figure_name);
        clf(fh);
        set(fh, savefigimages.figOptions);
        
        % Figure params
        figure_name = "Position_Calibration Error";
        
        % Figure
        fh = figure('Name',figure_name);
        clf(fh);
        set(fh, savefigimages.figOptions);
        
        % Figure params
        figure_name = "Bias_a Error";
        
        % Figure
        fh = figure('Name',figure_name);
        clf(fh);
        set(fh, savefigimages.figOptions);
        
        % Figure params
        figure_name = "Bias_w Error";
        
        % Figure
        fh = figure('Name',figure_name);
        clf(fh);
        set(fh, savefigimages.figOptions);
        
    end

    % ---------------------------------------------------------------------
    % Preallocate Filter energy plot
    % ---------------------------------------------------------------------

    % Figure params
    figure_name = "Energy";
    
    % Figure
    fh = figure('Name',figure_name);
    clf(fh);
    set(fh, savefigimages.figOptions);

    % ---------------------------------------------------------------------
    % Colormap
    % ---------------------------------------------------------------------

    N_colors = 8;
    mapColor = brewermap(N_colors,'Dark2');
    
    % LIEKF shift
    mapColor(9, :) = mapColor(7, :);
    mapColor(4:7, :) = mapColor(3:6, :);
    mapColor(3, :) = mapColor(9, :);
    mapColor(9, :) = [];

end

%% Loop through filters and parse data

% Variables declaration
estimates = [];
errors = [];
energies = [];
energies_nav = [];
energies_bias = [];
energies_pose = [];
if ~exist_gt
    gt_is_loaded = true;
else
    gt_is_loaded = false;
end

for file = filter_files

    load(data_path + file)
    
    est.R = zeros(3,3,length(R_est));
    for i = 1:length(R_est)
        est.R(:,:,i) = squeeze(R_est(i,:,:));
    end
    est.p = p_est';
    est.v = v_est';
    est.bw = bw_est';
    est.ba = ba_est';
    if (exist('cal_est', 'var') == 1)
        est.cal = cal_est';
    else
        est.cal = zeros(size(p_est'));
    end
    
    
    err.a = attitude_error;
    err.p = position_error;
    err.v = velocity_error;
    err.bw = bw_error;
    err.ba = ba_error;
    if (exist('cal_error', 'var') == 1)
        err.cal = cal_error;
    else
        err.cal = zeros(size(position_error));
    end
    
    estimates = [estimates, est];
    errors = [errors, err];
    [er, ec] = size(filter_energy);
    if er > ec
        filter_energy = filter_energy';
        filter_energy_nav = filter_energy_nav';
        filter_energy_bias = filter_energy_bias';
    end
    energies = [energies; filter_energy];
    energies_nav = [energies_nav; filter_energy_nav];
    energies_bias = [energies_bias; filter_energy_bias];
%     energies_pose = [energies_pose; filter_energy_pose];
    
    if ~gt_is_loaded
        
        gt = struct;
        gt = setfield(gt,'R',zeros(3,3,length(groundtruth)));
        gt.p = zeros(3,length(groundtruth));
        gt.v = zeros(3,length(groundtruth));
        gt.bw = zeros(3,length(groundtruth));
        gt.ba = zeros(3,length(groundtruth));
        gt.cal = zeros(3,length(groundtruth));
        for i = 1:length(groundtruth)
%             gt.R(:,:,i) = [groundtruth{i,1,1}; groundtruth{i,1,2}; groundtruth{i,1,3}];
%             gt.v(:,i) =  [groundtruth{i,2,1}; groundtruth{i,2,2}; groundtruth{i,2,3}];
%             gt.p(:,i) =  [groundtruth{i,3,1}; groundtruth{i,3,2}; groundtruth{i,3,3}];
%             gt.bw(:,i) =  [groundtruth{i,4,1}; groundtruth{i,4,2}; groundtruth{i,4,3}];
%             gt.ba(:,i) =  [groundtruth{i,5,1}; groundtruth{i,5,2}; groundtruth{i,5,3}];
%             if (s == 6)
%                 gt.cal(:,i) =  [groundtruth{i,6,1}; groundtruth{i,6,2}; groundtruth{i,6,3}];
%             end
            gt.R(:,:,i) = quat2rotm([groundtruth(i, 4) groundtruth(i, 1:3)]);
            gt.v(:,i) =  groundtruth(i, 5:7);
            gt.p(:,i) =  groundtruth(i, 8:10);
            gt.bw(:,i) =  groundtruth(i, 11:13);
            gt.ba(:,i) =  groundtruth(i, 14:16);
            gt.cal(:,i) =  groundtruth(i, 17:19);
        end
        
        gt_is_loaded = true;
    end

end

%% Compute RMSE

% for i = 1:length(filter_files)
%     rmse(t, errors(i).a, errors(i).p, errors(i).v, errors(i).bw, errors(i).ba, errors(i).cal, filter_labels(i), print_rmse);
% end

%% Plots

if plots

    t = t - t(1);
    
    % -----------------------------------------------------------------
    % Trajectory (position) plot
    % -----------------------------------------------------------------

    fh = findobj('Type','Figure','Name','Trajectory');
    set(groot,'CurrentFigure',fh);
    title("\textbf{UAV Trajectory}",'Interpreter','latex','fontsize',TitleFontSize);
    if exist_gt
        plot3(gt.p(1,:), gt.p(2,:), gt.p(3,:), '-.', 'LineWidth', 3.5, 'Color',mapColor(8,:))
    end
    grid on;
    hold on;
    for i = 1:length(estimates)
        plot3(estimates(i).p(1,:), estimates(i).p(2,:), estimates(i).p(3,:), 'LineWidth', 3.5, 'Color',mapColor(i,:))
    end
    axis equal
    
    % -----------------------------------------------------------------
    % Position (and error) plots
    % -----------------------------------------------------------------

    fh = findobj('Type','Figure','Name','Position');
    set(groot,'CurrentFigure',fh);
    subplot(3,1,1)
    title("\textbf{UAV Position, x}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    subplot(3,1,2)
    title("\textbf{UAV Position, y}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    subplot(3,1,3)
    title("\textbf{UAV Position, z}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    for j = 1:3
        subplot(3,1,j)
        if exist_gt
            plot(t, gt.p(j,:), 'LineWidth', 2, 'Color',mapColor(8,:))
        end
        for i = 1:length(estimates)
            plot(t, estimates(i).p(j,:), 'LineWidth', 2, 'Color',mapColor(i,:))
        end
    end

    if exist_gt
        fh = findobj('Type','Figure','Name','Position Error');
        set(groot,'CurrentFigure',fh);
        title("\textbf{UAV Position RMSE}",'Interpreter','latex','fontsize',TitleFontSize);
        grid on;
        hold on;
        for i = 1:length(errors)
            plot(t, errors(i).p, 'LineWidth', 2, 'Color',mapColor(i,:))
        end
    end
    
    % -----------------------------------------------------------------
    % Orientation (and error) plots
    % -----------------------------------------------------------------

    fh = findobj('Type','Figure','Name','Orientation');
    set(groot,'CurrentFigure',fh);
    subplot(3,1,1)
    title("\textbf{UAV Orientation, yaw}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    subplot(3,1,2)
    title("\textbf{UAV Orientation, pitch}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    subplot(3,1,3)
    title("\textbf{UAV Orientation, roll}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;

    if exist_gt
        GT_angles = rotm2eul(gt.R, 'ZYX')*180/pi;    
    end
    for j = 1:3
        subplot(3,1,j)
        if exist_gt
            plot(t, GT_angles(:,j), 'LineWidth', 2, 'Color',mapColor(8,:))
        end
        for i = 1:length(estimates)
            est_angles = rotm2eul(estimates(i).R, 'ZYX')*180/pi;
            plot(t, est_angles(:,j), 'LineWidth', 2, 'Color',mapColor(i,:))
        end
    end

    if exist_gt
        fh = findobj('Type','Figure','Name','Orientation Error');
        set(groot,'CurrentFigure',fh);
        title("\textbf{UAV Orientation RMSE}",'Interpreter','latex','fontsize',TitleFontSize);
        grid on;
        hold on;
        for i = 1:length(errors)
            plot(t, errors(i).a*180/pi, 'LineWidth', 2, 'Color',mapColor(i,:))
        end
    end

    % -----------------------------------------------------------------
    % Velocity (and error) plots
    % ----------------------------------------------------------------

    fh = findobj('Type','Figure','Name','Velocity');
    set(groot,'CurrentFigure',fh);
    subplot(3,1,1)
    title("\textbf{UAV Velocity, x}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    subplot(3,1,2)
    title("\textbf{UAV Velocity, y}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    subplot(3,1,3)
    title("\textbf{UAV Velocity, z}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    for j = 1:3
        subplot(3,1,j)
        if exist_gt
            plot(t, gt.v(j,:), 'LineWidth', 2, 'Color',mapColor(8,:))
        end
        for i = 1:length(estimates)
            plot(t, estimates(i).v(j,:), 'LineWidth', 2, 'Color',mapColor(i,:))
        end
    end
    
    if exist_gt
        fh = findobj('Type','Figure','Name','Velocity Error');
        set(groot,'CurrentFigure',fh);
        grid on;
        hold on;
        for i = 1:length(errors)
            plot(t, errors(i).v, 'LineWidth', 2, 'Color',mapColor(i,:))
        end
    end
    
    % -----------------------------------------------------------------
    % Calibration (and error) plots
    % -----------------------------------------------------------------
    
    fh = findobj('Type','Figure','Name','Position_Calibration');
    set(groot,'CurrentFigure',fh);
    subplot(3,1,1)
    title("\textbf{Position Calibration, x}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    subplot(3,1,2)
    title("\textbf{Position Calibration, y}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    subplot(3,1,3)
    title("\textbf{Position Calibration, z}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    for j = 1:3
        subplot(3,1,j)
        if exist_gt
            plot(t, gt.cal(j,:), 'LineWidth', 2, 'Color',mapColor(8,:))
        end
        for i = 1:length(estimates)
            plot(t, estimates(i).cal(j,:), 'LineWidth', 2, 'Color',mapColor(i,:))
        end
    end
    
    if exist_gt
        fh = findobj('Type','Figure','Name','Position_Calibration Error');
        set(groot,'CurrentFigure',fh);
        grid on;
        hold on;
        for i = 1:length(errors)
            plot(t, errors(i).cal, 'LineWidth', 2.5, 'Color',mapColor(i,:))
        end
    end
    
    % -----------------------------------------------------------------
    % biases (and error) plots
    % -----------------------------------------------------------------

    fh = findobj('Type','Figure','Name','Bias_w');
    set(groot,'CurrentFigure',fh);
    subplot(3,1,1)
    title("\textbf{Gyro Bias, bwx}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    subplot(3,1,2)
    title("\textbf{Gyro Bias, bwy}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    subplot(3,1,3)
    title("\textbf{Gyro Bias, bwz}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    for k = 1:3
        subplot(3,1,k)
        if exist_gt
            plot(t, gt.bw(k,:), 'LineWidth', 2, 'Color',mapColor(8,:))
        end
        for i = 1:length(estimates)
            plot(t, estimates(i).bw(k,:), 'LineWidth', 2, 'Color',mapColor(i,:))
        end
        ylim([-0.1,0.1])
    end


    if exist_gt
        fh = findobj('Type','Figure','Name','Bias_w Error');
        set(groot,'CurrentFigure',fh);
        grid on;
        hold on;
        for i = 1:length(errors)
            plot(t, errors(i).bw, 'LineWidth', 2, 'Color',mapColor(i,:))
        end
    end
    
    fh = findobj('Type','Figure','Name','Bias_a');
    set(groot,'CurrentFigure',fh);
    subplot(3,1,1)
    title("\textbf{Acc Bias, bax}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    subplot(3,1,2)
    title("\textbf{Acc Bias, bay}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    subplot(3,1,3)
    title("\textbf{Acc Bias, baz}",'Interpreter','latex','fontsize',TitleFontSize);
    grid on;
    hold on;
    for k = 1:3
        subplot(3,1,k)
        if exist_gt
            plot(t, gt.ba(k,:), 'LineWidth', 2, 'Color',mapColor(8,:))
        end
        for i = 1:length(estimates)
            plot(t, estimates(i).ba(k,:), 'LineWidth', 2, 'Color',mapColor(i,:))
        end
        ylim([-0.1,0.1])
    end

    if exist_gt
        fh = findobj('Type','Figure','Name','Bias_a Error');
        set(groot,'CurrentFigure',fh);
        grid on;
        hold on;
        for i = 1:length(errors)
            plot(t, errors(i).ba, 'LineWidth', 2, 'Color',mapColor(i,:))
        end
    end
    
end

% -----------------------------------------------------------------
% Filter energy plot
% -----------------------------------------------------------------

fh = findobj('Type','Figure','Name','Energy');
set(groot,'CurrentFigure',fh);
title("\textbf{Pose Energy (log plot)}",'Interpreter','latex','fontsize',TitleFontSize);
grid on;
hold on;
[r, ~] = size(energies);
for i = 1:r
    plot(t(1:length(energies(i, :))), energies(i, :), 'LineWidth', 2, 'Color',mapColor(i,:))
%     plot(t, energies_nav(i, :), 'LineWidth', 2, 'Color',mapColor(i,:))
%     plot(t, energies_bias(i, :), 'LineWidth', 2, 'Color',mapColor(i,:))
%     plot(t, energies_pose(i, :), 'LineWidth', 2, 'Color',mapColor(i,:))
end

% ---------------------------------------------------------------------
% Define legend labels
% ---------------------------------------------------------------------

gt_label = []
if exist_gt
    gt_label = ["Ground-Truth"];
end

% ---------------------------------------------------------------------
% Finalize Trajectory (position) plot
% ---------------------------------------------------------------------

figure_title = "\textbf{UAV trajectory}";
figure_x_label = "x [m]";
figure_y_label = "y [m]";
figure_z_label = "z [m]";
fh = findobj('Type','Figure','Name','Trajectory');
set(groot,'CurrentFigure',fh);
set(gca,'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
% title(figure_title,'Interpreter','latex','fontsize',TitleFontSize);
% legend([gt_label filter_labels],'Interpreter','latex','FontSize', LegendFontSize);
xlabel(figure_x_label,'Interpreter','latex');
ylabel(figure_y_label,'Interpreter','latex');
zlabel(figure_z_label,'Interpreter','latex');

% ---------------------------------------------------------------------
% Finalize Position plots
% ---------------------------------------------------------------------

fh = findobj('Type','Figure','Name','Position');
set(groot,'CurrentFigure',fh);
set(fh.Children(1),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
set(fh.Children(2),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
set(fh.Children(3),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
legend([gt_label filter_labels],'Interpreter','latex','FontSize', LegendFontSize);
figure_x_label = "t [s]";
xlabel(figure_x_label,'Interpreter','latex');

if exist_gt
    fh = findobj('Type','Figure','Name','Position Error');
    set(groot,'CurrentFigure',fh);
    set(gca,'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
    legend([filter_labels],'Interpreter','latex','FontSize', LegendFontSize);
    figure_x_label = "t [s]";
    figure_y_label = "RMSE [m]";
    xlabel(figure_x_label,'Interpreter','latex');
    ylabel(figure_y_label,'Interpreter','latex');
end

% ---------------------------------------------------------------------
% Finalize Orientation plots
% ---------------------------------------------------------------------

fh = findobj('Type','Figure','Name','Orientation');
set(groot,'CurrentFigure',fh);
set(fh.Children(1),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
set(fh.Children(2),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
set(fh.Children(3),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
legend([gt_label filter_labels],'Interpreter','latex','FontSize', LegendFontSize);
figure_x_label = "t [s]";
xlabel(figure_x_label,'Interpreter','latex');

if exist_gt
    fh = findobj('Type','Figure','Name','Orientation Error');
    set(groot,'CurrentFigure',fh);
    set(gca,'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
    legend([filter_labels],'Interpreter','latex','FontSize', LegendFontSize);
    figure_x_label = "t [s]";
    figure_y_label = "RMSE [deg]";
    xlabel(figure_x_label,'Interpreter','latex');
    ylabel(figure_y_label,'Interpreter','latex');
end

% ---------------------------------------------------------------------
% Finalize Velocity plots
% ---------------------------------------------------------------------

fh = findobj('Type','Figure','Name','Velocity');
set(groot,'CurrentFigure',fh);
set(fh.Children(1),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
set(fh.Children(2),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
set(fh.Children(3),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
legend([gt_label filter_labels],'Interpreter','latex','FontSize', LegendFontSize);
figure_x_label = "t [s]";
xlabel(figure_x_label,'Interpreter','latex');

if exist_gt
    fh = findobj('Type','Figure','Name','Velocity Error');
    set(groot,'CurrentFigure',fh);
    set(gca,'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
    legend([filter_labels],'Interpreter','latex','FontSize', LegendFontSize);
    figure_x_label = "t [s]";
    figure_y_label = "RMSE [m/s]";xlabel(figure_x_label,'Interpreter','latex');
    ylabel(figure_y_label,'Interpreter','latex');
end

% ---------------------------------------------------------------------
% Finalize Position calibration plots
% ---------------------------------------------------------------------

fh = findobj('Type','Figure','Name','Position_Calibration');
set(groot,'CurrentFigure',fh);
set(fh.Children(1),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
set(fh.Children(2),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
set(fh.Children(3),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
legend([gt_label filter_labels],'Interpreter','latex','FontSize', LegendFontSize);
figure_x_label = "t [s]";
xlabel(figure_x_label,'Interpreter','latex');

if exist_gt
    fh = findobj('Type','Figure','Name','Position_Calibration Error');
    set(groot,'CurrentFigure',fh);
    set(gca,'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
    legend([filter_labels],'Interpreter','latex','FontSize', LegendFontSize);
    figure_x_label = "t [s]";
    figure_y_label = "RMSE [m]";
    xlabel(figure_x_label,'Interpreter','latex');
    ylabel(figure_y_label,'Interpreter','latex');
end

% ---------------------------------------------------------------------
% Finalize bias bw
% ---------------------------------------------------------------------

fh = findobj('Type','Figure','Name','Bias_w');
set(groot,'CurrentFigure',fh);
set(fh.Children(1),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
set(fh.Children(2),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
set(fh.Children(3),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
legend([gt_label filter_labels],'Interpreter','latex','FontSize', LegendFontSize);
figure_x_label = "t [s]";
xlabel(figure_x_label,'Interpreter','latex');

if exist_gt
    fh = findobj('Type','Figure','Name','Bias_w Error');
    set(groot,'CurrentFigure',fh);
    set(gca,'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
    legend([filter_labels],'Interpreter','latex','FontSize', LegendFontSize);
    figure_x_label = "t [s]";
    figure_y_label = "RMSE [rad/s]";
    xlabel(figure_x_label,'Interpreter','latex');
    ylabel(figure_y_label,'Interpreter','latex');
end

% ---------------------------------------------------------------------
% Finalize bias ba
% ---------------------------------------------------------------------

fh = findobj('Type','Figure','Name','Bias_a');
set(groot,'CurrentFigure',fh);
set(fh.Children(1),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
set(fh.Children(2),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
set(fh.Children(3),'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
legend([gt_label filter_labels],'Interpreter','latex','FontSize', LegendFontSize);
figure_x_label = "t [s]";
xlabel(figure_x_label,'Interpreter','latex');

if exist_gt
    fh = findobj('Type','Figure','Name','Bias_a Error');
    set(groot,'CurrentFigure',fh);
    set(gca,'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
    legend([filter_labels],'Interpreter','latex','FontSize', LegendFontSize);
    figure_x_label = "t [s]";
    figure_y_label = "RMSE [m/$s^2$]";
    xlabel(figure_x_label,'Interpreter','latex');
    ylabel(figure_y_label,'Interpreter','latex');
end

% ---------------------------------------------------------------------
% Finalize Energy plot
% ---------------------------------------------------------------------

if exist_gt
    % Figure params
%     figure_title = "\textbf{Filter Energy}";
    figure_title = "\textbf{Innovation Energy}";
    figure_x_label = 't [s]';
    fh = findobj('Type','Figure','Name','Energy');
    set(groot,'CurrentFigure',fh);
    set(gca,'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1, 'YScale', 'log');
    title(figure_title,'Interpreter','latex','fontsize',TitleFontSize)
    xlabel(figure_x_label,'Interpreter','latex');
    legend([filter_labels],'Interpreter','latex','FontSize', LegendFontSize);
end

% ---------------------------------------------------------------------
% Save plots
% ---------------------------------------------------------------------

% Save
if save_plots
    
    if ~exist(save_dir, 'dir')
        fprintf("Info:" + save_dir + " does not exist. Creating folder...\n");
        mkdir(save_dir)
    end
    
    fh = findobj('Type','Figure','Name','Trajectory');
    export_name = regexprep(fh.Name,{'{','}',' '},{'','','_'});
    export_name = lower(export_name);
    export_fig(fh, save_dir+export_name, savefigimages.printOptions{:}, '-nocrop');
    savefig(fh, save_dir+export_name + ".fig");
    
    fh = findobj('Type','Figure','Name','Position');
    export_name = regexprep(fh.Name,{'{','}',' '},{'','','_'});
    export_name = lower(export_name);
    export_fig(fh, save_dir+export_name, savefigimages.printOptions{:}, '-nocrop');
    savefig(fh, save_dir+export_name + ".fig"); 
    
    fh = findobj('Type','Figure','Name','Orientation');
    export_name = regexprep(fh.Name,{'{','}',' '},{'','','_'});
    export_name = lower(export_name);
    export_fig(fh, save_dir+export_name, savefigimages.printOptions{:}, '-nocrop');
    savefig(fh, save_dir+export_name + ".fig");
    
    fh = findobj('Type','Figure','Name','Velocity');
    export_name = regexprep(fh.Name,{'{','}',' '},{'','','_'});
    export_name = lower(export_name);
    export_fig(fh, save_dir+export_name, savefigimages.printOptions{:}, '-nocrop');
    savefig(fh, save_dir+export_name + ".fig");
    
    fh = findobj('Type','Figure','Name','Bias_w');
    export_name = regexprep(fh.Name,{'{','}',' '},{'','','_'});
    export_name = lower(export_name);
    export_fig(fh, save_dir+export_name, savefigimages.printOptions{:}, '-nocrop');
    savefig(fh, save_dir+export_name + ".fig");
    
    fh = findobj('Type','Figure','Name','Bias_a');
    export_name = regexprep(fh.Name,{'{','}',' '},{'','','_'});
    export_name = lower(export_name);
    export_fig(fh, save_dir+export_name, savefigimages.printOptions{:}, '-nocrop');
    savefig(fh, save_dir+export_name + ".fig");
    
    fh = findobj('Type','Figure','Name','Position_Calibration');
    export_name = regexprep(fh.Name,{'{','}',' '},{'','','_'});
    export_name = lower(export_name);
    export_fig(fh, save_dir+export_name, savefigimages.printOptions{:}, '-nocrop');
    savefig(fh, save_dir+export_name + ".fig");
       
    if exist_gt
        
        fh = findobj('Type','Figure','Name','Position Error');
        export_name = regexprep(fh.Name,{'{','}',' '},{'','','_'});
        export_name = lower(export_name);
        export_fig(fh, save_dir+export_name, savefigimages.printOptions{:}, '-nocrop');
        savefig(fh, save_dir+export_name + ".fig");
        
        fh = findobj('Type','Figure','Name','Orientation Error');
        export_name = regexprep(fh.Name,{'{','}',' '},{'','','_'});
        export_name = lower(export_name);
        export_fig(fh, save_dir+export_name, savefigimages.printOptions{:}, '-nocrop');
        savefig(fh, save_dir+export_name + ".fig");
        
        fh = findobj('Type','Figure','Name','Velocity Error');
        export_name = regexprep(fh.Name,{'{','}',' '},{'','','_'});
        export_name = lower(export_name);
        export_fig(fh, save_dir+export_name, savefigimages.printOptions{:}, '-nocrop');
        savefig(fh, save_dir+export_name + ".fig");
        
        fh = findobj('Type','Figure','Name','Position_Calibration Error');
        export_name = regexprep(fh.Name,{'{','}',' '},{'','','_'});
        export_name = lower(export_name);
        export_fig(fh, save_dir+export_name, savefigimages.printOptions{:}, '-nocrop');
        savefig(fh, save_dir+export_name + ".fig");
        
        fh = findobj('Type','Figure','Name','Bias_w Error');
        export_name = regexprep(fh.Name,{'{','}',' '},{'','','_'});
        export_name = lower(export_name);
        export_fig(fh, save_dir+export_name, savefigimages.printOptions{:}, '-nocrop');
        savefig(fh, save_dir+export_name + ".fig");
        
        fh = findobj('Type','Figure','Name','Bias_a Error');
        export_name = regexprep(fh.Name,{'{','}',' '},{'','','_'});
        export_name = lower(export_name);
        export_fig(fh, save_dir+export_name, savefigimages.printOptions{:}, '-nocrop');
        savefig(fh, save_dir+export_name + ".fig");
        
        fh = findobj('Type','Figure','Name','Energy');
        export_name = regexprep(fh.Name,{'{','}',' '},{'','','_'});
        export_name = lower(export_name);
        export_fig(fh, save_dir+export_name, savefigimages.printOptions{:}, '-nocrop');
        savefig(fh, save_dir+export_name + ".fig");
        
    end
    
end