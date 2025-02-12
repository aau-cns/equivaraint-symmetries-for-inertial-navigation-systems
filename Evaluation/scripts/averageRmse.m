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

assert(length(filter_files) == length(filter_labels));
N = 600;
exclude = [12 79]; % Run to exclude in the montecarlo simulation

%% close old plots

close all

%% Preallocation

if plots

    % ---------------------------------------------------------------------
    % Preallocate position error plot
    % ---------------------------------------------------------------------

    % Figure params
    figure_name = "Position Error";

    % Figure
    fh = figure('Name',figure_name);
    clf(fh);
    set(fh, savefigimages.figOptions);

    % ---------------------------------------------------------------------
    % Preallocate orientation error plot
    % ---------------------------------------------------------------------

    % Figure params
    figure_name = "Orientation Error";

    % Figure
    fh = figure('Name',figure_name);
    clf(fh);
    set(fh, savefigimages.figOptions);


    % ---------------------------------------------------------------------
    % Preallocate velocity error plot
    % ---------------------------------------------------------------------

    % Figure params
    figure_name = "Velocity Error";

    % Figure
    fh = figure('Name',figure_name);
    clf(fh);
    set(fh, savefigimages.figOptions);


    % ---------------------------------------------------------------------
    % Preallocate position calibration (error) plot
    % ---------------------------------------------------------------------
    
    % Figure params
    figure_name = "Position_Calibration Error";
    
    % Figure
    fh = figure('Name',figure_name);
    clf(fh);
    set(fh, savefigimages.figOptions);

    % ---------------------------------------------------------------------
    % Preallocate ba error plot
    % ---------------------------------------------------------------------
    
    % Figure params
    figure_name = "Bias_a Error";
    
    % Figure
    fh = figure('Name',figure_name);
    clf(fh);
    set(fh, savefigimages.figOptions);

    % ---------------------------------------------------------------------
    % Preallocate bw error plot
    % ---------------------------------------------------------------------

    % Figure params
    figure_name = "Bias_w Error";
    
    % Figure
    fh = figure('Name',figure_name);
    clf(fh);
    set(fh, savefigimages.figOptions);
    
    % ---------------------------------------------------------------------
    % Preallocate bw error plot
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

    N_colors = max(8, length(filter_labels));
    mapColor = brewermap(N_colors,'Dark2');
    marker = ["o", "+", "*", ".", "x", "square", "diamond", "^"];
    
    % LIEKF shift
    mapColor(9, :) = mapColor(7, :);
    mapColor(4:7, :) = mapColor(3:6, :);
    mapColor(3, :) = mapColor(9, :);
    mapColor(9, :) = [];

end

%% Loop through filters, parse data and compute rmse

% Define averages of single run ATE RMSE
avg_rmse_R_angles_T = zeros(1, length(filter_files));
avg_rmse_p_T = zeros(1, length(filter_files));
avg_rmse_v_T = zeros(1, length(filter_files));
avg_rmse_bw_T = zeros(1, length(filter_files));
avg_rmse_ba_T = zeros(1, length(filter_files));
avg_rmse_cal_p_T = zeros(1, length(filter_files));
avg_rmse_R_angles_A = zeros(1, length(filter_files));
avg_rmse_p_A = zeros(1, length(filter_files));
avg_rmse_v_A = zeros(1, length(filter_files));
avg_rmse_bw_A = zeros(1, length(filter_files));
avg_rmse_ba_A = zeros(1, length(filter_files));
avg_rmse_cal_p_A = zeros(1, length(filter_files));
avg_rmse_R_angles = zeros(1, length(filter_files));
avg_rmse_p = zeros(1, length(filter_files));
avg_rmse_v = zeros(1, length(filter_files));
avg_rmse_bw = zeros(1, length(filter_files));
avg_rmse_ba = zeros(1, length(filter_files));
avg_rmse_cal_p = zeros(1, length(filter_files));

% Define errors for average RMSE over multiple runs
errors_full = [];
energies_full = [];

% Define shorter length
shorter_length = inf;

% Define variables to excude trajectories from average
removed = 0;

for k = 1:N
    
    % Continue if trajectory is excluded
    if sum(k == exclude) > 0
        removed = removed + 1;
        continue;
    end
    
    path = data_path + k + "/";
    
    % Variables declaration
    estimates = [];
    errors = [];
    energies = [];
    gt_is_loaded = false;
    
    for file = filter_files
        
        load(path + file)
        
        if max(position_error) > 5.0 || max(attitude_error(end-round(length(attitude_error)/2):end)) > 1.0
            fprintf("%s FAILURE: %d\n", file, k)
        end
        
        if ~gt_is_loaded
            
            gt = struct;
            s = size(groundtruth,2);
            gt = setfield(gt,'R',zeros(3,3,length(groundtruth)));
            gt.p = zeros(3,length(groundtruth));
            gt.v = zeros(3,length(groundtruth));
            gt.bw = zeros(3,length(groundtruth));
            gt.ba = zeros(3,length(groundtruth));
            gt.cal = zeros(3,length(groundtruth));
            for i = 1:length(groundtruth)
                gt.R(:,:,i) = [groundtruth{i,1,1}; groundtruth{i,1,2}; groundtruth{i,1,3}];
                gt.v(:,i) =  [groundtruth{i,2,1}; groundtruth{i,2,2}; groundtruth{i,2,3}];
                gt.p(:,i) =  [groundtruth{i,3,1}; groundtruth{i,3,2}; groundtruth{i,3,3}];
                gt.bw(:,i) =  [groundtruth{i,4,1}; groundtruth{i,4,2}; groundtruth{i,4,3}];
                gt.ba(:,i) =  [groundtruth{i,5,1}; groundtruth{i,5,2}; groundtruth{i,5,3}];
                if (s == 6)
                    gt.cal(:,i) = [groundtruth{i,6,1}; groundtruth{i,6,2}; groundtruth{i,6,3}];
                end
%                 q = [groundtruth(i,4) groundtruth(i,1) groundtruth(i,2) groundtruth(i,3)];
%                 gt.R(:,:,i) = quat2rotm(q);
%                 gt.v(:,i) =  [groundtruth(i,5) groundtruth(i,6) groundtruth(i,7)];
%                 gt.p(:,i) =  [groundtruth(i,8) groundtruth(i,9) groundtruth(i,10)];
%                 gt.bw(:,i) =  [groundtruth(i,11) groundtruth(i,12) groundtruth(i,13)];
%                 gt.ba(:,i) =  [groundtruth(i,14) groundtruth(i,15) groundtruth(i,16)];
%                 if (s == 6)
%                     gt.cal(:,i) = [groundtruth(i,17) groundtruth(i,18) groundtruth(i,19)];
%                 end
            end
            gt.eul = rotm2eul(gt.R);
        
            gt_is_loaded = true;
        end
        
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
        est.eul = rotm2eul(est.R);
        
        %
        % Note that these errors are already norm squared!
        %
        
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
        energies = [energies; filter_energy];
        
    end
    
    en.e = energies;
    
    % Fill errors_full
    errors_full = [errors_full; errors];
    
    % Fill energies full
    energies_full = [energies_full; en];
    
    % Assign shorter length
    if length(t) < shorter_length
        shorter_length = length(t);
    end
    
    % Compute RMSE (single trajectory ATE RMSE)
    for i = 1:length(filter_files)
        [rmse_R_angles_T, ...
         rmse_p_T, ...
         rmse_v_T, ...
         rmse_bw_T, ...
         rmse_ba_T, ...
         rmse_cal_p_T, ...
         rmse_R_angles_A, ...
         rmse_p_A, ...
         rmse_v_A, ...
         rmse_bw_A, ...
         rmse_ba_A, ...
         rmse_cal_p_A, ...
         rmse_R_angles, ...
         rmse_p, ...
         rmse_v, ...
         rmse_bw, ...
         rmse_ba, ...   
         rmse_cal_p, ...
         transient_endidx] ...
         = rmse(t, errors(i).a, errors(i).p, errors(i).v, errors(i).bw, errors(i).ba, errors(i).cal, filter_labels(i), print_rmse);
     
         % Cumulated sum
         avg_rmse_R_angles_T(1,i) = avg_rmse_R_angles_T(1,i) + rmse_R_angles_T;
         avg_rmse_p_T(1,i) = avg_rmse_p_T(1,i) + rmse_p_T;
         avg_rmse_v_T(1,i) = avg_rmse_v_T(1,i) + rmse_v_T;
         avg_rmse_bw_T(1,i) = avg_rmse_bw_T(1,i) + rmse_bw_T;
         avg_rmse_ba_T(1,i) = avg_rmse_ba_T(1,i) + rmse_ba_T;
         avg_rmse_cal_p_T(1,i) = avg_rmse_cal_p_T(1,i) + rmse_cal_p_T;
         avg_rmse_R_angles_A(1,i) = avg_rmse_R_angles_A(1,i) + rmse_R_angles_A;
         avg_rmse_p_A(1,i) = avg_rmse_p_A(1,i) + rmse_p_A;
         avg_rmse_v_A(1,i) = avg_rmse_v_A(1,i) + rmse_v_A;
         avg_rmse_bw_A(1,i) = avg_rmse_bw_A(1,i) + rmse_bw_A;
         avg_rmse_ba_A(1,i) = avg_rmse_ba_A(1,i) + rmse_ba_A;
         avg_rmse_cal_p_A(1,i) = avg_rmse_cal_p_A(1,i) + rmse_cal_p_A;
         avg_rmse_R_angles(1,i) = avg_rmse_R_angles(1,i) + rmse_R_angles;
         avg_rmse_p(1,i) = avg_rmse_p(1,i) + rmse_p;
         avg_rmse_v(1,i) = avg_rmse_v(1,i) + rmse_v;
         avg_rmse_bw(1,i) = avg_rmse_bw(1,i) + rmse_bw;
         avg_rmse_ba(1,i) = avg_rmse_ba(1,i) + rmse_ba;
         avg_rmse_cal_p(1,i) = avg_rmse_cal_p(1,i) + rmse_cal_p;
    end
    
end

% Number of evaluated trajectories
M = N-removed;

% Check
for k = 1:M
    for f = 1:length(filter_files)
        l = round(length(errors_full(k,f).a) / 2);
        if max(errors_full(k,f).a(end-l:end)) > 0.25 || max(errors_full(k,f).ba > 0.1)
            fprintf("%s FAILURE: %d\n", filter_files(f), k)
        end
    end
end

% Histogram
nees_hist = zeros(length(filter_files), M);
for f = 1:length(filter_files)
    for k = 1:M
        nees_hist(f,k) = mean(energies_full(k).e(f,end-110:end-10));
    end
    figure;
    H = histogram(nees_hist(f,:), 25);
    title(filter_labels(f))
    hold on;
    dof = 15;
    if f == 5
        dof = 18;
    end
    dist = chi2pdf(0:0.01:100,dof);
    dist = dist * (max(H.Values) / max(dist));
    idx = find(dist == max(dist));
    val = length(dist)/idx;
    plot(0:val/10000:val,dist)
end

if length(t) < shorter_length
    shorter_length = length(t);
end

% Average of single trajectory ATE RMSE
avg_rmse_R_angles_T = avg_rmse_R_angles_T./M;
avg_rmse_p_T = avg_rmse_p_T./M;
avg_rmse_v_T = avg_rmse_v_T./M;
avg_rmse_bw_T = avg_rmse_bw_T./M;
avg_rmse_ba_T = avg_rmse_ba_T./M;
avg_rmse_cal_p_T = avg_rmse_cal_p_T./M;
avg_rmse_R_angles_A = avg_rmse_R_angles_A./M;
avg_rmse_p_A = avg_rmse_p_A./M;
avg_rmse_v_A = avg_rmse_v_A./M;
avg_rmse_bw_A = avg_rmse_bw_A./M;
avg_rmse_ba_A = avg_rmse_ba_A./M;
avg_rmse_cal_p_A = avg_rmse_cal_p_A./M;

avg_rmse_R_angles = avg_rmse_R_angles./M;
avg_rmse_p = avg_rmse_p./M;
avg_rmse_v = avg_rmse_v./M;
avg_rmse_bw = avg_rmse_bw./M;
avg_rmse_ba = avg_rmse_ba./M;
avg_rmse_cal_p = avg_rmse_cal_p./M;

% Compute average RMSE over multiple runs
L = shorter_length-10;
a_summed = zeros(length(filter_files),L);
p_summed = zeros(length(filter_files),L);
v_summed = zeros(length(filter_files),L);
bw_summed = zeros(length(filter_files),L);
ba_summed = zeros(length(filter_files),L);
cal_summed = zeros(length(filter_files),L);
en_summed = zeros(length(filter_files),L);
for k = 1:M
    for j = 1:length(filter_files)
        a_summed(j,:) = a_summed(j,:) + errors_full(k,j).a(1:L);
        p_summed(j,:) = p_summed(j,:) + errors_full(k,j).p(1:L);
        v_summed(j,:) = v_summed(j,:) + errors_full(k,j).v(1:L);
        bw_summed(j,:) = bw_summed(j,:) + errors_full(k,j).bw(1:L);
        ba_summed(j,:) = ba_summed(j,:) + errors_full(k,j).ba(1:L);
        cal_summed(j,:) = cal_summed(j,:) + errors_full(k,j).cal(1:L);
        en_summed(j,:) = en_summed(j,:) + energies_full(k).e(j,1:L);
    end
end
avg_a = a_summed./M;
avg_p = p_summed./M;
avg_v = v_summed./M;
avg_bw = bw_summed./M;
avg_ba = ba_summed./M;
avg_cal = cal_summed./M;
energies_summed = en_summed./M;


% std_a = zeros(length(filter_files),L);
% std_p = zeros(length(filter_files),L);
% std_v = zeros(length(filter_files),L);
% std_bw = zeros(length(filter_files),L);
% std_ba = zeros(length(filter_files),L);
% std_cal = zeros(length(filter_files),L);
% for k = 1:M
%     for j = 1:length(filter_files)
%         std_a(j, :) = std_a(j, :) + (errors_full(k,j).a(1:L) - avg_a(j,:)).^2;
%         std_p(j, :) = std_p(j, :) + (errors_full(k,j).p(1:L) - avg_p(j,:)).^2;
%         std_v(j, :) = std_v(j, :) + (errors_full(k,j).v(1:L) - avg_v(j,:)).^2;
%         std_bw(j, :) = std_bw(j, :) + (errors_full(k,j).bw(1:L) - avg_bw(j,:)).^2;
%         std_ba(j, :) = std_ba(j, :) + (errors_full(k,j).ba(1:L) - avg_ba(j,:)).^2;
%         std_cal(j, :) = std_cal(j, :) + (errors_full(k,j).cal(1:L) - avg_cal(j,:)).^2;
%     end
% end
% std_a = sqrt(std_a./(M-1));
% std_p = sqrt(std_p./(M-1));
% std_v = sqrt(std_v./(M-1));
% std_bw = sqrt(std_bw./(M-1));
% std_ba = sqrt(std_ba./(M-1));
% std_cal = sqrt(std_cal./(M-1));

std_a = zeros(length(filter_files),L);
std_p = zeros(length(filter_files),L);
std_v = zeros(length(filter_files),L);
std_bw = zeros(length(filter_files),L);
std_ba = zeros(length(filter_files),L);
std_cal = zeros(length(filter_files),L);
for j = 1:length(filter_files)
    errs_a = [];
    errs_p = [];
    errs_v = [];
    errs_bw = [];
    errs_ba = [];
    for k = 1:M
        errs_a = [errs_a; errors_full(k,j).a(1:L)];
        errs_p = [errs_p; errors_full(k,j).p(1:L)];
        errs_v = [errs_v; errors_full(k,j).v(1:L)];
        errs_bw = [errs_bw; errors_full(k,j).bw(1:L)];
        errs_ba = [errs_ba; errors_full(k,j).ba(1:L)];
    end
    std_a(j, :) = std(errs_a);
    std_p(j, :) = std(errs_p);
    std_v(j, :) = std(errs_v);
    std_bw(j, :) = std(errs_bw);
    std_ba(j, :) = std(errs_ba);
end

% std_a = std_a.*(180/pi);
% std_a = avg_a.*(180/pi);

%
% Compute percentage (transient/asympthotic) on time to get below 10% of
% max error
%

idx_a = zeros(length(filter_files),1);
idx_p = zeros(length(filter_files),1);
idx_v = zeros(length(filter_files),1);
idx_bw = zeros(length(filter_files),1);
idx_ba = zeros(length(filter_files),1);

perc = 0.1;

for j = 1:length(filter_files)
    idx_a(j) = find(avg_a(j,:) > perc * max(avg_a(j,:)), 1, 'last');
    idx_p(j) = find(avg_p(j,:) > perc * max(avg_p(j,:)), 1, 'last');
    idx_v(j) = find(avg_v(j,:) > perc * max(avg_v(j,:)), 1, 'last');
    idx_bw(j) = find(avg_bw(j,:) > perc * max(avg_bw(j,:)), 1, 'last');
    idx_ba(j) = find(avg_ba(j,:) > perc * max(avg_ba(j,:)), 1, 'last');
end


% for j = 1:length(filter_files)
%     figure; plot(t(1:L), avg_a(j,:)); grid on; hold on; plot(t(1:idx_a(j)), avg_a(j,1:idx_a(j)))
%     figure; plot(t(1:L), avg_p(j,:)); grid on; hold on; plot(t(1:idx_p(j)), avg_p(j,1:idx_p(j)))
%     figure; plot(t(1:L), avg_v(j,:)); grid on; hold on; plot(t(1:idx_v(j)), avg_v(j,1:idx_v(j)))
%     figure; plot(t(1:L), avg_bw(j,:)); grid on; hold on; plot(t(1:idx_bw(j)), avg_bw(j,1:idx_bw(j)))
%     figure; plot(t(1:L), avg_ba(j,:)); grid on; hold on; plot(t(1:idx_ba(j)), avg_ba(j,1:idx_ba(j)))
% end

transient_times = zeros(length(filter_files),5);

for j = 1:length(filter_files)
    transient_times(j, 1) = t(idx_a(j));
    transient_times(j, 2) = t(idx_p(j));
    transient_times(j, 3) = t(idx_v(j));
    transient_times(j, 4) = t(idx_bw(j));
    transient_times(j, 5) = t(idx_ba(j));
end

%% Print average RMSE

avg_rmse_transient = [(sum(avg_a(:, 1:transient_endidx),2) ./ length(1:transient_endidx))'; ...
                      (sum(avg_p(:, 1:transient_endidx),2) ./ length(1:transient_endidx))'; ...
                      (sum(avg_v(:, 1:transient_endidx),2) ./ length(1:transient_endidx))'; ...
                      (sum(avg_bw(:, 1:transient_endidx),2) ./ length(1:transient_endidx))'; ...
                      (sum(avg_ba(:, 1:transient_endidx),2) ./ length(1:transient_endidx))'];
avg_rmse_transient_perc = (avg_rmse_transient ./ repmat(avg_rmse_transient(:,1),1,length(filter_files))) .* 100;

% Latex table

table_avg_rmse_transient(:, 1:2:2*length(filter_files)) = avg_rmse_transient;
table_avg_rmse_transient(:, 2:2:2*length(filter_files)) = avg_rmse_transient_perc;

% disp(cell2mat(compose('& $%.4f\\;(%.0f\\%%)$ ',table_avg_rmse_transient)))


for i = 1:length(filter_labels) 
    fprintf("\n**************************************************\n");
    fprintf(" - Transient (t = [0, %.3f)) RMSE and Energy: \n", t(transient_endidx));
    fprintf("**************************************************\n");
    
    fprintf("\n**************************************************\n");
    fprintf(" - %s Orientation RMSE = [%.3f]\n", filter_labels(i), avg_rmse_R_angles_T(1,i));
    fprintf(" - %s Position RMSE = [%.3f]\n", filter_labels(i), avg_rmse_p_T(1,i));
    fprintf(" - %s Velocity RMSE = [%.3f]\n", filter_labels(i), avg_rmse_v_T(1,i));
    fprintf(" - %s Gyro Bias RMSE = [%.3f]\n", filter_labels(i), avg_rmse_bw_T(1,i));
    fprintf(" - %s Acc Bias RMSE = [%.3f]\n", filter_labels(i), avg_rmse_ba_T(1,i));
    fprintf(" - %s Calibration position RMSE = [%.3f]\n", filter_labels(i), avg_rmse_cal_p_T(1,i));
    fprintf(" - %s Average Energy = [%.3f]\n", filter_labels(i), mean(energies_summed(i,1:(round(L/2) + 1))));
    fprintf("**************************************************\n");
    
    fprintf("\n**************************************************\n");
    fprintf(" - Asymptotic (t = [%.3f, end] RMSE and Energy: \n", t(transient_endidx));
    fprintf("**************************************************\n");
    
    fprintf("\n**************************************************\n");
    fprintf(" - %s Orientation RMSE = [%.3f]\n", filter_labels(i), avg_rmse_R_angles_A(1,i));
    fprintf(" - %s Position RMSE = [%.3f]\n", filter_labels(i), avg_rmse_p_A(1,i));
    fprintf(" - %s Velocity RMSE = [%.3f]\n", filter_labels(i), avg_rmse_v_A(1,i));
    fprintf(" - %s Gyro Bias RMSE = [%.3f]\n", filter_labels(i), avg_rmse_bw_A(1,i));
    fprintf(" - %s Acc Bias RMSE = [%.3f]\n", filter_labels(i), avg_rmse_ba_A(1,i));
    fprintf(" - %s Calibration position RMSE = [%.3f]\n", filter_labels(i), avg_rmse_cal_p_A(1,i));
    fprintf(" - %s Average Energy = [%.3f]\n", filter_labels(i), mean(energies_summed(i,(round(L/2) + 1):L)));
    fprintf("**************************************************\n");
    
%     fprintf("\n**************************************************\n");
%     fprintf(" - Full RMSE: \n");
%     fprintf("**************************************************\n");
%     
%     fprintf("\n**************************************************\n");
%     fprintf(" - %s Orientation RMSE = [%.3f]\n", filter_labels(i), avg_rmse_R_angles(1,i));
%     fprintf(" - %s Position RMSE = [%.3f]\n", filter_labels(i), avg_rmse_p(1,i));
%     fprintf(" - %s Velocity RMSE = [%.3f]\n", filter_labels(i), avg_rmse_v(1,i));
%     fprintf(" - %s Gyro Bias RMSE = [%.3f]\n", filter_labels(i), avg_rmse_bw(1,i));
%     fprintf(" - %s Acc Bias RMSE = [%.3f]\n", filter_labels(i), avg_rmse_ba(1,i));
%     fprintf(" - %s Calibration position RMSE = [%.3f]\n", filter_labels(i), avg_rmse_cal_p(1,i));
%     fprintf("**************************************************\n");

end

%% Plots

if plots
    
    % -----------------------------------------------------------------
    % Position and error plots
    % -----------------------------------------------------------------

    fh = findobj('Type','Figure','Name','Position Error');
    set(groot,'CurrentFigure',fh);
    grid on;
    hold on;
    for i = 1:length(filter_labels)
        shade_std(t(1:L), avg_p(i,:), std_p(i,:), 0.15, mapColor(i,:))
        plot(t(1:L), avg_p(i,:), 'LineWidth', 1, 'Color',mapColor(i,:))%,'Marker',marker(i),'MarkerIndices',1:300:L)
    end
   
    % -----------------------------------------------------------------
    % Orientation and error plots
    % -----------------------------------------------------------------

    fh = findobj('Type','Figure','Name','Orientation Error');
    set(groot,'CurrentFigure',fh);
    grid on;
    hold on;
    for i = 1:length(filter_labels)
        shade_std(t(1:L), avg_a(i,:)*180/pi, std_a(i,:)*180/pi, 0.15, mapColor(i,:))
        plot(t(1:L), avg_a(i,:)*180/pi, 'LineWidth', 1, 'Color',mapColor(i,:))%,'Marker',marker(i),'MarkerIndices',1:300:L)
    end

    % -----------------------------------------------------------------
    % Velocity and error plots
    % ----------------------------------------------------------------

    fh = findobj('Type','Figure','Name','Velocity Error');
    set(groot,'CurrentFigure',fh);
    grid on;
    hold on;
    for i = 1:length(filter_labels)
        shade_std(t(1:L), avg_v(i,:), std_v(i,:), 0.15, mapColor(i,:))
        plot(t(1:L), avg_v(i,:), 'LineWidth', 1, 'Color',mapColor(i,:))%,'Marker',marker(i),'MarkerIndices',1:300:L)
    end
    
    % -----------------------------------------------------------------
    % Calibration and error plots
    % -----------------------------------------------------------------
    
    fh = findobj('Type','Figure','Name','Position_Calibration Error');
    set(groot,'CurrentFigure',fh);
    grid on;
    hold on;
    for i = 1:length(filter_labels)
        shade_std(t(1:L), avg_cal(i,:), std_cal(i,:), 0.15, mapColor(i,:))
        plot(t(1:L), avg_cal(i,:), 'LineWidth', 1, 'Color',mapColor(i,:))%,'Marker',marker(i),'MarkerIndices',1:300:L)
    end
    
    % -----------------------------------------------------------------
    % biases and error plots
    % -----------------------------------------------------------------

    fh = findobj('Type','Figure','Name','Bias_w Error');
    set(groot,'CurrentFigure',fh);
    grid on;
    hold on;
    for i = 1:length(filter_labels)
        shade_std(t(1:L), avg_bw(i,:), std_bw(i,:), 0.15, mapColor(i,:))
        plot(t(1:L), avg_bw(i,:), 'LineWidth', 1, 'Color',mapColor(i,:))%,'Marker',marker(i),'MarkerIndices',1:300:L)
    end

    fh = findobj('Type','Figure','Name','Bias_a Error');
    set(groot,'CurrentFigure',fh);
    grid on;
    hold on;
    for i = 1:length(filter_labels)
        shade_std(t(1:L), avg_ba(i,:), std_ba(i,:), 0.15, mapColor(i,:))
        plot(t(1:L), avg_ba(i,:), 'LineWidth', 1, 'Color',mapColor(i,:))%,'Marker',marker(i),'MarkerIndices',1:300:L)
    end
    
    
    % -----------------------------------------------------------------
    %Energy plot
    % -----------------------------------------------------------------
    
%     % 95% WH approximation (https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.142.6049&rep=rep1&type=pdfplo)
% %     dbcl = [(1-(2/(9*18*N))-1.96*sqrt(2/(9*18*N)))^3, ...
% %         (1-(2/(9*18*N))+1.96*sqrt(2/(9*18*N)))^3];
% 
    fh = findobj('Type','Figure','Name','Energy');
    set(groot,'CurrentFigure',fh);
    set(gca, 'YScale', 'log')
    grid on;
    hold on;
    for i = 1:length(filter_labels)
%         plot(t(1:L), 10*log10(energies_summed(i,:)), 'LineWidth', 2, 'Color',mapColor(i,:))%,'Marker',marker(i),'MarkerIndices',1:300:L)
        plot(t(1:L), energies_summed(i,:), 'LineWidth', 1, 'Color',mapColor(i,:))%,'Marker',marker(i),'MarkerIndices',1:300:L)
    end
% %     yline(dbcl(1),'--')
% %     yline(dbcl(2),'--')

end

legend_ = [];
p_idxs = [];
cnt = 0;
for label = filter_labels
%     legend_ = [legend_ label+" $\sigma$" label];
    legend_ = [legend_, label];
    p_idxs = [p_idxs 2*(cnt+1)];
    cnt = cnt+1;
end

% ---------------------------------------------------------------------
% Finalize Position plots
% ---------------------------------------------------------------------

fh = findobj('Type','Figure','Name','Position Error');
set(groot,'CurrentFigure',fh);
p = flip(get(gca, 'Children'));
set(gca,'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
legend(p(p_idxs), legend_,'Interpreter','latex','FontSize', LegendFontSize);
figure_title = "\textbf{UAV Position RMSE}";
figure_x_label = "t [s]";
figure_y_label = "RMSE [m]";
title(figure_title,'Interpreter','latex','fontsize',TitleFontSize);
xlabel(figure_x_label,'Interpreter','latex');
ylabel(figure_y_label,'Interpreter','latex');

% ---------------------------------------------------------------------
% Finalize Orientation plots
% ---------------------------------------------------------------------

fh = findobj('Type','Figure','Name','Orientation Error');
set(groot,'CurrentFigure',fh);
p = flip(get(gca, 'Children'));
set(gca,'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
legend(p(p_idxs), legend_,'Interpreter','latex','FontSize', LegendFontSize);
figure_title = "\textbf{UAV Orientation RMSE}";

figure_x_label = "t [s]";
figure_y_label = "RMSE [deg]";
title(figure_title,'Interpreter','latex','fontsize',TitleFontSize);
xlabel(figure_x_label,'Interpreter','latex');
ylabel(figure_y_label,'Interpreter','latex');

% ---------------------------------------------------------------------
% Finalize Velocity plots
% ---------------------------------------------------------------------

fh = findobj('Type','Figure','Name','Velocity Error');
set(groot,'CurrentFigure',fh);
p = flip(get(gca, 'Children'));
set(gca,'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
legend(p(p_idxs), legend_,'Interpreter','latex','FontSize', LegendFontSize);
figure_title = "\textbf{UAV Velocity RMSE}";
figure_x_label = "t [s]";
figure_y_label = "RMSE [m/s]";
title(figure_title,'Interpreter','latex','fontsize',TitleFontSize);
xlabel(figure_x_label,'Interpreter','latex');
ylabel(figure_y_label,'Interpreter','latex');
% ---------------------------------------------------------------------
% Finalize Position calibration plots
% ---------------------------------------------------------------------

fh = findobj('Type','Figure','Name','Position_Calibration Error');
set(groot,'CurrentFigure',fh);
p = flip(get(gca, 'Children'));
set(gca,'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
legend(p(p_idxs), legend_,'Interpreter','latex','FontSize', LegendFontSize);
figure_title = "\textbf{GNSS Calibration (lever arm) RMSE}";
figure_x_label = "t [s]";
figure_y_label = "RMSE [m]";
title(figure_title,'Interpreter','latex','fontsize',TitleFontSize);
xlabel(figure_x_label,'Interpreter','latex');
ylabel(figure_y_label,'Interpreter','latex');

% ---------------------------------------------------------------------
% Finalize bias bw
% ---------------------------------------------------------------------

fh = findobj('Type','Figure','Name','Bias_w Error');
set(groot,'CurrentFigure',fh);
p = flip(get(gca, 'Children'));
set(gca,'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
legend(p(p_idxs), legend_,'Interpreter','latex','FontSize', LegendFontSize);
figure_title = "\textbf{Gyro Bias RMSE}";
figure_x_label = "t [s]";
figure_y_label = "RMSE [rad/s]";
title(figure_title,'Interpreter','latex','fontsize',TitleFontSize);
xlabel(figure_x_label,'Interpreter','latex');
ylabel(figure_y_label,'Interpreter','latex');

% ---------------------------------------------------------------------
% Finalize bias ba
% ---------------------------------------------------------------------

fh = findobj('Type','Figure','Name','Bias_a Error');
set(groot,'CurrentFigure',fh);
p = flip(get(gca, 'Children'));
set(gca,'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
legend(p(p_idxs), legend_,'Interpreter','latex','FontSize', LegendFontSize);
figure_title = "\textbf{Acc Bias RMSE}";
figure_x_label = "t [s]";
figure_y_label = "RMSE [m/$s^2$]";
title(figure_title,'Interpreter','latex','fontsize',TitleFontSize);
xlabel(figure_x_label,'Interpreter','latex');
ylabel(figure_y_label,'Interpreter','latex');

% ---------------------------------------------------------------------
% Energy
% ---------------------------------------------------------------------

fh = findobj('Type','Figure','Name','Energy');
set(groot,'CurrentFigure',fh);
set(gca,'TickLabelInterpreter','latex','FontSize',AxisTickFontSize,'LineWidth', 1);
legend(legend_,'Interpreter','latex','FontSize', LegendFontSize);
figure_title = "\textbf{Average Energy (Log plot)}";
figure_x_label = "t [s]";
figure_y_label = "";
title(figure_title,'Interpreter','latex','fontsize',TitleFontSize);
xlabel(figure_x_label,'Interpreter','latex');
ylabel(figure_y_label,'Interpreter','latex');

% ---------------------------------------------------------------------
% Save plots
% ---------------------------------------------------------------------

% Save
if save_plots
    
    if ~exist(save_dir, 'dir')
        fprintf("Info:" + save_dir + " does not exist. Creating folder...\n");
        mkdir(save_dir)
    end
    
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
        
    fh = findobj('Type','Figure','Name','Position_Calibration Error');
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