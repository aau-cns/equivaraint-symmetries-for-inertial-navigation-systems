function [rmse_R_angles_T, rmse_p_T, rmse_v_T, rmse_bw_T, rmse_ba_T, rmse_cal_p_T, ...
          rmse_R_angles_A, rmse_p_A, rmse_v_A, rmse_bw_A, rmse_ba_A, rmse_cal_p_A, ...
          rmse_R_angles,   rmse_p,   rmse_v,   rmse_bw,   rmse_ba,   rmse_cal_p, ...
          idx] ...
          = rmse(t, e_angles, e_p, e_v, e_bw, e_ba, e_cal_t, filter, print_rmse)
      
      
      %
      % Errors are given squared already
      %
      
      rmse_R_angles_T = 0;
      rmse_p_T = 0;
      rmse_v_T = 0;
      rmse_bw_T = 0;
      rmse_ba_T = 0;
      rmse_cal_p_T = 0;
      
      
      rmse_R_angles_A = 0;
      rmse_p_A = 0;
      rmse_v_A = 0;
      rmse_bw_A = 0;
      rmse_ba_A = 0;
      rmse_cal_p_A = 0;
      
      
      % Transient
      
      starting_point_rmse = 1;
%       ending_point_rmse = round(length(t)/2);
      ending_point_rmse = find(t < 30, 1, 'last');
      idx = ending_point_rmse;
      n_points_in_rmse = ending_point_rmse-starting_point_rmse+1;
      
      if print_rmse
          fprintf("\n**************************************************\n");
          fprintf(" - Starting point (step) for computing RMSE: #%d - %d \n", starting_point_rmse, ending_point_rmse);
          fprintf("**************************************************\n");
      end
      
      for it = starting_point_rmse:ending_point_rmse
          rmse_R_angles_T = rmse_R_angles_T + e_angles(it);
          rmse_p_T = rmse_p_T + e_p(it);
          rmse_v_T = rmse_v_T + e_v(it);
          rmse_bw_T = rmse_bw_T + e_bw(it);
          rmse_ba_T = rmse_ba_T + e_ba(it);
          rmse_cal_p_T = rmse_cal_p_T + e_cal_t(it);
      end
      rmse_R_angles_T = sqrt(rmse_R_angles_T/n_points_in_rmse);
      rmse_p_T = sqrt(rmse_p_T/n_points_in_rmse);
      rmse_v_T = sqrt(rmse_v_T/n_points_in_rmse);
      rmse_bw_T = sqrt(rmse_bw_T/n_points_in_rmse);
      rmse_ba_T = sqrt(rmse_ba_T/n_points_in_rmse);
      rmse_cal_p_T = sqrt(rmse_cal_p_T/n_points_in_rmse);
      
      if print_rmse
          fprintf("\n**************************************************\n");
          fprintf(" - %s Orientation RMSE = [%.3f]\n", filter, rmse_R_angles_T);
          fprintf(" - %s Position RMSE = [%.3f]\n", filter, rmse_p_T);
          fprintf(" - %s Velocity RMSE = [%.3f]\n", filter, rmse_v_T);
          fprintf(" - %s Gyro Bias RMSE = [%.3f]\n", filter, rmse_bw_T);
          fprintf(" - %s Acc Bias RMSE = [%.3f]\n", filter, rmse_ba_T);
          fprintf(" - %s Calibration position RMSE = [%.3f]\n", filter, rmse_cal_p_T);
          fprintf("**************************************************\n");
      end
      
      % Asymptotic
      
%       starting_point_rmse = round(length(t)/2) + 1;
      starting_point_rmse = find(t >= 30, 1, 'first');
      ending_point_rmse = round(length(t));
      n_points_in_rmse = ending_point_rmse-starting_point_rmse+1;
      
      if print_rmse
          fprintf("\n**************************************************\n");
          fprintf(" - Starting point (step) for computing RMSE: #%d - %d \n", starting_point_rmse, ending_point_rmse);
          fprintf("**************************************************\n");
      end
      
      for it = starting_point_rmse:ending_point_rmse
          rmse_R_angles_A = rmse_R_angles_A + e_angles(it);
          rmse_p_A = rmse_p_A + e_p(it);
          rmse_v_A = rmse_v_A + e_v(it);
          rmse_bw_A = rmse_bw_A + e_bw(it);
          rmse_ba_A = rmse_ba_A + e_ba(it);
          rmse_cal_p_A = rmse_cal_p_A + e_cal_t(it);
      end
      rmse_R_angles_A = sqrt(rmse_R_angles_A/n_points_in_rmse);
      rmse_p_A = sqrt(rmse_p_A/n_points_in_rmse);
      rmse_v_A = sqrt(rmse_v_A/n_points_in_rmse);
      rmse_bw_A = sqrt(rmse_bw_A/n_points_in_rmse);
      rmse_ba_A = sqrt(rmse_ba_A/n_points_in_rmse);
      rmse_cal_p_A = sqrt(rmse_cal_p_A/n_points_in_rmse);
      
      if print_rmse
          fprintf("\n**************************************************\n");
          fprintf(" - %s Orientation RMSE = [%.3f]\n", filter, rmse_R_angles_A);
          fprintf(" - %s Position RMSE = [%.3f]\n", filter, rmse_p_A);
          fprintf(" - %s Velocity RMSE = [%.3f]\n", filter, rmse_v_A);
          fprintf(" - %s Gyro Bias RMSE = [%.3f]\n", filter, rmse_bw_A);
          fprintf(" - %s Acc Bias RMSE = [%.3f]\n", filter, rmse_ba_A);
          fprintf(" - %s Calibration position RMSE = [%.3f]\n", filter, rmse_cal_p_A);
          fprintf("**************************************************\n");
      end
      
      starting_point_rmse = 1;
      ending_point_rmse = round(length(t));
      n_points_in_rmse = ending_point_rmse-starting_point_rmse+1;
      
      rmse_R_angles = 0;
      rmse_p = 0;
      rmse_v = 0;
      rmse_bw = 0;
      rmse_ba = 0;
      rmse_cal_p = 0;
      
      if print_rmse
          fprintf("\n**************************************************\n");
          fprintf(" - Starting point (step) for computing RMSE: #%d - %d \n", starting_point_rmse, ending_point_rmse);
          fprintf("**************************************************\n");
      end
      
      for it = starting_point_rmse:ending_point_rmse
          rmse_R_angles = rmse_R_angles + e_angles(it);
          rmse_p = rmse_p + e_p(it);
          rmse_v = rmse_v + e_v(it);
          rmse_bw = rmse_bw + e_bw(it);
          rmse_ba = rmse_ba + e_ba(it);
          rmse_cal_p = rmse_cal_p + e_cal_t(it);
      end
      rmse_R_angles = sqrt(rmse_R_angles/n_points_in_rmse);
      rmse_p = sqrt(rmse_p/n_points_in_rmse);
      rmse_v = sqrt(rmse_v/n_points_in_rmse);
      rmse_bw = sqrt(rmse_bw/n_points_in_rmse);
      rmse_ba = sqrt(rmse_ba/n_points_in_rmse);
      rmse_cal_p = sqrt(rmse_cal_p/n_points_in_rmse);
      
      if print_rmse
          fprintf("\n**************************************************\n");
          fprintf(" - %s Orientation RMSE = [%.3f]\n", filter, rmse_R_angles);
          fprintf(" - %s Position RMSE = [%.3f]\n", filter, rmse_p);
          fprintf(" - %s Velocity RMSE = [%.3f]\n", filter, rmse_v);
          fprintf(" - %s Gyro Bias RMSE = [%.3f]\n", filter, rmse_bw);
          fprintf(" - %s Acc Bias RMSE = [%.3f]\n", filter, rmse_ba);
          fprintf(" - %s Calibration position RMSE = [%.3f]\n", filter, rmse_cal_p);
          fprintf("**************************************************\n");
      end
      
      
      
      
end

