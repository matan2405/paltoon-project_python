% Vehicle parameters (unchanged) 
params.mass = 1305;        
params.Iz = 2500;         
params.lf = 1.2525;       
params.lr = 1.2525;       
params.Caf = 15000;       
params.Car = 18000;       
params.height = 1.353;    
params.wheelbase = 2.505; 
params.dragCoefficient = 0.30;    
params.frontalArea = 2.09;        
params.roadSlope = 0;             
params.maxVelocity = 120;


% Read input data from CSV
input_data = readtable('Inputs.csv');
t_data = input_data.Time;
lambda_data = input_data.Lambda; 
Fx_data = input_data.Fx;

% Read output data from simulation CSV
output_data = readtable('Outputs.csv');
t_sim = output_data.Time;
y_sim = output_data.Y;
yDot_sim = output_data.YDot;
psi_sim = output_data.Psi;
psiDot_sim = output_data.PsiDot;
vx_sim = output_data.Vx;
ax_sim = output_data.Ax;
pos_x=output_data.PositionX;
pos_z=output_data.PositionZ;
% Create interpolation functions for inputs
params.Fx = @(t) interp1(t_data, Fx_data, t, 'spline');
params.lambda = @(t) interp1(t_data, lambda_data, t, 'spline');

% Compare interpolated inputs with original inputs
Fx_interpolated = params.Fx(t_data);
lambda_interpolated = params.lambda(t_data);

% Initial conditions
vx0 = 0;  % Initial longitudinal velocity
tspan = [min(t_data) max(t_data)];
%options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
disp(['Simulation time range: ' num2str(min(t_data)) ' to ' num2str(max(t_data)) ' seconds']);
disp(['Number of input data points: ' num2str(height(input_data))]);
disp(['Time range: ' num2str(min(t_data)) ' to ' num2str(max(t_data))]);
disp(['Lambda range: ' num2str(min(lambda_data)) ' to ' num2str(max(lambda_data))]);
disp(['Fx range: ' num2str(min(Fx_data)) ' to ' num2str(max(Fx_data))]);

% Solve longitudinal dynamics with direct velocity limiting
%options = odeset('RelTol', 1e-8, 'AbsTol', 1e-10, 'MaxStep', 0.01);
options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8, 'MaxStep', 0.05);
[t_vx, vx] = ode45(@(t,vx) longitudinalDynamics(t, vx, params), tspan, vx0, options);

% Apply direct velocity limit to ensure results remain within bounds
vx(vx > params.maxVelocity/3.6) = params.maxVelocity/3.6; % Limit to maxVelocity km/h
vx(vx < 0) = 0; % Prevent negative velocities

% Store vx solution for lateral dynamics
vx_solution.x = t_vx;
vx_solution.y = vx;

% Then solve lateral dynamics using the vx solution
x0_lat = [0; 0; 0; 0];  % Initial lateral states [y, ydot, psi, psidot]
[t, x_lat] = ode45(@(t,x) lateralDynamics(t, x, vx_solution, params), tspan, x0_lat, options);
% [t, x_lat] = ode45(@(t,x) lateralDynamics(t, x, vx_solution, params), tspan, x0_lat);

% Extract results
y = x_lat(:,1);
ydot = x_lat(:,2);
psi = x_lat(:,3);
psiDot = x_lat(:,4);

% Plot comparison
figure;

% Fx comparison
subplot(2,1,1);
plot(t_data, Fx_data, 'b*-', 'DisplayName', 'Original Fx','LineWidth',0.05);
hold on;
plot(t_data, Fx_interpolated, 'r--', 'DisplayName', 'Interpolated Fx','LineWidth',0.05);
title('Comparison of Longitudinal Force (Fx)');
xlabel('Time (s)');
ylabel('Force (N)');
legend();
grid on; % Add grid

% Lambda comparison
subplot(2,1,2);
plot(t_data, lambda_data, 'b*-', 'DisplayName', 'Original Lambda');
hold on;
plot(t_data, lambda_interpolated, 'r--', 'DisplayName', 'Interpolated Lambda');
title('Comparison of Steering Angle (Lambda)');
xlabel('Time (s)');
ylabel('Angle (rad)');
legend();
grid on; % Add grid

% Plot comparison of simulation results
figure;

subplot(3,2,1);
plot(t, y, 'b-', t_sim, y_sim, 'r--');
title('Lateral Position (y)');
xlabel('Time (s)');
ylabel('Position (m)');
legend('MATLAB', 'Simulation');
grid on; % Add grid

subplot(3,2,2);
plot(t, ydot, 'b-', t_sim, yDot_sim, 'r--');
title('Lateral Velocity (yDot)');
xlabel('Time (s)');
ylabel('Velocity (m/s)');
legend('MATLAB', 'Simulation');
grid on; % Add grid

subplot(3,2,3);
plot(t, psi*180/pi, 'b-', t_sim, psi_sim*180/pi, 'r--');
title('Yaw Angle (Psi)');
xlabel('Time (s)');
ylabel('Angle (deg)');
legend('MATLAB', 'Simulation');
grid on; % Add grid

subplot(3,2,4);
plot(t, psiDot*180/pi, 'b-', t_sim, psiDot_sim*180/pi, 'r--');
title('Yaw Rate (PsiDot)');
xlabel('Time (s)');
ylabel('Rate (deg/s)');
legend('MATLAB', 'Simulation');
grid on; % Add grid

subplot(3,2,5);
plot(t_vx, vx*3.6, 'b-', t_sim, vx_sim*3.6, 'r--');
title('Longitudinal Velocity (Vx)');
xlabel('Time (s)');
ylabel('Speed (km/h)');
legend('MATLAB', 'Simulation');
grid on; % Add grid

ax = zeros(size(t_vx));
for i = 1:length(t_vx)
    ax(i) = longitudinalDynamics(t_vx(i), vx(i), params);
end

subplot(3,2,6);
plot(t_vx, ax, 'b-', t_sim, ax_sim, 'r--');
title('Longitudinal Acceleration (Ax)');
xlabel('Time (s)');
ylabel('Acceleration (m/s²)');
legend('MATLAB', 'Simulation');
grid on; % Add grid

% Plot X-Z trajectory (X vs. Y)
figure;
plot(pos_x, pos_z, 'b-', 'LineWidth', 1.5);
xlabel('X Position (m)');
ylabel('Z Position (m)');
title('Vehicle Trajectory in X-Z Plane');
grid on;
legend('Trajectory');
% Set X-axis limit
%xlim([8 -8]);

% Longitudinal dynamics function with integrated velocity limiting
function dvx_dt = longitudinalDynamics(t, vx, params)
    % Get input force
    Fx_tot = params.Fx(t);
    
    % Calculate resistive forces
    airDensity = 1.225;
    Faero = 0.5 * airDensity * params.dragCoefficient * params.frontalArea * vx^2;
    rollingResistance = params.mass * 9.81 * 0.015 *10* (vx > 0.001);
    slopeForce = params.mass * 9.81 * sin(params.roadSlope * pi/180);
    
    % Calculate raw acceleration
    dvx_dt_raw = (Fx_tot - Faero - rollingResistance - slopeForce) / params.mass;
    
    % Apply velocity limits directly
    if vx >= params.maxVelocity/3.6 && dvx_dt_raw > 0
        dvx_dt = 0; % No further acceleration beyond maximum velocity
        vx = params.maxVelocity/3.6;
    elseif vx < 0 && dvx_dt_raw < 0
        dvx_dt = max(dvx_dt_raw, 0); % No further deceleration below zero
        vx = 0;
    else
        dvx_dt = dvx_dt_raw; % Normal dynamics within limits
    end
end

% Lateral dynamics function
function dx_lat_dt = lateralDynamics(t, x_lat, vx_t, params)
    y = x_lat(1);
    ydot = x_lat(2);
    psi = x_lat(3);
    psidot = x_lat(4);
    
    vx = interp1(vx_t.x, vx_t.y, t, 'linear');
    lambda = params.lambda(t);
    
    if abs(vx) > 0.01
        A = [0, 1, 0, 0;
             0, -(2*params.Caf + 2*params.Car)/(params.mass*vx), 0, ...
                -vx - (2*params.Caf*params.lf - 2*params.Car*params.lr)/(params.mass*vx);
             0, 0, 0, 1;
             0, -(2*params.lf*params.Caf - 2*params.lr*params.Car)/(params.Iz*vx), 0, ...
                -(2*params.lf^2*params.Caf + 2*params.lr^2*params.Car)/(params.Iz*vx)];
        B = [0;
             2*params.Caf/params.mass;
             0;
             2*params.lf*params.Caf/params.Iz];
        
        dx_lat_dt = A * x_lat + B * lambda;
    else
        dx_lat_dt = zeros(4,1);
    end
end

%mcc -m BicycleModelStateSpace_ode45.m
