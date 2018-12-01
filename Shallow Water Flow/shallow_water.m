%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Applied High Perfomance Computing
%
%
%
% Problem:  Shallow water equation
%
% Method:   Sixth order semi finite discretization,
%           Fouth-order Runga Kutta Method
%
% Author: Anubhav Singh
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function assignment1()

    clear all; close all; clc;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%% Fourth-order runge-kutta stability analysis %%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    [X, Y]          = meshgrid(-4:0.1:4, -4:0.1:4);
    lambda_dt       = X + i*Y;
    sigma           = 1 + lambda_dt + (lambda_dt.^2)/2 + (lambda_dt.^3)/6 + (lambda_dt.^4)/24;
    Z               = abs(sigma);

    % Plot the stability diagram
    figure('WindowStyle', 'docked');

    contourf(X, Y, Z, [1 1],'-k');
    hold on;
    axis('equal', [-4 4 -4 4]);
    grid on;
    xlabel('\lambda_{Re}\Delta t');
    ylabel('\lambda_{Im}\Delta t');
    title("Fourth-order runge-kutta stability analysis");

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%% Fourth-order runge-kutta error analysis %%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    lambda_dt          = i*Y;
    sigma           = 1 + lambda_dt + (lambda_dt.^2)/2 + (lambda_dt.^3)/6 + (lambda_dt.^4)/24;
    Z               = abs(sigma);
    phase_sigma     = angle(sigma);
    phase_err       = phase_sigma - wrapToPi(Y);
    amp_err         = Z;

    % plot the phase-error
    figure('WindowStyle', 'docked');

    plot(Y, phase_err, 'LineWidth', 2, 'MarkerSize', 20);
    hold on;
    axis('equal', [-4 4 -7 7]);
    grid on;
    xlabel('\lambda_{Im}\Delta t');
    ylabel('\Delta \theta');
    title("Fourth-order runge-kutta phase error analysis");

    % plot the amplitude-error
    figure('WindowStyle', 'docked');

    plot(Y, amp_err, 'LineWidth', 2, 'MarkerSize', 20);
    hold on;
    axis('equal', [-4 4 -7 7]);
    grid on;
    xlabel('\lambda_{Im}\Delta t');
    ylabel('\Delta Amp');
    title("Fourth-order runge-kutta amplitude error analysis");

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%% Shallow water equation solution %%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Simulation parameters
    x_min           =   0.00;
    x_max           = 100.00;
    y_min           =   0.00;
    y_max           = 100.00;
    t_min           =   0.00;
    t_max           = 100.00;

    Delta_x         =  1;
    Delta_y         =  1;
    Delta_t         =  0.01;

    N_x=length(x_min:Delta_x:x_max); N_y=length(y_min:Delta_y:y_max);  %number of grid points

    N_t=length(t_min:Delta_t:t_max);  %number of time steps

    % Create meshgrid which hads 6 more gridpoints on both x and y axis.
    % The extra gridpoints accomodate for periodic boundary and calculation
    % of sixth order finite difference at gridpoints near the boundaries.
    [p,q]=meshgrid(cat(2, [x_min-3*Delta_y x_min-2*Delta_y x_min-Delta_y],...
            x_min:Delta_y:x_max,[x_min+Delta_y x_min+2*Delta_y x_min+3*Delta_y]),...
            cat(2, [y_min-3*Delta_y y_min-2*Delta_y y_min-Delta_y],...
            y_min:Delta_y:y_max,[y_min+Delta_y y_min+2*Delta_y y_min+3*Delta_y]));

    g=9.81;      % gravitational constant

    % initial condition
    x=4:N_x+3; y=4:N_y+3;
    time=0; %time in seconds
    h(1:N_x+6, 1:N_y+6)=1 + 0.5*exp(-1/25*((p-30).^2 + (q-30).^2));
    v_x(1:N_x+6, 1:N_y+6)=0;
    v_y(1:N_x+6, 1:N_y+6)=0;

    % plot the result
    swe_plot = figure('WindowStyle', 'docked');
    Solution = surf(x,y, h(x,y),'linestyle','none','EdgeColor','none'); view(-60,90);
    hold on;
    axis([4 N_x+3 4 N_y+3 -10 10]);
    xlabel('x'); ylabel('y'); zlabel('z');
    hold off;
    grid off;
    t = title(['t = ' num2str(time) ' s']);

    % time marching loop
    for n=1:N_t
        % Runge-Kutta fourth order
        k1_vx       = f_vx(v_x,v_y,h,N_x,N_y,Delta_x,Delta_y,g);
        k1_vy       = f_vy(v_x,v_y,h,N_x,N_y,Delta_x,Delta_y,g);
        k1_h        = f_h(v_x,v_y,h,N_x,N_y,Delta_x,Delta_y);
        v_x_t=v_x; v_y_t=v_y; h_t=h;
        v_x_t(x,y)  =v_x(x,y)+0.5*Delta_t*k1_vx;
        v_y_t(x,y)  =v_y(x,y)+0.5*Delta_t*k1_vy;
        h_t(x,y)    =h(x,y)+0.5*Delta_t*k1_h;
        v_x_t = apply_periodic_boundary(v_x_t);
        v_y_t = apply_periodic_boundary(v_y_t);
        h_t  = apply_periodic_boundary(h_t);

        k2_vx       =f_vx(v_x_t,v_y_t,h_t,N_x,N_y,Delta_x,Delta_y,g);
        k2_vy       =f_vy(v_x_t,v_y_t,h_t,N_x,N_y,Delta_x,Delta_y,g);
        k2_h        =f_h(v_x_t,v_y_t,h_t,N_x,N_y,Delta_x,Delta_y);
        v_x_t(x,y)  =v_x(x,y)+0.5*Delta_t*k2_vx;
        v_y_t(x,y)  =v_y(x,y)+0.5*Delta_t*k2_vy;
        h_t(x,y)    =h(x,y)+0.5*Delta_t*k2_h;
        v_x_t = apply_periodic_boundary(v_x_t);
        v_y_t = apply_periodic_boundary(v_y_t);
        h_t  = apply_periodic_boundary(h_t);

        k3_vx       =f_vx(v_x_t,v_y_t,h_t,N_x,N_y,Delta_x,Delta_y,g);
        k3_vy       =f_vy(v_x_t,v_y_t,h_t,N_x,N_y,Delta_x,Delta_y,g);
        k3_h        =f_h(v_x_t,v_y_t,h_t,N_x,N_y,Delta_x,Delta_y);
        v_x_t(x,y)  =v_x(x,y)+Delta_t*k3_vx;
        v_y_t(x,y)  =v_y(x,y)+Delta_t*k3_vy;
        h_t(x,y)    =h(x,y)+Delta_t*k3_h;
        v_x_t = apply_periodic_boundary(v_x_t);
        v_y_t = apply_periodic_boundary(v_y_t);
        h_t  = apply_periodic_boundary(h_t);

        k4_vx       =f_vx(v_x_t,v_y_t,h_t,N_x,N_y,Delta_x,Delta_y,g);
        k4_vy       =f_vy(v_x_t,v_y_t,h_t,N_x,N_y,Delta_x,Delta_y,g);
        k4_h        =f_h(v_x_t,v_y_t,h_t,N_x,N_y,Delta_x,Delta_y);
        v_x(x,y)    =v_x(x,y)+Delta_t*(k1_vx/6+k2_vx/3+k3_vx/3+k4_vx/6);
        v_y(x,y)    =v_y(x,y)+Delta_t*(k1_vy/6+k2_vy/3+k3_vy/3+k4_vy/6);
        h(x,y)      =h(x,y)+Delta_t*(k1_h/6+k2_h/3+k3_h/3+k4_h/6);
        v_x_t = apply_periodic_boundary(v_x_t);
        v_y_t = apply_periodic_boundary(v_y_t);
        h_t  = apply_periodic_boundary(h_t);

        time=time+Delta_t;

        % plot results

        set(Solution,  'ZData', h(x,y));
        t.String = ['t = ' num2str(time) ' s'];
        drawnow;
    end

return
end

function f=f_vx(v_x,v_y,h,N_x,N_y,Delta_x,Delta_y,g)
	x=4:N_x+3; y=4:N_y+3;
    % Sixth order finite stencil
	f = (-v_x(x,y).*((1/60)*v_x(x+3,y)-(1/60)*v_x(x-3,y)...
                            -(3/20)*v_x(x+2,y)+(3/20)*v_x(x-2,y)...
                            +(3/4)*v_x(x+1,y)-(3/4)*v_x(x-1,y))/(Delta_x) ...
           -v_y(x,y).*((1/60)*v_x(x,y+3)-(1/60)*v_x(x,y-3)...
                            -(3/20)*v_x(x,y+2)+(3/20)*v_x(x,y-2)...
                            +(3/4)*v_x(x,y+1)-(3/4)*v_x(x,y-1))/(Delta_y)) ...
          -g*((1/60)*h(x+3,y)-(1/60)*h(x-3,y)...
                            -(3/20)*h(x+2,y)+(3/20)*h(x-2,y)...
                            +(3/4)*h(x+1,y)-(3/4)*h(x-1,y))/(Delta_x);
end

function f=f_vy(v_x,v_y,h,N_x,N_y,Delta_x,Delta_y,g)
	x=4:N_x+3; y=4:N_y+3;
    % Sixth order finite stencil
	f = (-v_x(x,y).*((1/60)*v_y(x+3,y)-(1/60)*v_y(x-3,y)...
                            -(3/20)*v_y(x+2,y)+(3/20)*v_y(x-2,y)...
                            +(3/4)*v_y(x+1,y)-(3/4)*v_y(x-1,y))/(Delta_x) ...
           -v_y(x,y).*((1/60)*v_y(x,y+3)-(1/60)*v_y(x,y-3)...
                            -(3/20)*v_y(x,y+2)+(3/20)*v_y(x,y-2)...
                            +(3/4)*v_y(x,y+1)-(3/4)*v_y(x,y-1))/(Delta_y)) ...
          -g*((1/60)*h(x,y+3)-(1/60)*h(x,y-3)...
                            -(3/20)*h(x,y+2)+(3/20)*h(x,y-2)...
                            +(3/4)*h(x,y+1)-(3/4)*h(x,y-1))/(Delta_y);
end

function f=f_h(v_x,v_y,h,N_x,N_y,Delta_x,Delta_y)
	x=4:N_x+3; y=4:N_y+3;
    % Sixth order finite stencil
	f =   (-v_x(x,y).*((1/60)*h(x+3,y)-(1/60)*h(x-3,y)...
                        -(3/20)*h(x+2,y)+(3/20)*h(x-2,y)...
                        +(3/4)*h(x+1,y)-(3/4)*h(x-1,y))/(Delta_x) ...
             -v_y(x,y).*((1/60)*h(x,y+3)-(1/60)*h(x,y-3)...
                        -(3/20)*h(x,y+2)+(3/20)*h(x,y-2)...
                        +(3/4)*h(x,y+1)-(3/4)*h(x,y-1))/(Delta_y) ) ...
             -h(x,y).*(((1/60)*v_x(x+3,y)-(1/60)*v_x(x-3,y)...
                        -(3/20)*v_x(x+2,y)+(3/20)*v_x(x-2,y)...
                        +(3/4)*v_x(x+1,y)-(3/4)*v_x(x-1,y))/(Delta_x)...
                        +((1/60)*v_y(x,y+3)-(1/60)*v_y(x,y-3)...
                        -(3/20)*v_y(x,y+2)+(3/20)*v_y(x,y-2)...
                        +(3/4)*v_y(x,y+1)-(3/4)*v_y(x,y-1))/(Delta_y));
end

function input=apply_periodic_boundary(input)
        input(3,:)=input(end-3,:); input(:,3)=input(:,end-3); input(end-2,:)=input(4,:); input(:,end-2)=input(:,4);
        input(2,:)=input(end-4,:); input(:,2)=input(:,end-4); input(end-1,:)=input(5,:); input(:,end-1)=input(:,5);
        input(1,:)=input(end-5,:); input(:,1)=input(:,end-5); input(end,:)=input(6,:); input(:,end)=input(:,6);

end
