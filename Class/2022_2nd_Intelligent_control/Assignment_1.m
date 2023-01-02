%% ANN HW - Intelligent Control

clear, clc

% define sigmoid(x)
sigmoid = @(x) 1./(1+exp(-x));

% data initializing (Input, Output)
node_input = eye(8,8);   % unit matrix
target = node_input;
node_output = zeros(8,8);  % unit matrix
node_hidden = zeros(3,8); % 3x8 matrix (3 hidden node in each network)
weight_i2h = rand(3,8,8);       % 3-dimension is for saving each input binary set
weight_h2o = rand(8,3,8);       % 3-dimension is for saving each input binary set

error_output = zeros(8,8);
error_output_redefined = zeros(1,8);
error_gradient_k = zeros(8,8);
error_gradient_h = zeros(3,8);

% initialize parameter
learning_rate = 0.5;
iteration = 100;

% for Graphs
G_HIDDEN = zeros(3,iteration,8);
G_OUTPUT = zeros(8,iteration,8);
G_ERROR_EACH_OUTPUT = zeros(8,iteration,8);
G_ERROR_REDEFINED = zeros(1,iteration,8);
G_ERROR_GRADIENT_K = zeros(8,iteration,8);
G_ERROR_GRADIENT_H = zeros(3,iteration,8);
G_WEIGHT_I2H = zeros(3,8,iteration,8);
G_WEIGHT_H2O = zeros(8,3,iteration,8);


% ------------------- PROCESS ---------------------%
% -------------------  START  ---------------------%

for N = 1:1:8
    
    for rep = 1:1:iteration
        
        % <FORWARD PROPAGATION>
        % calculate hidden node values with sigmoid(x)
        for i=1:1:3
            node_hidden(i,N) = sigmoid( weight_i2h(i,:,N)*node_input(:,N) );
            G_HIDDEN(i,rep,N) = node_hidden(i,N);
        end

        % calculate output node values with sigmoid(x)
        for i=1:1:8
            node_output(i,N) = sigmoid( weight_h2o(i,:,N)*node_hidden(:,N) )
            G_OUTPUT(i,rep,N) = node_output(i,N);
        end

        % calculate output error
        for i=1:1:8
            error_output(i,N) = (1/2).*( target(i,N)-node_output(i,N) ).^2;
            G_ERROR_EACH_OUTPUT(i,rep,N) = error_output(i,N);
        end
        
        % Summation of Output errors
        error_output_redefined(N) = sum(error_output(:,N));
        G_ERROR_REDEFINED(1,rep,N) = error_output_redefined(N);

        % <BACK PROPAGATION>
        % Calculate Error Gradient and update weights 
        % output node
        for i=1:1:8
            error_gradient_k(i,N) = node_output(i,N).*(1-node_output(i,N)).*(target(i,N)-node_output(i,N));
            G_ERROR_GRADIENT_K(i,rep,N) = error_gradient_k(i,N);
        end
        % hidden node
        for i=1:1:3
            error_gradient_h(i,N) = node_hidden(i,N).*(1-node_hidden(i,N)).* sum( weight_h2o(:,:,N)'*error_gradient_k(:,N) );
            G_ERROR_GRADIENT_H(i,rep,N) = error_gradient_h(i,N);
        end

        % update weights
        for i=1:1:8
            for j=1:1:3
                % weights of hidden to output
                weight_h2o(i,j,N) = weight_h2o(i,j,N) + learning_rate*error_gradient_k(i,N)*node_hidden(j,N);
                G_WEIGHT_H2O(i,j,rep,N) = weight_h2o(i,j,N);
                % weights of input to hidden
                weight_i2h(j,i,N) = weight_i2h(j,i,N) + learning_rate*error_gradient_h(j,N)*node_input(i,N);
                G_WEIGHT_I2H(j,i,rep,N) = weight_i2h(j,i,N);
            end
        end
    end
end

% ------------------- PROCESS ---------------------%
% -------------------   END   ---------------------%

figure
% plot3(G_ERROR_EACH_OUTPUT(:,:,:))
% plot(G_ERROR_EACH_OUTPUT(2,:,1))
hold on

% plot(1:iteration, G_ERROR_REDEFINED(1,:,1))
% plot(1:iteration, G_ERROR_REDEFINED(1,:,2))
% plot(1:iteration, G_ERROR_REDEFINED(1,:,3))
% plot(1:iteration, G_ERROR_REDEFINED(1,:,4))
% plot(1:iteration, G_ERROR_REDEFINED(1,:,5))
% plot(1:iteration, G_ERROR_REDEFINED(1,:,6))

% plot(1:100, G_WEIGHT_H2O(8,2,:,1))

% plot(1:iteration, G_HIDDEN(1,:,1))
% plot(1:iteration, G_HIDDEN(2,:,1))
% plot(1:iteration, G_HIDDEN(3,:,1))

plot(1:iteration, G_OUTPUT(1,:,1))
plot(1:iteration, G_OUTPUT(2,:,1))
plot(1:iteration, G_OUTPUT(3,:,1))
plot(1:iteration, G_OUTPUT(4,:,1))
plot(1:iteration, G_OUTPUT(5,:,1))
plot(1:iteration, G_OUTPUT(6,:,1))
plot(1:iteration, G_OUTPUT(7,:,1))
plot(1:iteration, G_OUTPUT(8,:,1))

% hold on




















