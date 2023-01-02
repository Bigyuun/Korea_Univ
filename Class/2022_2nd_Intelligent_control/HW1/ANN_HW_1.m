%% ANN 
close, clear, clc

% define sigmoid(x)
sigmoid = @(x) 1./(1+exp(-x));

% data set
input = eye(8,8);
target = input;

output = zeros(8);   %unit matrix
hidden = zeros(3,8); % 3x8 matrix (3 hidden node in each network)

err_output_all = zeros(8,8);
err_output_redef = zeros(1,8);

err_grad_k = zeros(8,1);
err_grad_h = zeros(3,1);

w_ih = rand(3,8);
w_ho = rand(8,3);

% init parameters
iteration = 800000;
lr = 0.3;

% save values
G_OUTPUT = zeros(8, iteration, 8);
G_ERROR = zeros(1, iteration);
G_W_IH = zeros(3,iteration,8);
G_W_HO = zeros(8,iteration,3);

%------------- PROCESS START -------------------
for rep = 1:1:iteration
    
    % 8 inputs
    for i = 1:1:8 
        %forward propagation
        for j=1:1:3
            hidden(j,i) = sigmoid(w_ih(j,:)*input(:,i));
        end

        % compute output layer components
        output(:,i) = sigmoid(w_ho * hidden(:, i));
        G_OUTPUT(:,rep,i) = output(:,i);
    end
    
    % output error (redefined)
    for i = 1:1:8
        G_ERROR(:,rep) = G_ERROR(:,rep) + (1/2)*sum( (target(:,i)-output(:,i)).^2 );
    end

    % redefined output errors
    for i = 1:1:8
        err_grad_k = output(:,i) .* (1-output(:,i)) .* (target(:,i) - output(:,i));

        for j = 1:1:3
            err_grad_h(j) = hidden(j,i)*(1-hidden(j,i))*sum(w_ho(:,j).*err_grad_k);
        end

        % update weights
        for k = 1:1:3
            w_ih(k,:) = w_ih(k,:) + lr*err_grad_h(k,:)*input(:,i)';
            w_ho(:,k) = w_ho(:,k) + lr*err_grad_k*hidden(k,i);
        end
        
    end
    G_W_IH(:,rep,:) = w_ih;
    G_W_HO(:,rep,:) = w_ho;
end

%------------- PROCESS END -------------------

% graphs

% output values
figure(1)
for i=1:1:8 
    for j=1:1:8
        hold on
        subplot(2,4,i)
        title('Output Values')
        xlabel('iteration');ylabel('value')
        ylim([0 1])
        semilogx(1:iteration, G_OUTPUT(j,:,i)')
%         plot(G_OUTPUT(:,:,i)')
    end
    hold off
end

% output error
figure(2)
for i=1:1:8 
    for j=1:1:8
        hold on
        subplot(2,4,i)
        title('Output Error')
        xlabel('iteration');ylabel('value')
        ylim([0 1])
        semilogx(1:iteration, G_ERROR(:))
    end
    hold off
end

figure(3)
% hold on
% semilogx(1:iteration, G_W_HO(1,1,:))
for i=1:1:8 
    for j=1:1:3
        hold on
        title('Weight of Hidden to Output')
        xlabel('iteration');ylabel('value')
%         ylim([-3 3])
%         semilogx(1:iteration, G_W_HO(i,:,j))
        semilogx(G_W_IH(j,:,i))
    end
end










