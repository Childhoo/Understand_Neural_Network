% backpropagation example based on XOR net..


% prepare the input and output
clear all;
input_data = [1 0; 0 1; 0 0; 1 1];
target = [1;1;0;0];
% input_data = randn(4,2);
% target = randn(4,1);
%set the input
input = input_data;

% build the network and initilize them
weight_hidden_layer_1 = randn(2,5);
weight_hidden_layer_2 = randn(5,1);
z_1 = input*weight_hidden_layer_1;
a_1 = activation(z_1);
z_2 = a_1*weight_hidden_layer_2;
a_2 = activation(z_2);
Num_samples = 4;
%measure the error term
loss = 0.5*(target - a_2).^2;
%delta output equals the derivatives of loss wrt. network output
delta_output = a_2 - target; 
disp(mean(loss));

%start backpropgation

% initial momentum as zeros
vt_w1 = zeros(2,5);
vt_w2 = zeros(5,1);

%maxmize graph for better visualization
h=figure('units','normalized','outerposition',[0 0 1 1]);
% prepare for saving gif
filename = 'Train_XOR_Forward_BackWard_Prop.gif';
% plot(loss,'*b');
for i=1:50000
    delta_z2 = delta_output.*activation_deriv(z_2);
    
    delta_z2_to_w2 = a_1;
%     delta_w2 = delta_z2 * delta_z2_to_w2;
    delta_w2 = zeros(Num_samples,5,1);
    for ind = 1:Num_samples
        delta_w2(ind,:) = delta_z2(ind,:) * delta_z2_to_w2(ind,:);
    end
    delta_w2 = reshape(delta_w2, Num_samples, 5, 1);
    
    delta_z2_to_a1 = weight_hidden_layer_2;
%     delta_a1 = delta_z2 * delta_z2_to_a1;
    delta_a1 = zeros(Num_samples,5,1);
    for ind = 1:Num_samples
        delta_a1(ind,:) = delta_z2(ind,:) * delta_z2_to_a1';
    end
    
    act_der_a_1_z_1 = activation_deriv(z_1);
    delta_z1 = zeros(Num_samples,5,1);
    
    for ind = 1:Num_samples
        delta_z1(ind,:,:) = delta_a1(ind,:).* act_der_a_1_z_1(ind,:);
    end
%     delta_z1 = delta_a1 * activation_deriv(z_1);
    
    delta_z1_to_w1 = input;
%     delta_w1 =  delta_z1*delta_z1_to_w1;
    delta_w1 = zeros(Num_samples,5,2);
    for ind = 1:Num_samples
        delta_w1(ind,:,:) = delta_z1(ind,:)'* delta_z1_to_w1(ind,:);
    end
    
    % calculate the accumulated mean gradient
    gradient_w1 = reshape(mean(delta_w1,1),5,2)';
    gradient_w2 = mean(delta_w2, 1)';
    %update according to learning rate
    
    % use momentum based gradient descent
    vt_w1 = 0.95*vt_w1 + 0.01*gradient_w1;
    vt_w2 = 0.95*vt_w2 + 0.01*gradient_w2;
    weight_hidden_layer_1 = weight_hidden_layer_1 - vt_w1;
    weight_hidden_layer_2 = weight_hidden_layer_2 - vt_w2;
    
    % forward propagation again 
    z_1 = input*weight_hidden_layer_1;
    a_1 = activation(z_1);
    z_2 = a_1*weight_hidden_layer_2;
    a_2 = activation(z_2);
    %measure the error term
    loss = 0.5*(target - a_2).^2;
    if mod(i,500)==0
        disp(mean(loss));
    end
    
    % all the remain code in loop are only for visulization....
    loss_rec(i) = mean(loss);
    if mod(i,1000)==0
        allPlots = findall(0, 'Type', 'figure', 'FileName', []);
        clf();

%         delete(allPlots);
%         hold on;
%         subplot(2,2,1);
%         refreshdata;
%         plot(loss,'*g');
%         
%         subplot(2,2,2);
%         plot(loss_rec,'r');
%         title('loss over iteration');

        radius = 3;
        plot_x_start = 20;
        plot_y_start = 20;
        plot_x_step = 200;
        plot_y_step = 80;
        
        subplot(1,2,1);
        axis off;
        title(['Forward Propagation(' num2str(i) 'th iter)'],'FontSize',15);
        hold on;
        
        % draw forward propagation first
        % draw neurons first
        for kk = 1:2
             layer1_x(kk) = plot_x_start;
             layer1_y(kk) = plot_y_start + (kk-1)*plot_y_step + 1.5*plot_y_step;
             viscircles([layer1_x(kk),layer1_y(kk)],radius,'EdgeColor','black');
        end
        
        text(layer1_x(1)-10,layer1_y(2)+20,'input','FontSize',15);    
        
        for kk = 1:5
             layer2_x(kk) = plot_x_start + plot_x_step;
             layer2_y(kk) = plot_y_start + (kk-1)*plot_y_step;
             viscircles([layer2_x(kk),layer2_y(kk)],radius,'EdgeColor','b');
        end        
        text(layer2_x(5)-20,layer2_y(5)+5,'z1=w1*input','FontSize',15); 
        
        
        layer3_x = plot_x_start + 2*plot_x_step;
        layer3_y = plot_y_start + 2*plot_y_step;
        viscircles([layer3_x, layer3_y],radius,'EdgeColor','b');
        text(layer3_x-20,layer3_y+35,'z2=w2*a1','FontSize',15); 
        % draw edges (weights) then
        for kk = 1:2
            for jj=1:5
                a=plot([layer1_x(kk) layer2_x(jj)],...
                    [layer1_y(kk) layer2_y(jj)],'LineWidth',2);
                 txt_x = layer1_x(kk) + 0.65*(layer2_x(jj)-layer1_x(kk));
                 txt_y = layer1_y(kk) + 0.65*(layer2_y(jj)-layer1_y(kk));
                 text(txt_x,txt_y,...
                    [num2str(weight_hidden_layer_1(kk,jj))],'FontSize',12);
            end
        end
        
        text(txt_x,txt_y+20,'w1','FontSize',15); 
        
        for kk=1:5
            plot([layer2_x(kk) layer3_x],...
                [layer2_y(kk) layer3_y],'LineWidth',2);
            txt_x = layer2_x(kk) + 0.5*(layer3_x-layer2_x(kk));
            txt_y = layer2_y(kk) + 0.5*(layer3_y-layer2_y(kk));
            % text the current weight
            text(txt_x,txt_y,...
                [num2str(weight_hidden_layer_2(kk))],'FontSize',12);
            if kk==5
                text(txt_x,txt_y+20,'w2','FontSize',15); 
            end
            % text activation with red color
            txt_x = layer2_x(kk) + 0.1*(layer3_x-layer2_x(kk));
            txt_y = layer2_y(kk) + 0.1*(layer3_y-layer2_y(kk));
            text(txt_x,txt_y,...
                [num2str(a_1(1,kk))],'FontSize',12,'Color',[1.0 0 0]);
        end
        text(txt_x,txt_y+10,'a1=sigmoid(z1)','FontSize',15); 
               
        text(layer1_x(1)-10,layer1_y(1)-10,...
            ['' num2str(input_data(1,1))],'FontSize',12);
        
        text(layer1_x(2)-10,layer1_y(2)+10,...
            ['' num2str(input_data(1,2))],'FontSize',12);       
        
        text(layer3_x+10,layer3_y+20,...
            'ouput=sigmoid(z2)','FontSize',15);
        
        text(layer3_x+10,layer3_y+5,...
            ['ouput=' num2str(a_2(1))],'FontSize',12);
        
        text(layer3_x+10,layer3_y-15,...
            ['loss=' num2str(loss(1))],'FontSize',12, 'Color',[0 0 0]);
        
                      

        % draw backward propagation
%         figure(2);
        subplot(1,2,2);
        axis off;
        title(['Backward Prop(' num2str(i) 'th iter)'],'FontSize',15);
%         set(get(gca,'title'),'Position',[1.5 0.50 2.00011])
        hold on;
        for kk = 1:2
             layer1_x(kk) = plot_x_start;
             layer1_y(kk) = plot_y_start + (kk-1)*plot_y_step + 1.5*plot_y_step;
             viscircles([layer1_x(kk),layer1_y(kk)],radius,'EdgeColor','b');
        end
        text(layer1_x(1)-10,layer1_y(2)+20,'input','FontSize',15);         
        for kk = 1:5
             layer2_x(kk) = plot_x_start + plot_x_step;
             layer2_y(kk) = plot_y_start + (kk-1)*plot_y_step;
             viscircles([layer2_x(kk),layer2_y(kk)],radius,'EdgeColor','b');
        end
        text(layer2_x(5)-25,layer2_y(5)+5,'d_z_1=d_l_o_s_s/d_a_1*d_a_1/d_z_1','FontSize',15); 
        
        layer3_x = plot_x_start + 2*plot_x_step;
        layer3_y = plot_y_start + 2*plot_y_step;
        viscircles([layer3_x, layer3_y],radius,'EdgeColor','b');
        
        % draw input first
        text(layer1_x(1)-10,layer1_y(1)-10,...
            ['' num2str(input_data(1,1))],'FontSize',12);
        
        text(layer1_x(2)-10,layer1_y(2)+10,...
            ['' num2str(input_data(1,2))],'FontSize',12);
        
        text(layer3_x-5,layer3_y+30,...
            ['d_l_o_s_s/d_o_u_t_p_t = outout-target'],'FontSize',15);
       
        text(layer3_x+10,layer3_y+15,...
            ['d_l_o_s_s/d_o_u_t_p_t=' num2str(delta_output(1))],'FontSize',12);
        
        text(layer3_x+10,layer3_y-5,...
            ['d_l_o_s_s/d_z_2=' num2str(delta_z2(1))],'FontSize',12);
        
        
        % draw d_loss/d_a_1
        for kk=1:5
            plot([layer2_x(kk) layer3_x],...
                [layer2_y(kk) layer3_y],'LineWidth',2);
            txt_x = layer2_x(kk) + 0.5*(layer3_x-layer2_x(kk));
            txt_y = layer2_y(kk) + 0.5*(layer3_y-layer2_y(kk));
            if kk==5;
                text(txt_x,txt_y+35,...
                    ['d_l_o_s_s/d_w_2=d_l_o_s_s/d_z_2 * a_1 '],'FontSize',15);
                text(txt_x,txt_y+20,...
                    ['a_1 = d_z_2/d_w_2'],'FontSize',15);
            end

            % text the current weight
            text(txt_x,txt_y,...
                [num2str(delta_w2(1,kk))],'FontSize',12,'Color',...
                [1.0 0.0 1],'FontWeight','bold');
            % text activation with red color
            txt_x = layer2_x(kk) + 0.15*(layer3_x-layer2_x(kk));
            txt_y = layer2_y(kk) + 0.15*(layer3_y-layer2_y(kk));
            text(txt_x,txt_y,...
                [num2str(delta_a1(1,kk))],'FontSize',12,'Color',[1.0 0.0 0]);
            if kk==5;
                text(txt_x,txt_y+15,...
                ['d_l_o_s_s/d_a_1=d_l_o_s_s/d_z_2 * w_2'],'FontSize',15);
            end

        end
        
        
        for kk = 1:2
            for jj=1:5
                a=plot([layer1_x(kk) layer2_x(jj)],...
                    [layer1_y(kk) layer2_y(jj)],'LineWidth',2);
                 txt_x = layer1_x(kk) + 0.60*(layer2_x(jj)-layer1_x(kk));
                 txt_y = layer1_y(kk) + 0.60*(layer2_y(jj)-layer1_y(kk));
                 text(txt_x,txt_y,...
                    [num2str(delta_w1(1,jj,kk))],'FontSize',12,'Color',...
                    [1.0 0.0 1],'FontWeight','bold');
            end
        end
        text(layer1_x(kk),txt_y + 25,...
                    ['d_l_o_s_s/d_w_1=d_l_o_s_s/d_z_1 * input '],'FontSize',15);

        
%         clf;
        if exist('ht','var')==1
            delete(ht);
        end
%         set(ht,'Visible','off');
%         text(40,25,'                 ','FontSize',15);
        text(0,0,['Note: each training sample contributes to the gradients. The gradients over 4 samples are averaged for real updating.'],...
            'FontSize',12);
        refreshdata;
%         ht = text(40,25,num2str(weight_hidden_layer_1(2,1)),'FontSize',10);
        
        drawnow
        frame = getframe(h); 
        im = frame2im(frame); 
        [imind,cm] = rgb2ind(im,256); 
        % Write to the GIF File      
        if i == 1000 
            imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
        else
            imwrite(imind,cm,filename,'gif','WriteMode','append');
        end 
        
        
    end
    
    
    %delta output equals the derivatives of loss wrt. network output
    delta_output = a_2 - target; 
    

end


