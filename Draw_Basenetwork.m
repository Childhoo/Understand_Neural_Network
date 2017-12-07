
axis off;
%         title(['bas'],'FontSize',15);
        hold on;
        
        % draw forward propagation first
        % draw neurons first
        for kk = 1:2
             layer1_x(kk) = plot_x_start;
             layer1_y(kk) = plot_y_start + (kk-1)*plot_y_step + 1.5*plot_y_step;
             viscircles([layer1_x(kk),layer1_y(kk)],radius,'EdgeColor','black');
        end
        
        text(layer1_x(1)-20,layer1_y(2)+20,'input','FontSize',15);    
        
        for kk = 1:5
             layer2_x(kk) = plot_x_start + plot_x_step;
             layer2_y(kk) = plot_y_start + (kk-1)*plot_y_step;
             viscircles([layer2_x(kk),layer2_y(kk)],radius,'EdgeColor','b');
        end        
        text(layer2_x(5)-20,layer2_y(5)+15,'z_1=w_1*input','FontSize',15); 
        
        
        layer3_x = plot_x_start + 2*plot_x_step;
        layer3_y = plot_y_start + 2*plot_y_step;
        viscircles([layer3_x, layer3_y],radius,'EdgeColor','b');
        text(layer3_x-20,layer3_y+35,'z_2=w_2*a_1','FontSize',15); 
        % draw edges (weights) then
        for kk = 1:2
            for jj=1:5
                a=plot([layer1_x(kk) layer2_x(jj)],...
                    [layer1_y(kk) layer2_y(jj)],'LineWidth',2);
                 txt_x = layer1_x(kk) + 0.65*(layer2_x(jj)-layer1_x(kk));
                 txt_y = layer1_y(kk) + 0.65*(layer2_y(jj)-layer1_y(kk));
%                  text(txt_x,txt_y,...
%                     sprintf('%0.3f',weight_hidden_layer_1(kk,jj)),'FontSize',15);
            end
        end
        
        text(txt_x,txt_y+20,'w_1','FontSize',15); 
        
        for kk=1:5
            plot([layer2_x(kk) layer3_x],...
                [layer2_y(kk) layer3_y],'LineWidth',2);
%             txt_x = layer2_x(kk) + 0.5*(layer3_x-layer2_x(kk));
%             txt_y = layer2_y(kk) + 0.5*(layer3_y-layer2_y(kk));
%             % text the current weight
%             text(txt_x,txt_y,...
%                 sprintf('%0.3f',weight_hidden_layer_2(kk)),'FontSize',15);
%             if kk==5
%                 text(txt_x,txt_y+20,'w_2','FontSize',15); 
%             end
            % text activation with red color
            txt_x = layer2_x(kk) + 0.1*(layer3_x-layer2_x(kk));
            txt_y = layer2_y(kk) + 0.1*(layer3_y-layer2_y(kk));
%             text(txt_x,txt_y,...
%                 sprintf('%0.4f',a_1(1,kk)),'FontSize',15,'Color',[1.0 0 0]);
        end
        text(txt_x,txt_y+5,'a_1=sigmoid(z_1)','FontSize',15); 
               
%         text(layer1_x(1)-10,layer1_y(1)-10,...
%             ['' num2str(input_data(1,1))],'FontSize',15);
%         
%         text(layer1_x(2)-10,layer1_y(2)-10,...
%             ['' num2str(input_data(1,2))],'FontSize',15);       
        
        text(layer3_x-20,layer3_y-20,...
            'ouput=sigmoid(z_2)','FontSize',15);
        
%         text(layer3_x+10,layer3_y+5,...
%             ['ouput=' num2str(a_2(1))],'FontSize',15);
        
        text(layer3_x-40,layer3_y-35,...
            ['loss=1/2 * (target-output)^2'],'FontSize',15, 'Color',[0 0 0]);
        
%         text(layer3_x+10,layer3_y-30,...
%             ['loss=' num2str(loss(1))],'FontSize',15, 'Color',[0 0 0]);