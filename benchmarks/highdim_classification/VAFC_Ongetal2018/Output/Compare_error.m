restr=zeros(20,2,3);
rests=zeros(20,2,3);

rettr=cell(3,2);
retts=cell(3,2);


for p = [4 20]
  for data_pick = 1:3
    for seedi=1:20
          if (p == 4)
            pi=1;
              switch data_pick
                  case 1
                      disp('Running Colon data... (p = 4)')
                      load(strcat('F_Colon_p4_',num2str(seedi),'.mat'))
                      plot_constant = 500;
                  case 2
                      disp('Running Leuk data... (p = 4)')
                      load(strcat('F_leuk_p4_',num2str(seedi),'.mat'))
                      plot_constant = 50000;
                  case 3
                      disp('Running Duke data... (p = 4)')
                      load(strcat('F_duke_p4_',num2str(seedi),'.mat'))
                      plot_constant = 50000;
                  otherwise
                      disp('Please pick one of the three datasets')
              end
          end

          if (p == 20)
            pi=2;
              switch data_pick
                  case 1
                      disp('Running Colon data... (p = 20)')
                      load(strcat('F_Colon_p20_',num2str(seedi),'.mat'))
                      plot_constant = 500;
                  case 2
                      disp('Running Leuk data... (p = 20)')
                      load(strcat('F_leuk_p20_',num2str(seedi),'.mat'))
                      plot_constant = 50000;
                  case 3
                      disp('Running Duke data... (p = 20)')
                      load(strcat('F_duke_p20_',num2str(seedi),'.mat'))
                      plot_constant = 50000;
                  otherwise
                      disp('Please pick one of the three datasets')
              end
          end

          [Ntr, ~] = size(Xtr);
          [Nts, ~] = size(Xts);


          F_mu = F_mu(1:(length(F_mu)/2));
          Str = sigmoid(Xtr*F_mu);
          Sts = sigmoid(Xts*F_mu);
          rettr{data_pick,pi}=horzcat(rettr{data_pick,pi},Str);
          retts{data_pick,pi}=horzcat(retts{data_pick,pi},Sts);
          fprintf('VAFC Train error: %d/%d\n',sum(abs([(Ytr+1)/2 - round(Str)])), Ntr);
          fprintf('VAFC Test error: %d/%d\n',sum(abs([(Yts+1)/2 - round(Sts)])), Nts);
          restr(seedi,pi,data_pick)=sum(abs([(Ytr+1)/2 - round(Str)]));
          rests(seedi,pi,data_pick)=sum(abs([(Yts+1)/2 - round(Sts)]));
    end
  end
end;

save('resVAFC.mat','restr','rests');
retts2=cell2struct(retts,{'colon','leukemia','duke'},1);
rettr2=cell2struct(rettr,{'colon','leukemia','duke'},1);
save('resVAFC2.mat','retts2','rettr2','-v7');