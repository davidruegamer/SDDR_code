function output = log_oneplus_x(x)
%%% to evaluate log(1 + exp(x)) when x is large
   output = zeros(length(x),1);
for i = 1:length(x)
   if x(i) > 35
      output(i) = x(i);
   else
      output(i) = log(1+exp(x(i)));
   end
end