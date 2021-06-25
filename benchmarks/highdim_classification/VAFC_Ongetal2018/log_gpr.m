function [g dg] = logMLGPR(theta, covfunc, X, Y)
%

[nlml dnlml] = gpr(theta/2, covfunc, X, Y);
g = - nlml;
dg = - 0.5*dnlml;

