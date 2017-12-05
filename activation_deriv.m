function a = activation_deriv(x)
    b = 1./(1.+exp(-x));
    a = b.*(1-b);
end