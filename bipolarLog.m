function output = bipolarLog(input,deriv)
    if((nargin == 2) && strcmp(deriv,'deriv'))
        output = 0.5*(1-input.*input);
    else
        output = (2./(1+exp(-input))) - 1;
    end
end