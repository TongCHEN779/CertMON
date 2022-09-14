% Fix-point iteration for implicit layer
function z = solution_fix_point(x0, W, U, u, eta, alpha, method)
    if method == "forward-backward"
        d = size(W, 2); z = zeros(d,1); err = 1;
        while err > eta
            z_new = (1-alpha)*z + alpha*(W*z+U*x0+u);
            z_new = z_new .* (z_new >= 0);
            err = norm(z_new-z, 2)/norm(z_new, 2);
            z = z_new;
        end
    elseif method == "Peaceman-Rachford"
        d = size(W, 2); z = zeros(d,1); y = zeros(d,1); err = 1;
        V = (eye(d)+alpha*(eye(d)-W))^(-1);
        while err > eta
            y_new1 = 2*z-y; z_new1 = V*(y_new1+alpha*(U*x0+u));
            y_new2 = 2*z_new1-y_new1; z_new2 = y_new2 .* (y_new2 >= 0);
            err = norm(z_new2-z, 2)/norm(z_new2, 2);
            z = z_new2; y = y_new2;
        end
    end
end