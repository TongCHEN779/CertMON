using JuMP
using LinearAlgebra
using MosekTools
using MAT
using MLDatasets

function cert_monDEQ(x0, W, U, u, c, ϵ; nrm = "l2")
    d_in = size(U, 2); d = size(U, 1);
    model = Model(Mosek.Optimizer)
    @variable(model, P[1:1+d_in+d, 1:1+d_in+d], Symmetric)
    idx_x = 2:1+d_in; idx_z = 2+d_in:1+d_in+d;
    lx = x0.-ϵ*ones(d_in,1); ux = x0.+ϵ*ones(d_in,1);
    obj = c'*P[1,idx_z]
    @objective(model, Max, obj)
    if nrm == "l2"
        @constraint(model, sum(diag(P[idx_x, idx_x])) .- 2*P[idx_x, 1]'*x0 .+ x0'*x0 .<= ϵ^2)
    elseif nrm == "linf"
        @constraint(model, diag(P[idx_x, idx_x]) .- (lx+ux).*P[idx_x, 1] .+ lx.*ux .<= 0)
    else
        error("NotImplementedError")
    end
    @constraints(model, begin
        P[1,1] == 1
        P in PSDCone()
        P[idx_z, 1] .>= 0
        P[idx_z, 1] .>= W*P[idx_z, 1] + U*P[idx_x, 1] + u
        diag(P[idx_z, idx_z]) .== diag(W*P[idx_z, idx_z] + U*P[idx_x, idx_z]) + u.*P[idx_z, 1]
    end)
    optimize!(model)
    return objective_value(model)
end

function lip_monDEQ(W, U, u, C, ϵ; nrm = "l2")
    d_in = size(U, 2); d = size(U, 1);
    me = 0.1307; std = 0.3081;
    model = Model(Mosek.Optimizer)
    lx = (0-ϵ-me)/std*ones(d_in,1);
    ux = (1+ϵ-me)/std*ones(d_in,1);
    if nrm == "l2"
        @variable(model, P[1:1+2*d_in+4*d+10, 1:1+2*d_in+4*d+10], Symmetric)
        idx_t = 2:1+d_in; idx_x = 2+d_in:1+2*d_in; idx_z = 2+2*d_in:1+2*d_in+d;
        idx_s = 2+2*d_in+d:1+2*d_in+2*d; idx_y = 2+2*d_in+2*d:1+2*d_in+3*d;
        idx_u = 2+2*d_in+3*d:1+2*d_in+4*d; idx_c = 2+2*d_in+4*d:1+2*d_in+4*d+10;
        @constraints(model, begin
            sum(diag(P[idx_t, idx_t])) .<= 1
            sum(diag(P[idx_c, idx_c])) .<= 1
            diag(P[idx_x, idx_x]) .- (lx+ux).*P[idx_x, 1] .+ lx.*ux .<= 0
            sum(diag(P[idx_u, idx_u])) .<= 1
            sum(diag(P[idx_y, idx_y])) .<= 1
        end)
    elseif nrm == "linf"
        @variable(model, P[1:1+2*d_in+4*d+20, 1:1+2*d_in+4*d+20], Symmetric)
        idx_t = 2:1+d_in; idx_x = 2+d_in:1+2*d_in; idx_z = 2+2*d_in:1+2*d_in+d;
        idx_s = 2+2*d_in+d:1+2*d_in+2*d; idx_y = 2+2*d_in+2*d:1+2*d_in+3*d;
        idx_u = 2+2*d_in+3*d:1+2*d_in+4*d; idx_c = 2+2*d_in+4*d:1+2*d_in+4*d+10;
        idx_w = 2+2*d_in+4*d+10:1+2*d_in+4*d+20;
        @constraints(model, begin
            diag(P[idx_t, idx_t]) .<= 1
            diag(P[idx_w, idx_w]) .<= 1
            sum(diag(P[idx_c, idx_c])) .<= 1
            sum(diag(P[idx_c, idx_w])) .<= 1
            diag(P[idx_x, idx_x]) .- (lx+ux).*P[idx_x, 1] .+ lx.*ux .<= 0
            sum(diag(P[idx_u, idx_u])) .<= 1
            sum(diag(P[idx_y, idx_y])) .<= 1
        end)
    else
        error("NotImplementedError")
    end
    obj = sum(U' .* P[idx_t, idx_y])
    @objective(model, Max, obj)
    @constraints(model, begin
        P[1,1] == 1
        P in PSDCone()
        P[idx_u, 1] - W'*P[idx_y, 1] .== C'*P[idx_c, 1]
        diag(P[idx_u, idx_u]) .== diag(W'*P[idx_y, idx_u]) + diag(C'*P[idx_c, idx_u])
        diag(P[idx_y, idx_u]) .== diag(W'*P[idx_y, idx_y]) + diag(C'*P[idx_c, idx_y])
        diag(P[idx_s, idx_u]) .== diag(W'*P[idx_y, idx_s]) + diag(C'*P[idx_c, idx_s])
        diag(P[idx_z, idx_u]) .== diag(W'*P[idx_y, idx_z]) + diag(C'*P[idx_c, idx_z])
        diag(P[idx_c, idx_u]) .== diag(W'*P[idx_y, idx_c]) + diag(C'*P[idx_c, idx_c])
        P[idx_y, 1] .== diag(P[idx_s, idx_u])
        diag(P[idx_s, idx_s]) .<= P[idx_s, 1]
        diag(W*P[idx_z, idx_s] .+ U*P[idx_x, idx_s]) .+ u.*P[idx_s, 1] .>= 0
        diag(W*P[idx_z, idx_s] .+ U*P[idx_x, idx_s]) .+ u.*P[idx_s, 1] .>= 1 .* (W*P[idx_z, 1] .+ U*P[idx_x, 1] .+ u)
        P[idx_z, 1] .>= 0
        P[idx_z, 1] .>= W*P[idx_z, 1] + U*P[idx_x, 1] + u
        diag(P[idx_z, idx_z]) .== diag(W*P[idx_z, idx_z] + U*P[idx_x, idx_z]) + u.*P[idx_z, 1]
    end)
    optimize!(model)
    return objective_value(model)
end

function solution_fix_point(x0, W, U, u; η = 1e-10, α = 1e-2, method = "forward-backward")
    if method == "forward-backward"
        d = size(U, 1); z = zeros(d,1); err = 1;
        while err > η
            z_new = (1-α)*z + α*(W*z+U*x0+u);
            z_new = z_new .* (z_new .>= 0);
            err = norm(z_new-z, 2)/norm(z_new, 2);
            z = z_new
        end
    elseif method == "Peaceman-Rachford"
        d = size(U, 1); z = zeros(d,1); y = zeros(d,1); err = 1;
        V = (Matrix(I(d))+α*(Matrix(I(d))-W))^(-1)
        while err > η
            y_new1 = 2*z-y; z_new1 = V*(y_new1+α*(U*x0+u))
            y_new2 = 2*z_new1-y_new1; z_new2 = y_new2 .* (y_new2 .>= 0)
            err = norm(z_new2-z, 2)/norm(z_new2, 2);
            z = z_new2; y = y_new2
        end
    else
        error("NotImplementedError")
    end
    return z
end

function cert(W, U, u, C, n, ϵ, set_input, set_output; me = 0.1307, std = 0.3081)
    n_suc = 0; idx_suc = [];
    for k = 1:n
        x0 = (vec(set_input[:,:,k]).-me)./std;
        for i = 1:10
            if i == set_output[k]+1
                continue
            else
                c = C[i,:] - C[set_output[k]+1,:];
                obj = cert_monDEQ(x0, W, U, u, c, ϵ; nrm = "linf")
                if obj >= 0
                    break
                elseif (i == 10 && set_output[k] <= 8) || (i == 9 && set_output[k] == 9)
                    n_suc += 1
                    idx_suc = vcat(idx_suc, k)
                end
            end
        end
    end
    ratio = n_suc/n
    return ratio
end

function cert_lip(W, U, u, C, c, n, ϵ, L, set_input, set_output; me = 0.1307, std = 0.3081)
    n_suc = 0; idx_suc = [];
    for k = 1:n
        x0 = (vec(set_input[:,:,k]).-me)./std;
        z0 = solution_fix_point(x0, W, U, u; α = 1, method = "Peaceman-Rachford");
        y0 = C*z0+c;
        margin = y0[set_output[k]+1] - maximum(y0[setdiff(1:10, [set_output[k]+1])]);
        if 2*L*ϵ < margin
            n_suc += 1
            idx_suc = vcat(idx_suc, k)
        else
            continue
        end
    end
    ratio = n_suc/n;
    return ratio
end
