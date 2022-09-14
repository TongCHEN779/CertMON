%% Compute outer approximation ellipsoids
load('MNIST_test.mat');
me = 0.1307; std = 0.3081; eps = 0.01; m = 20; nrm = "linf"; 
load(sprintf('deq_MNIST_SingleFcNet_m=%d.mat', m)); 
p0 = size(U,2); p = size(U,1); d = size(C,1); n = 1+p0+p; 
W = (1-m)*eye(p)-A'*A+B-B';
for k = 1:100
    digit = images(:,:,k);
    str = sprintf("params_monDEQ/Ellip_momDEQ_eg%d_m%d_%s_eps%.2f.mat", k, m, nrm, eps);
    eps = eps/std; x0 = (vec(double(digit)')/255-me)/std;
    cvx_begin
    cvx_solver mosek
    cvx_precision low
    variable Q(d,d) symmetric;
    variables b(d,1) lambda(p,1);
    variable nu(p,1) nonnegative;
    variable eta(p,1) nonnegative;
    variable gama(p*(p-1)/2) nonnegative;
    E = eye(p); T = zeros(p);
    for i = 1:p
        T = T + lambda(i)*(E(i,:)'*E(i,:));
    end
    M3 = sparse([zeros(p) T -nu; T -2*T nu+eta; -nu' nu'+eta' 0]);
    B_M3 = [U W u; zeros(p,p0) eye(p) zeros(p,1); zeros(1,p0+p) 1];
    M3 = B_M3'*M3*B_M3;
    if nrm == "l2"
        variable delta(1,1) nonnegative;
        M2 = delta * [-eye(p0) zeros(p0,p) x0; zeros(p,n); x0' zeros(1,p) (eps^2-x0'*x0)];
    elseif nrm == "linf"
        variable delta(p0,1) nonnegative;
        M2 = [-diag(delta) zeros(p0,p) delta.*x0; zeros(p,n); delta'.*x0' zeros(1,p) delta'*(eps^2-x0.^2)];
    end
    e = sparse([zeros(p0+p,1);1]); MM = M2+M3;
    M = sparse([[MM-e*e' [zeros(p0,d); C'*Q; c'*Q+b']]; [zeros(d,p0) Q*C Q*c+b -eye(d)]]);
    maximize(log_det(-Q));
    -M == semidefinite(n+d);
    cvx_end
end
%% Compute the ratio of certified examples
m = 20; nrm = "linf"; eps = 0.01;
n = 100; n_suc = 0; idx_suc = [];
for i = 1:n
    load(sprintf('params_ellips/Ellip_momDEQ_eg%d_m%d_%s_eps%.2f.mat', i, m, nrm, eps)); lbl = labels(i);
    for j = 1:10
        if j ~= lbl+1
            cvx_begin quiet
                cvx_solver mosek
                variable x(10)
                minimize(x(lbl+1)-x(j))
                subject to
                    x'*Q^2*x+2*b'*Q*x+b'*b-1 <= 0;
            cvx_end
            if (x(lbl+1)-x(j)) <= 0
                break
            elseif (j == 10 && lbl <= 8) || (j == 9 && lbl == 9)
                n_suc = n_suc + 1;
                idx_suc = [idx_suc; i];
            end
        end
    end
end
ratio = n_suc / n;
%% Plot the landscape of projections
eps = 0.1; k = 1; nrm = "l2"; j = 4;
digit = images(:,:,k); digit = (vec(double(digit)')/255-me)/std; lbl = labels(k);
load(sprintf('params_ellips/Ellip_momDEQ_eg%d_m%d_%s_eps%.2f.mat', k, m, nrm, eps));
load(sprintf('border_%s_%.2f.mat', nrm, eps)); load(sprintf('attack_%s_%.2f.mat', nrm, eps));
eps = eps/std; eta = 1e-2; alpha = 1; method = "Peaceman-Rachford";
theta = 0:0.01:2*pi+0.01; y0 = [cos(theta); sin(theta)];
set1 = [lbl+1,j]; set2 = 1:d; set2 = set2(intersect(set2~=lbl+1, set2~=j)); 
QQ = Q^2; J = QQ(set1, set1); K = QQ(set2, set2); L = QQ(set1, set2);
Q1 = (J-L*K^(-1)*L')^(1/2); b1 = Q^(-1)*b;
x0 = Q1^(-1)*y0-b1(set1);
axis equal
plot(x0(1,:), x0(2,:), 'b')
hold on
M = 50; y = zeros(d,M+size(border,3));
for i = 1:M
    if nrm == "l2"
        y0 = normrnd(0,1, [p0,1]); y0 = y0/sqrt(y0'*y0);
        x = y0*eps+digit;
    elseif nrm == "linf"
        y0 = normrnd(0,1, [p0,1]); y0 = y0/abs(max(y0));
        x = y0*eps+digit;
    end
    z = solution_fix_point(x, W, U, u, eta, alpha, method);
    y(:,i) = C*z+c;
end
for i = 1:size(border,3)
    x = vec(border(:,:,i)');
    z = solution_fix_point(x, W, U, u, eta, alpha, method);
    y(:,M+i) = C*z+c;
end
scatter(y(lbl+1,:), y(j,:), 'red', '.')
y = zeros(d,size(attack,3));
for i = 1:size(attack,3)
    x = vec(attack(:,:,i)');
    z = solution_fix_point(x, W, U, u, eta, alpha, method);
    y(:,i) = C*z+c;
end
scatter(y(lbl+1,:), y(j,:), 'black', 'o')
if nrm == "linf"
    line([-15,25], [-15,25], 'Color', 'blue', 'LineStyle', '--')
else
    line([3,4], [3,4], 'Color', 'blue', 'LineStyle', '--')
end
legend('Ellipsiod', 'Outputs', 'Attacks', 'Threshold', 'FontSize',12);
hold off
xlabel(sprintf('label %d', lbl), 'FontSize',14); ylabel(sprintf('label %d', j-1), 'FontSize',14);
title('Outputs, outer ellipsoid and threshold line', 'FontSize',14);