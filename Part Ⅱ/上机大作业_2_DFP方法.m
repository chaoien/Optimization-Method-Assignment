% 上机大作业_2_DFP方法

% 目标函数
f = @(x) (x(1) + 10 * x(2)) ^ 2  + ((x(3) - x(4)) * sqrt(5)) ^ 2 + ((x(2) - x(3) * 2) ^ 2) ^ 2 + ((x(1) - x(4)) ^ 2 * sqrt(10)) ^ 2;

% 初始点
x = [3; -1; 0; 1];

% 精度
tor = 1e-4;

% 最大迭代次数
max_iter = 10000;

% Wolfe-Powell 准则参数
c1 = 1e-4;
c2 = 0.9;
alpha = 1;
rho = 0.5;

% 初始化梯度和函数值
g = grad(f, x);
H = eye(4);

% 初始化梯度和函数值
g = grad(f, x);
fx = f(x);
gx = transpose(g) * g;

% 迭代
for k = 1:max_iter

    % 判断是否满足精度
    if norm(g) < tor
        break;
    end

    % 尝试步长
    x_try = x - alpha * H * g;
    fx_try = f(x_try);
    g_try = grad(f, x_try);
    gx_try = transpose(g_try) * g_try;

    % Wolfe-Powell 准则
    while fx_try > fx + c1 * alpha * gx || transpose(g_try) * g < c2 * gx_try
        alpha = rho * alpha;
        x_try = x - alpha * g;
        fx_try = f(x_try);
        g_try = grad(f, x_try);
        gx_try = transpose(g_try) * g_try;
    end

    % 更新近似 Hessian 矩阵
    del_x = x_try - x;
    del_y = g_try - g;
    % Broyden – Fletcher 公式太长，分开写防出错
    BF1 = (del_x * transpose(del_x)) / (transpose(del_x) * del_y);
    BF2 = (H * del_y * transpose(del_y) * H) / (transpose(del_y) * H * del_y);
    H = H + BF1 - BF2;

    % 更新梯度和函数值
    x = x_try;
    fx = fx_try;
    g = g_try;
    gx = gx_try;


end

% 输出结果
fprintf('迭代次数为 %d\n', k);
fprintf('最小值为 %.3f\n', fx);

% (引)计算梯度的函数
function g = grad(f, x)
    h = 1e-6;
    g = zeros(size(x));
    for i = 1:length(x)
        x1 = x;
        x2 = x;
        x1(i) = x1(i) - h;
        x2(i) = x2(i) + h;
        g(i) = (f(x2) - f(x1)) / (2 * h);
    end
end