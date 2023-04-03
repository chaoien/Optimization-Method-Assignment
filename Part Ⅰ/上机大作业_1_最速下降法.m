% 上机大作业_1_最速下降法

% 学号
id = 32206192;

% 维数
n = 2 * 100 + mod(id,100);

% 参数
a = unidrnd(10,n,1);
G = a * transpose(a) + unidrnd(2) * eye(n);
b = 0.5 * G * ones(n,1);

% 目标函数
f = @(x) 1/2 * transpose(x) * G * x + transpose(b) * x;

% 梯度
grad = @(x) G * x + b;

% 初始化
x = zeros(n, 1);

% 最大迭代次数
max_iter = 1000;

% 精度
tor = 1e-6;

% 求解
for k = 1:max_iter

    % 判断是否满足精度
    if norm(grad(x)) < tor
        break;
    end

    % 下降方向
    d = -grad(x);
    
    % 步长
    alpha = (transpose(grad(x)) * grad(x)) / (transpose(grad(x)) * G * grad(x));
    
    % 更新点
    x = x + alpha * d;

end

% 输出结果
fprintf('迭代次数为 %d\n', k);
fprintf('最小值为 %.3f\n', f(x));
