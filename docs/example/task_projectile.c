/**
 * task_projectile.c — 任务 3：抛体运动落点预测
 *
 * 目标：给定初速度 (vx, vy)，预测物体落地的水平距离。
 *
 * 物理规律：从已有的 main.c 提取。
 *
 *   物体以初速度 (vx, vy) 发射（像素/秒），受重力 g = 98 影响。
 *   运动方程（步长 dt = 20ms）：
 *     vy' = vy - g × dt
 *     x'  = x  + vx × dt
 *     y'  = y  + vy × dt
 *   当 y < 0 时落地，此时的 x 即为水平位移。
 *
 * 输入：2 个值 (vx/200, vy/200)  —— 归一化到 [0.25, 1]
 * 输出：1 个值 (落地距离/1000)    —— 归一化到 ~[0, 1]
 *
 * 为什么需要归一化？
 *   - 神经网络对输入值的量级敏感。如果 vx 是 200，权重初始化是 ~0.01，
 *     加权和是 200×0.01=2，经过 Sigmoid 后接近饱和（梯度近乎 0），
 *     学习停滞。
 *   - 归一化把所有值压缩到 [0,1] 附近，让激活函数的"活跃区"能正常工作。
 *
 * 建议网络结构（与你的 main.c 一致）：
 *   nn_add_layer(nn, 2, ACT_LINEAR);    // 输入：(vx/200, vy/200)
 *   nn_add_layer(nn, 64, ACT_SIGMOID);  // 隐藏层 1
 *   nn_add_layer(nn, 128, ACT_SIGMOID); // 隐藏层 2
 *   nn_add_layer(nn, 1, ACT_LINEAR);    // 输出：落点距离
 */
#include "nn.h"
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double rand_range(double min, double max) {
    return min + (max - min) * ((double)rand() / RAND_MAX);
}

static void generate(double *input, double *target) {
    /* 随机初始速度，范围 [50, 200] */
    double vx = rand_range(50, 200);
    double vy = rand_range(50, 200);

    /* 物理仿真：迭代直到落地 */
    double x = 0, y = 0;
    double cvx = vx, cvy = vy;
    double dt = 20.0 / 1000.0;   /* 步长 20ms */
    double g  = 98.0;            /* 重力加速度 */

    while (1) {
        cvy -= g * dt;
        x   += cvx * dt;
        y   += cvy * dt;
        if (y < 0) break;
    }

    /* 归一化 */
    input[0]  = vx / 200.0;      /* vx ∈ [0.25, 1.0] */
    input[1]  = vy / 200.0;      /* vy ∈ [0.25, 1.0] */
    target[0] = x  / 1000.0;     /* 落地距离 ∈ ~[0, ~2.0] */
}

Task task_projectile = {
    .name       = "抛体运动落点预测",
    .input_dim  = 2,
    .output_dim = 1,
    .generate   = generate,
};
