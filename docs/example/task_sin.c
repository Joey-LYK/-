/**
 * task_sin.c — 任务 1：sin(x) 函数拟合
 *
 * 目标：让网络学会 sin(x) 的曲线形状。
 *
 * 输入：1 个值 x ∈ [0, 2π]
 * 输出：1 个值 y ≈ sin(x)
 *
 * 为什么这是最好的入门案例？
 *
 * 1. 输入只有 1 维：可视化时就是 x→y 一条曲线，一目了然。
 * 2. 数据无限生成：rand() × 2π→sin()，想要多少有多少。
 * 3. 立即可验证：训练后的网络画出来应该是一条光滑的 sin 线。
 * 4. 调参效果明显：层太少→拟合差，层太多→过拟合（每个点都对但中间乱晃）。
 *
 * 建议网络结构：
 *   nn_add_layer(nn, 1, ACT_LINEAR);   // 输入：x
 *   nn_add_layer(nn, 16, ACT_RELU);    // 隐藏层
 *   nn_add_layer(nn, 16, ACT_RELU);    // 隐藏层
 *   nn_add_layer(nn, 1, ACT_LINEAR);   // 输出：sin(x)
 */
#include "nn.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void generate(double *input, double *target) {
    double x = ((double)rand() / RAND_MAX) * 2.0 * M_PI;
    input[0]  = x;
    target[0] = sin(x);
}

Task task_sin = {
    .name       = "sin(x) 函数拟合",
    .input_dim  = 1,
    .output_dim = 1,
    .generate   = generate,
};
