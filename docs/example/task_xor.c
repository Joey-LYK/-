/**
 * task_xor.c — 任务 2：XOR 逻辑门
 *
 * 目标：让网络学会 XOR（异或）的真值表。
 *
 * XOR 真值表：
 *   (0,0) → 0    (0,1) → 1
 *   (1,0) → 1    (1,1) → 0
 *
 * 为什么 XOR 是神经网络的"存在理由"？
 *
 * 单层感知器（无隐藏层）无法解决 XOR——因为 XOR 不是线性可分的。
 * 你在 2D 平面上画 (0,0)(0,1)(1,0)(1,1) 四个点，无法用一条直线把 0 和 1 分开。
 *
 * 加上隐藏层后，网络实际上是在"折叠"输入空间，把线性不可分的问题
 * 映射到高维空间变成线性可分。
 *
 * 建议网络结构：
 *   nn_add_layer(nn, 2, ACT_LINEAR);   // 输入：(x1, x2)
 *   nn_add_layer(nn, 4, ACT_SIGMOID);  // 隐藏层：Sigmoid 做非线性变换
 *   nn_add_layer(nn, 1, ACT_SIGMOID);  // 输出：0 或 1
 *
 * 训练后会看到：损失从 ~0.25 降到 ~0.001 以下。
 * 如果去掉隐藏层（2→1），损失停在 ~0.25——这就是"不能线性可分"的证据。
 */
#include "nn.h"
#include <stdlib.h>

static void generate(double *input, double *target) {
    int a = rand() % 2;
    int b = rand() % 2;
    input[0]  = (double)a;
    input[1]  = (double)b;
    target[0] = (double)(a ^ b);  /* XOR */
}

Task task_xor = {
    .name       = "XOR 逻辑门",
    .input_dim  = 2,
    .output_dim = 1,
    .generate   = generate,
};
