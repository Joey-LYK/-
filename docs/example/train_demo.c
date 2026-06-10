/**
 * train_demo.c — 神经网络多任务训练演示
 *
 * 编译（MSVC）：
 *   cl /nologo /W4 train_demo.c nn.c task_sin.c task_xor.c task_projectile.c
 *
 * 编译（GCC / MinGW）：
 *   gcc -Wall -O2 train_demo.c nn.c task_sin.c task_xor.c task_projectile.c -o train_demo.exe -lm
 *
 * 运行后选择要训练的任务，观察损失值随训练轮次下降。
 *
 * 学习路径建议：
 *   先跑 sin(x) → 理解基本流程
 *   再跑 XOR  → 理解"非线性"的必要性
 *   最后跑抛体 → 理解归一化和大数据量训练
 */

#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <windows.h>

/* ====== 三个任务的声明 ====== */
extern Task task_sin;
extern Task task_xor;
extern Task task_projectile;

/* ====== 工具函数 ====== */

/**
 * 训练一个任务并打印结果。
 */
static void train_task(Task *task,
                       int input_size, int *hidden_layers, int hidden_count,
                       int output_size, Activation output_act,
                       double lr, int epochs, int samples_per_epoch)
{
    /* 创建网络 */
    NeuralNet *nn = nn_create(lr);

    /* 输入层 */
    nn_add_layer(nn, input_size, ACT_LINEAR);

    /* 隐藏层 */
    for (int i = 0; i < hidden_count; i++) {
        nn_add_layer(nn, hidden_layers[i], ACT_RELU);
    }

    /* 输出层 */
    nn_add_layer(nn, output_size, output_act);

    nn_print_structure(nn);

    printf("开始训练: %s\n", task->name);
    printf("训练参数: epochs=%d, samples/epoch=%d, lr=%.4f\n\n",
           epochs, samples_per_epoch, lr);

    double *input  = (double*)malloc(sizeof(double) * task->input_dim);
    double *target = (double*)malloc(sizeof(double) * task->output_dim);
    double *output = (double*)malloc(sizeof(double) * task->output_dim);

    int report_interval = epochs / 10;
    if (report_interval < 1) report_interval = 1;

    clock_t start_time = clock();

    for (int ep = 0; ep < epochs; ep++) {
        double total_loss = 0.0;

        for (int s = 0; s < samples_per_epoch; s++) {
            task->generate(input, target);
            total_loss += nn_train_step(nn, input, target);
        }

        double avg_loss = total_loss / samples_per_epoch;

        /* 定期打印进度 */
        if ((ep + 1) % report_interval == 0 || ep == 0) {
            int bar_width = 20;
            int filled = (ep + 1) * bar_width / epochs;
            printf("\r[Epoch %4d/%4d] |", ep + 1, epochs);
            for (int b = 0; b < bar_width; b++)
                printf("%s", b < filled ? "█" : "░");
            printf("|  Loss: %10.6f", avg_loss);
            fflush(stdout);
        }
    }

    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("训练完成！用时 %.2f 秒\n\n", elapsed);

    /* ──── 验证：用几个样本测试预测准确性 ──── */
    printf("── 预测验证（10 个随机样本）──\n");
    printf("%-20s %-20s %-10s\n", "输入", "预测", "真实值");
    printf("%-20s %-20s %-10s\n", "────", "────", "──────");

    for (int i = 0; i < 10; i++) {
        task->generate(input, target);
        nn_predict(nn, input, output);

        /* 打印输入 */
        printf("[");
        for (int j = 0; j < task->input_dim; j++) {
            printf("%.3f%s", input[j],
                   j < task->input_dim - 1 ? ", " : "]");
        }
        printf("  →  ");

        /* 打印预测 */
        printf("[");
        for (int j = 0; j < task->output_dim; j++) {
            printf("%.4f%s", output[j],
                   j < task->output_dim - 1 ? ", " : "]");
        }
        printf("  真实: [");
        for (int j = 0; j < task->output_dim; j++) {
            printf("%.4f%s", target[j],
                   j < task->output_dim - 1 ? ", " : "]");
        }
        printf("\n");
    }

    printf("\n");

    free(input);
    free(target);
    free(output);
    nn_destroy(nn);
}

/* ====== 主菜单 ====== */

int main(void) {
    SetConsoleOutputCP(65001);  /* 设置控制台输出编码为 UTF-8，解决中文乱码 */
    srand((unsigned int)time(NULL));

    printf("\n");
    printf("╔══════════════════════════════════════════╗\n");
    printf("║   神经网络学习平台 — 多任务训练演示       ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  1. sin(x) 函数拟合 （入门）             ║\n");
    printf("║  2. XOR 逻辑门     （理解非线性）        ║\n");
    printf("║  3. 抛体运动        （回归实战）         ║\n");
    printf("║  0. 退出                                 ║\n");
    printf("╚══════════════════════════════════════════╝\n");
    printf("\n请选择任务 [0-3]: ");

    int choice;
    if (scanf("%d", &choice) != 1) {
        printf("无效输入\n");
        return 1;
    }

    switch (choice) {
        case 1: {
            /* sin(x) 拟合：
             *   1 输入 → 2×16 隐藏层(ReLU) → 1 输出(Linear)
             *   简单任务，5000 轮足以收敛
             */
            int hidden[] = {16, 16};
            train_task(&task_sin, 1, hidden, 2, 1, ACT_LINEAR,
                       0.01, 5000, 50);
            break;
        }
        case 2: {
            /* XOR 逻辑：
             *   2 输入 → 4 隐藏层(Sigmoid) → 1 输出(Sigmoid)
             *   Sigmoid 输出适合 [0,1] 的逻辑结果
             */
            int hidden[] = {4};
            train_task(&task_xor, 2, hidden, 1, 1, ACT_SIGMOID,
                       0.1, 2000, 20);
            break;
        }
        case 3: {
            /* 抛体运动：
             *   2 输入 → 64→128 隐藏层(Sigmoid) → 1 输出(Linear)
             *   与你的 main.c 完全相同结构，学习率 0.005
             */
            int hidden[] = {64, 128};
            train_task(&task_projectile, 2, hidden, 2, 1, ACT_LINEAR,
                       0.005, 100, 500);
            break;
        }
        case 0:
            printf("再见！\n");
            return 0;
        default:
            printf("无效选择: %d\n", choice);
            return 1;
    }

    return 0;
}
