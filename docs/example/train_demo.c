/**
 * train_demo.c — 神经网络多任务训练演示
 *
 * 编译（MSVC）：
 *   cl /nologo /W4 train_demo.c nn.c ^
 *     task_sin.c task_xor.c task_projectile.c ^
 *     task_comfort.c task_sevenseg.c task_temp_comp.c ^
 *     task_battery.c task_gesture.c task_voice.c
 *
 * 编译（GCC / MinGW）：
 *   gcc -Wall -O2 train_demo.c nn.c \
 *     task_sin.c task_xor.c task_projectile.c \
 *     task_comfort.c task_sevenseg.c task_temp_comp.c \
 *     task_battery.c task_gesture.c task_voice.c \
 *     -o train_demo.exe -lm
 *
 * 学习路径建议：
 *   基础入门：sin(x) → XOR → 舒适度分类 → 数码管识别
 *   进阶实战：温度补偿 → 电池SOC
 *   高级应用：动作识别 → 语音命令
 */

#include "nn.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <windows.h>

/* ====== 任务声明 ====== */
/* 原有 3 个基础任务 */
extern Task task_sin;
extern Task task_xor;
extern Task task_projectile;
/* 新增 6 个实战案例 */
extern Task task_comfort;
extern Task task_sevenseg;
extern Task task_temp_comp;
extern Task task_battery;
extern Task task_gesture;
extern Task task_voice;

/* ====== 工具函数 ====== */

/**
 * 训练一个任务并打印结果。
 */
static void train_task(Task *task,
                       int input_size, int *hidden_layers, int hidden_count,
                       Activation hidden_act,
                       int output_size, Activation output_act,
                       double lr, int epochs, int samples_per_epoch)
{
    /* 创建网络 */
    NeuralNet *nn = nn_create(lr);

    /* 输入层 */
    nn_add_layer(nn, input_size, ACT_LINEAR);

    /* 隐藏层 */
    for (int i = 0; i < hidden_count; i++) {
        nn_add_layer(nn, hidden_layers[i], hidden_act);
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

        /* 定期打印进度（换行打印，保留历史记录） */
        if ((ep + 1) % report_interval == 0 || ep == 0) {
            int bar_width = 20;
            int filled = (ep + 1) * bar_width / epochs;
            printf("[Epoch %4d/%4d] |", ep + 1, epochs);
            for (int b = 0; b < bar_width; b++)
                printf("%s", b < filled ? "█" : "░");
            printf("|  Loss: %10.6f\n", avg_loss);
            fflush(stdout);
        }
    }

    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("训练完成！用时 %.2f 秒\n\n", elapsed);

    /* ──── 验证：用几个样本测试预测准确性 ──── */
    int is_softmax = (output_act == ACT_SOFTMAX);
    int correct = 0;
    int test_count = 10;

    if (is_softmax) {
        printf("── 分类验证（%d 个随机样本）──\n", test_count);
        printf("%-25s %-12s %-12s\n", "输入", "预测类别", "真实类别");
        printf("%-25s %-12s %-12s\n", "───────", "────────", "────────");
    } else {
        printf("── 回归验证（%d 个随机样本）──\n", test_count);
        printf("%-25s %-20s %-20s\n", "输入", "预测", "真实值");
        printf("%-25s %-20s %-20s\n", "───────", "────", "──────");
    }

    for (int i = 0; i < test_count; i++) {
        task->generate(input, target);
        nn_predict(nn, input, output);

        /* 打印输入 */
        printf("[");
        for (int j = 0; j < task->input_dim; j++) {
            printf("%.3f%s", input[j],
                   j < task->input_dim - 1 ? ", " : "]");
        }

        if (is_softmax) {
            /* Softmax 输出：找 argmax */
            int pred_class = 0, true_class = 0;
            for (int j = 1; j < task->output_dim; j++) {
                if (output[j] > output[pred_class])  pred_class  = j;
                if (target[j] > target[true_class])  true_class = j;
            }
            if (pred_class == true_class) correct++;
            printf("  →  类别 %-2d    真实: 类别 %-2d", pred_class, true_class);
        } else {
            /* 回归输出：打印数值 */
            printf("  →  [");
            for (int j = 0; j < task->output_dim; j++) {
                printf("%.4f%s", output[j],
                       j < task->output_dim - 1 ? ", " : "]");
            }
            printf("  真实: [");
            for (int j = 0; j < task->output_dim; j++) {
                printf("%.4f%s", target[j],
                       j < task->output_dim - 1 ? ", " : "]");
            }
        }
        printf("\n");
    }

    if (is_softmax) {
        printf("\n准确率: %d / %d (%.1f%%)\n", correct, test_count,
               100.0 * correct / test_count);
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
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║   神经网络学习平台 — 多任务训练演示                    ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║ ── 基础入门 ────────────────────────────────────────────  ║\n");
    printf("║  1. sin(x) 函数拟合     （理解基本训练流程）           ║\n");
    printf("║  2. XOR 逻辑门          （理解非线性的必要性）         ║\n");
    printf("║  3. 抛体运动            （理解归一化和大数据量）       ║\n");
    printf("║ ── 基础实战 ────────────────────────────────────────────  ║\n");
    printf("║  4. 环境舒适度分类      （真实场景分类入门）           ║\n");
    printf("║  5. 数码管字符识别      （传感器噪声容忍）             ║\n");
    printf("║ ── 进阶实战 ────────────────────────────────────────────  ║\n");
    printf("║  6. 传感器温度补偿      （回归 + 非线性校正）          ║\n");
    printf("║  7. 电池SOC估计         （嵌入式BMS核心功能）          ║\n");
    printf("║ ── 高级应用 ────────────────────────────────────────────  ║\n");
    printf("║  8. 手势动作识别        （IMU时序特征 + 分类）         ║\n");
    printf("║  9. 语音命令识别        （音频特征 + 多类分类）         ║\n");
    printf("║ ──────────────────────────────────────────────────────  ║\n");
    printf("║  0. 退出                                          ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("\n请选择任务 [0-9]: ");

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
            train_task(&task_sin, 1, hidden, 2, ACT_RELU, 1, ACT_LINEAR,
                       0.01, 5000, 50);
            break;
        }
        case 2: {
            /* XOR 逻辑：
             *   2 输入 → 4 隐藏层(Sigmoid) → 1 输出(Sigmoid)
             *   Sigmoid 输出适合 [0,1] 的逻辑结果
             */
            int hidden[] = {4};
            train_task(&task_xor, 2, hidden, 1, ACT_SIGMOID, 1, ACT_SIGMOID,
                       0.1, 2000, 20);
            break;
        }
        case 3: {
            /* 抛体运动：
             *   2 输入 → 64→128 隐藏层(Sigmoid) → 1 输出(Linear)
             *   与你的 main.c 完全相同结构，学习率 0.005
             */
            int hidden[] = {64, 128};
            train_task(&task_projectile, 2, hidden, 2, ACT_SIGMOID, 1, ACT_LINEAR,
                       0.005, 100, 500);
            break;
        }
        /* ── 基础实战 ── */
        case 4: {
            /* 环境舒适度分类：
             *   2 输入 → 16→16 隐藏层(ReLU) → 4 输出(Softmax)
             *   分类任务，需要较多轮次收敛
             */
            int hidden[] = {16, 16};
            train_task(&task_comfort, 2, hidden, 2, ACT_RELU, 4, ACT_SOFTMAX,
                       0.01, 3000, 50);
            break;
        }
        case 5: {
            /* 数码管字符识别：
             *   7 输入 → 32→16 隐藏层(ReLU) → 10 输出(Softmax)
             *   10% 噪声容错训练
             */
            int hidden[] = {32, 16};
            train_task(&task_sevenseg, 7, hidden, 2, ACT_RELU, 10, ACT_SOFTMAX,
                       0.01, 3000, 30);
            break;
        }
        /* ── 进阶实战 ── */
        case 6: {
            /* 传感器温度补偿：
             *   3 输入 → 32→16 隐藏层(ReLU) → 1 输出(Linear)
             *   回归任务，非线性补偿
             */
            int hidden[] = {32, 16};
            train_task(&task_temp_comp, 3, hidden, 2, ACT_RELU, 1, ACT_LINEAR,
                       0.005, 5000, 100);
            break;
        }
        case 7: {
            /* 电池 SOC 估计：
             *   3 输入 → 64→32→16 隐藏层(ReLU) → 1 输出(Linear)
             *   深层网络拟合非线性 OCV-SOC 曲线
             */
            int hidden[] = {64, 32, 16};
            train_task(&task_battery, 3, hidden, 3, ACT_RELU, 1, ACT_LINEAR,
                       0.005, 5000, 100);
            break;
        }
        /* ── 高级应用 ── */
        case 8: {
            /* 手势动作识别：
             *   12 输入 → 32→32 隐藏层(ReLU) → 4 输出(Softmax)
             *   IMU 统计特征分类
             */
            int hidden[] = {32, 32};
            train_task(&task_gesture, 12, hidden, 2, ACT_RELU, 4, ACT_SOFTMAX,
                       0.005, 5000, 100);
            break;
        }
        case 9: {
            /* 语音命令识别：
             *   13 输入 → 64→32 隐藏层(ReLU) → 5 输出(Softmax)
             *   模拟 MFCC 特征的多类分类
             */
            int hidden[] = {64, 32};
            train_task(&task_voice, 13, hidden, 2, ACT_RELU, 5, ACT_SOFTMAX,
                       0.005, 5000, 200);
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
