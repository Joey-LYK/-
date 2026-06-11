/**
 * task_comfort.c — 案例 1：环境舒适度分类
 *
 * 目标：根据温度和湿度，判断人体舒适度。
 *
 * 输入：温度 (°C, 归一化到 [0,1]) + 湿度 (%, 归一化到 [0,1])
 * 输出：4 类 one-hot —— 热(3) / 闷(2) / 舒适(1) / 冷(0)
 *
 * 为什么这个案例有意义？
 *
 * 1. 真实需求：智能家居空调控制、仓库环境监控、农业大棚管理等都需要
 *    根据温湿度判断环境状态。
 * 2. 无法用精确公式描述：人体舒适度是主观的，受温度、湿度、风速、
 *    穿着、个体差异等影响，不存在一个数学公式能精确描述。
 * 3. 边界模糊：25°C + 70% 湿度，有人觉得闷，有人觉得还行。
 *    正是这种"模糊地带"让神经网络比 if-else 规则更有优势。
 *
 * 分类规则（简化版，加入随机噪声模拟主观性）：
 *   冷：温度 < 18°C
 *   舒适：18°C ≤ 温度 ≤ 28°C 且 湿度 ≤ 65%
 *   闷：温度 18~33°C 且 湿度 > 65%（或温度 > 28°C 且湿度 > 55%）
 *   热：温度 > 33°C
 *
 * 建议网络结构：
 *   nn_add_layer(nn, 2, ACT_LINEAR);    // 输入：温度, 湿度
 *   nn_add_layer(nn, 16, ACT_RELU);     // 隐藏层 1
 *   nn_add_layer(nn, 16, ACT_RELU);     // 隐藏层 2
 *   nn_add_layer(nn, 4, ACT_SOFTMAX);   // 输出：4 类概率
 *
 * CSV 数据集格式（可选外部数据）：
 *   第一行：2,4          (input_dim,output_dim)
 *   后续行：温度,湿度,热,闷,舒适,冷
 *   示例：25.0,50.0,0,0,1,0
 */
#include "nn.h"
#include <stdlib.h>
#include <math.h>

/* ====== 内置数据集 ======
 * 在温湿度空间均匀采样，按规则打标签。
 * 边界区域加入随机扰动，模拟真实场景的模糊性。
 */
#define DATASET_SIZE 160

static double dataset[DATASET_SIZE][2];  /* 温度, 湿度 */
static int    labels[DATASET_SIZE];      /* 0=冷, 1=舒适, 2=闷, 3=热 */

/* 归一化参数 */
#define TEMP_MIN   5.0
#define TEMP_MAX   42.0
#define HUMI_MIN   15.0
#define HUMI_MAX   100.0

static void build_dataset(void)
{
    static int built = 0;
    if (built) return;
    built = 1;

    int idx = 0;
    for (int t = 5; t <= 40; t += 3) {          /* 温度 5~40°C，步长 3 */
        for (int h = 15; h <= 100; h += 8) {     /* 湿度 15~100%，步长 8 */
            if (idx >= DATASET_SIZE) break;

            dataset[idx][0] = (double)t + ((rand() % 3 - 1) * 0.5); /* ±0.5 随机扰动 */
            dataset[idx][1] = (double)h + ((rand() % 3 - 1) * 1.0);

            double temp = dataset[idx][0];
            double humi = dataset[idx][1];

            /* 分类规则（带随机边界抖动） */
            int jitter = (rand() % 3 - 1); /* -1, 0, 1 */
            int label;

            if (temp > (33 + jitter)) {
                label = 3;  /* 热 */
            } else if (temp < (18 - jitter)) {
                label = 0;  /* 冷 */
            } else if (humi <= (65 + jitter * 3) && temp <= (28 + jitter)) {
                label = 1;  /* 舒适 */
            } else {
                label = 2;  /* 闷 */
            }

            labels[idx] = label;
            idx++;
        }
    }

    /* 补齐剩余数据（如果网格没填满） */
    while (idx < DATASET_SIZE) {
        dataset[idx][0] = 5.0 + ((double)rand() / RAND_MAX) * 37.0;
        dataset[idx][1] = 15.0 + ((double)rand() / RAND_MAX) * 85.0;

        double temp = dataset[idx][0];
        double humi = dataset[idx][1];

        if (temp > 33) labels[idx] = 3;
        else if (temp < 18) labels[idx] = 0;
        else if (humi <= 65 && temp <= 28) labels[idx] = 1;
        else labels[idx] = 2;

        idx++;
    }
}

static void generate(double *input, double *target)
{
    build_dataset();

    /* 随机选取一条数据 */
    int idx = rand() % DATASET_SIZE;

    /* 归一化到 [0, 1] */
    input[0] = (dataset[idx][0] - TEMP_MIN) / (TEMP_MAX - TEMP_MIN);
    input[1] = (dataset[idx][1] - HUMI_MIN) / (HUMI_MAX - HUMI_MIN);

    /* one-hot 编码 */
    for (int i = 0; i < 4; i++) {
        target[i] = (i == labels[idx]) ? 1.0 : 0.0;
    }
}

Task task_comfort = {
    .name       = "环境舒适度分类",
    .input_dim  = 2,
    .output_dim = 4,
    .generate   = generate,
};
