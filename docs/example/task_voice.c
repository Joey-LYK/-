/**
 * task_voice.c — 案例 6：语音命令识别（简化版）
 *
 * 目标：根据音频信号的简化声学特征，识别语音命令。
 *
 * 输入：13 个特征（均归一化到 [0,1]），模拟 MFCC 简化特征
 *   [0~12]：13 个频带的能量系数（类似 MFCC 的前 13 维）
 *
 * 输出：5 类 one-hot
 *   类别 0："开灯"   — 低频突出、能量中等
 *   类别 1："关灯"   — 中频突出、能量中等
 *   类别 2："增大"   — 高频爬升、能量较大
 *   类别 3："减小"   — 高频下降、能量较大
 *   类别 4：噪声/无关 — 频谱平坦、能量较低
 *
 * 为什么这个案例有意义？
 *
 * 1. "Hey Siri" 的简化版：智能音箱、手机语音助手、车载语音控制的核心功能。
 *    嵌入式端语音识别是当前最热门的 AI 应用方向之一。
 * 2. 特征工程的核心案例：音频波形不能直接喂给 MLP（维度太高、变长）。
 *    必须先提取 MFCC 特征——这展示了"为什么需要特征工程"。
 * 3. 从原理到部署的桥梁：实际产品中用 TensorFlow Lite Micro 或 CMSIS-NN
 *    部署，但原理和这里完全一致。
 *
 * 数据生成策略：
 *   不同语音命令有不同的频谱特征模式。
 *   "开" = 元音 /ai/，低频共振峰突出
 *   "关" = 元音 /uan/，中频突出
 *   "增" = 辅音 /z/ + 元音 /eng/，高频成分
 *   "减" = 辅音 /j/ + 元音 /ian/，中高频
 *   噪声 = 白噪声，频谱平坦
 *   每个命令的频谱特征用数学模型模拟，加入随机变化。
 *
 * 建议网络结构：
 *   nn_add_layer(nn, 13, ACT_LINEAR);    // 输入：13 个频带能量
 *   nn_add_layer(nn, 64, ACT_RELU);      // 隐藏层 1
 *   nn_add_layer(nn, 32, ACT_RELU);      // 隐藏层 2
 *   nn_add_layer(nn, 5, ACT_SOFTMAX);    // 输出：5 类概率
 *
 * CSV 数据集格式（可选外部数据）：
 *   第一行：13,5
 *   后续行：13个MFCC特征,5个one-hot值
 */
#include "nn.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define N_FEATURES  13
#define N_CLASSES   5

/* ====== 高斯随机数 ====== */
static double gauss_rand(double mean, double std)
{
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 < 1e-10) u1 = 1e-10;
    return mean + sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2) * std;
}

/**
 * 生成某个语音命令的模拟 MFCC 特征
 *
 * 核心思想：不同语音有不同的共振峰（formant）分布。
 * 共振峰决定了元音的特征。辅音则贡献高频噪声包络。
 *
 * 这里用简化的高斯峰模型来模拟频谱包络：
 *   每个命令有 2-3 个共振峰，分布在不同频率位置。
 *   MFCC 特征 = 对数频谱的 DCT 系数 ≈ 频谱包络的低维表示。
 */
static void gen_voice_features(int command, double *features)
{
    /* 频谱包络模型：用几个高斯峰的和来模拟 */
    double peaks[N_FEATURES];

    /* 初始化为低基底能量 + 噪声 */
    double base_energy = 0.1 + ((double)rand() / RAND_MAX) * 0.1;
    for (int i = 0; i < N_FEATURES; i++) {
        peaks[i] = base_energy + gauss_rand(0, 0.02);
    }

    switch (command) {
        case 0: /* "开灯" (kai deng) — /ai/ 元音：低频共振峰强 */
            /* F1 ≈ 频带 2-3, F2 ≈ 频带 6-7 */
            for (int i = 0; i < N_FEATURES; i++) {
                double f = (double)i / N_FEATURES;
                peaks[i] += 0.6 * exp(-pow((f - 0.2), 2) / (2 * 0.05 * 0.05));  /* F1 */
                peaks[i] += 0.4 * exp(-pow((f - 0.5), 2) / (2 * 0.08 * 0.08));  /* F2 */
            }
            break;

        case 1: /* "关灯" (guan deng) — /uan/ 元音：中频共振峰 */
            /* F1 ≈ 频带 3-4, F2 ≈ 频带 7-8 */
            for (int i = 0; i < N_FEATURES; i++) {
                double f = (double)i / N_FEATURES;
                peaks[i] += 0.5 * exp(-pow((f - 0.3), 2) / (2 * 0.06 * 0.06));  /* F1 */
                peaks[i] += 0.5 * exp(-pow((f - 0.6), 2) / (2 * 0.07 * 0.07));  /* F2 */
            }
            break;

        case 2: /* "增大" (zeng da) — /z/ 辅音 + /eng/ 元音：高频突出 */
            /* 辅音噪声 + F1 偏低, F2 高 */
            for (int i = 0; i < N_FEATURES; i++) {
                double f = (double)i / N_FEATURES;
                peaks[i] += 0.3 * exp(-pow((f - 0.2), 2) / (2 * 0.05 * 0.05));  /* F1 */
                peaks[i] += 0.5 * exp(-pow((f - 0.7), 2) / (2 * 0.06 * 0.06));  /* F2 */
                peaks[i] += 0.3 * f;  /* 高频偏移：辅音贡献 */
            }
            break;

        case 3: /* "减小" (jian xiao) — /j/ 辅音 + /ian/ 元音：中高频 */
            /* /j/ 的高频前导 + /ian/ 的中频共振峰 */
            for (int i = 0; i < N_FEATURES; i++) {
                double f = (double)i / N_FEATURES;
                peaks[i] += 0.2 * exp(-pow((f - 0.15), 2) / (2 * 0.04 * 0.04)); /* F1 */
                peaks[i] += 0.5 * exp(-pow((f - 0.55), 2) / (2 * 0.06 * 0.06)); /* F2 */
                peaks[i] += 0.4 * exp(-pow((f - 0.85), 2) / (2 * 0.05 * 0.05)); /* F3 */
            }
            break;

        case 4: /* 噪声/无关 */
            /* 频谱平坦，无明显共振峰 */
            for (int i = 0; i < N_FEATURES; i++) {
                peaks[i] += gauss_rand(0, 0.05);
            }
            break;
    }

    /* 添加个体差异和通道噪声 */
    double speaker_var = 0.8 + ((double)rand() / RAND_MAX) * 0.4; /* 声音大小差异 */
    for (int i = 0; i < N_FEATURES; i++) {
        features[i] = peaks[i] * speaker_var + gauss_rand(0, 0.03);
        if (features[i] < 0) features[i] = 0;
    }
}

static void generate(double *input, double *target)
{
    /* 随机选择一个命令 */
    int command = rand() % N_CLASSES;

    /* 生成特征 */
    double raw_features[N_FEATURES];
    gen_voice_features(command, raw_features);

    /* 归一化到 [0, 1] */
    /* 预估特征范围 [0, ~1.5] */
    for (int i = 0; i < N_FEATURES; i++) {
        input[i] = raw_features[i] / 1.5;
        if (input[i] > 1.0) input[i] = 1.0;
    }

    /* one-hot 编码 */
    for (int i = 0; i < N_CLASSES; i++) {
        target[i] = (i == command) ? 1.0 : 0.0;
    }
}

Task task_voice = {
    .name       = "语音命令识别",
    .input_dim  = N_FEATURES,
    .output_dim = N_CLASSES,
    .generate   = generate,
};
