/**
 * task_temp_comp.c — 案例 3：传感器温度补偿
 *
 * 目标：根据 ADC 原始读数、环境参考温度和供电电压，推算传感器的真实温度。
 *
 * 输入：3 个值（均归一化到 [0,1]）
 *   [0] ADC 原始值
 *   [1] 环境参考温度（如外壳温度）
 *   [2] 供电电压
 *
 * 输出：1 个值 —— 补偿后的真实温度（归一化到 [0,1]，映射回实际温度范围）
 *
 * 为什么这个案例有意义？
 *
 * 1. 嵌入式核心需求：NTC 热敏电阻、DS18B20 等温度传感器在实际使用中受供电
 *    电压波动、自发热、环境温度梯度影响，ADC 读数与真实温度之间存在非线性偏差。
 * 2. 传统方案：查表 + 线性插值，需要人工标定，精度有限。
 * 3. 神经网络方案：用标定数据训练网络，自动学习非线性补偿曲线，
 *    精度更高且无需人工编写补偿公式。
 *
 * 模拟模型：
 *   真实温度 T ∈ [-20°C, 80°C]
 *   ADC 理想值 = (T + 20) / 100 × 4095  （12-bit ADC）
 *   供电电压影响：Vcc 偏离 3.3V 时 ADC 按比例缩放
 *   自发热效应：与环境温度差成正比的小偏移
 *   非线性误差：温度两端的灵敏度下降（热敏电阻特性）
 *   噪声：高斯随机噪声
 *
 * 建议网络结构：
 *   nn_add_layer(nn, 3, ACT_LINEAR);    // 输入：ADC, 参考温度, 电压
 *   nn_add_layer(nn, 32, ACT_RELU);     // 隐藏层 1
 *   nn_add_layer(nn, 16, ACT_RELU);     // 隐藏层 2
 *   nn_add_layer(nn, 1, ACT_LINEAR);    // 输出：补偿后温度
 *
 * CSV 数据集格式（可选外部数据）：
 *   第一行：3,1
 *   后续行：adc_raw,ref_temp,voltage,真实温度
 */
#include "nn.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ====== 模拟参数 ====== */
#define REAL_TEMP_MIN  (-20.0)   /* 真实温度范围 */
#define REAL_TEMP_MAX  (80.0)
#define VCC_NOMINAL    (3.3)     /* 标称供电电压 */
#define ADC_MAX        (4095.0)  /* 12-bit ADC 满量程 */
#define SELF_HEAT_K    (0.3)     /* 自发热系数 (°C/°C) */
#define NOISE_STD      (3.0)     /* ADC 噪声标准差 (LSB) */

/**
 * Box-Muller 变换生成高斯随机数
 */
static double gauss_rand(double mean, double std)
{
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 < 1e-10) u1 = 1e-10;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mean + z * std;
}

static void generate(double *input, double *target)
{
    /* 1. 随机生成真实温度 */
    double real_temp = REAL_TEMP_MIN + ((double)rand() / RAND_MAX) * (REAL_TEMP_MAX - REAL_TEMP_MIN);

    /* 2. 随机生成环境参考温度（与真实温度接近但有偏差） */
    double ref_temp = real_temp + gauss_rand(0, 5.0);
    ref_temp = REAL_TEMP_MIN + fmod(fabs(ref_temp - REAL_TEMP_MIN), (REAL_TEMP_MAX - REAL_TEMP_MIN)) + REAL_TEMP_MIN;

    /* 3. 随机供电电压（3.0V ~ 3.6V） */
    double vcc = 3.0 + ((double)rand() / RAND_MAX) * 0.6;

    /* 4. 模拟 ADC 读数 */
    /* 4a. 理想 ADC 值 */
    double adc_ideal = (real_temp - REAL_TEMP_MIN) / (REAL_TEMP_MAX - REAL_TEMP_MIN) * ADC_MAX;

    /* 4b. 供电电压影响：ADC 读数按 Vcc/Vcc_nominal 缩放 */
    double voltage_factor = vcc / VCC_NOMINAL;
    double adc_voltage = adc_ideal * voltage_factor;

    /* 4c. 自发热效应：传感器测量自身温度 + 环境热传导 */
    double self_heat = SELF_HEAT_K * (real_temp - ref_temp);
    double adc_selfheat = adc_voltage + self_heat * (ADC_MAX / (REAL_TEMP_MAX - REAL_TEMP_MIN));

    /* 4d. 非线性误差：温度两端灵敏度下降（模拟 NTC 特性） */
    double temp_center = (REAL_TEMP_MIN + REAL_TEMP_MAX) / 2.0;
    double temp_norm = (real_temp - temp_center) / ((REAL_TEMP_MAX - REAL_TEMP_MIN) / 2.0);
    double nonlin_error = -15.0 * temp_norm * temp_norm * (ADC_MAX / (REAL_TEMP_MAX - REAL_TEMP_MIN));

    /* 4e. 添加噪声 */
    double adc_noisy = adc_selfheat + nonlin_error + gauss_rand(0, NOISE_STD);

    /* 4f. 钳位 */
    if (adc_noisy < 0) adc_noisy = 0;
    if (adc_noisy > ADC_MAX) adc_noisy = ADC_MAX;

    /* 归一化输出 */
    input[0] = adc_noisy / ADC_MAX;                        /* ADC 值 [0, 1] */
    input[1] = (ref_temp - REAL_TEMP_MIN) / (REAL_TEMP_MAX - REAL_TEMP_MIN);  /* 参考温度 [0, 1] */
    input[2] = (vcc - 3.0) / 0.6;                        /* 电压 [0, 1] */
    target[0] = (real_temp - REAL_TEMP_MIN) / (REAL_TEMP_MAX - REAL_TEMP_MIN); /* 真实温度 [0, 1] */
}

Task task_temp_comp = {
    .name       = "传感器温度补偿",
    .input_dim  = 3,
    .output_dim = 1,
    .generate   = generate,
};
