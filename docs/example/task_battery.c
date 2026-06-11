/**
 * task_battery.c — 案例 4：电池剩余容量（SOC）估计
 *
 * 目标：根据电池的端电压、放电电流和温度，估计电池的剩余电量百分比。
 *
 * 输入：3 个值（均归一化到 [0,1]）
 *   [0] 端电压 (V)
 *   [1] 放电电流 (A)
 *   [2] 温度 (°C)
 *
 * 输出：1 个值 —— SOC（剩余电量百分比，0~1）
 *
 * 为什么这个案例有意义？
 *
 * 1. 嵌入式 BMS 核心功能：电池管理系统（BMS）的 SOC 估计是电动汽车、
 *    储能系统、无人机等产品的关键技术。不准确会导致过放损坏或续航焦虑。
 * 2. 无法用简单公式：电池 SOC 与电压的关系是非线性的——放电初期电压平缓，
 *    后期断崖式下降。温度影响内阻、电流影响压降，耦合关系复杂。
 * 3. 工业界现状：Tesla、比亚迪等厂商都在用神经网络做 SOC 估计。
 *
 * 模拟模型（简化锂离子电池）：
 *   开路电压 (OCV) vs SOC 曲线：非线性 S 形
 *   端电压 = OCV - I × R_internal(T, SOC)
 *   内阻 R 受温度和 SOC 影响
 *   温度越低、SOC 越低 → 内阻越大
 *
 * 建议网络结构：
 *   nn_add_layer(nn, 3, ACT_LINEAR);    // 输入：电压, 电流, 温度
 *   nn_add_layer(nn, 64, ACT_RELU);     // 隐藏层 1
 *   nn_add_layer(nn, 32, ACT_RELU);     // 隐藏层 2
 *   nn_add_layer(nn, 16, ACT_RELU);     // 隐藏层 3
 *   nn_add_layer(nn, 1, ACT_LINEAR);    // 输出：SOC
 *
 * CSV 数据集格式（可选外部数据）：
 *   第一行：3,1
 *   后续行：voltage,current,temperature,soc
 */
#include "nn.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ====== 电池模型参数 ====== */
#define V_FULL     (4.2)    /* 满电开路电压 (V) */
#define V_EMPTY    (3.0)    /* 空电开路电压 (V) */
#define V_NOMINAL  (3.7)    /* 标称电压 (V) */
#define R_BASE     (0.05)   /* 基准内阻 (Ω) */
#define TEMP_NOM   (25.0)   /* 标称温度 (°C) */
#define I_MAX      (5.0)    /* 最大放电电流 (A) */
#define TEMP_MIN   (-10.0)  /* 最低温度 */
#define TEMP_MAX   (45.0)   /* 最高温度 */
#define NOISE_V    (0.005)  /* 电压测量噪声 (V) */
#define NOISE_I    (0.02)   /* 电流测量噪声 (A) */

/**
 * OCV-SOC 曲线：开路电压随 SOC 变化的非线性关系
 * 使用经验公式模拟典型锂离子电池的放电曲线
 */
static double ocv_from_soc(double soc)
{
    /* 三段式 OCV 曲线：
     *   SOC 0~10%：电压从 3.0V 快速上升到 ~3.3V
     *   SOC 10~90%：电压从 3.3V 缓慢下降到 ~3.7V（平台区）
     *   SOC 90~100%：电压从 3.7V 快速上升到 4.2V
     */
    double v;
    if (soc < 0.1) {
        v = V_EMPTY + (3.3 - V_EMPTY) * (soc / 0.1);
    } else if (soc < 0.9) {
        v = 3.3 - 0.05 * ((soc - 0.1) / 0.8);
    } else {
        v = 3.25 + (V_FULL - 3.25) * ((soc - 0.9) / 0.1);
    }
    return v;
}

/**
 * 内阻模型：R 受温度和 SOC 影响
 * 温度越低 → 内阻越大（Arrhenius 简化）
 * SOC 越低 → 内阻越大
 */
static double internal_resistance(double soc, double temp_c)
{
    /* 温度因子：25°C 时为 1，-10°C 时约 3 倍，45°C 时约 0.6 倍 */
    double temp_factor = exp(-0.02 * (temp_c - TEMP_NOM));

    /* SOC 因子：SOC 越低，内阻越大 */
    double soc_factor = 1.0 + 0.5 * (1.0 - soc);

    return R_BASE * temp_factor * soc_factor;
}

/**
 * Box-Muller 高斯随机数
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
    /* 1. 随机 SOC (0~1) */
    double soc = (double)rand() / RAND_MAX;

    /* 2. 随机温度 */
    double temp = TEMP_MIN + ((double)rand() / RAND_MAX) * (TEMP_MAX - TEMP_MIN);

    /* 3. 随机放电电流 (0.1A ~ I_MAX) */
    double current = 0.1 + ((double)rand() / RAND_MAX) * (I_MAX - 0.1);

    /* 4. 计算端电压 */
    double ocv = ocv_from_soc(soc);
    double r = internal_resistance(soc, temp);
    double v_terminal = ocv - current * r;

    /* 5. 添加测量噪声 */
    v_terminal += gauss_rand(0, NOISE_V);
    current += gauss_rand(0, NOISE_I);

    /* 归一化到 [0, 1] */
    input[0] = (v_terminal - (V_EMPTY - 0.2)) / (V_FULL + 0.2 - (V_EMPTY - 0.2));
    input[1] = current / I_MAX;
    input[2] = (temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN);

    /* 钳位 */
    if (input[0] < 0) input[0] = 0;
    if (input[0] > 1) input[0] = 1;

    target[0] = soc;
}

Task task_battery = {
    .name       = "电池SOC估计",
    .input_dim  = 3,
    .output_dim = 1,
    .generate   = generate,
};
