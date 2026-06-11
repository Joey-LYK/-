/**
 * task_gesture.c — 案例 5：手势/动作识别（简化版）
 *
 * 目标：根据加速度计的时序统计特征，识别用户的手势动作。
 *
 * 输入：12 个特征（均归一化到 [0,1]），来自三轴加速度计
 *   X 轴：均值, 方差, 峰值绝对值
 *   Y 轴：均值, 方差, 峰值绝对值
 *   Z 轴：均值, 方差, 峰值绝对值
 *   合成轴：总能量, 过零率, 峰峰差
 *
 * 输出：4 类 one-hot
 *   类别 0：静止
 *   类别 1：挥手（水平左右）
 *   类别 2：敲击（快速上下）
 *   类别 3：摇晃（不规则多方向）
 *
 * 为什么这个案例有意义？
 *
 * 1. 可穿戴设备核心功能：智能手表、手环、AR 眼镜等产品中，
 *    手势识别是最常用的交互方式。
 * 2. 特征工程入门：这是从"原始传感器数据"到"神经网络输入"的关键步骤。
 *    为什么不能直接把加速度波形喂给 MLP？因为时序长度不固定、噪声大。
 *    必须先提取统计特征，把变长序列变成定长向量。
 * 3. 实际部署可行：12 个 double = 96 字节，STM32F4 上运行毫无压力。
 *
 * 数据生成策略：
 *   每种手势有独特的加速度模式，通过数学函数模拟，加入噪声。
 *   这模拟了真实的 IMU 传感器数据（如 MPU6050）。
 *
 * 建议网络结构：
 *   nn_add_layer(nn, 12, ACT_LINEAR);    // 输入：12 个统计特征
 *   nn_add_layer(nn, 32, ACT_RELU);      // 隐藏层 1
 *   nn_add_layer(nn, 32, ACT_RELU);      // 隐藏层 2
 *   nn_add_layer(nn, 4, ACT_SOFTMAX);   // 输出：4 类概率
 *
 * CSV 数据集格式（可选外部数据）：
 *   第一行：12,4
 *   后续行：12个特征值,4个one-hot值
 */
#include "nn.h"
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ====== 采样参数 ====== */
#define SAMPLE_COUNT   50    /* 每次手势采样点数 */
#define SAMPLE_RATE    100   /* 模拟采样率 Hz */
#define DT             (1.0 / SAMPLE_RATE)
#define GRAVITY        (9.81) /* 重力加速度 m/s² */
#define NOISE_STD      (0.5)  /* 加速度计噪声标准差 m/s² */

/* ====== 高斯随机数 ====== */
static double gauss_rand(double mean, double std)
{
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    if (u1 < 1e-10) u1 = 1e-10;
    return mean + sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2) * std;
}

/**
 * 计算统计特征
 * ax/ay/az: 采样数组
 * n: 采样点数
 * features[12]: 输出特征向量
 */
static void extract_features(const double *ax, const double *ay, const double *az,
                              int n, double *features)
{
    /* 对每个轴计算：均值、方差、峰值绝对值 */
    for (int axis = 0; axis < 3; axis++) {
        const double *data = (axis == 0) ? ax : (axis == 1) ? ay : az;

        double sum = 0, sum2 = 0, peak = 0;
        for (int i = 0; i < n; i++) {
            sum += data[i];
            sum2 += data[i] * data[i];
            double abs_val = fabs(data[i]);
            if (abs_val > peak) peak = abs_val;
        }
        double mean = sum / n;
        double var  = sum2 / n - mean * mean;

        features[axis * 3 + 0] = mean;
        features[axis * 3 + 1] = var;
        features[axis * 3 + 2] = peak;
    }

    /* 合成特征：总能量、过零率、峰峰差 */
    double energy = 0, zcr = 0;
    double mag_min = 1e10, mag_max = -1e10;
    for (int i = 0; i < n; i++) {
        double mag = sqrt(ax[i]*ax[i] + ay[i]*ay[i] + az[i]*az[i]);
        energy += mag * mag;
        if (mag < mag_min) mag_min = mag;
        if (mag > mag_max) mag_max = mag;
        if (i > 0) {
            double prev_mag = sqrt(ax[i-1]*ax[i-1] + ay[i-1]*ay[i-1] + az[i-1]*az[i-1]);
            if ((mag - prev_mag) * (mag - prev_mag) > 0.01) zcr++;
        }
    }
    features[9]  = energy / n;                       /* 总能量 */
    features[10] = zcr / (n - 1);                    /* 过零率 */
    features[11] = mag_max - mag_min;                /* 峰峰差 */
}

/**
 * 生成特定手势的模拟加速度数据
 */
static void gen_gesture(int gesture, double *ax, double *ay, double *az)
{
    double phase = ((double)rand() / RAND_MAX) * 2.0 * M_PI;  /* 随机初始相位 */
    double amp   = 5.0 + ((double)rand() / RAND_MAX) * 10.0;  /* 随机振幅 */

    for (int i = 0; i < SAMPLE_COUNT; i++) {
        double t = i * DT;
        double nx = gauss_rand(0, NOISE_STD);
        double ny = gauss_rand(0, NOISE_STD);
        double nz = gauss_rand(0, NOISE_STD);

        switch (gesture) {
            case 0: /* 静止 */
                ax[i] = nx;
                ay[i] = GRAVITY + ny;
                az[i] = nz;
                break;

            case 1: /* 挥手：X 轴正弦波 */
                ax[i] = amp * sin(2.0 * M_PI * 3.0 * t + phase) + nx;
                ay[i] = GRAVITY + ny * 0.5;
                az[i] = amp * 0.3 * cos(2.0 * M_PI * 3.0 * t + phase) + nz;
                break;

            case 2: /* 敲击：Z 轴快速脉冲 */
            {
                double pulse = amp * 1.5 * sin(2.0 * M_PI * 8.0 * t + phase)
                            * exp(-3.0 * (t - 0.25));  /* 衰减 */
                ax[i] = nx * 0.5;
                ay[i] = GRAVITY + ny * 0.5;
                az[i] = pulse + nz;
                break;
            }

            case 3: /* 摇晃：多轴不规则运动 */
                ax[i] = amp * sin(2.0 * M_PI * 5.0 * t + phase)
                      + amp * 0.5 * sin(2.0 * M_PI * 7.3 * t + phase * 1.7) + nx;
                ay[i] = GRAVITY
                      + amp * 0.4 * cos(2.0 * M_PI * 4.1 * t + phase * 0.8) + ny;
                az[i] = amp * 0.7 * sin(2.0 * M_PI * 6.7 * t + phase * 1.3) + nz;
                break;
        }
    }
}

static void generate(double *input, double *target)
{
    /* 随机选择一种手势 */
    int gesture = rand() % 4;

    /* 生成模拟加速度数据 */
    double ax[SAMPLE_COUNT], ay[SAMPLE_COUNT], az[SAMPLE_COUNT];
    gen_gesture(gesture, ax, ay, az);

    /* 提取特征 */
    double raw_features[12];
    extract_features(ax, ay, az, SAMPLE_COUNT, raw_features);

    /* 归一化特征到 [0, 1] */
    /* 预估各特征的范围 */
    double feature_max[12] = {
        15, 100, 25,    /* X 轴: 均值, 方差, 峰值 */
        15, 100, 25,    /* Y 轴 */
        15, 100, 25,    /* Z 轴 */
        500, 1.0, 40    /* 合成: 能量, 过零率, 峰峰差 */
    };
    for (int i = 0; i < 12; i++) {
        input[i] = fabs(raw_features[i]) / feature_max[i];
        if (input[i] > 1) input[i] = 1;
    }

    /* one-hot 编码 */
    for (int i = 0; i < 4; i++) {
        target[i] = (i == gesture) ? 1.0 : 0.0;
    }
}

Task task_gesture = {
    .name       = "手势动作识别",
    .input_dim  = 12,
    .output_dim = 4,
    .generate   = generate,
};
