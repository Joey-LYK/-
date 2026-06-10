/**
 * nn.c — 神经网络核心实现
 *
 * 文件结构：
 *   第 1 节：数学工具（高斯随机数）
 *   第 2 节：激活函数及其导数
 *   第 3 节：网络生命周期（create / add_layer / destroy）
 *   第 4 节：前向传播
 *   第 5 节：反向传播
 *   第 6 节：权重更新
 *   第 7 节：训练 API
 */

#include "nn.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* =================================================================
 *  第 1 节：数学工具
 *
 *  Box-Muller 算法：从均匀分布 [0,1) 生成标准正态分布 N(0,1)
 *  用于初始化权重——权重需要以 0 为中心随机分布
 * ================================================================= */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double gauss_rand(void) {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    /* 避免 log(0) */
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/* =================================================================
 *  第 2 节：激活函数及其导数
 *
 *  这里只有"正向"激活函数（forward pass 用）。
 *  导数在反向传播中直接计算（因为常用简化形式）。
 * ================================================================= */

static double activate(double x, Activation type) {
    switch (type) {
        case ACT_SIGMOID: return 1.0 / (1.0 + exp(-x));
        case ACT_TANH:    return tanh(x);
        case ACT_RELU:    return x > 0 ? x : 0;
        case ACT_LINEAR:  return x;
        case ACT_SOFTMAX: return x; /* Softmax 在 forward 中单独处理 */
        default:          return x;
    }
}

/**
 * 激活函数的导数 f'(x)，其中 x 已经是 f(x)（即：已激活的输出值）。
 *
 * 这样设计是因为反向传播中拿到的是 out=f(net)，而很多激活函数的导数
 * 可以用 out 直接算，不需要再算一遍 f(net)：
 *
 *   Sigmoid: f'(net) = out * (1 - out)        ← 用 out 算
 *   Tanh:    f'(net) = 1 - out²               ← 用 out 算
 *   ReLU:    f'(net) = (net > 0) ? 1 : 0       ← 需要原始 net
 *   Linear:  f'(net) = 1                      ← 常数
 */
static double activate_derivative(double activated_output, double raw_net, Activation type) {
    switch (type) {
        case ACT_SIGMOID:
            return activated_output * (1.0 - activated_output);
        case ACT_TANH:
            return 1.0 - activated_output * activated_output;
        case ACT_RELU:
            return raw_net > 0 ? 1.0 : 0;
        case ACT_LINEAR:
            return 1.0;
        case ACT_SOFTMAX:
            return 1.0; /* 交叉熵 + softmax 组合的导数在 backward 中单独处理 */
        default:
            return 1.0;
    }
}

/* =================================================================
 *  第 3 节：网络生命周期
 * ================================================================= */

NeuralNet* nn_create(double learning_rate) {
    NeuralNet *nn = (NeuralNet*)malloc(sizeof(NeuralNet));
    if (!nn) return NULL;
    memset(nn, 0, sizeof(NeuralNet));
    nn->learning_rate = learning_rate;
    return nn;
}

void nn_add_layer(NeuralNet *nn, int size, Activation activation) {
    int old_count = nn->layer_count;

    /* 扩容层数组 */
    nn->layers = (NNLayer*)realloc(nn->layers, sizeof(NNLayer) * (old_count + 1));
    nn->layer_count = old_count + 1;

    NNLayer *layer = &nn->layers[old_count];
    memset(layer, 0, sizeof(NNLayer));
    layer->size = size;
    layer->activation = activation;

    /* 如果这不是输入层（第一层），需要创建权重矩阵 */
    if (old_count > 0) {
        NNLayer *prev = &nn->layers[old_count - 1];
        layer->input_size = prev->size;

        int w_count = size * prev->size;
        layer->weights = (double*)malloc(sizeof(double) * w_count);
        layer->biases  = (double*)malloc(sizeof(double) * size);
        layer->net     = (double*)malloc(sizeof(double) * size);
        layer->out     = (double*)malloc(sizeof(double) * size);
        layer->error   = (double*)malloc(sizeof(double) * size);

        /* ──── 权重初始化：Xavier（Glorot）初始化 ────
         *
         * 原理：保持前向传播的方差不变，避免信号在层间爆炸或消失。
         * 
         * Xavier:  w ~ N(0, sqrt(2/(fan_in + fan_out)))
         * 均匀分布版: w ~ U(-sqrt(6/(fan_in+fan_out)), +sqrt(6/(fan_in+fan_out)))
         */
        double scale = sqrt(2.0 / (prev->size + size));

        for (int i = 0; i < w_count; i++) {
            layer->weights[i] = gauss_rand() * scale;
        }

        /* 偏置初始化为 0（ReLU 常用，Sigmoid/Tanh 也可用小微量） */
        for (int i = 0; i < size; i++) {
            layer->biases[i] = 0.0;
        }

        nn->total_params += w_count + size;
    } else {
        /* 输入层：只有 net 和 out 缓冲区（无权重无偏置） */
        layer->net = (double*)malloc(sizeof(double) * size);
        layer->out = (double*)malloc(sizeof(double) * size);
    }
}

void nn_destroy(NeuralNet *nn) {
    if (!nn) return;
    for (int i = 0; i < nn->layer_count; i++) {
        NNLayer *layer = &nn->layers[i];
        free(layer->weights);
        free(layer->biases);
        free(layer->net);
        free(layer->out);
        free(layer->error);
    }
    free(nn->layers);
    free(nn);
}

/* =================================================================
 *  第 4 节：前向传播
 *
 *  逐层计算：
 *
 *    第 i 层第 k 个神经元：
 *      net[k] = Σ(前层输出[j] × 权重[k*input_size + j]) + 偏置[k]
 *      out[k] = 激活函数(net[k])
 *
 *  最终输出层的 out 就是网络的预测结果。
 * ================================================================= */

void nn_forward(NeuralNet *nn, double *input, double *output) {
    NNLayer *layers = nn->layers;
    int n = nn->layer_count;

    /* ── 第 0 层（输入层）：直接把原始输入当作 "激活后输出" ── */
    for (int i = 0; i < layers[0].size; i++) {
        layers[0].net[i] = input[i];
        layers[0].out[i] = input[i];
    }

    /* ── 第 1 层到第 n-1 层：标准前向传播 ── */
    for (int L = 1; L < n; L++) {
        NNLayer *cur  = &layers[L];
        NNLayer *prev = &layers[L - 1];

        for (int i = 0; i < cur->size; i++) {
            /* 加权求和 */
            double sum = 0.0;
            for (int j = 0; j < prev->size; j++) {
                sum += prev->out[j] * cur->weights[i * cur->input_size + j];
            }
            cur->net[i] = sum + cur->biases[i];

            /* 激活 */
            cur->out[i] = activate(cur->net[i], cur->activation);
        }

        /* Softmax 整个层一起处理 */
        if (cur->activation == ACT_SOFTMAX) {
            double max_val = cur->net[0];
            for (int i = 1; i < cur->size; i++)
                if (cur->net[i] > max_val) max_val = cur->net[i];

            double sum_exp = 0.0;
            for (int i = 0; i < cur->size; i++) {
                cur->out[i] = exp(cur->net[i] - max_val); /* 减去 max 防溢出 */
                sum_exp += cur->out[i];
            }
            for (int i = 0; i < cur->size; i++) {
                cur->out[i] /= sum_exp;
            }
        }
    }

    /* 把输出层的 out 复制给调用者 */
    if (output) {
        NNLayer *out_layer = &layers[n - 1];
        for (int i = 0; i < out_layer->size; i++) {
            output[i] = out_layer->out[i];
        }
    }
}

/* =================================================================
 *  第 5 节：反向传播
 *
 *  核心思想：从输出层开始，逐层往回传"误差信号"。
 *
 *  输出层误差 δ_k：
 *    δ_k = (out_k - target_k) × f'(out_k)
 *    其中 f' 是激活函数的导数。
 *
 *  隐藏层误差 δ_j（链式法则）：
 *    δ_j = (Σ(下一层的 δ_k × 权重[j→k])) × f'(out_j)
 *
 *  注意：δ 存在 layer->error[] 中。
 *
 *  ⚠️ 关键理解：反向传播是在"分配责任"——
 *    "输出层的误差，按权重比例分摊给前一层每个神经元"
 * ================================================================= */

double nn_backward(NeuralNet *nn, double *target) {
    NNLayer *layers = nn->layers;
    int n = nn->layer_count;

    /* ── Step 1：算输出层的误差 δ ── */
    NNLayer *out_layer = &layers[n - 1];
    double loss = 0.0;

    if (out_layer->activation == ACT_SOFTMAX) {
        /* 分类问题：Softmax + 交叉熵损失
         *
         * 这个组合有一个优雅的简化：
         *   ∂L/∂net = out - target  （直接就是差值！）
         *
         * 不需要乘激活函数导数——这正是 Softmax+CE 成为标准的原因。
         */
        for (int i = 0; i < out_layer->size; i++) {
            out_layer->error[i] = out_layer->out[i] - target[i];
            loss -= target[i] * log(out_layer->out[i] + 1e-10); /* 交叉熵 */
        }
    } else {
        /* 回归问题：均方误差 (MSE)
         *
         * 损失 L = ½ Σ(target - out)²
         * 导数 ∂L/∂out = -(target - out) = out - target
         *
         * 完整的 δ = ∂L/∂out × f'(net)
         */
        for (int i = 0; i < out_layer->size; i++) {
            double diff = out_layer->out[i] - target[i];
            out_layer->error[i] = diff * activate_derivative(out_layer->out[i],
                                                              out_layer->net[i],
                                                              out_layer->activation);
            loss += diff * diff; /* MSE（不除 2，倍数不影响方向） */
        }
    }

    /* ── Step 2：逐层往回传误差 ── */
    for (int L = n - 2; L >= 1; L--) {
        NNLayer *cur  = &layers[L];
        NNLayer *next = &layers[L + 1];

        for (int j = 0; j < cur->size; j++) {
            /* 来自下一层的"责任分摊" */
            double back_sum = 0.0;
            for (int k = 0; k < next->size; k++) {
                /* 第 j 个神经元收到的"错误反馈" =
                 *   下一层第 k 个神经元的误差 × 连接它们之间的权重 */
                back_sum += next->error[k] * next->weights[k * cur->size + j];
            }
            cur->error[j] = back_sum * activate_derivative(cur->out[j],
                                                            cur->net[j],
                                                            cur->activation);
        }
    }

    nn->total_loss = loss;
    return loss;
}

/* =================================================================
 *  第 6 节：梯度下降（权重更新）
 *
 *  利用反向传播算出的 error[] 来更新权重和偏置。
 *
 *  公式（梯度下降）：
 *    新权重 = 旧权重 - 学习率 × (本层误差 × 前层输出)
 *
 *  为什么要乘以前层输出？——如果前层输出为 0，这个权重就不需要调整，因为它没用上。
 *  如果前层输出很大，说明这个连接"激活"了，权重要背很大的责任。
 *
 *  偏置更新：
 *    新偏置 = 旧偏置 - 学习率 × 本层误差
 *  偏置的梯度就是误差本身（因为偏置不乘任何输入）。
 * ================================================================= */

void nn_update(NeuralNet *nn) {
    NNLayer *layers = nn->layers;
    int n = nn->layer_count;
    double lr = nn->learning_rate;

    for (int L = 1; L < n; L++) {
        NNLayer *cur  = &layers[L];
        NNLayer *prev = &layers[L - 1];

        for (int i = 0; i < cur->size; i++) {
            double delta = cur->error[i];

            for (int j = 0; j < prev->size; j++) {
                /* 权重梯度 = 本层误差 × 前层输出
                 *
                 * 直观理解：
                 *   误差 δ_i 说"这个神经元输出有问题"
                 *   前层输出 out_j 说"这个输入信号有多强"
                 *   两者相乘 → "这个连接的响应该调多少"
                 */
                double gradient = delta * prev->out[j];
                cur->weights[i * cur->input_size + j] -= lr * gradient;
            }

            /* 偏置更新 */
            cur->biases[i] -= lr * delta;
        }
    }
}

/* =================================================================
 *  第 7 节：训练 API
 *
 *  将前向、反向、更新串成一步训练的便捷接口。
 * ================================================================= */

double nn_train_step(NeuralNet *nn, double *input, double *target) {
    /* 前向传播：结果写入 nn 内部各层的 out[]，不返回给调用者 */
    nn_forward(nn, input, NULL);
    /* 反向传播：计算各层误差梯度，存入 error[] */
    double loss = nn_backward(nn, target);
    /* 权重更新：用 error[] 梯度更新 weights 和 biases */
    nn_update(nn);
    return loss;
}

void nn_predict(NeuralNet *nn, double *input, double *output) {
    nn_forward(nn, input, output);
}

void nn_train(NeuralNet *nn, Task *task, int epochs, int samples_per_epoch,
              void (*on_epoch)(int epoch, double avg_loss))
{
    double *input  = (double*)malloc(sizeof(double) * task->input_dim);
    double *target = (double*)malloc(sizeof(double) * task->output_dim);

    for (int ep = 0; ep < epochs; ep++) {
        double total_loss = 0.0;

        for (int s = 0; s < samples_per_epoch; s++) {
            /* 让任务生成一个训练样本 */
            task->generate(input, target);
            /* 训练一步 */
            total_loss += nn_train_step(nn, input, target);
        }

        double avg_loss = total_loss / samples_per_epoch;

        if (on_epoch) {
            on_epoch(ep + 1, avg_loss);
        }
    }

    free(input);
    free(target);
}

/* =================================================================
 *  工具函数
 * ================================================================= */

void nn_print_structure(NeuralNet *nn) {
    const char *act_names[] = {
        [ACT_SIGMOID] = "Sigmoid",
        [ACT_TANH]    = "Tanh",
        [ACT_RELU]    = "ReLU",
        [ACT_LINEAR]  = "Linear",
        [ACT_SOFTMAX] = "Softmax",
    };

    printf("\n");
    printf("╔══════════════════════════════════════════╗\n");
    printf("║        Neural Network Structure          ║\n");
    printf("╠══════════════════════════════════════════╣\n");
    printf("║  Total layers:    %-3d                   ║\n", nn->layer_count);
    printf("║  Total params:    %-6d                ║\n", nn->total_params);
    printf("║  Learning rate:   %-8.5f              ║\n", nn->learning_rate);
    printf("╠══════════════════════════════════════════╣\n");

    for (int L = 0; L < nn->layer_count; L++) {
        NNLayer *layer = &nn->layers[L];
        const char *role = (L == 0) ? "Input " :
                           (L == nn->layer_count - 1) ? "Output" : "Hidden";

        printf("║  L%d %s: %3d neurons  [%s]    ║\n",
               L, role, layer->size,
               act_names[layer->activation]);

        if (L > 0) {
            int w = layer->size * layer->input_size;
            printf("║       ↳ weights: %d, biases: %d        ║\n",
                    w, layer->size);
        }
    }
    printf("╚══════════════════════════════════════════╝\n\n");
}

/* =================================================================
 *  第 8 节：序列化
 *
 *  二进制格式（与 PainterEngine 的 PX_ANN 格式不同，独立设计）：
 *
 *    ┌──────────────────────────────────────┐
 *    │ magic: "NN01"          (4 bytes)      │ ← 版本标识，防止误加载
 *    │ layer_count: int       (4 bytes)      │
 *    │ learning_rate: double  (8 bytes)      │
 *    ├──────────────────────────────────────┤
 *    │ per layer:                            │
 *    │   size:       int     (4 bytes)      │
 *    │   activation: int     (4 bytes)      │
 *    │   input_size: int     (4 bytes)      │
 *    │   (若 input_size > 0)               │
 *    │     weights: double[]                │ ← size * input_size 个
 *    │     biases:  double[]                │ ← size 个
 *    └──────────────────────────────────────┘
 *
 *  注意：只序列化"可训练参数"（权重和偏置），不序列化运行时缓存（net/out/error）。
 *        导入后 net/out/error 会在第一次前向传播时自动填充。
 * ================================================================= */

#define NN_MAGIC_0 'N'
#define NN_MAGIC_1 'N'
#define NN_MAGIC_2 0x00
#define NN_MAGIC_3 0x01

int nn_export(NeuralNet *nn, void *buffer, int *size) {
    unsigned char *wb = (unsigned char *)buffer;
    int total = 0;

    if (!nn || !size) return 0;

    /* ── 文件头：magic + layer_count + learning_rate ── */
    /* 4 + 4 + 8 = 16 bytes */
    if (wb) {
        wb[0] = NN_MAGIC_0;
        wb[1] = NN_MAGIC_1;
        wb[2] = NN_MAGIC_2;
        wb[3] = NN_MAGIC_3;
        memcpy(wb + 4, &nn->layer_count, 4);
        memcpy(wb + 8, &nn->learning_rate, 8);
        wb += 16;
    }
    total += 16;

    /* ── 逐层写入 ── */
    for (int L = 0; L < nn->layer_count; L++) {
        NNLayer *layer = &nn->layers[L];

        /* 层头：size + activation + input_size = 12 bytes */
        if (wb) {
            memcpy(wb,     &layer->size,       4);
            memcpy(wb + 4, &layer->activation, 4);
            memcpy(wb + 8, &layer->input_size, 4);
            wb += 12;
        }
        total += 12;

        /* 非输入层：写入权重和偏置 */
        if (layer->input_size > 0) {
            int w_count = layer->size * layer->input_size;
            int w_bytes = w_count * (int)sizeof(double);
            int b_bytes = layer->size * (int)sizeof(double);

            if (wb) {
                memcpy(wb, layer->weights, w_bytes);
                wb += w_bytes;
                memcpy(wb, layer->biases, b_bytes);
                wb += b_bytes;
            }
            total += w_bytes + b_bytes;
        }
    }

    *size = total;
    return 1;
}

NeuralNet* nn_import(void *buffer, int size) {
    unsigned char *rb = (unsigned char *)buffer;
    NeuralNet *nn;

    if (!buffer || size < 16) return NULL;

    /* ── 验证 magic ── */
    if (rb[0] != NN_MAGIC_0 || rb[1] != NN_MAGIC_1 ||
        rb[2] != NN_MAGIC_2 || rb[3] != NN_MAGIC_3)
        return NULL;

    /* ── 读取文件头 ── */
    int layer_count;
    double learning_rate;
    memcpy(&layer_count, rb + 4, 4);
    memcpy(&learning_rate, rb + 8, 8);
    rb += 16;
    size -= 16;

    /* ── 创建网络 ── */
    nn = nn_create(learning_rate);
    if (!nn) return NULL;

    /* ── 逐层导入 ── */
    for (int L = 0; L < layer_count; L++) {
        int l_size, activation, input_size;

        /* 读取层头 */
        if (size < 12) { nn_destroy(nn); return NULL; }
        memcpy(&l_size,     rb,     4);
        memcpy(&activation, rb + 4, 4);
        memcpy(&input_size, rb + 8, 4);
        rb += 12;
        size -= 12;

        /* 添加层（会自动创建权重矩阵和缓冲区） */
        nn_add_layer(nn, l_size, (Activation)activation);

        /* 非输入层：覆盖权重和偏置（用导入值替换随机初始化值） */
        if (input_size > 0) {
            int w_count = l_size * input_size;
            int w_bytes = w_count * (int)sizeof(double);
            int b_bytes = l_size * (int)sizeof(double);

            if (size < w_bytes + b_bytes) { nn_destroy(nn); return NULL; }

            NNLayer *layer = &nn->layers[L];
            memcpy(layer->weights, rb, w_bytes);
            rb += w_bytes;
            size -= w_bytes;

            memcpy(layer->biases, rb, b_bytes);
            rb += b_bytes;
            size -= b_bytes;
        }
    }

    return nn;
}
