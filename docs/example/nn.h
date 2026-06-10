/**
 * nn.h — 教学级神经网络库
 * 
 * 设计理念：代码即文档。每个函数的注释解释"它在做什么"而非"怎么做的"。
 * 
 * 架构概览：
 * 
 *   Task（任务层）                      NeuralNet（网络层）
 *   ┌──────────────────┐               ┌──────────────────┐
 *   │ .generate()      │──→ 数据 ──→   │ nn_forward()     │ 前向传播
 *   │  生成训练样本     │               │ nn_backward()    │ 反向传播
 *   └──────────────────┘               │ nn_update()      │ 权重更新
 *                                      │ nn_train_step()  │ 训练一步
 *                                      └──────────────────┘
 * 
 * 层结构图解：
 * 
 *   输入层    隐藏层1    隐藏层2    输出层
 *   [○ ○]  →  [○ ○ ○] → [○ ○ ○] →  [○]
 *              ↕ 权重      ↕ 权重     ↕ 权重
 *
 * 一个神经元的计算：
 *   net = Σ(前层输出_j × 权重_ij) + 偏置_i
 *   out = 激活函数(net)
 */

#ifndef NN_H
#define NN_H

/* ========================== 激活函数 ========================== */

typedef enum {
    ACT_SIGMOID,   /* S 形曲线，输出 (0,1)。适合：概率、归一化特征 */
    ACT_TANH,      /* S 形曲线，输出 (-1,1)。适合：数据中心化的场景 */
    ACT_RELU,      /* max(0,x)。适合：深层网络（计算快、梯度不消失） */
    ACT_LINEAR,    /* f(x)=x。适合：回归问题的输出层 */
    ACT_SOFTMAX,   /* 多分类输出层专用。转成概率分布，输出和为 1 */
} Activation;

/* ========================== 神经网络层 ========================== */

typedef struct {
    int input_size;    /* 前一层神经元数量（输入层此项为 0） */
    int size;          /* 本层神经元数量 */

    double *weights;   /* 权重矩阵 [size × input_size]，行优先 */
                       /* weights[i*input_size + j] = 前层第j个神经元 → 本层第i个神经元的权重 */

    double *biases;    /* 偏置向量 [size] */

    double *net;       /* 前向传播缓存：加权和（激活前） [size] */
    double *out;       /* 前向传播缓存：激活后输出 [size] */

    double *error;     /* 反向传播缓存：本层误差项 δ [size] */
                       /* 物理意义："这个神经元对最终误差负多大责任" */

    Activation activation; /* 本层激活函数 */
} NNLayer;

/* ========================== 神经网络 ========================== */

typedef struct {
    NNLayer *layers;      /* 层数组，layers[0] 是输入层 */
    int layer_count;      /* 层数（含输入层） */
    double learning_rate; /* 学习率 η */
    double total_loss;    /* 最近一次训练的损失值 */
    int total_params;     /* 总参数量（所有权重 + 偏置） */
} NeuralNet;

/* ========================== 任务接口 ========================== */

/**
 * 数据生成器：产生一对 (输入, 期望输出) 用于训练。
 * @param input  输出参数，长度为 input_dim
 * @param target 输出参数，长度为 output_dim
 */
typedef void (*DataGenerator)(double *input, double *target);

typedef struct {
    const char    *name;        /* 任务名称（如 "sin(x) 拟合"） */
    int            input_dim;   /* 输入向量维度 */
    int            output_dim;  /* 输出向量维度 */
    DataGenerator  generate;    /* 生成一对训练样本 */
} Task;

/* ========================== 网络 API ========================== */

/**
 * 创建一个神经网络。
 * @param learning_rate 学习率，典型值 0.001 ~ 0.1
 * @return 已初始化的网络对象
 */
NeuralNet* nn_create(double learning_rate);

/**
 * 添加一层神经元到网络。
 * @param nn         网络
 * @param size       本层神经元数量
 * @param activation 激活函数类型
 * 
 * 调用顺序即为层的排列顺序：
 *   nn_add_layer(nn, 2, ACT_LINEAR);   // 第 0 层（输入层）：2 个输入
 *   nn_add_layer(nn, 16, ACT_RELU);    // 第 1 层（隐藏层）：16 个神经元
 *   nn_add_layer(nn, 1, ACT_LINEAR);   // 第 2 层（输出层）：1 个输出
 */
void nn_add_layer(NeuralNet *nn, int size, Activation activation);

/**
 * 销毁网络，释放所有内存。
 */
void nn_destroy(NeuralNet *nn);

/* ========================== 训练 API ========================== */

/**
 * 前向传播：给定输入，计算网络输出。
 * @param nn     网络
 * @param input  输入向量，长度 = 输入层神经元数
 * @param output 输出参数，长度 = 输出层神经元数
 */
void nn_forward(NeuralNet *nn, double *input, double *output);

/**
 * 反向传播：计算每一层每个神经元的误差梯度。
 * @param nn     网络
 * @param target 期望输出（ground truth），长度 = 输出层神经元数
 * @return 损失值（均方误差 MSE）
 */
double nn_backward(NeuralNet *nn, double *target);

/**
 * 梯度下降：用反向传播算出的梯度更新所有权重和偏置。
 * @param nn 网络
 */
void nn_update(NeuralNet *nn);

/**
 * 训练一步：前向 → 反向 → 更新权重。等效于一次完整的 SGD 迭代。
 * @param nn     网络
 * @param input  输入向量
 * @param target 期望输出
 * @return 损失值（MSE）
 */
double nn_train_step(NeuralNet *nn, double *input, double *target);

/**
 * 训练多个 epoch，每个 epoch 包含一批随机样本。
 * @param nn           网络
 * @param task         任务（用于生成训练数据）
 * @param epochs       训练轮数
 * @param samples_per_epoch 每轮训练样本数
 * @param on_epoch     每轮结束的回调（可选，传 NULL 忽略）
 *                    参数：(epoch, avg_loss) —— 可用于打印或绘图
 */
void nn_train(NeuralNet *nn, Task *task, int epochs, int samples_per_epoch,
              void (*on_epoch)(int epoch, double avg_loss));

/**
 * 获取网络的预测输出（不改变内部状态）。
 * 实际上是 nn_forward 的便捷包装。
 */
void nn_predict(NeuralNet *nn, double *input, double *output);

/**
 * 打印网络结构（调试用）。
 */
void nn_print_structure(NeuralNet *nn);

/* ========================== 序列化 API ========================== */

/**
 * 将神经网络导出为二进制数据。
 * 两步调用：先传 buffer=NULL 获取所需大小，再传 buffer 分配的内存写入数据。
 *
 * 二进制格式：
 *   [magic "NN01" 4B] [layer_count 4B] [learning_rate 8B]
 *   per layer: [size 4B] [activation 4B] [input_size 4B]
 *              (若非输入层) [weights] [biases]
 *
 * @param nn     网络
 * @param buffer 输出缓冲区，传 NULL 仅计算大小
 * @param size   输出参数：所需/已写字节数
 * @return 1 成功，0 失败
 */
int nn_export(NeuralNet *nn, void *buffer, int *size);

/**
 * 从二进制数据导入神经网络。
 * @param buffer 包含 nn_export 格式数据的缓冲区
 * @param size   数据字节数
 * @return 新创建的 NeuralNet，失败返回 NULL
 */
NeuralNet* nn_import(void *buffer, int size);

#endif /* NN_H */
