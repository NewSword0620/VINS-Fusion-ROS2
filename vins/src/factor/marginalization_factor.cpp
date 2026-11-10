/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "marginalization_factor.h"

void ResidualBlockInfo::Evaluate()
{
    residuals.resize(cost_function->num_residuals());

    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());

    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
    {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
        //dim += block_sizes[i] == 7 ? 6 : block_sizes[i];
    }
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    //std::vector<int> tmp_idx(block_sizes.size());
    //Eigen::MatrixXd tmp(dim, dim);
    //for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
    //{
    //    int size_i = localSize(block_sizes[i]);
    //    Eigen::MatrixXd jacobian_i = jacobians[i].leftCols(size_i);
    //    for (int j = 0, sub_idx = 0; j < static_cast<int>(parameter_blocks.size()); sub_idx += block_sizes[j] == 7 ? 6 : block_sizes[j], j++)
    //    {
    //        int size_j = localSize(block_sizes[j]);
    //        Eigen::MatrixXd jacobian_j = jacobians[j].leftCols(size_j);
    //        tmp_idx[j] = sub_idx;
    //        tmp.block(tmp_idx[i], tmp_idx[j], size_i, size_j) = jacobian_i.transpose() * jacobian_j;
    //    }
    //}
    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(tmp);
    //std::cout << saes.eigenvalues() << std::endl;
    //ROS_ASSERT(saes.eigenvalues().minCoeff() >= -1e-6);

    // imu因子约束不需要loss_function,视觉需要
    if (loss_function)
    {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();// 平方和
        // rho 是长度为 3 的输出数组：rho[0]=ρ(s)（损失值）、rho[1]=ρ'(s)，rho[2]=ρ''(s)（二阶导）
        // 以HuberLoss(1.0)为例，rho[0] = 1.0 * (sqrt(s) - 0.5) if s > 1.0 else 1.0 * s
        loss_function->Evaluate(sq_norm, rho);
        //printf("sq_norm: %f, rho[0]: %f, rho[1]: %f, rho[2]: %f\n", sq_norm, rho[0], rho[1], rho[2]);

        double sqrt_rho1_ = sqrt(rho[1]);

        // Huber 永远走 if 分支，因为 rho[2] <= 0
        if ((sq_norm == 0.0) || (rho[2] <= 0.0))
        {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
        {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}

MarginalizationInfo::~MarginalizationInfo()
{
    //ROS_WARN("release marginlizationinfo");
    
    for (auto it = parameter_block_data.begin(); it != parameter_block_data.end(); ++it)
    {
        delete it->second;
        it->second = nullptr;
    }
    
    for (int i = 0; i < (int)factors.size(); i++)
    {

        delete[] factors[i]->raw_jacobians;
        factors[i]->raw_jacobians = nullptr;
        
        delete factors[i]->cost_function;
        factors[i]->cost_function = nullptr;

        factors[i] = nullptr;
    }
}

// 记录所有与待边缘化参数有关的parameter_blocks对应的地址和大小
// parameter_block_idx中记录了要边缘化的参数的地址
void MarginalizationInfo::addResidualBlockInfo(ResidualBlockInfo *residual_block_info)
{
    factors.emplace_back(residual_block_info);

    std::vector<double *> &parameter_blocks = residual_block_info->parameter_blocks;
    std::vector<int> parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    for (int i = 0; i < static_cast<int>(residual_block_info->parameter_blocks.size()); i++)
    {
        double *addr = parameter_blocks[i];
        int size = parameter_block_sizes[i];
        parameter_block_size[reinterpret_cast<long>(addr)] = size;
    }

    for (int i = 0; i < static_cast<int>(residual_block_info->drop_set.size()); i++)
    {
        double *addr = parameter_blocks[residual_block_info->drop_set[i]];
        parameter_block_idx[reinterpret_cast<long>(addr)] = 0;
    }
}

void MarginalizationInfo::preMarginalize()
{
    for (auto it : factors)
    {
        it->Evaluate();

        // 记录所有与待边缘化参数有关的parameter_blocks对应的地址和数据备份
        // 与parameter_block_size（参数地址：size），parameter_block_idx（要边缘化的参数地址）对应
        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++)
        {
            long addr = reinterpret_cast<long>(it->parameter_blocks[i]);
            int size = block_sizes[i];
            if (parameter_block_data.find(addr) == parameter_block_data.end())
            {
                double *data = new double[size];
                memcpy(data, it->parameter_blocks[i], sizeof(double) * size);
                parameter_block_data[addr] = data;// [原地址]=新地址，新地址上的内容与原地址相同
            }
        }
    }
}

int MarginalizationInfo::localSize(int size) const
{
    return size == 7 ? 6 : size;
}

int MarginalizationInfo::globalSize(int size) const
{
    return size == 6 ? 7 : size;
}

// 单个线程的累加任务：把分到的残差块累加进线程私有的 A（pos×pos）和 b（pos）。
//   A = Σ JᵀJ  信息矩阵（正规方程左端）
//   b = Σ Jᵀr  梯度（正规方程右端）
// 利用之前整理的parameter_block_idx，需要边缘化的参数块的信息矩阵被挪到了左上角
void* ThreadsConstructA(void* threadsstruct)
{
    ThreadsStruct* p = ((ThreadsStruct*)threadsstruct);   // pthread 接口只认 void*，转回结构体
    for (auto it : p->sub_factors)                        // 本线程负责的残差块
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            // 该参数块在全局 pos 维向量中的起始下标与维度
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])];
            if (size_i == 7)
                size_i = 6;                                   // 位姿 7 存 6 用：切掉四元数多余分量
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                // j 从 i 开始：只算上三角，对称位置靠下面镜像，省一半矩阵乘
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])];
                if (size_j == 7)
                    size_j = 6;
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;   // 对角块 JᵢᵀJᵢ
                else
                {
                    p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;   // 上三角 JᵢᵀJⱼ
                    // 下三角镜像。这里用 = 而非 +=：该位置在本线程内从未被写过（j 只从 i 起）
                    p->A.block(idx_j, idx_i, size_j, size_i) = p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            // 梯度只在 i 循环里累加一次；若放内层 j 循环会重复 n 次
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    return threadsstruct;
}

void MarginalizationInfo::marginalize()
{
    // 实际是对所有因子进行操作，这里只以2帧1相机举例，理解流程。实际pos的统计包含了所有待边缘化的factor
    // 以para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]
    // 维度：7 7 7 1 1, 假设地址为a b c d e, drop_set<0,3>为例
    // parameter_block_idx:addr = a  d
    // parameter_block_idx:idx  = 0  7
    //                      pos = 7  8
    int pos = 0;
    for (auto &it : parameter_block_idx)
    {
        it.second = pos;
        pos += localSize(parameter_block_size[it.first]);
    }

    // m = pos = 8
    m = pos; // 边缘化的参数总维度

    // parameter_block_idx:addr = a  d  b  c  e
    // parameter_block_idx:idx  = 0  7  8 15 22
    //                      pos = 7  8 15 22 23
    for (const auto &it : parameter_block_size)
    {
        if (parameter_block_idx.find(it.first) == parameter_block_idx.end())
        {
            parameter_block_idx[it.first] = pos;
            pos += localSize(it.second);
        }
    }

    // n = pos - m = 23 - 8 = 15
    n = pos - m; // 剩余的参数总维度
    //ROS_INFO("marginalization, pos: %d, m: %d, n: %d, size: %d", pos, m, n, (int)parameter_block_idx.size());
    if(m == 0)
    {
        valid = false;
        printf("unstable tracking...\n");
        return;
    }

    TicToc t_summing;
    Eigen::MatrixXd A(pos, pos);
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();
    /*
    for (auto it : factors)
    {
        for (int i = 0; i < static_cast<int>(it->parameter_blocks.size()); i++)
        {
            int idx_i = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[i])];
            int size_i = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[i])]);
            Eigen::MatrixXd jacobian_i = it->jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it->parameter_blocks.size()); j++)
            {
                int idx_j = parameter_block_idx[reinterpret_cast<long>(it->parameter_blocks[j])];
                int size_j = localSize(parameter_block_size[reinterpret_cast<long>(it->parameter_blocks[j])]);
                Eigen::MatrixXd jacobian_j = it->jacobians[j].leftCols(size_j);
                if (i == j)
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                else
                {
                    A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                    A.block(idx_j, idx_i, size_j, size_i) = A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            b.segment(idx_i, size_i) += jacobian_i.transpose() * it->residuals;
        }
    }
    ROS_INFO("summing up costs %f ms", t_summing.toc());
    */
    //multi thread


    TicToc t_thread_summing;
    pthread_t tids[NUM_THREADS];
    ThreadsStruct threadsstruct[NUM_THREADS];
    int i = 0;
    // 把所有因子平均分给 NUM_THREADS 个线程
    for (auto it : factors)
    {
        threadsstruct[i].sub_factors.push_back(it);
        i++;
        i = i % NUM_THREADS;
    }
    for (int i = 0; i < NUM_THREADS; i++)
    {
        TicToc zero_matrix;
        threadsstruct[i].A = Eigen::MatrixXd::Zero(pos,pos);
        threadsstruct[i].b = Eigen::VectorXd::Zero(pos);
        threadsstruct[i].parameter_block_size = parameter_block_size;
        threadsstruct[i].parameter_block_idx = parameter_block_idx;
        // 计算每个线程的 A=JTJ,b=JTr
        int ret = pthread_create( &tids[i], NULL, ThreadsConstructA ,(void*)&(threadsstruct[i]));
        if (ret != 0)
        {
            ROS_WARN("pthread_create error");
            // ROS_BREAK();
        }
    }
    // 主线程累加A b
    for( int i = NUM_THREADS - 1; i >= 0; i--)  
    {
        pthread_join( tids[i], NULL ); 
        A += threadsstruct[i].A;
        b += threadsstruct[i].b;
    }
    //ROS_DEBUG("thread summing up costs %f ms", t_thread_summing.toc());
    //ROS_INFO("A diff %f , b diff %f ", (A - tmp_A).sum(), (b - tmp_b).sum());


    // 利用舒尔消元，消掉边缘化的部分，重新计算A b
    //TODO
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);

    //ROS_ASSERT_MSG(saes.eigenvalues().minCoeff() >= -1e-4, "min eigenvalue %f", saes.eigenvalues().minCoeff());

    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() * saes.eigenvectors().transpose();
    //printf("error1: %f\n", (Amm * Amm_inv - Eigen::MatrixXd::Identity(m, m)).sum());

    Eigen::VectorXd bmm = b.segment(0, m);
    Eigen::MatrixXd Amr = A.block(0, m, m, n);
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);
    // A*dxn = -b
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;

    // 利用特征分解把 A 分解成 JᵀJ，b 分解成 Jᵀr，得到线性化的雅可比和残差
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    linearized_jacobians = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    // r = JT^-1*b
    linearized_residuals = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
    //std::cout << A << std::endl
    //          << std::endl;
    //std::cout << linearized_jacobians << std::endl;
    //printf("error2: %f %f\n", (linearized_jacobians.transpose() * linearized_jacobians - A).sum(),
    //      (linearized_jacobians.transpose() * linearized_residuals - b).sum());
}

std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift)
{
    std::vector<double *> keep_block_addr;
    keep_block_size.clear();
    keep_block_idx.clear();
    keep_block_data.clear();

    for (const auto &it : parameter_block_idx)
    {
        if (it.second >= m)
        {
            keep_block_size.push_back(parameter_block_size[it.first]);
            keep_block_idx.push_back(parameter_block_idx[it.first]);
            // 原地址内容的备份，作为evalueate的起点
            keep_block_data.push_back(parameter_block_data[it.first]);
            // 剔除边缘化后的参数块左移，占用的地址，原来1的内容现在占用位置0
            keep_block_addr.push_back(addr_shift[it.first]);
        }
    }
    sum_block_size = std::accumulate(std::begin(keep_block_size), std::end(keep_block_size), 0);

    return keep_block_addr;
}

MarginalizationFactor::MarginalizationFactor(MarginalizationInfo* _marginalization_info):marginalization_info(_marginalization_info)
{
    int cnt = 0;
    for (auto it : marginalization_info->keep_block_size)
    {
        mutable_parameter_block_sizes()->push_back(it);
        cnt += it;
    }
    //printf("residual size: %d, %d\n", cnt, n);
    set_num_residuals(marginalization_info->n);
};

// 边缘化先验的求值。与视觉/IMU 因子不同，这里的残差和雅可比不是现场推导的 ——
// 它们在 MarginalizationInfo::marginalize() 里已算好并冻结为常数：
//     J  = linearized_jacobians   （JᵀJ = Λ'，即 Schur 补出的信息矩阵）
//     r₀ = linearized_residuals   （Jᵀr₀ = b'，即 Schur 补出的梯度）
// 本函数只做三件事：
//   ① dx = x ⊖ x_lin（相对"边缘化那一刻"的增量）
//   ② r = r₀ + J · dx
//   ③ 把 J 按块切片拷进 jacobians[]
bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    //printf("internal addr,%d, %d\n", (int)parameter_block_sizes().size(), num_residuals());
    //for (int i = 0; i < static_cast<int>(keep_block_size.size()); i++)
    //{
    //    //printf("unsigned %x\n", reinterpret_cast<unsigned long>(parameters[i]));
    //    //printf("signed %x\n", reinterpret_cast<long>(parameters[i]));
    //printf("jacobian %x\n", reinterpret_cast<long>(jacobians));
    //printf("residual %x\n", reinterpret_cast<long>(residuals));
    //}
    int n = marginalization_info->n;                 // 先验维数 = 保留变量的切空间总维数
    int m = marginalization_info->m;                 // 被消元变量的总维数（keep_block_idx 以此为基准）
    // dx = x ⊖ x_lin：当前状态相对"冻结的线性化点"的增量，长度 n
    // 注意这里用的是切空间维度：位姿块 7 个数只贡献 6 维
    Eigen::VectorXd dx(n);
    for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
    {
        int size = marginalization_info->keep_block_size[i];   // ambient 尺寸：位姿 7、速度零偏 9、深度/td 1
        int idx = marginalization_info->keep_block_idx[i] - m; // 全局索引(dx.segment 起点) → 先验内部 0 基下标
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);                          // 当前值（Ceres 传入）
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info->keep_block_data[i], size);  // 线性化点（深拷贝）
        if (size != 7)
            dx.segment(idx, size) = x - x0;                    // 欧氏量（速度/零偏/深度/td）直接相减
        else
        {
            // 位姿块：7 维存（p + 四元数）→ 6 维用（p + 旋转向量）
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();                                  // 平移直接相减
            // 旋转：2·vec(q₀⁻¹ ⊗ q)，即 Log 的一阶近似（与全项目约定一致）
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            // 四元数双覆盖：若 q₀⁻¹q 的实部 < 0，上面算出的是"绕远路"的旋转，需整体取反
            if (!((Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).w() >= 0))
            {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse() * Eigen::Quaterniond(x(6), x(3), x(4), x(5))).vec();
            }
        }
    }
    // 残差 r = r₀ + J·dx（n 维），J 与 r₀ 都是边缘化时预存的常量。展开代价：
    //   ½‖r‖² = ½dxᵀ(JᵀJ)dx + (r₀ᵀJ)dx + ½‖r₀‖²
    // 而 JᵀJ = Λ'、r₀ᵀJ = b'ᵀ（Λ'、b' 是 Schur 补的结果），
    // 所以这个代价正好等于 ½dxᵀΛ'dx + b'ᵀdx + 常数 —— 与原二次型只差一个常数
    Eigen::Map<Eigen::VectorXd>(residuals, n) = marginalization_info->linearized_residuals + marginalization_info->linearized_jacobians * dx;
    if (jacobians)
    {

        for (int i = 0; i < static_cast<int>(marginalization_info->keep_block_size.size()); i++)
        {
            if (jacobians[i])
            {
                int size = marginalization_info->keep_block_size[i], local_size = marginalization_info->localSize(size);   // 7 → 6
                int idx = marginalization_info->keep_block_idx[i] - m;
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobian(jacobians[i], n, size);
                // 雅可比就是 J（linearized_jacobians）的一个列块切片，无需任何推导
                // setZero 是必需的：位姿块 size=7 而 local_size=6，第 7 列(qw)要留 0
                jacobian.setZero();
                jacobian.leftCols(local_size) = marginalization_info->linearized_jacobians.middleCols(idx, local_size);
            }
        }
    }
    return true;
}
