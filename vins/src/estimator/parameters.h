/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>
#include <map>

using namespace std;

// 从 ROS1 迁移时留下的兼容层：把 ROS_INFO/ROS_WARN/ROS_ERROR 映射到 ROS2 的 rcutils 日志宏
#define ROS_INFO RCUTILS_LOG_INFO
#define ROS_WARN RCUTILS_LOG_WARN
#define ROS_ERROR RCUTILS_LOG_ERROR

// ==================== 编译期常量（不在配置文件里） ====================
// 归一化平面与像素之间的近似换算基准。
// 用于：MIN_PARALLAX 换算、特征点像素坐标还原（FOCAL_LENGTH*x+COL/2）、
//       重投影误差信息矩阵（sqrt_info = FOCAL_LENGTH/1.5）。
// 注意：这个 460.0 是按 EuRoC 焦距（≈461）选的，换相机不匹配会带来系统性偏差。
const double FOCAL_LENGTH = 460.0;
// 滑动窗口内保留的关键帧数量，决定 Ps/Vs/Bas/Bgs 等数组的长度
const int WINDOW_SIZE = 10;
// 单帧特征点容量的硬上限，决定 estimator.h 中定长数组 para_Feature[NUM_OF_F][1] 的大小。
// 必须 >= 配置文件里的 max_cnt，否则会越界写。
const int NUM_OF_F = 1000;
// 打开后重投影误差改用单位球面（归一化坐标）度量，替代乘 FOCAL_LENGTH 的像素度量
//#define UNIT_SPHERE_ERROR

// ==================== 全局运行参数 ====================
// 全部在这里用 extern 声明，真正的定义集中在 parameters.cpp，值由
// readParameters() 从配置 yaml 读入。用 extern 是为了让十几个 .cpp 共享同一份
// 存储，避免每个翻译单元各自定义导致重复定义链接错误。
// 因为都是静态存储期的全局变量，未赋值时会被零初始化为 0 / 空串。

// ---------- 初始化与关键帧 ----------
extern double INIT_DEPTH;       // 单目初始化时三角化点的默认深度，parameters.cpp 里硬编码 5.0
extern double MIN_PARALLAX;     // 关键帧判定视差阈值，配置给像素值，读入后除以 FOCAL_LENGTH
extern int ESTIMATE_EXTRINSIC;  // 0=固定外参 1=以外参为初值一起优化 2=无先验在线标定

// ---------- GPU 加速开关 ----------
extern int USE_GPU;           // 光流 GPU 加速
extern int USE_GPU_ACC_FLOW;  // GPU 金字塔光流
extern int USE_GPU_CERES;     // Ceres 的 CUDA 版求解

// ---------- IMU 噪声参数（连续时间）----------
extern double ACC_N, ACC_W;  // 加速度计：测量噪声标准差 / bias 随机游走噪声
extern double GYR_N, GYR_W;  // 陀螺仪：测量噪声标准差 / bias 随机游走噪声

// ---------- 外参与重力 ----------
extern std::vector<Eigen::Matrix3d> RIC;  // 各相机到 IMU 的旋转 R_imu_cam[i]，由 body_T_camX 拆出
extern std::vector<Eigen::Vector3d> TIC;  // 各相机到 IMU 的平移 t_imu_cam[i]
extern Eigen::Vector3d G;                 // 重力向量；初值只在 z 分量给了 9.8，读取后用 g_norm 覆盖

// ---------- 重置阈值与优化器 ----------
extern double BIAS_ACC_THRESHOLD;  // 加速度计 bias 突变检测阈值（硬编码 0.1），超限触发重置
extern double BIAS_GYR_THRESHOLD;  // 陀螺仪 bias 突变检测阈值（硬编码 0.1）
extern double SOLVER_TIME;         // 单次 Ceres 求解时间上限（秒），为实时性牺牲精度
extern int NUM_ITERATIONS;         // Ceres 最大迭代次数上限

// ---------- 输出路径 ----------
extern std::string EX_CALIB_RESULT_PATH;  // 外参在线标定结果 csv（仅 estimate_extrinsic!=0 时有意义）
extern std::string VINS_RESULT_PATH;      // 位姿结果 csv，固定为 OUTPUT_FOLDER/vio.csv
extern std::string OUTPUT_FOLDER;         // 输出根目录，来自配置的 output_path

// ---------- 话题名与时间偏移 ----------
extern std::string IMU_TOPIC;  // IMU 话题名，只在 USE_IMU 为真时被赋值，否则为空串
extern double TD;              // 相机与 IMU 的时间偏移（秒），读图时钟 + td = 真实时钟
extern int ESTIMATE_TD;        // 是否把 td 作为待优化量在线估计；无 IMU 时被强制置 0
extern int ROLLING_SHUTTER;    // 滚动快门补偿开关（本版本只声明，全项目未使用）

// ---------- 图像尺寸与传感器组合 ----------
extern int ROW, COL;      // 图像行/列数。注意 ROW 实际存的是 image_height，COL 是 image_width
extern int NUM_OF_CAM;    // 相机数量，只允许 1 或 2，否则 assert
extern int STEREO;        // 1=双目 0=单目，由 NUM_OF_CAM 推导得出
extern int USE_IMU;       // 0=纯视觉 1=视觉惯性，决定估计器走哪条分支，可由 /vins_imu_switch 运行时改写
extern int MULTIPLE_THREAD;  // 光流多线程加速开关

// 真值轨迹（调试用）。本仓库只有读取没有写入，恒为空，不影响正常运行
extern map<int, Eigen::Vector3d> pts_gt;

// ---------- 图像话题与特征跟踪 ----------
extern std::string IMAGE0_TOPIC, IMAGE1_TOPIC;  // 左/右目图像话题
extern std::string FISHEYE_MASK;                // 鱼眼掩膜图像路径（本版本只声明，全项目未使用）
extern std::vector<std::string> CAM_NAMES;      // 各相机内参 yaml 的完整路径，供 camodocal 解析
extern int MAX_CNT;              // 每帧最大特征点数
extern int MIN_DIST;             // 特征点之间最小像素间距，用于生成掩膜抑制聚集
extern double F_THRESHOLD;       // 光流跟踪结果的对极几何/RANSAC 阈值（像素）
extern int SHOW_TRACK;           // 是否发布带跟踪标记的图像
extern int FLOW_BACK;            // 是否做前向后向光流校验，剔除跟踪失败点

// ---------- TF 坐标系名称（配置缺省时分别回退为 world / body / camera）----------
extern std::string WORLD_FRAME_ID;
extern std::string BODY_FRAME_ID;
extern std::string CAMERA_FRAME_ID;

void readParameters(std::string config_file);

// 各类待优化参数块的维度，Ceres 添加参数块时按这个切分
enum SIZE_PARAMETERIZATION
{
    SIZE_POSE = 7,       // 单帧位姿：3 维平移 + 4 维四元数(xyzw)，用 Manifold 保证流形约束
    SIZE_SPEEDBIAS = 9,  // 速度与零偏：3 速度 + 3 加速度计 bias + 3 陀螺仪 bias
    SIZE_FEATURE = 1     // 单个特征点：只优化逆深度
};

// 状态量在同一帧参数块中的起始下标（para_Pose[i]、para_SpeedBias[i] 内部偏移）
enum StateOrder
{
    O_P = 0,   // 位置
    O_R = 3,   // 旋转（四元数）
    O_V = 6,   // 速度
    O_BA = 9,  // 加速度计 bias
    O_BG = 12  // 陀螺仪 bias
};

// IMU 噪声参数在 para_Noise 参数块中的起始下标
enum NoiseOrder
{
    O_AN = 0,  // 加速度计测量噪声
    O_GN = 3,  // 陀螺仪测量噪声
    O_AW = 6,  // 加速度计 bias 随机游走
    O_GW = 9   // 陀螺仪 bias 随机游走
};
