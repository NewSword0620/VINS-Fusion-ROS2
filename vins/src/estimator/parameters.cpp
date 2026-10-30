/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

int USE_GPU;
int USE_GPU_ACC_FLOW;
int USE_GPU_CERES;

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string OUTPUT_FOLDER;
std::string IMU_TOPIC;
int ROW, COL;
double TD;
int NUM_OF_CAM;
int STEREO;
int USE_IMU;
int MULTIPLE_THREAD;
map<int, Eigen::Vector3d> pts_gt;
std::string IMAGE0_TOPIC, IMAGE1_TOPIC;
std::string FISHEYE_MASK;
std::vector<std::string> CAM_NAMES;
int MAX_CNT;
int MIN_DIST;
double F_THRESHOLD;
int SHOW_TRACK;
int FLOW_BACK;

std::string WORLD_FRAME_ID;
std::string BODY_FRAME_ID;
std::string CAMERA_FRAME_ID;

template <typename T>
T readParam(rclcpp::Node::SharedPtr n, std::string name)
{
    T ans;
    if (n->get_parameter(name, ans))
    {
        ROS_INFO("Loaded %s: ", name);
        std::cout << ans << std::endl;
    }
    else
    {
        ROS_ERROR("Failed to load %s", name);
        rclcpp::shutdown();
    }
    return ans;
}

void readParameters(std::string config_file)
{
    // 先用 fopen 粗查文件是否存在。因为后面的 FileStorage 打开失败时只打一行错误，
    // 不返回，所有参数会静默变成 0/空串，所以这里提前拦截
    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL){
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        // ROS_BREAK();
        return;          
    }
    fclose(fh);

    // 用 OpenCV 的 FileStorage 解析 YAML（文件首行必须是 %YAML:1.0）
    // 取值有两种写法：fs["k"] >> var（流式）和 var = fs["k"]（隐式类型转换）
    // 注意：键不存在不会报错，int 得 0、string 得空串，拼错键名不会有任何提示
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        // 只打印不返回：上面 fopen 已经保证文件存在，这里几乎不会触发
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    // ---------- 图像话题：vins_node 据此创建订阅 ----------
    fsSettings["image0_topic"] >> IMAGE0_TOPIC;   // 左目图像话题
    fsSettings["image1_topic"] >> IMAGE1_TOPIC;   // 右目图像话题（单目时可为空）

    // ---------- 特征跟踪参数：featureTracker.trackImage() 使用 ----------
    MAX_CNT = fsSettings["max_cnt"];          // 每帧最大特征点数
    MIN_DIST = fsSettings["min_dist"];        // 特征点之间最小像素间距，用于生成掩膜抑制聚集
    F_THRESHOLD = fsSettings["F_threshold"];  // 光流跟踪结果的对极几何/RANSAC 阈值（像素）
    SHOW_TRACK = fsSettings["show_track"];    // 是否发布带跟踪标记的图像
    FLOW_BACK = fsSettings["flow_back"];      // 是否做前向后向光流校验，剔除跟踪失败点

    MULTIPLE_THREAD = fsSettings["multiple_thread"];  // 光流多线程加速开关

    // ---------- GPU 加速开关：本仓库未启用 CUDA 版 OpenCV，配 1 也不会生效 ----------
    USE_GPU = fsSettings["use_gpu"];                    // 光流 GPU 加速
    USE_GPU_ACC_FLOW = fsSettings["use_gpu_acc_flow"];  // GPU 金字塔光流
    USE_GPU_CERES = fsSettings["use_gpu_ceres"];        // Ceres 的 CUDA 版求解

    // ---------- IMU 相关：USE_IMU 决定整个系统是 VIO 还是纯视觉 ----------
    USE_IMU = fsSettings["imu"];       // 0=纯视觉（单目/双目），非 0=视觉惯性
    printf("USE_IMU: %d\n", USE_IMU);  // 打印出来便于确认配置是否真的生效
    if(USE_IMU)
    {
        fsSettings["imu_topic"] >> IMU_TOPIC;  // rosNodeTest.cpp 用它创建 IMU 订阅
        printf("IMU_TOPIC: %s\n", IMU_TOPIC.c_str());
        ACC_N = fsSettings["acc_n"];  // 加速度计测量噪声标准差（连续时间）
        ACC_W = fsSettings["acc_w"];  // 加速度计 bias 随机游走噪声
        GYR_N = fsSettings["gyr_n"];  // 陀螺仪测量噪声标准差
        GYR_W = fsSettings["gyr_w"];  // 陀螺仪 bias 随机游走噪声
        G.z() = fsSettings["g_norm"]; // 重力模长，只覆盖 z 分量，约定重力方向为 +Z
    }
    // USE_IMU == 0 时上述量与 IMU_TOPIC 保持全局零初始化值（0 / 空串），不会被访问

    // ---------- 优化器参数：为保证实时性做的硬性上限 ----------
    SOLVER_TIME = fsSettings["max_solver_time"];        // 单次求解时间上限（秒），超时提前收敛
    NUM_ITERATIONS = fsSettings["max_num_iterations"];  // 最大迭代次数上限

    // 关键帧判定阈值：配置里给的是像素单位，这里换算到归一化平面单位
    // 注意 FOCAL_LENGTH 是 parameters.h:27 写死的 460.0，并非从标定文件读取，
    // 换成非 EuRoC 相机（焦距差得多）时这个换算会引入偏差
    MIN_PARALLAX = fsSettings["keyframe_parallax"];
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH;

    // ---------- 输出路径：启动时就创建/清空 vio.csv ----------
    // 注意不会自动创建目录，output_path 不存在时 ofstream 静默失败，后面写结果才会暴露
    fsSettings["output_path"] >> OUTPUT_FOLDER;
    VINS_RESULT_PATH = OUTPUT_FOLDER + "/vio.csv";
    std::cout << "result path " << VINS_RESULT_PATH << std::endl;
    std::ofstream fout(VINS_RESULT_PATH, std::ios::out);
    fout.close();

    // ---------- IMU-相机外参（RIC/TIC）：相机系到 IMU 系的 R、t ----------
    // estimate_extrinsic: 0=固定外参; 1=以初值为起点一起优化; 2=完全没有先验，在线标定
    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"];
    if (ESTIMATE_EXTRINSIC == 2)
    {
        // 无先验：给单位阵作为初值，估计结果另行输出到 csv
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity());
        TIC.push_back(Eigen::Vector3d::Zero());
        EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
    }
    else 
    {
        // 1=优化外参；0=完全信任配置里的值
        if ( ESTIMATE_EXTRINSIC == 1)
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_FOLDER + "/extrinsic_parameter.csv";
        }
        if (ESTIMATE_EXTRINSIC == 0)
            ROS_WARN(" fix extrinsic param ");

        // 从 YAML 读 4x4 齐次矩阵 body_T_cam0，拆成旋转块和平移块存入 RIC/TIC
        cv::Mat cv_T;
        fsSettings["body_T_cam0"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);             // cv::Mat -> Eigen 转换
        RIC.push_back(T.block<3, 3>(0, 0)); // 左上 3x3 即 R_imu_cam
        TIC.push_back(T.block<3, 1>(0, 3)); // 右上 3x1 即 t_imu_cam
    } 
    
    // ---------- 相机数量：只有 1 或 2 两种合法取值 ----------
    NUM_OF_CAM = fsSettings["num_of_cam"];
    printf("camera number %d\n", NUM_OF_CAM);

    if(NUM_OF_CAM != 1 && NUM_OF_CAM != 2)
    {
        // 用 assert(0) 直接中断，Release 下若定义了 NDEBUG 则会被忽略
        printf("num_of_cam should be 1 or 2\n");
        assert(0);
    }


    // ---------- 相机内参文件路径：相对于主配置所在目录解析 ----------
    // 即 cam0_calib: "cam0_mei.yaml" 会拼成 <配置目录>/cam0_mei.yaml
    // 若 config_file 只是纯文件名（不含 '/'），pn 为 npos，路径会算错
    int pn = config_file.find_last_of('/');
    std::string configPath = config_file.substr(0, pn);
    
    // cam0 内参文件（如 cam0_mei.yaml），只存路径，真正解析在 Estimator::setParameter 里
    std::string cam0Calib;
    fsSettings["cam0_calib"] >> cam0Calib;
    std::string cam0Path = configPath + "/" + cam0Calib;
    CAM_NAMES.push_back(cam0Path);

    // ---------- 双目配置：多一路内参 + 右目外参，并把 STEREO 置 1 ----------
    if(NUM_OF_CAM == 2)
    {
        STEREO = 1;
        std::string cam1Calib;
        fsSettings["cam1_calib"] >> cam1Calib;
        std::string cam1Path = configPath + "/" + cam1Calib; 
        //printf("%s cam1 path\n", cam1Path.c_str() );
        CAM_NAMES.push_back(cam1Path);
        
        // 同理读 body_T_cam1，作为 RIC[1]/TIC[1]
        cv::Mat cv_T;
        fsSettings["body_T_cam1"] >> cv_T;
        Eigen::Matrix4d T;
        cv::cv2eigen(cv_T, T);
        RIC.push_back(T.block<3, 3>(0, 0));
        TIC.push_back(T.block<3, 1>(0, 3));
    }

    // ---------- 几个不在配置文件里、直接写死的常量 ----------
    INIT_DEPTH = 5.0;            // 单目初始化时三角化点的默认深度（米）
    BIAS_ACC_THRESHOLD = 0.1;    // 加速度计 bias 跳变检测阈值，超限触发重置
    BIAS_GYR_THRESHOLD = 0.1;    // 陀螺仪 bias 跳变检测阈值，超限触发重置

    // ---------- 相机与 IMU 的时间偏移 td ----------
    // estimate_td=1 时把 td 作为待优化量在线估计；=0 时固定为配置值
    TD = fsSettings["td"];
    ESTIMATE_TD = fsSettings["estimate_td"];
    if (ESTIMATE_TD)
        ROS_INFO("Unsynchronized sensors, online estimate time offset, initial td: %f", TD);
    else
        ROS_INFO("Synchronized sensors, fix time offset: %f", TD);

    // 注意命名与常规相反：ROW 存的是图像高度，COL 存的是宽度
    ROW = fsSettings["image_height"];
    COL = fsSettings["image_width"];
    ROS_INFO("ROW: %d COL: %d ", ROW, COL);

    // ---------- 无 IMU 时的强制降级 ----------
    if(!USE_IMU)
    {
        // 没有 IMU 就没有 IMU-相机外参和时间偏移可标，
        // 因此在前面读过配置之后在这里再覆盖一次，避免配置写错导致无意义优化
        ESTIMATE_EXTRINSIC = 0;
        ESTIMATE_TD = 0;
        printf("no imu, fix extrinsic param; no time offset calibration\n");
    }

    // ---------- TF 坐标系名字：配置里没写就用默认值 ----------
    // 这里用 "cond ? a = x : a;" 的写法做「为空则赋默认值」，靠赋值表达式的副作用完成
    // （FileStorage 取不到键时字符串保持空，所以 empty() 判断是有效的）
    fsSettings["world_frame_id"] >> WORLD_FRAME_ID;
    WORLD_FRAME_ID.empty()? WORLD_FRAME_ID = "world" : WORLD_FRAME_ID;
    fsSettings["body_frame_id"] >> BODY_FRAME_ID;   
    BODY_FRAME_ID.empty()? BODY_FRAME_ID = "body" : BODY_FRAME_ID;
    fsSettings["camera_frame_id"] >> CAMERA_FRAME_ID;
    CAMERA_FRAME_ID.empty()? CAMERA_FRAME_ID = "camera" : CAMERA_FRAME_ID;
    
    ROS_INFO("frame_ids: world=%s body=%s camera=%s", WORLD_FRAME_ID.c_str(),
             BODY_FRAME_ID.c_str(), CAMERA_FRAME_ID.c_str());

    fsSettings.release();  // 显式关闭文件（析构时也会关，这里只是写明意图）
}
