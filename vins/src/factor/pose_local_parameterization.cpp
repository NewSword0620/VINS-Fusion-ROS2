/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 *
 * This file is part of VINS.
 *
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#include "pose_local_parameterization.h"

// 求雅可比处，平移本来导数就是 1，旋转相关的导数在因子里已经算完了，
// 因此雅可比对应位置写的单位阵，相当于复制。
// J_final = J_factor * J_plus, J_factor 已经算完了旋所有导数

// ambient（3+4） ⊕ tangent（3+3） → ambient
bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}

// Plus(x, delta) 对delta的雅可比
bool PoseLocalParameterization::PlusJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}

// ambient ⊖ ambient → tangent
bool PoseLocalParameterization::Minus(const double *y, const double *x, double *y_minus_x) const
{
    Eigen::Map<const Eigen::Vector3d> p_y(y);
    Eigen::Map<const Eigen::Quaterniond> q_y(y + 3);
    Eigen::Map<const Eigen::Vector3d> p_x(x);
    Eigen::Map<const Eigen::Quaterniond> q_x(x + 3);

    Eigen::Map<Eigen::Vector3d> dp(y_minus_x);
    dp = p_y - p_x;

    Eigen::Quaterniond dq = q_x.inverse() * q_y;
    // Extract the rotation vector (angle-axis) from the quaternion
    Eigen::AngleAxisd aa(dq);
    Eigen::Map<Eigen::Vector3d> dtheta(y_minus_x + 3);
    dtheta = aa.angle() * aa.axis();

    return true;
}

// Minus(y, x) 对 y 的雅可比
bool PoseLocalParameterization::MinusJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> j(jacobian);
    j.setZero();
    j.topLeftCorner<3, 3>().setIdentity();
    // For small perturbations, the Jacobian of Minus w.r.t. quaternion is approximately 2 * [qw*I + [qxyz]_x]^-1
    // but for the purpose of optimization, identity on the tangent space is sufficient
    j.block<3, 3>(3, 3).setIdentity();

    return true;
}
