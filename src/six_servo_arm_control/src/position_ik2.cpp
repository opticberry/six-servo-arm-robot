#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/joint_model_group.h>
#include <moveit/robot_state/robot_state.h>

namespace
{
constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
}

static void PrintStm32RadCommands(
  const rclcpp::Logger& logger,
  const std::string& label,
  const std::vector<double>& q,
  int time_ms);

static bool SolvePositionIK(
  const moveit::core::RobotModelConstPtr& robot_model,
  const moveit::core::JointModelGroup* jmg,
  const std::vector<double>& seed_q,
  double target_x,
  double target_y,
  double target_z,
  const Eigen::Vector3d& desired_dir,
  double ik_direction_weight,
  int ik_max_iters,
  double ik_position_tolerance,
  double ik_damping,
  double ik_step_scale,
  std::vector<double>& solution_q,
  double& final_pos_error,
  double& final_dir_error_deg);


int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared(
    "approx_ik_with_pregrasp_flow",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  double target_x = 0.0;
  double target_y = 0.0;
  double target_z = 0.20;
  double dir_x = 0.0;
  double dir_y = 0.0;
  double dir_z = -1.0;

  // User-provided pregrasp joint target.
  double pregrasp_j0 = 0.0;
  double pregrasp_j1 = 0.9;
  double pregrasp_j2 = 1.57;
  double pregrasp_j3 = -1.57;
  double pregrasp_j4 = 0.0;

  // Position IK parameters.
  int ik_max_iters = 120;
  double ik_position_tolerance = 0.005;  // 5 mm
  double ik_damping = 0.05;
  double ik_step_scale = 0.5;
  double ik_direction_weight = 0.0;

  int pregrasp_time_ms = 2000;
  int target_time_ms = 2500;

  node->get_parameter_or("x", target_x, target_x);
  node->get_parameter_or("y", target_y, target_y);
  node->get_parameter_or("z", target_z, target_z);
  node->get_parameter_or("dir_x", dir_x, dir_x);
  node->get_parameter_or("dir_y", dir_y, dir_y);
  node->get_parameter_or("dir_z", dir_z, dir_z);

  node->get_parameter_or("pregrasp_j0", pregrasp_j0, pregrasp_j0);
  node->get_parameter_or("pregrasp_j1", pregrasp_j1, pregrasp_j1);
  node->get_parameter_or("pregrasp_j2", pregrasp_j2, pregrasp_j2);
  node->get_parameter_or("pregrasp_j3", pregrasp_j3, pregrasp_j3);
  node->get_parameter_or("pregrasp_j4", pregrasp_j4, pregrasp_j4);

  node->get_parameter_or("ik_max_iters", ik_max_iters, ik_max_iters);
  node->get_parameter_or("ik_position_tolerance", ik_position_tolerance, ik_position_tolerance);
  node->get_parameter_or("ik_damping", ik_damping, ik_damping);
  node->get_parameter_or("ik_step_scale", ik_step_scale, ik_step_scale);
  node->get_parameter_or("ik_direction_weight", ik_direction_weight, ik_direction_weight);

  node->get_parameter_or("pregrasp_time_ms", pregrasp_time_ms, pregrasp_time_ms);
  node->get_parameter_or("target_time_ms", target_time_ms, target_time_ms);

  Eigen::Vector3d desired_dir(dir_x, dir_y, dir_z);
  if (desired_dir.norm() < 1e-9)
  {
    RCLCPP_ERROR(node->get_logger(), "Desired direction vector norm is zero.");
    rclcpp::shutdown();
    return 1;
  }
  desired_dir.normalize();

  moveit::planning_interface::MoveGroupInterface move_group(node, "arm");
  move_group.setPlanningTime(10.0);
  move_group.setNumPlanningAttempts(20);
  move_group.setMaxVelocityScalingFactor(0.1);
  move_group.setMaxAccelerationScalingFactor(0.1);

  const moveit::core::RobotModelConstPtr robot_model = move_group.getRobotModel();
  const moveit::core::JointModelGroup* jmg = robot_model->getJointModelGroup("arm");
  if (!jmg)
  {
    RCLCPP_ERROR(node->get_logger(), "Joint model group 'arm' not found.");
    rclcpp::shutdown();
    return 1;
  }


  //Move to pregrasp_q
  std::vector<double> pregrasp_q = {
    pregrasp_j0, pregrasp_j1, pregrasp_j2, pregrasp_j3, pregrasp_j4
  };

  PrintStm32RadCommands(node->get_logger(), "pregrasp", pregrasp_q, pregrasp_time_ms);

  RCLCPP_INFO(
    node->get_logger(),
    "Moving to pregrasp: [%.4f, %.4f, %.4f, %.4f, %.4f]",
    pregrasp_q[0], pregrasp_q[1], pregrasp_q[2], pregrasp_q[3], pregrasp_q[4]);

  move_group.setStartStateToCurrentState();
  if (!move_group.setJointValueTarget(pregrasp_q))
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to set pregrasp joint target.");
    rclcpp::shutdown();
    return 1;
  }

  moveit::planning_interface::MoveGroupInterface::Plan pregrasp_plan;
  if (!static_cast<bool>(move_group.plan(pregrasp_plan)))
  {
    RCLCPP_ERROR(node->get_logger(), "Planning to pregrasp failed.");
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "Pregrasp planning succeeded. Executing...");
  if (move_group.execute(pregrasp_plan) != moveit::core::MoveItErrorCode::SUCCESS)
  {
    RCLCPP_ERROR(node->get_logger(), "Execution to pregrasp failed.");
    rclcpp::shutdown();
    return 1;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  const auto& active_joint_models = jmg->getActiveJointModels();
  if (active_joint_models.size() != pregrasp_q.size())
  {
    RCLCPP_ERROR(node->get_logger(), "Unexpected joint count mismatch.");
    rclcpp::shutdown();
    return 1;
  }


  //IK
  std::vector<double> ik_q;
  double final_pos_error = std::numeric_limits<double>::infinity();
  double final_dir_error_deg = 0.0;

  RCLCPP_INFO(
    node->get_logger(),
    "Solving position IK for tool0 target xyz: [%.4f, %.4f, %.4f]",
    target_x, target_y, target_z);
  RCLCPP_INFO(
    node->get_logger(),
    "Soft direction target (tool0 local Z): [%.4f, %.4f, %.4f], weight=%.3f",
    desired_dir.x(), desired_dir.y(), desired_dir.z(), ik_direction_weight);

  if (!SolvePositionIK(
        robot_model,
        jmg,
        pregrasp_q,
        target_x,
        target_y,
        target_z,
        desired_dir,
        ik_direction_weight,
        ik_max_iters,
        ik_position_tolerance,
        ik_damping,
        ik_step_scale,
        ik_q,
        final_pos_error,
        final_dir_error_deg))
  {
    RCLCPP_ERROR(
      node->get_logger(),
      "Position IK failed. Final position error = %.4f m (%.2f cm), final direction error = %.2f deg",
      final_pos_error,
      final_pos_error * 100.0,
      final_dir_error_deg);
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(
    node->get_logger(),
    "Position IK succeeded. Final position error = %.4f m (%.2f cm), final direction error = %.2f deg",
    final_pos_error,
    final_pos_error * 100.0,
    final_dir_error_deg);

  RCLCPP_INFO(
    node->get_logger(),
    "IK joint target: [%.4f, %.4f, %.4f, %.4f, %.4f]",
    ik_q[0], ik_q[1], ik_q[2], ik_q[3], ik_q[4]);


  moveit::core::RobotState solved_state(robot_model);
  solved_state.setToDefaultValues();
  solved_state.setJointGroupPositions(jmg, ik_q);
  solved_state.update();

  const Eigen::Isometry3d& solved_tf = solved_state.getGlobalLinkTransform("tool0");
  const Eigen::Vector3d solved_p = solved_tf.translation();
  Eigen::Vector3d solved_dir = solved_tf.rotation().col(2);
  solved_dir.normalize();
  const double solved_dot = std::clamp(solved_dir.dot(desired_dir), -1.0, 1.0);
  const double solved_dir_error_deg = std::acos(solved_dot) * kRadToDeg;

   RCLCPP_INFO(
    node->get_logger(),
    "Predicted tool0 xyz: [%.4f, %.4f, %.4f]",
    solved_p.x(), solved_p.y(), solved_p.z());
  RCLCPP_INFO(
    node->get_logger(),
    "Predicted tool0 dir: [%.4f, %.4f, %.4f], dir err = %.2f deg",
    solved_dir.x(), solved_dir.y(), solved_dir.z(), solved_dir_error_deg);

  PrintStm32RadCommands(node->get_logger(), "target", ik_q, target_time_ms);

  move_group.setStartStateToCurrentState();
  if (!move_group.setJointValueTarget(ik_q))
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to set IK joint target.");
    rclcpp::shutdown();
    return 1;
  }

  moveit::planning_interface::MoveGroupInterface::Plan target_plan;
  if (!static_cast<bool>(move_group.plan(target_plan)))
  {
    RCLCPP_ERROR(node->get_logger(), "MoveIt planning to IK target failed.");
    rclcpp::shutdown();
    return 1;
  }

   RCLCPP_INFO(node->get_logger(), "Planning succeeded. Executing target motion...");
  if (move_group.execute(target_plan) != moveit::core::MoveItErrorCode::SUCCESS)
  {
    RCLCPP_ERROR(node->get_logger(), "Execution to IK target failed.");
    rclcpp::shutdown();
    return 1;
  }

  moveit::core::RobotState executed_state(robot_model);
  executed_state.setToDefaultValues();
  executed_state.setJointGroupPositions(jmg, ik_q);
  executed_state.update();

  const Eigen::Isometry3d& executed_tf = executed_state.getGlobalLinkTransform("tool0");
  const Eigen::Vector3d executed_p = executed_tf.translation();
  Eigen::Vector3d executed_dir = executed_tf.rotation().col(2);
  executed_dir.normalize();

  const double ex = executed_p.x() - target_x;
  const double ey = executed_p.y() - target_y;
  const double ez = executed_p.z() - target_z;
  const double exec_pos_err = std::sqrt(ex * ex + ey * ey + ez * ez);
  const double executed_dot = std::clamp(executed_dir.dot(desired_dir), -1.0, 1.0);
  const double exec_dir_err_deg = std::acos(executed_dot) * kRadToDeg;

  RCLCPP_INFO(
    node->get_logger(),
    "Executed xyz:     [%.4f, %.4f, %.4f]",
    executed_p.x(), executed_p.y(), executed_p.z());
  RCLCPP_INFO(
    node->get_logger(),
    "Executed dir:     [%.4f, %.4f, %.4f]",
    executed_dir.x(), executed_dir.y(), executed_dir.z());
  RCLCPP_INFO(
    node->get_logger(),
    "Executed pos err: %.4f m (%.2f cm)",
    exec_pos_err,
    exec_pos_err * 100.0);
  RCLCPP_INFO(
    node->get_logger(),
    "Executed dir err: %.2f deg",
    exec_dir_err_deg);

  rclcpp::shutdown();
  return 0;
}

static void PrintStm32RadCommands(
  const rclcpp::Logger& logger,
  const std::string& label,
  const std::vector<double>& q,
  int time_ms)
{
  RCLCPP_INFO(logger, "STM32 %s commands:", label.c_str());

  for (std::size_t i = 0; i < q.size(); ++i)
  {
    RCLCPP_INFO(logger, "joint %zu %d %d\n", i, (int)(q[i] * 1000), time_ms);
  }
}

static bool SolvePositionIK(
  const moveit::core::RobotModelConstPtr& robot_model,
  const moveit::core::JointModelGroup* jmg,
  const std::vector<double>& seed_q,
  double target_x,
  double target_y,
  double target_z,
  const Eigen::Vector3d& desired_dir,
  double ik_direction_weight,
  int ik_max_iters,
  double ik_position_tolerance,
  double ik_damping,
  double ik_step_scale,
  std::vector<double>& solution_q,
  double& final_pos_error,
  double& final_dir_error_deg)
{
  final_pos_error = std::numeric_limits<double>::infinity();
  final_dir_error_deg = 0.0;

  if (!robot_model || !jmg)
  {
    return false;
  }

  const auto& active_joint_models = jmg->getActiveJointModels();
  if (active_joint_models.size() != seed_q.size())
  {
    return false;
  }

  std::vector<double> q = seed_q;
  Eigen::Vector3d target(target_x, target_y, target_z);
  const bool use_direction_soft_constraint = ik_direction_weight > 0.0;

  moveit::core::RobotState state(robot_model);
  state.setToDefaultValues();

  for (int iter = 0; iter < ik_max_iters; ++iter)
  {
    state.setJointGroupPositions(jmg, q);
    state.enforceBounds(jmg);
    state.update();

    const Eigen::Isometry3d& tf = state.getGlobalLinkTransform("tool0");
    const Eigen::Vector3d current = tf.translation();
    const Eigen::Vector3d error = target - current;
    Eigen::Vector3d tool_dir = tf.rotation().col(2);
    tool_dir.normalize();

    final_pos_error = error.norm();
    final_dir_error_deg =
      std::acos(std::clamp(tool_dir.dot(desired_dir), -1.0, 1.0)) * kRadToDeg;

    if (final_pos_error < ik_position_tolerance)
    {
      state.copyJointGroupPositions(jmg, q);
      solution_q = q;
      return true;
    }

    Eigen::MatrixXd jacobian;
    const Eigen::Vector3d reference_point_position(0.0, 0.0, 0.0);

    if (!state.getJacobian(
          jmg,
          state.getLinkModel("tool0"),
          reference_point_position,
          jacobian))
    {
      return false;
    }

    if (jacobian.rows() < 3 || jacobian.cols() != static_cast<int>(q.size()))
    {
      return false;
    }

    Eigen::VectorXd task_error;
    Eigen::MatrixXd task_jacobian;

    if (use_direction_soft_constraint)
    {
      task_error.resize(6);
      task_jacobian.resize(6, jacobian.cols());

      task_error.head<3>() = error;
      task_error.tail<3>() = ik_direction_weight * tool_dir.cross(desired_dir);

      task_jacobian.topRows(3) = jacobian.topRows(3);
      task_jacobian.bottomRows(3) = ik_direction_weight * jacobian.bottomRows(3);
    }
    else
    {
      task_error = error;
      task_jacobian = jacobian.topRows(3);
    }

    // Damped least-squares:
    // dq = J^T * (J*J^T + lambda^2 * I)^(-1) * error
    Eigen::MatrixXd A =
      task_jacobian * task_jacobian.transpose() +
      (ik_damping * ik_damping) * Eigen::MatrixXd::Identity(task_jacobian.rows(), task_jacobian.rows());

    Eigen::VectorXd y = A.ldlt().solve(task_error);
    Eigen::VectorXd dq = task_jacobian.transpose() * y;

    if (!dq.allFinite())
    {
      return false;
    }

    for (std::size_t i = 0; i < q.size(); ++i)
    {
      q[i] += ik_step_scale * dq(static_cast<int>(i));
    }

    state.setJointGroupPositions(jmg, q);
    state.enforceBounds(jmg);
    state.copyJointGroupPositions(jmg, q);
  }

  solution_q = q;
  return false;
}
