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
  int ik_max_iters,
  double ik_position_tolerance,
  double ik_damping,
  double ik_step_scale,
  std::vector<double>& solution_q,
  double& final_pos_error);


int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared(
    "approx_ik_with_pregrasp_flow",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  double target_x = 0.0;
  double target_y = 0.0;
  double target_z = 0.20;

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

  int pregrasp_time_ms = 2000;
  int target_time_ms = 2500;

  node->get_parameter_or("x", target_x, target_x);
  node->get_parameter_or("y", target_y, target_y);
  node->get_parameter_or("z", target_z, target_z);

  node->get_parameter_or("pregrasp_j0", pregrasp_j0, pregrasp_j0);
  node->get_parameter_or("pregrasp_j1", pregrasp_j1, pregrasp_j1);
  node->get_parameter_or("pregrasp_j2", pregrasp_j2, pregrasp_j2);
  node->get_parameter_or("pregrasp_j3", pregrasp_j3, pregrasp_j3);
  node->get_parameter_or("pregrasp_j4", pregrasp_j4, pregrasp_j4);

  node->get_parameter_or("ik_max_iters", ik_max_iters, ik_max_iters);
  node->get_parameter_or("ik_position_tolerance", ik_position_tolerance, ik_position_tolerance);
  node->get_parameter_or("ik_damping", ik_damping, ik_damping);
  node->get_parameter_or("ik_step_scale", ik_step_scale, ik_step_scale);

  node->get_parameter_or("pregrasp_time_ms", pregrasp_time_ms, pregrasp_time_ms);
  node->get_parameter_or("target_time_ms", target_time_ms, target_time_ms);

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

  RCLCPP_INFO(
    node->get_logger(),
    "Solving position IK for tool0 target xyz: [%.4f, %.4f, %.4f]",
    target_x, target_y, target_z);

  if (!SolvePositionIK(
        robot_model,
        jmg,
        pregrasp_q,
        target_x,
        target_y,
        target_z,
        ik_max_iters,
        ik_position_tolerance,
        ik_damping,
        ik_step_scale,
        ik_q,
        final_pos_error))
  {
    RCLCPP_ERROR(
      node->get_logger(),
      "Position IK failed. Final position error = %.4f m (%.2f cm)",
      final_pos_error,
      final_pos_error * 100.0);
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(
    node->get_logger(),
    "Position IK succeeded. Final position error = %.4f m (%.2f cm)",
    final_pos_error,
    final_pos_error * 100.0);

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

   RCLCPP_INFO(
    node->get_logger(),
    "Predicted tool0 xyz: [%.4f, %.4f, %.4f]",
    solved_p.x(), solved_p.y(), solved_p.z());

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

  const double ex = executed_p.x() - target_x;
  const double ey = executed_p.y() - target_y;
  const double ez = executed_p.z() - target_z;
  const double exec_pos_err = std::sqrt(ex * ex + ey * ey + ez * ez);

  RCLCPP_INFO(
    node->get_logger(),
    "Executed xyz:     [%.4f, %.4f, %.4f]",
    executed_p.x(), executed_p.y(), executed_p.z());
  RCLCPP_INFO(
    node->get_logger(),
    "Executed pos err: %.4f m (%.2f cm)",
    exec_pos_err,
    exec_pos_err * 100.0);

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
  int ik_max_iters,
  double ik_position_tolerance,
  double ik_damping,
  double ik_step_scale,
  std::vector<double>& solution_q,
  double& final_pos_error)
{
  final_pos_error = std::numeric_limits<double>::infinity();

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

    final_pos_error = error.norm();

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

    // Only use the linear-position part of the Jacobian: 3 x N
    Eigen::MatrixXd J = jacobian.topRows(3);

    // Damped least-squares:
    // dq = J^T * (J*J^T + lambda^2 * I)^(-1) * error
    Eigen::Matrix3d A =
      J * J.transpose() + (ik_damping * ik_damping) * Eigen::Matrix3d::Identity();

    Eigen::Vector3d y = A.ldlt().solve(error);
    Eigen::VectorXd dq = J.transpose() * y;

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
