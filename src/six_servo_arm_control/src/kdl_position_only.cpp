#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model/joint_model_group.h>
#include <moveit/robot_state/robot_state.h>

static void PrintStm32RadCommands(
  const rclcpp::Logger& logger,
  const std::string& label,
  const std::vector<double>& q,
  int time_ms)
{
  RCLCPP_INFO(logger, "STM32 %s commands:", label.c_str());

  for (std::size_t i = 0; i < q.size(); ++i)
  {
    RCLCPP_INFO(logger, "joint %zu %d %d", i, static_cast<int>(q[i] * 1000.0), time_ms);
  }
}

static bool GetLastPlanJointTarget(
  const moveit::planning_interface::MoveGroupInterface::Plan& plan,
  std::vector<double>& q_out)
{
  const auto& traj = plan.trajectory_.joint_trajectory;
  if (traj.points.empty())
  {
    return false;
  }

  q_out = traj.points.back().positions;
  return !q_out.empty();
}

static bool ComputeTool0Position(
  const moveit::core::RobotModelConstPtr& robot_model,
  const moveit::core::JointModelGroup* jmg,
  const std::vector<double>& q,
  Eigen::Vector3d& pos_out)
{
  if (!robot_model || !jmg)
  {
    return false;
  }

  moveit::core::RobotState state(robot_model);
  state.setToDefaultValues();
  state.setJointGroupPositions(jmg, q);
  state.update();

  const Eigen::Isometry3d& tf = state.getGlobalLinkTransform("tool0");
  pos_out = tf.translation();
  return true;
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared(
    "kdl_position_only",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  double target_x = 0.0;
  double target_y = 0.0;
  double target_z = 0.20;

  double pregrasp_j0 = 0.0;
  double pregrasp_j1 = 0.9;
  double pregrasp_j2 = 1.57;
  double pregrasp_j3 = -1.57;
  double pregrasp_j4 = 0.0;

  double goal_position_tolerance = 0.005;
  double planning_time = 10.0;
  int num_planning_attempts = 20;

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

  node->get_parameter_or("goal_position_tolerance", goal_position_tolerance, goal_position_tolerance);
  node->get_parameter_or("planning_time", planning_time, planning_time);
  node->get_parameter_or("num_planning_attempts", num_planning_attempts, num_planning_attempts);

  node->get_parameter_or("pregrasp_time_ms", pregrasp_time_ms, pregrasp_time_ms);
  node->get_parameter_or("target_time_ms", target_time_ms, target_time_ms);

  moveit::planning_interface::MoveGroupInterface move_group(node, "arm");
  move_group.setEndEffectorLink("tool0");
  move_group.setPoseReferenceFrame("base_link");

  move_group.setPlanningTime(planning_time);
  move_group.setNumPlanningAttempts(num_planning_attempts);
  move_group.setMaxVelocityScalingFactor(0.1);
  move_group.setMaxAccelerationScalingFactor(0.1);
  move_group.setGoalPositionTolerance(goal_position_tolerance);

  const moveit::core::RobotModelConstPtr robot_model = move_group.getRobotModel();
  const moveit::core::JointModelGroup* jmg = robot_model->getJointModelGroup("arm");
  if (!jmg)
  {
    RCLCPP_ERROR(node->get_logger(), "Joint model group 'arm' not found.");
    rclcpp::shutdown();
    return 1;
  }

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

  RCLCPP_INFO(
    node->get_logger(),
    "Using default KDL position-only IK for tool0 target xyz: [%.4f, %.4f, %.4f]",
    target_x, target_y, target_z);

  move_group.clearPoseTargets();
  move_group.setStartStateToCurrentState();

  if (!move_group.setPositionTarget(target_x, target_y, target_z, "tool0"))
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to set position-only target for tool0.");
    rclcpp::shutdown();
    return 1;
  }

  auto t0 = std::chrono::steady_clock::now();

  moveit::planning_interface::MoveGroupInterface::Plan target_plan;
  if (!static_cast<bool>(move_group.plan(target_plan)))
  {
    RCLCPP_ERROR(node->get_logger(), "KDL position-only planning failed.");
    rclcpp::shutdown();
    return 1;
  }

  auto t1 = std::chrono::steady_clock::now();
  const double plan_time_ms =
    std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

  std::vector<double> kdl_q;
  if (!GetLastPlanJointTarget(target_plan, kdl_q))
  {
    RCLCPP_ERROR(node->get_logger(), "Planned trajectory is empty.");
    rclcpp::shutdown();
    return 1;
  }

  if (kdl_q.size() != pregrasp_q.size())
  {
    RCLCPP_ERROR(
      node->get_logger(),
      "Unexpected joint count in planned target: got %zu, expected %zu",
      kdl_q.size(),
      pregrasp_q.size());
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "KDL planning time: %.1f ms", plan_time_ms);
  RCLCPP_INFO(
    node->get_logger(),
    "KDL joint target: [%.4f, %.4f, %.4f, %.4f, %.4f]",
    kdl_q[0], kdl_q[1], kdl_q[2], kdl_q[3], kdl_q[4]);

  Eigen::Vector3d predicted_p;
  if (!ComputeTool0Position(robot_model, jmg, kdl_q, predicted_p))
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to compute FK for KDL joint target.");
    rclcpp::shutdown();
    return 1;
  }

  const double px = predicted_p.x() - target_x;
  const double py = predicted_p.y() - target_y;
  const double pz = predicted_p.z() - target_z;
  const double predicted_pos_err = std::sqrt(px * px + py * py + pz * pz);

  RCLCPP_INFO(
    node->get_logger(),
    "Predicted tool0 xyz: [%.4f, %.4f, %.4f]",
    predicted_p.x(), predicted_p.y(), predicted_p.z());
  RCLCPP_INFO(
    node->get_logger(),
    "Predicted pos err:  %.4f m (%.2f cm)",
    predicted_pos_err,
    predicted_pos_err * 100.0);

  PrintStm32RadCommands(node->get_logger(), "target", kdl_q, target_time_ms);

  RCLCPP_INFO(node->get_logger(), "KDL planning succeeded. Executing target motion...");
  if (move_group.execute(target_plan) != moveit::core::MoveItErrorCode::SUCCESS)
  {
    RCLCPP_ERROR(node->get_logger(), "Execution to KDL target failed.");
    rclcpp::shutdown();
    return 1;
  }

  Eigen::Vector3d executed_p;
  if (!ComputeTool0Position(robot_model, jmg, kdl_q, executed_p))
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to compute executed FK.");
    rclcpp::shutdown();
    return 1;
  }

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