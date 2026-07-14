#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <iomanip>
#include <thread>

#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model/joint_model_group.h>
#include <moveit/robot_state/robot_state.h>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared(
    "approx_ik_joint_search",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  double target_x = 0.0;
  double target_y = 0.0;
  double target_z = 0.20;
  int samples = 4000;
  double goal_tolerance = 0.02;

  node->get_parameter_or("x", target_x, target_x);
  node->get_parameter_or("y", target_y, target_y);
  node->get_parameter_or("z", target_z, target_z);
  node->get_parameter_or("samples", samples, samples);
  node->get_parameter_or("goal_tolerance", goal_tolerance, goal_tolerance);

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

  moveit::core::RobotStatePtr current_state(new moveit::core::RobotState(robot_model));
  current_state->setToDefaultValues();

  std::vector<double> current_q;
  current_state->copyJointGroupPositions(jmg, current_q);

  const auto& active_joint_models = jmg->getActiveJointModels();
  if (active_joint_models.size() != current_q.size())
  {
    RCLCPP_ERROR(node->get_logger(), "Unexpected joint count mismatch.");
    rclcpp::shutdown();
    return 1;
  }

  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<double> unit01(0.0, 1.0);

  double best_score = std::numeric_limits<double>::infinity();
  double best_pos_error = std::numeric_limits<double>::infinity();
  std::vector<double> best_q = current_q;

  // 我们在当前姿态附近多采样，也偶尔全局采样。
  for (int i = 0; i < samples; ++i)
  {
    std::vector<double> q = current_q;

    for (std::size_t j = 0; j < active_joint_models.size(); ++j)
    {
      const auto* joint_model = active_joint_models[j];
      const std::string var_name = joint_model->getVariableNames()[0];
      const auto& bounds = joint_model->getVariableBounds(var_name);

      const double low = bounds.min_position_;
      const double high = bounds.max_position_;

      if (i < samples * 0.8)
      {
        // 80% 的样本在当前关节附近搜索，便于找到相邻可行解
        const double local_span = 0.35;  // rad
        const double local_low = std::max(low, current_q[j] - local_span);
        const double local_high = std::min(high, current_q[j] + local_span);
        std::uniform_real_distribution<double> local_dist(local_low, local_high);
        q[j] = local_dist(rng);
      }
      else
      {
        // 20% 做全局搜索，避免卡在局部
        std::uniform_real_distribution<double> global_dist(low, high);
        q[j] = global_dist(rng);
      }
    }

    moveit::core::RobotState candidate_state(*current_state);
    candidate_state.setJointGroupPositions(jmg, q);
    candidate_state.update();

    const Eigen::Isometry3d& tf = candidate_state.getGlobalLinkTransform("tool0");
    const Eigen::Vector3d p = tf.translation();

    const double dx = p.x() - target_x;
    const double dy = p.y() - target_y;
    const double dz = p.z() - target_z;
    const double pos_error = std::sqrt(dx * dx + dy * dy + dz * dz);

    // 关节变化惩罚：在同等位置误差下，优先选离当前姿态更近的解
    double joint_reg = 0.0;
    for (std::size_t j = 0; j < q.size(); ++j)
    {
      joint_reg += std::abs(q[j] - current_q[j]);
    }

    const double score = pos_error + 0.02 * joint_reg;

    if (score < best_score)
    {
      best_score = score;
      best_pos_error = pos_error;
      best_q = q;
    }

    if (best_pos_error < goal_tolerance)
    {
      break;
    }
  }

  RCLCPP_INFO(
    node->get_logger(),
    "Best position error: %.4f m, best joint target: [%.4f, %.4f, %.4f, %.4f, %.4f]",
    best_pos_error, best_q[0], best_q[1], best_q[2], best_q[3], best_q[4]);

  moveit::core::RobotState best_state(*current_state);
  best_state.setJointGroupPositions(jmg, best_q);
  best_state.update();

  const Eigen::Isometry3d& best_tf = best_state.getGlobalLinkTransform("tool0");
  const Eigen::Vector3d best_p = best_tf.translation();

  const double pred_dx = best_p.x() - target_x;
  const double pred_dy = best_p.y() - target_y;
  const double pred_dz = best_p.z() - target_z;
  const double pred_err = std::sqrt(pred_dx * pred_dx + pred_dy * pred_dy + pred_dz * pred_dz);

  RCLCPP_INFO(
    node->get_logger(),
    "Target tool0 xyz:  [%.4f, %.4f, %.4f]",
    target_x, target_y, target_z);

  RCLCPP_INFO(
    node->get_logger(),
    "Predicted xyz:     [%.4f, %.4f, %.4f]",
    best_p.x(), best_p.y(), best_p.z());

  RCLCPP_INFO(
    node->get_logger(),
    "Predicted error:   %.4f m (%.2f cm)",
    pred_err, pred_err * 100.0);

  if (best_pos_error > 0.05)
  {
    RCLCPP_ERROR(
      node->get_logger(),
      "No good approximate IK solution found. Best error is still %.4f m",
      best_pos_error);
    rclcpp::shutdown();
    return 1;
  }

  const bool target_set = move_group.setJointValueTarget(best_q);
  if (!target_set)
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to set joint target from approximate IK.");
    rclcpp::shutdown();
    return 1;
  }

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  const bool success = static_cast<bool>(move_group.plan(plan));

  if (!success)
  {
    RCLCPP_ERROR(node->get_logger(), "Joint-space planning failed after approximate IK.");
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "Planning succeeded. Executing...");
  const auto result = move_group.execute(plan);

  if (result == moveit::core::MoveItErrorCode::SUCCESS)
{
  RCLCPP_INFO(node->get_logger(), "Execution succeeded.");

  // 给 joint_states 一点时间更新
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  auto executed_state = move_group.getCurrentState(2.0);
  if (executed_state)
  {
    executed_state->update();
    const Eigen::Isometry3d& executed_tf = executed_state->getGlobalLinkTransform("tool0");
    const Eigen::Vector3d executed_p = executed_tf.translation();

    const double act_dx = executed_p.x() - target_x;
    const double act_dy = executed_p.y() - target_y;
    const double act_dz = executed_p.z() - target_z;
    const double act_err = std::sqrt(act_dx * act_dx + act_dy * act_dy + act_dz * act_dz);

    RCLCPP_INFO(
      node->get_logger(),
      "Executed xyz:      [%.4f, %.4f, %.4f]",
      executed_p.x(), executed_p.y(), executed_p.z());

    RCLCPP_INFO(
      node->get_logger(),
      "Executed error:    %.4f m (%.2f cm)",
      act_err, act_err * 100.0);
  }
  else
  {
    RCLCPP_WARN(
      node->get_logger(),
      "Execution succeeded, but current state was unavailable. Predicted xyz/error above is still valid.");
  }
}
  else
  {
    RCLCPP_ERROR(node->get_logger(), "Execution failed.");
  }

  rclcpp::shutdown();
  return 0;
}