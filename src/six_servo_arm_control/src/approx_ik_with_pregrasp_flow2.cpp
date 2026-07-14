#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>

#include <moveit/collision_detection/collision_common.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/joint_model_group.h>
#include <moveit/robot_state/robot_state.h>

struct Candidate
{
  std::vector<double> q;
  Eigen::Vector3d pos;
  Eigen::Vector3d dir;
  Eigen::Vector3d side;
  double pos_error;
  double dir_error;
  double side_error;
  double score;
};

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

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared(
    "approx_ik_with_pregrasp_flow",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  double target_x = 0.0;
  double target_y = 0.0;
  double target_z = 0.20;

  // Default forward-down direction.
  double dir_x = 1.0;
  double dir_y = 0.0;
  double dir_z = -1.0;

  // Keep the fingers left-right rather than up-down.
  double side_x = 0.0;
  double side_y = 1.0;
  double side_z = 0.0;

  // User-provided pregrasp joint target.
  double pregrasp_j0 = 0.0;
  double pregrasp_j1 = 0.9;
  double pregrasp_j2 = 1.57;
  double pregrasp_j3 = -1.57;
  double pregrasp_j4 = 0.0;

  int samples = 8000;
  int top_k = 30;
  double accept_position_error = 0.05;

  double weight_position = 1.0;
  double weight_direction = 0.10;
  double weight_side = 0.15;
  double weight_joint_reg = 0.02;

  double side_alignment_min = 0.60;
  double local_ratio = 0.80;
  double local_span = 0.60;

  node->get_parameter_or("x", target_x, target_x);
  node->get_parameter_or("y", target_y, target_y);
  node->get_parameter_or("z", target_z, target_z);

  node->get_parameter_or("dir_x", dir_x, dir_x);
  node->get_parameter_or("dir_y", dir_y, dir_y);
  node->get_parameter_or("dir_z", dir_z, dir_z);

  node->get_parameter_or("side_x", side_x, side_x);
  node->get_parameter_or("side_y", side_y, side_y);
  node->get_parameter_or("side_z", side_z, side_z);

  node->get_parameter_or("pregrasp_j0", pregrasp_j0, pregrasp_j0);
  node->get_parameter_or("pregrasp_j1", pregrasp_j1, pregrasp_j1);
  node->get_parameter_or("pregrasp_j2", pregrasp_j2, pregrasp_j2);
  node->get_parameter_or("pregrasp_j3", pregrasp_j3, pregrasp_j3);
  node->get_parameter_or("pregrasp_j4", pregrasp_j4, pregrasp_j4);

  node->get_parameter_or("samples", samples, samples);
  node->get_parameter_or("top_k", top_k, top_k);
  node->get_parameter_or("accept_position_error", accept_position_error, accept_position_error);

  node->get_parameter_or("weight_position", weight_position, weight_position);
  node->get_parameter_or("weight_direction", weight_direction, weight_direction);
  node->get_parameter_or("weight_side", weight_side, weight_side);
  node->get_parameter_or("weight_joint_reg", weight_joint_reg, weight_joint_reg);

  node->get_parameter_or("side_alignment_min", side_alignment_min, side_alignment_min);
  node->get_parameter_or("local_ratio", local_ratio, local_ratio);
  node->get_parameter_or("local_span", local_span, local_span);

  Eigen::Vector3d desired_dir(dir_x, dir_y, dir_z);
  if (desired_dir.norm() < 1e-9)
  {
    RCLCPP_ERROR(node->get_logger(), "Desired direction vector norm is zero.");
    rclcpp::shutdown();
    return 1;
  }
  desired_dir.normalize();

  Eigen::Vector3d desired_side(side_x, side_y, side_z);
  if (desired_side.norm() < 1e-9)
  {
    RCLCPP_ERROR(node->get_logger(), "Desired side vector norm is zero.");
    rclcpp::shutdown();
    return 1;
  }
  desired_side.normalize();

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

  std::vector<double> pregrasp_q = {
    pregrasp_j0, pregrasp_j1, pregrasp_j2, pregrasp_j3, pregrasp_j4
  };

  PrintStm32RadCommands(node->get_logger(), "pregrasp", pregrasp_q, 2000);

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

  // MoveIt execute() returns after execution finishes. We still wait briefly to
  // keep the simulated state and our search seed nicely aligned.
  std::this_thread::sleep_for(std::chrono::milliseconds(300));

  const auto& active_joint_models = jmg->getActiveJointModels();
  if (active_joint_models.size() != pregrasp_q.size())
  {
    RCLCPP_ERROR(node->get_logger(), "Unexpected joint count mismatch.");
    rclcpp::shutdown();
    return 1;
  }

  moveit::core::RobotState seed_state(robot_model);
  seed_state.setToDefaultValues();
  seed_state.setJointGroupPositions(jmg, pregrasp_q);
  seed_state.enforceBounds(jmg);
  seed_state.copyJointGroupPositions(jmg, pregrasp_q);
  seed_state.update();

  planning_scene::PlanningScene planning_scene(robot_model);
  collision_detection::CollisionRequest collision_request;
  collision_detection::CollisionResult collision_result;

  std::mt19937 rng(std::random_device{}());
  std::vector<Candidate> top_candidates;
  top_candidates.reserve(top_k);

  auto try_insert_candidate = [&](Candidate&& candidate) {
    top_candidates.push_back(std::move(candidate));
    std::sort(
      top_candidates.begin(),
      top_candidates.end(),
      [](const Candidate& a, const Candidate& b) { return a.score < b.score; });

    if (static_cast<int>(top_candidates.size()) > top_k)
    {
      top_candidates.resize(top_k);
    }
  };

  for (int i = 0; i < samples; ++i)
  {
    std::vector<double> q = pregrasp_q;

    for (std::size_t j = 0; j < active_joint_models.size(); ++j)
    {
      const auto* joint_model = active_joint_models[j];
      const std::string var_name = joint_model->getVariableNames()[0];
      const auto& bounds = joint_model->getVariableBounds(var_name);

      const double low = bounds.min_position_;
      const double high = bounds.max_position_;

      if (static_cast<double>(i) < static_cast<double>(samples) * local_ratio)
      {
        const double local_low = std::max(low, pregrasp_q[j] - local_span);
        const double local_high = std::min(high, pregrasp_q[j] + local_span);
        std::uniform_real_distribution<double> local_dist(local_low, local_high);
        q[j] = local_dist(rng);
      }
      else
      {
        std::uniform_real_distribution<double> global_dist(low, high);
        q[j] = global_dist(rng);
      }
    }

    moveit::core::RobotState candidate_state(seed_state);
    candidate_state.setJointGroupPositions(jmg, q);
    if (!candidate_state.satisfiesBounds(jmg))
    {
      continue;
    }

    candidate_state.update();

    collision_result.clear();
    planning_scene.checkSelfCollision(collision_request, collision_result, candidate_state);
    if (collision_result.collision)
    {
      continue;
    }

    const Eigen::Isometry3d& tf = candidate_state.getGlobalLinkTransform("tool0");
    const Eigen::Vector3d p = tf.translation();
    Eigen::Vector3d tool_dir = tf.rotation() * Eigen::Vector3d::UnitZ();
    tool_dir.normalize();
    Eigen::Vector3d tool_side = tf.rotation() * Eigen::Vector3d::UnitY();
    tool_side.normalize();

    const double dx = p.x() - target_x;
    const double dy = p.y() - target_y;
    const double dz = p.z() - target_z;
    const double pos_error = std::sqrt(dx * dx + dy * dy + dz * dz);

    const double dot = std::clamp(tool_dir.dot(desired_dir), -1.0, 1.0);
    const double dir_error = 1.0 - dot;

    const double side_alignment = std::clamp(std::abs(tool_side.dot(desired_side)), 0.0, 1.0);
    if (side_alignment < side_alignment_min)
    {
      continue;
    }
    const double side_error = 1.0 - side_alignment;

    double joint_reg = 0.0;
    for (std::size_t j = 0; j < q.size(); ++j)
    {
      joint_reg += std::abs(q[j] - pregrasp_q[j]);
    }

    const double score =
      weight_position * pos_error +
      weight_direction * dir_error +
      weight_side * side_error +
      weight_joint_reg * joint_reg;

    try_insert_candidate(Candidate{q, p, tool_dir, tool_side, pos_error, dir_error, side_error, score});
  }

  if (top_candidates.empty())
  {
    RCLCPP_ERROR(node->get_logger(), "No collision-free candidate was found during search.");
    rclcpp::shutdown();
    return 1;
  }

  constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;

  RCLCPP_INFO(node->get_logger(), "Target xyz:        [%.4f, %.4f, %.4f]", target_x, target_y, target_z);
  RCLCPP_INFO(node->get_logger(), "Desired direction: [%.4f, %.4f, %.4f]", desired_dir.x(), desired_dir.y(), desired_dir.z());
  RCLCPP_INFO(node->get_logger(), "Desired side:      [%.4f, %.4f, %.4f]", desired_side.x(), desired_side.y(), desired_side.z());
  RCLCPP_INFO(node->get_logger(), "Pregrasp seed:     [%.4f, %.4f, %.4f, %.4f, %.4f]",
              pregrasp_q[0], pregrasp_q[1], pregrasp_q[2], pregrasp_q[3], pregrasp_q[4]);
  RCLCPP_INFO(node->get_logger(), "Kept top %zu candidates. Trying them one by one in MoveIt...", top_candidates.size());

  for (std::size_t i = 0; i < top_candidates.size(); ++i)
  {
    const Candidate& c = top_candidates[i];
    const double dir_dot = std::clamp(c.dir.dot(desired_dir), -1.0, 1.0);
    const double dir_deg = std::acos(dir_dot) * kRadToDeg;
    const double side_dot = std::clamp(std::abs(c.side.dot(desired_side)), 0.0, 1.0);
    const double side_deg = std::acos(side_dot) * kRadToDeg;

    RCLCPP_INFO(
      node->get_logger(),
      "Candidate %zu: pos_err=%.4f m, dir_err=%.2f deg, side_err=%.2f deg, score=%.4f, q=[%.4f, %.4f, %.4f, %.4f, %.4f]",
      i, c.pos_error, dir_deg, side_deg, c.score, c.q[0], c.q[1], c.q[2], c.q[3], c.q[4]);

    move_group.setStartStateToCurrentState();
    if (!move_group.setJointValueTarget(c.q))
    {
      RCLCPP_WARN(node->get_logger(), "Candidate %zu: setJointValueTarget failed.", i);
      continue;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (!static_cast<bool>(move_group.plan(plan)))
    {
      RCLCPP_WARN(node->get_logger(), "Candidate %zu: MoveIt planning failed.", i);
      continue;
    }

    PrintStm32RadCommands(node->get_logger(), "target", c.q, 2500);
    RCLCPP_INFO(node->get_logger(), "Candidate %zu: planning succeeded. Executing...", i);
    if (move_group.execute(plan) != moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_WARN(node->get_logger(), "Candidate %zu: execution failed.", i);
      continue;
    }

    moveit::core::RobotState executed_state(robot_model);
    executed_state.setToDefaultValues();
    executed_state.setJointGroupPositions(jmg, c.q);
    executed_state.update();

    const Eigen::Isometry3d& executed_tf = executed_state.getGlobalLinkTransform("tool0");
    const Eigen::Vector3d executed_p = executed_tf.translation();
    Eigen::Vector3d executed_dir = executed_tf.rotation() * Eigen::Vector3d::UnitZ();
    executed_dir.normalize();
    Eigen::Vector3d executed_side = executed_tf.rotation() * Eigen::Vector3d::UnitY();
    executed_side.normalize();

    const double ex = executed_p.x() - target_x;
    const double ey = executed_p.y() - target_y;
    const double ez = executed_p.z() - target_z;
    const double exec_pos_err = std::sqrt(ex * ex + ey * ey + ez * ez);

    const double exec_dir_dot = std::clamp(executed_dir.dot(desired_dir), -1.0, 1.0);
    const double exec_dir_deg = std::acos(exec_dir_dot) * kRadToDeg;
    const double exec_side_dot = std::clamp(std::abs(executed_side.dot(desired_side)), 0.0, 1.0);
    const double exec_side_deg = std::acos(exec_side_dot) * kRadToDeg;

    RCLCPP_INFO(node->get_logger(), "Executed xyz:      [%.4f, %.4f, %.4f]",
                executed_p.x(), executed_p.y(), executed_p.z());
    RCLCPP_INFO(node->get_logger(), "Executed dir:      [%.4f, %.4f, %.4f]",
                executed_dir.x(), executed_dir.y(), executed_dir.z());
    RCLCPP_INFO(node->get_logger(), "Executed side:     [%.4f, %.4f, %.4f]",
                executed_side.x(), executed_side.y(), executed_side.z());
    RCLCPP_INFO(node->get_logger(), "Executed pos err:  %.4f m (%.2f cm)",
                exec_pos_err, exec_pos_err * 100.0);
    RCLCPP_INFO(node->get_logger(), "Executed dir err:  %.2f deg", exec_dir_deg);
    RCLCPP_INFO(node->get_logger(), "Executed side err: %.2f deg", exec_side_deg);

    rclcpp::shutdown();
    return 0;
  }

  RCLCPP_ERROR(node->get_logger(), "All top-%d candidates failed in MoveIt planning/execution.", top_k);
  rclcpp::shutdown();
  return 1;
}
