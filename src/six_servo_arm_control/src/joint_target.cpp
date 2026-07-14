#include <memory>
#include <string>
#include <vector>

#include <moveit/move_group_interface/move_group_interface.h>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared(
    "joint_target",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  double j0 = 0.0, j1 = 0.0, j2 = 0.0, j3 = 0.0, j4 = 0.0;

  node->get_parameter("j0", j0);
  node->get_parameter("j1", j1);
  node->get_parameter("j2", j2);
  node->get_parameter("j3", j3);
  node->get_parameter("j4", j4);

  moveit::planning_interface::MoveGroupInterface move_group(node, "arm");

  move_group.setStartStateToCurrentState();
  move_group.setPlanningTime(10.0);
  move_group.setNumPlanningAttempts(20);
  move_group.setMaxVelocityScalingFactor(0.1);
  move_group.setMaxAccelerationScalingFactor(0.1);

  std::vector<double> joint_values = {j0, j1, j2, j3, j4};

  RCLCPP_INFO(
    node->get_logger(),
    "Planning to joint target: [%.4f, %.4f, %.4f, %.4f, %.4f]",
    j0, j1, j2, j3, j4);

  bool target_set = move_group.setJointValueTarget(joint_values);

  if (!target_set)
  {
    RCLCPP_ERROR(node->get_logger(), "Failed to set joint target.");
    rclcpp::shutdown();
    return 1;
  }

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool success = static_cast<bool>(move_group.plan(plan));

  if (!success)
  {
    RCLCPP_ERROR(node->get_logger(), "Joint-space planning failed.");
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "Planning succeeded. Executing...");
  auto result = move_group.execute(plan);

  if (result == moveit::core::MoveItErrorCode::SUCCESS)
  {
    RCLCPP_INFO(node->get_logger(), "Execution succeeded.");
  }
  else
  {
    RCLCPP_ERROR(node->get_logger(), "Execution failed.");
  }

  rclcpp::shutdown();
  return 0;
}