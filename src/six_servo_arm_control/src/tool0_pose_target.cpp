#include <cmath>
#include <memory>
#include <string>

#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared(
    "tool0_pose_target",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  const double x = node->get_parameter("x").as_double();
  const double y = node->get_parameter("y").as_double();
  const double z = node->get_parameter("z").as_double();

  double qx = node->get_parameter("qx").as_double();
  double qy = node->get_parameter("qy").as_double();
  double qz = node->get_parameter("qz").as_double();
  double qw = node->get_parameter("qw").as_double();

  const double qnorm = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
  if (qnorm < 1e-9)
  {
    RCLCPP_ERROR(node->get_logger(), "Invalid quaternion: norm is zero.");
    rclcpp::shutdown();
    return 1;
  }

  qx /= qnorm;
  qy /= qnorm;
  qz /= qnorm;
  qw /= qnorm;

  moveit::planning_interface::MoveGroupInterface move_group(node, "arm");

  move_group.setEndEffectorLink("tool0");
  move_group.setPoseReferenceFrame("base_link");
  move_group.setStartStateToCurrentState();

  move_group.setPlanningTime(10.0);
  move_group.setNumPlanningAttempts(20);
  move_group.setMaxVelocityScalingFactor(0.1);
  move_group.setMaxAccelerationScalingFactor(0.1);

  move_group.setGoalPositionTolerance(0.01);
  move_group.setGoalOrientationTolerance(0.10);

  geometry_msgs::msg::Pose target_pose;
  target_pose.position.x = x;
  target_pose.position.y = y;
  target_pose.position.z = z;
  target_pose.orientation.x = qx;
  target_pose.orientation.y = qy;
  target_pose.orientation.z = qz;
  target_pose.orientation.w = qw;

  RCLCPP_INFO(
    node->get_logger(),
    "Planning to tool0 pose: pos=(%.3f, %.3f, %.3f), quat=(%.3f, %.3f, %.3f, %.3f)",
    x, y, z, qx, qy, qz, qw);

  move_group.setPoseTarget(target_pose, "tool0");

  moveit::planning_interface::MoveGroupInterface::Plan plan;
  const bool success = static_cast<bool>(move_group.plan(plan));

  if (!success)
  {
    RCLCPP_ERROR(node->get_logger(), "Planning failed.");
    move_group.clearPoseTargets();
    rclcpp::shutdown();
    return 1;
  }

  RCLCPP_INFO(node->get_logger(), "Planning succeeded. Executing...");
  const auto result = move_group.execute(plan);

  if (result == moveit::core::MoveItErrorCode::SUCCESS)
  {
    RCLCPP_INFO(node->get_logger(), "Execution succeeded.");
  }
  else
  {
    RCLCPP_ERROR(node->get_logger(), "Execution failed.");
  }

  move_group.clearPoseTargets();
  rclcpp::shutdown();
  return 0;
}