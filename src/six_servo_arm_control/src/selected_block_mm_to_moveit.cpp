#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <rclcpp/rclcpp.hpp>

class SelectedBlockMoveItNode : public rclcpp::Node
{
public:
  SelectedBlockMoveItNode()
  : Node("selected_block_mm_to_moveit_node")
  {
    // A4 plane -> base_link mapping.
    declare_parameter("a4_origin_x_m", 0.10);
    declare_parameter("a4_origin_y_m", 0.00);
    declare_parameter("a4_x_sign", 1.0);
    declare_parameter("a4_y_sign", 1.0);
    declare_parameter("table_z_m", 0.02);

    // Grasp strategy.
    declare_parameter("pregrasp_z_offset_m", 0.06);
    declare_parameter("grasp_z_offset_m", 0.01);

    // MoveIt settings.
    declare_parameter("move_group_name", "arm");
    declare_parameter("end_effector_link", "tool0");
    declare_parameter("reference_frame", "base_link");
    declare_parameter("planning_time_s", 8.0);
    declare_parameter("velocity_scaling", 0.10);
    declare_parameter("acceleration_scaling", 0.10);
    declare_parameter("goal_position_tolerance", 0.01);
    declare_parameter("goal_orientation_tolerance", 1.57);

    // Safety switch.
    declare_parameter("execute_on_receive", false);

    // Fixed "downward" tool pose for a 5-DOF arm.
    declare_parameter("target_qx", 0.0);
    declare_parameter("target_qy", 1.0);
    declare_parameter("target_qz", 0.0);
    declare_parameter("target_qw", 0.0);

    a4_origin_x_m_ = get_parameter("a4_origin_x_m").as_double();
    a4_origin_y_m_ = get_parameter("a4_origin_y_m").as_double();
    a4_x_sign_ = get_parameter("a4_x_sign").as_double();
    a4_y_sign_ = get_parameter("a4_y_sign").as_double();
    table_z_m_ = get_parameter("table_z_m").as_double();

    pregrasp_z_offset_m_ = get_parameter("pregrasp_z_offset_m").as_double();
    grasp_z_offset_m_ = get_parameter("grasp_z_offset_m").as_double();

    move_group_name_ = get_parameter("move_group_name").as_string();
    end_effector_link_ = get_parameter("end_effector_link").as_string();
    reference_frame_ = get_parameter("reference_frame").as_string();
    planning_time_s_ = get_parameter("planning_time_s").as_double();
    velocity_scaling_ = get_parameter("velocity_scaling").as_double();
    acceleration_scaling_ = get_parameter("acceleration_scaling").as_double();
    goal_position_tolerance_ = get_parameter("goal_position_tolerance").as_double();
    goal_orientation_tolerance_ = get_parameter("goal_orientation_tolerance").as_double();
    execute_on_receive_ = get_parameter("execute_on_receive").as_bool();

    target_qx_ = get_parameter("target_qx").as_double();
    target_qy_ = get_parameter("target_qy").as_double();
    target_qz_ = get_parameter("target_qz").as_double();
    target_qw_ = get_parameter("target_qw").as_double();

    selected_block_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      "/selected_block_mm",
      10,
      std::bind(&SelectedBlockMoveItNode::onSelectedBlock, this, std::placeholders::_1));

    selected_block_base_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
      "/selected_block_base", 10);

    RCLCPP_INFO(get_logger(), "selected_block_mm -> base_link -> MoveIt node ready.");
    RCLCPP_INFO(
      get_logger(),
      "execute_on_receive=%s, move_group=%s, ee_link=%s, frame=%s",
      execute_on_receive_ ? "true" : "false",
      move_group_name_.c_str(),
      end_effector_link_.c_str(),
      reference_frame_.c_str());
  }

  void initMoveIt()
  {
    move_group_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), move_group_name_);

    move_group_->setPoseReferenceFrame(reference_frame_);
    move_group_->setPlanningTime(planning_time_s_);
    move_group_->setNumPlanningAttempts(10);
    move_group_->setMaxVelocityScalingFactor(velocity_scaling_);
    move_group_->setMaxAccelerationScalingFactor(acceleration_scaling_);
    move_group_->setGoalPositionTolerance(goal_position_tolerance_);
    move_group_->setGoalOrientationTolerance(goal_orientation_tolerance_);
    move_group_->setEndEffectorLink(end_effector_link_);

    RCLCPP_INFO(get_logger(), "MoveIt interface initialized.");
  }

private:
  void onSelectedBlock(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    if (!msg)
    {
      return;
    }

    // Input topic uses A4-plane millimeters in point.x/point.y.
    const double x_mm = msg->point.x;
    const double y_mm = msg->point.y;

    const double x_base = a4_origin_x_m_ + a4_x_sign_ * (x_mm * 0.001);
    const double y_base = a4_origin_y_m_ + a4_y_sign_ * (y_mm * 0.001);
    const double z_base = table_z_m_;

    geometry_msgs::msg::PointStamped base_msg;
    base_msg.header.stamp = now();
    base_msg.header.frame_id = reference_frame_;
    base_msg.point.x = x_base;
    base_msg.point.y = y_base;
    base_msg.point.z = z_base;
    selected_block_base_pub_->publish(base_msg);

    RCLCPP_INFO(
      get_logger(),
      "A4(mm)=(%.2f, %.2f) -> base_link(m)=(%.3f, %.3f, %.3f)",
      x_mm, y_mm, x_base, y_base, z_base);

    if (!execute_on_receive_)
    {
      return;
    }

    if (!move_group_)
    {
      RCLCPP_ERROR(get_logger(), "MoveIt is not initialized.");
      return;
    }

    // Framework choice: first move to a safe pregrasp pose above the block.
    const double pregrasp_z = z_base + pregrasp_z_offset_m_;
    const double grasp_z = z_base + grasp_z_offset_m_;

    if (!planAndExecutePregrasp(x_base, y_base, pregrasp_z))
    {
      RCLCPP_ERROR(get_logger(), "Pregrasp planning/execution failed.");
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "Pregrasp succeeded. Grasp candidate would be (%.3f, %.3f, %.3f).",
      x_base, y_base, grasp_z);

    // TODO:
    // 1) move down to grasp_z
    // 2) close gripper
    // 3) move back to pregrasp_z
  }

  bool planAndExecutePregrasp(double x, double y, double z)
  {
    geometry_msgs::msg::Pose target_pose;
    target_pose.position.x = x;
    target_pose.position.y = y;
    target_pose.position.z = z;
    target_pose.orientation.x = target_qx_;
    target_pose.orientation.y = target_qy_;
    target_pose.orientation.z = target_qz_;
    target_pose.orientation.w = target_qw_;

    RCLCPP_INFO(
      get_logger(),
      "Planning pregrasp pose: pos=(%.3f, %.3f, %.3f), quat=(%.3f, %.3f, %.3f, %.3f)",
      target_pose.position.x,
      target_pose.position.y,
      target_pose.position.z,
      target_pose.orientation.x,
      target_pose.orientation.y,
      target_pose.orientation.z,
      target_pose.orientation.w);

    move_group_->setStartStateToCurrentState();
    move_group_->setPoseTarget(target_pose, end_effector_link_);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool success = static_cast<bool>(move_group_->plan(plan));
    if (!success)
    {
      RCLCPP_ERROR(get_logger(), "MoveIt planning failed.");
      move_group_->clearPoseTargets();
      return false;
    }

    RCLCPP_INFO(get_logger(), "MoveIt planning succeeded.");

    const auto result = move_group_->execute(plan);
    move_group_->clearPoseTargets();

    if (result != moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_ERROR(get_logger(), "MoveIt execution failed.");
      return false;
    }

    RCLCPP_INFO(get_logger(), "MoveIt execution succeeded.");
    return true;
  }

private:
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr selected_block_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr selected_block_base_pub_;
  std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  double a4_origin_x_m_{0.0};
  double a4_origin_y_m_{0.0};
  double a4_x_sign_{1.0};
  double a4_y_sign_{1.0};
  double table_z_m_{0.0};
  double pregrasp_z_offset_m_{0.06};
  double grasp_z_offset_m_{0.01};

  std::string move_group_name_{"arm"};
  std::string end_effector_link_{"tool0"};
  std::string reference_frame_{"base_link"};

  double planning_time_s_{8.0};
  double velocity_scaling_{0.10};
  double acceleration_scaling_{0.10};
  double goal_position_tolerance_{0.01};
  double goal_orientation_tolerance_{1.57};
  bool execute_on_receive_{false};

  double target_qx_{0.0};
  double target_qy_{1.0};
  double target_qz_{0.0};
  double target_qw_{0.0};
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<SelectedBlockMoveItNode>();
  node->initMoveIt();

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
