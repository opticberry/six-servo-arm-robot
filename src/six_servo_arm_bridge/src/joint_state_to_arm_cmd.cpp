#include <rclcpp/rclcpp.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <six_servo_arm_msgs/msg/arm_joint_command.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <string>
#include <unordered_map>

class JointStateToArmCmdNode : public rclcpp::Node
{
public:
  JointStateToArmCmdNode()
  : Node("joint_state_to_arm_cmd")
  {
    publish_period_ms_ = declare_parameter<int>("publish_period_ms", 40);
    move_time_ms_ = declare_parameter<int>("move_time_ms", 60);
    deadband_mrad_ = declare_parameter<int>("deadband_mrad", 5);
    require_gripper_ = declare_parameter<bool>("require_gripper", false);
    control_gripper_ = declare_parameter<bool>("control_gripper", false);
    gripper_hold_position_rad_ = static_cast<float>(
      declare_parameter<double>("gripper_hold_position_rad", -0.488));

    joint_name_to_index_ = {
      {"joint_0_base_yaw", 0},
      {"joint_1_shoulder_pitch", 1},
      {"joint_2_elbow_pitch", 2},
      {"joint_3_wrist_pitch", 3},
      {"joint_4_wrist_yaw", 4},
      {"joint_5_gripper", 5},
    };

    latest_rad_.fill(0.0f);
    latest_rad_[5] = gripper_hold_position_rad_;
    last_sent_mrad_.fill(0);
    joint_seen_.fill(false);
    joint_seen_[5] = !control_gripper_;

    arm_cmd_pub_ =
      create_publisher<six_servo_arm_msgs::msg::ArmJointCommand>(
        "/arm_joint_cmd",
        10);

    joint_state_sub_ =
      create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states",
        10,
        std::bind(&JointStateToArmCmdNode::joint_state_callback, this, std::placeholders::_1));

    timer_ =
      create_wall_timer(
        std::chrono::milliseconds(publish_period_ms_),
        std::bind(&JointStateToArmCmdNode::timer_callback, this));

    RCLCPP_INFO(
      get_logger(),
      "joint_state_to_arm_cmd started. control_gripper=%s hold_gripper=%.3f rad",
      control_gripper_ ? "true" : "false",
      gripper_hold_position_rad_);
  }

private:
  void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    const auto n = std::min(msg->name.size(), msg->position.size());

    for (size_t i = 0; i < n; ++i)
    {
      auto it = joint_name_to_index_.find(msg->name[i]);
      if (it == joint_name_to_index_.end())
      {
        continue;
      }

      const size_t joint_index = static_cast<size_t>(it->second);
      if (joint_index == 5 && !control_gripper_)
      {
        continue;
      }
      latest_rad_[joint_index] = static_cast<float>(msg->position[i]);
      joint_seen_[joint_index] = true;
      has_joint_state_ = true;
    }
  }

  bool has_significant_change() const
  {
    const size_t required_count = require_gripper_ ? 6 : 5;

    for (size_t i = 0; i < required_count; ++i)
    {
      if (!joint_seen_[i])
      {
        continue;
      }

      const int mrad = static_cast<int>(std::lround(latest_rad_[i] * 1000.0f));
      if (std::abs(mrad - last_sent_mrad_[i]) >= deadband_mrad_)
      {
        return true;
      }
    }

    return false;
  }

  bool has_all_required_joints() const
  {
    const size_t required_count = require_gripper_ ? 6 : 5;

    for (size_t i = 0; i < required_count; ++i)
    {
      if (!joint_seen_[i])
      {
        return false;
      }
    }

    return true;
  }

  void timer_callback()
  {
    if (!has_joint_state_)
    {
      return;
    }

    if (!has_all_required_joints())
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(),
        *get_clock(),
        2000,
        "Waiting for required joints in /joint_states...");
      return;
    }

    if (!has_significant_change())
    {
      return;
    }

    six_servo_arm_msgs::msg::ArmJointCommand cmd;

    for (size_t i = 0; i < latest_rad_.size(); ++i)
    {
      if (i == 5 && !control_gripper_)
      {
        cmd.position[i] = gripper_hold_position_rad_;
      }
      else
      {
        cmd.position[i] = latest_rad_[i];
      }
      last_sent_mrad_[i] = static_cast<int>(std::lround(cmd.position[i] * 1000.0f));
    }

    cmd.time_ms = static_cast<uint16_t>(move_time_ms_);
    arm_cmd_pub_->publish(cmd);

    RCLCPP_INFO(
      get_logger(),
      "Published /arm_joint_cmd: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f], time=%d ms",
      cmd.position[0],
      cmd.position[1],
      cmd.position[2],
      cmd.position[3],
      cmd.position[4],
      cmd.position[5],
      cmd.time_ms);
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Publisher<six_servo_arm_msgs::msg::ArmJointCommand>::SharedPtr arm_cmd_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::unordered_map<std::string, int> joint_name_to_index_;
  std::array<float, 6> latest_rad_;
  std::array<int, 6> last_sent_mrad_;
  std::array<bool, 6> joint_seen_;

  bool has_joint_state_{false};
  bool require_gripper_{false};
  bool control_gripper_{false};
  int publish_period_ms_{200};
  int move_time_ms_{200};
  int deadband_mrad_{5};
  float gripper_hold_position_rad_{-0.488f};
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<JointStateToArmCmdNode>());
  rclcpp::shutdown();
  return 0;
}
