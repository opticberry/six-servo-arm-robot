#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model/joint_model_group.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <moveit/trajectory_processing/ruckig_traj_smoothing.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <rclcpp/rclcpp.hpp>

class SelectedBlockBusyPositionIkNode : public rclcpp::Node
{
public:
  struct Target
  {
    double x_m{0.0};
    double y_m{0.0};
    double z_m{0.0};
  };

  SelectedBlockBusyPositionIkNode()
  : Node("selected_block_base_busy_moveit_node")
  {
    declare_parameter("selected_block_topic", "/selected_block_base");
    declare_parameter("locked_target_topic", "/locked_selected_block_base");

    declare_parameter("sample_window_size", 6);
    declare_parameter("sample_spread_threshold_m", 0.010);
    declare_parameter("target_scale_x", 1.0);
    declare_parameter("target_scale_y", 1.0);
    declare_parameter("target_scale_z", 1.0);
    declare_parameter("target_offset_x_m", 0.0);
    declare_parameter("target_offset_y_m", 0.0);
    declare_parameter("target_offset_z_m", 0.0);
    declare_parameter("auto_execute", true);
    declare_parameter("enable_local_servo", true);
    declare_parameter("local_servo_mode", "rule");
    declare_parameter("visual_servo_error_topic", "/visual_servo_error_base");
    declare_parameter("ppo_delta_topic", "/local_servo_delta_base");
    declare_parameter("rule_servo_gain", 0.6);
    declare_parameter("ppo_fallback_to_rule", true);
    declare_parameter("local_servo_x_sign", 1.0);
    declare_parameter("local_servo_y_sign", 1.0);
    declare_parameter("local_servo_max_step_mm", 4.0);
    declare_parameter("local_servo_stop_error_mm", 2.0);
    declare_parameter("local_servo_max_iters", 8);
    declare_parameter("local_servo_error_timeout_s", 0.5);
    declare_parameter("local_servo_delta_timeout_s", 0.5);
    declare_parameter("local_servo_wait_timeout_s", 3.0);
    declare_parameter("local_servo_limit_step_by_error", true);
    declare_parameter("local_servo_step_limit_ratio", 1.0);
    declare_parameter("local_servo_ik_position_tolerance", 0.0005);
    declare_parameter("local_servo_settle_ms", 300);
    declare_parameter("local_servo_time_ms", 700);

    declare_parameter("move_group_name", "arm");
    declare_parameter("gripper_group_name", "gripper");
    declare_parameter("reference_frame", "base_link");
    declare_parameter("planning_time_s", 10.0);
    declare_parameter("num_planning_attempts", 20);
    declare_parameter("velocity_scaling", 0.10);
    declare_parameter("acceleration_scaling", 0.10);
    declare_parameter("moveit_retime_method", "totg");

    declare_parameter("pregrasp_j0", 0.0);
    declare_parameter("pregrasp_j1", 0.9);
    declare_parameter("pregrasp_j2", 1.57);
    declare_parameter("pregrasp_j3", -1.57);
    declare_parameter("pregrasp_j4", 0.0);

    declare_parameter("postgrasp_j0", 1.57);
    declare_parameter("postgrasp_j1", 0.89);
    declare_parameter("postgrasp_j2", 1.57);
    declare_parameter("postgrasp_j3", -1.57);
    declare_parameter("postgrasp_j4", 0.0);

    declare_parameter("gripper_close", 0.488);
    declare_parameter("gripper_open", -0.488);
    declare_parameter("gripper_settle_ms", 500);

    declare_parameter("pregrasp_time_ms", 2000);
    declare_parameter("target_time_ms", 2500);
    declare_parameter("postgrasp_time_ms", 2500);

    declare_parameter("ik_max_iters", 120);
    declare_parameter("ik_position_tolerance", 0.005);
    declare_parameter("ik_damping", 0.05);
    declare_parameter("ik_step_scale", 0.5);

    selected_block_topic_ = get_parameter("selected_block_topic").as_string();
    locked_target_topic_ = get_parameter("locked_target_topic").as_string();

    sample_window_size_ = get_parameter("sample_window_size").as_int();
    sample_spread_threshold_m_ = get_parameter("sample_spread_threshold_m").as_double();
    target_scale_x_ = get_parameter("target_scale_x").as_double();
    target_scale_y_ = get_parameter("target_scale_y").as_double();
    target_scale_z_ = get_parameter("target_scale_z").as_double();
    target_offset_x_m_ = get_parameter("target_offset_x_m").as_double();
    target_offset_y_m_ = get_parameter("target_offset_y_m").as_double();
    target_offset_z_m_ = get_parameter("target_offset_z_m").as_double();
    auto_execute_ = get_parameter("auto_execute").as_bool();
    enable_local_servo_ = get_parameter("enable_local_servo").as_bool();
    local_servo_mode_ = get_parameter("local_servo_mode").as_string();
    visual_servo_error_topic_ = get_parameter("visual_servo_error_topic").as_string();
    ppo_delta_topic_ = get_parameter("ppo_delta_topic").as_string();
    rule_servo_gain_ = get_parameter("rule_servo_gain").as_double();
    ppo_fallback_to_rule_ = get_parameter("ppo_fallback_to_rule").as_bool();
    local_servo_x_sign_ = get_parameter("local_servo_x_sign").as_double();
    local_servo_y_sign_ = get_parameter("local_servo_y_sign").as_double();
    local_servo_max_step_m_ = get_parameter("local_servo_max_step_mm").as_double() * 0.001;
    local_servo_stop_error_m_ = get_parameter("local_servo_stop_error_mm").as_double() * 0.001;
    local_servo_max_iters_ = get_parameter("local_servo_max_iters").as_int();
    local_servo_error_timeout_s_ = get_parameter("local_servo_error_timeout_s").as_double();
    local_servo_delta_timeout_s_ = get_parameter("local_servo_delta_timeout_s").as_double();
    local_servo_wait_timeout_s_ = get_parameter("local_servo_wait_timeout_s").as_double();
    local_servo_limit_step_by_error_ = get_parameter("local_servo_limit_step_by_error").as_bool();
    local_servo_step_limit_ratio_ = get_parameter("local_servo_step_limit_ratio").as_double();
    local_servo_ik_position_tolerance_ =
      get_parameter("local_servo_ik_position_tolerance").as_double();
    local_servo_settle_ms_ = get_parameter("local_servo_settle_ms").as_int();
    local_servo_time_ms_ = get_parameter("local_servo_time_ms").as_int();

    move_group_name_ = get_parameter("move_group_name").as_string();
    gripper_group_name_ = get_parameter("gripper_group_name").as_string();
    reference_frame_ = get_parameter("reference_frame").as_string();
    planning_time_s_ = get_parameter("planning_time_s").as_double();
    num_planning_attempts_ = get_parameter("num_planning_attempts").as_int();
    velocity_scaling_ = get_parameter("velocity_scaling").as_double();
    acceleration_scaling_ = get_parameter("acceleration_scaling").as_double();
    moveit_retime_method_ = get_parameter("moveit_retime_method").as_string();

    pregrasp_q_ = {
      get_parameter("pregrasp_j0").as_double(),
      get_parameter("pregrasp_j1").as_double(),
      get_parameter("pregrasp_j2").as_double(),
      get_parameter("pregrasp_j3").as_double(),
      get_parameter("pregrasp_j4").as_double()
    };
    postgrasp_q_ = {
      get_parameter("postgrasp_j0").as_double(),
      get_parameter("postgrasp_j1").as_double(),
      get_parameter("postgrasp_j2").as_double(),
      get_parameter("postgrasp_j3").as_double(),
      get_parameter("postgrasp_j4").as_double()
    };

    gripper_close_ = get_parameter("gripper_close").as_double();
    gripper_open_ = get_parameter("gripper_open").as_double();
    gripper_settle_ms_ = get_parameter("gripper_settle_ms").as_int();

    pregrasp_time_ms_ = get_parameter("pregrasp_time_ms").as_int();
    target_time_ms_ = get_parameter("target_time_ms").as_int();
    postgrasp_time_ms_ = get_parameter("postgrasp_time_ms").as_int();

    ik_max_iters_ = get_parameter("ik_max_iters").as_int();
    ik_position_tolerance_ = get_parameter("ik_position_tolerance").as_double();
    ik_damping_ = get_parameter("ik_damping").as_double();
    ik_step_scale_ = get_parameter("ik_step_scale").as_double();

    selected_block_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      selected_block_topic_,
      10,
      std::bind(&SelectedBlockBusyPositionIkNode::onSelectedBlockBase, this, std::placeholders::_1));
    visual_servo_error_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      visual_servo_error_topic_,
      10,
      std::bind(&SelectedBlockBusyPositionIkNode::onVisualServoErrorBase, this, std::placeholders::_1));
    ppo_delta_sub_ = create_subscription<geometry_msgs::msg::PointStamped>(
      ppo_delta_topic_,
      10,
      std::bind(&SelectedBlockBusyPositionIkNode::onPpoDeltaBase, this, std::placeholders::_1));

    locked_target_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
      locked_target_topic_, 10);
    local_servo_target_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
      "/local_servo_target_base", 10);

    RCLCPP_INFO(get_logger(), "Busy position-IK node with gripper/postgrasp ready.");
    RCLCPP_INFO(
      get_logger(),
      "Subscribing base target topic: %s, locked target topic: %s",
      selected_block_topic_.c_str(), locked_target_topic_.c_str());
    RCLCPP_INFO(
      get_logger(),
      "Target correction: scale=(%.4f, %.4f, %.4f), offset=(%.3f, %.3f, %.3f) m",
      target_scale_x_, target_scale_y_, target_scale_z_,
      target_offset_x_m_, target_offset_y_m_, target_offset_z_m_);
    RCLCPP_INFO(
      get_logger(),
      "Local servo: enabled=%s mode=%s error_topic=%s ppo_delta_topic=%s gain=%.3f sign=(%.1f, %.1f) max_step=%.1fmm stop=%.1fmm max_iters=%d",
      enable_local_servo_ ? "true" : "false",
      local_servo_mode_.c_str(),
      visual_servo_error_topic_.c_str(),
      ppo_delta_topic_.c_str(),
      rule_servo_gain_,
      local_servo_x_sign_,
      local_servo_y_sign_,
      local_servo_max_step_m_ * 1000.0,
      local_servo_stop_error_m_ * 1000.0,
      local_servo_max_iters_);
    RCLCPP_INFO(
      get_logger(),
      "Local servo step limiter: enabled=%s ratio=%.2f local_ik_tolerance=%.1fmm",
      local_servo_limit_step_by_error_ ? "true" : "false",
      local_servo_step_limit_ratio_,
      local_servo_ik_position_tolerance_ * 1000.0);
    RCLCPP_INFO(
      get_logger(),
      "Local servo execution: MoveIt plan only. STM32 output should be handled by joint_state_to_arm_cmd.");
    RCLCPP_INFO(
      get_logger(),
      "pregrasp=[%.3f, %.3f, %.3f, %.3f, %.3f], postgrasp=[%.3f, %.3f, %.3f, %.3f, %.3f]",
      pregrasp_q_[0], pregrasp_q_[1], pregrasp_q_[2], pregrasp_q_[3], pregrasp_q_[4],
      postgrasp_q_[0], postgrasp_q_[1], postgrasp_q_[2], postgrasp_q_[3], postgrasp_q_[4]);
    RCLCPP_INFO(
      get_logger(),
      "MoveIt trajectory retime: method=%s velocity_scaling=%.3f acceleration_scaling=%.3f",
      moveit_retime_method_.c_str(),
      velocity_scaling_,
      acceleration_scaling_);
  }

  void initMoveIt()
  {
    move_group_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), move_group_name_);
    gripper_group_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), gripper_group_name_);

    configureMoveGroup(*move_group_);
    configureMoveGroup(*gripper_group_);
    move_group_->setPoseReferenceFrame(reference_frame_);

    robot_model_ = move_group_->getRobotModel();
    if (!robot_model_)
    {
      throw std::runtime_error("Failed to get robot model from MoveGroupInterface");
    }

    jmg_ = robot_model_->getJointModelGroup(move_group_name_);
    if (!jmg_)
    {
      throw std::runtime_error("Joint model group not found: " + move_group_name_);
    }

    RCLCPP_INFO(
      get_logger(),
      "MoveIt initialized for arm group '%s' and gripper group '%s'.",
      move_group_name_.c_str(),
      gripper_group_name_.c_str());
  }

  void startWorker()
  {
    worker_running_ = true;
    worker_thread_ = std::thread(&SelectedBlockBusyPositionIkNode::workerLoop, this);
  }

  void stopWorker()
  {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      worker_running_ = false;
    }
    cv_.notify_all();
    if (worker_thread_.joinable())
    {
      worker_thread_.join();
    }
  }

  ~SelectedBlockBusyPositionIkNode() override
  {
    stopWorker();
  }

private:
  void configureMoveGroup(moveit::planning_interface::MoveGroupInterface& group)
  {
    group.setPlanningTime(planning_time_s_);
    group.setNumPlanningAttempts(num_planning_attempts_);
    group.setMaxVelocityScalingFactor(velocity_scaling_);
    group.setMaxAccelerationScalingFactor(acceleration_scaling_);
  }

  static Target computeMean(const std::deque<Target>& samples)
  {
    Target mean;
    if (samples.empty())
    {
      return mean;
    }

    for (const auto& s : samples)
    {
      mean.x_m += s.x_m;
      mean.y_m += s.y_m;
      mean.z_m += s.z_m;
    }

    const double inv = 1.0 / static_cast<double>(samples.size());
    mean.x_m *= inv;
    mean.y_m *= inv;
    mean.z_m *= inv;
    return mean;
  }

  static double computeSpreadMeters(const std::deque<Target>& samples, const Target& mean)
  {
    double max_dist = 0.0;
    for (const auto& s : samples)
    {
      const double dx = s.x_m - mean.x_m;
      const double dy = s.y_m - mean.y_m;
      const double dz = s.z_m - mean.z_m;
      const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
      max_dist = std::max(max_dist, dist);
    }
    return max_dist;
  }

  static double clampAbs(double value, double max_abs)
  {
    return std::max(-max_abs, std::min(max_abs, value));
  }

  static void clampStepNorm(double& x, double& y, double max_norm)
  {
    if (max_norm <= 0.0)
    {
      x = 0.0;
      y = 0.0;
      return;
    }

    const double norm = std::sqrt(x * x + y * y);
    if (norm <= max_norm || norm <= 1e-9)
    {
      return;
    }

    const double scale = max_norm / norm;
    x *= scale;
    y *= scale;
  }

  void onSelectedBlockBase(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    if (!msg)
    {
      return;
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (busy_)
    {
      return;
    }

    if (!msg->header.frame_id.empty() && msg->header.frame_id != reference_frame_)
    {
      RCLCPP_WARN(
        get_logger(),
        "Received target frame '%s', expected '%s'. Using values as %s coordinates.",
        msg->header.frame_id.c_str(),
        reference_frame_.c_str(),
        reference_frame_.c_str());
    }

    const double x_base = msg->point.x;
    const double y_base = msg->point.y;
    const double z_base = msg->point.z;

    if (!std::isfinite(x_base) || !std::isfinite(y_base) || !std::isfinite(z_base))
    {
      RCLCPP_WARN(get_logger(), "Received non-finite base target. Ignored.");
      return;
    }

    sample_buffer_.push_back(Target{x_base, y_base, z_base});
    if (static_cast<int>(sample_buffer_.size()) > sample_window_size_)
    {
      sample_buffer_.pop_front();
    }

    if (static_cast<int>(sample_buffer_.size()) < sample_window_size_)
    {
      RCLCPP_INFO(
        get_logger(),
        "Collecting samples %zu/%d, latest base=(%.3f, %.3f, %.3f)",
        sample_buffer_.size(), sample_window_size_, x_base, y_base, z_base);
      return;
    }

    const Target mean = computeMean(sample_buffer_);
    const double spread = computeSpreadMeters(sample_buffer_, mean);
    if (spread > sample_spread_threshold_m_)
    {
      RCLCPP_WARN(
        get_logger(),
        "Spread too large: %.4f m > %.4f m, reset buffer.",
        spread, sample_spread_threshold_m_);
      sample_buffer_.clear();
      return;
    }

    const Target adjusted_target{
      mean.x_m * target_scale_x_ + target_offset_x_m_,
      mean.y_m * target_scale_y_ + target_offset_y_m_,
      mean.z_m * target_scale_z_ + target_offset_z_m_};

    geometry_msgs::msg::PointStamped base_msg;
    base_msg.header.stamp = now();
    base_msg.header.frame_id = reference_frame_;
    base_msg.point.x = adjusted_target.x_m;
    base_msg.point.y = adjusted_target.y_m;
    base_msg.point.z = adjusted_target.z_m;
    locked_target_pub_->publish(base_msg);

    pending_target_ = adjusted_target;
    pending_ready_ = true;
    busy_ = true;
    sample_buffer_.clear();
    lock.unlock();
    cv_.notify_one();

    RCLCPP_INFO(
      get_logger(),
      "Locked target: raw=(%.3f, %.3f, %.3f), adjusted=(%.3f, %.3f, %.3f), spread=%.4f m",
      mean.x_m, mean.y_m, mean.z_m,
      adjusted_target.x_m, adjusted_target.y_m, adjusted_target.z_m,
      spread);
  }

  void onVisualServoErrorBase(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    if (!msg)
    {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    latest_error_x_m_ = msg->point.x;
    latest_error_y_m_ = msg->point.y;
    latest_error_z_m_ = msg->point.z;
    latest_error_time_ = std::chrono::steady_clock::now();
    has_latest_error_ = true;
  }

  void onPpoDeltaBase(const geometry_msgs::msg::PointStamped::SharedPtr msg)
  {
    if (!msg)
    {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    latest_delta_x_m_ = msg->point.x;
    latest_delta_y_m_ = msg->point.y;
    latest_delta_z_m_ = msg->point.z;
    latest_delta_time_ = std::chrono::steady_clock::now();
    has_latest_delta_ = true;
  }

  bool getLatestVisualServoError(
    double& error_x_m,
    double& error_y_m,
    double& error_z_m,
    double& age_s)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_latest_error_)
    {
      return false;
    }

    age_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - latest_error_time_).count();
    if (age_s > local_servo_error_timeout_s_)
    {
      return false;
    }

    error_x_m = latest_error_x_m_;
    error_y_m = latest_error_y_m_;
    error_z_m = latest_error_z_m_;
    return true;
  }

  bool getLatestPpoDelta(
    double& delta_x_m,
    double& delta_y_m,
    double& delta_z_m,
    double& age_s)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!has_latest_delta_)
    {
      return false;
    }

    age_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - latest_delta_time_).count();
    if (age_s > local_servo_delta_timeout_s_)
    {
      return false;
    }

    delta_x_m = latest_delta_x_m_;
    delta_y_m = latest_delta_y_m_;
    delta_z_m = latest_delta_z_m_;
    return true;
  }

  bool waitForFreshVisualServoError(
    double& error_x_m,
    double& error_y_m,
    double& error_z_m,
    double& age_s)
  {
    const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration<double>(local_servo_wait_timeout_s_);
    while (rclcpp::ok())
    {
      if (getLatestVisualServoError(error_x_m, error_y_m, error_z_m, age_s))
      {
        return true;
      }

      if (std::chrono::steady_clock::now() >= deadline)
      {
        return false;
      }

      std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }

    return false;
  }

  void publishLocalServoTargetBase(const Target& target)
  {
    geometry_msgs::msg::PointStamped msg;
    msg.header.stamp = now();
    msg.header.frame_id = reference_frame_;
    msg.point.x = target.x_m;
    msg.point.y = target.y_m;
    msg.point.z = target.z_m;
    local_servo_target_pub_->publish(msg);
  }

  void workerLoop()
  {
    while (true)
    {
      Target target;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() { return pending_ready_ || !worker_running_; });
        if (!worker_running_)
        {
          break;
        }

        target = pending_target_.value();
        pending_ready_ = false;
      }

      if (auto_execute_)
      {
        runPregraspAndTargetFlow(target);
      }
      else
      {
        RCLCPP_INFO(
          get_logger(),
          "Auto execute disabled. Locked target base=(%.3f, %.3f, %.3f)",
          target.x_m, target.y_m, target.z_m);
      }

      {
        std::lock_guard<std::mutex> lock(mutex_);
        busy_ = false;
      }
    }
  }

  bool runPregraspAndTargetFlow(const Target& target)
  {
    RCLCPP_INFO(
      get_logger(),
      "Step 1/6: move to pregrasp, target xyz=(%.3f, %.3f, %.3f)",
      target.x_m, target.y_m, target.z_m);

    if (!moveArmToJointTarget(pregrasp_q_, "pregrasp", pregrasp_time_ms_))
    {
      return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    std::vector<double> ik_q;
    double final_pos_error = std::numeric_limits<double>::infinity();
    if (!solvePositionIK(pregrasp_q_, target.x_m, target.y_m, target.z_m, ik_q, final_pos_error))
    {
      RCLCPP_ERROR(
        get_logger(),
        "Position IK failed. Final position error = %.4f m (%.2f cm)",
        final_pos_error,
        final_pos_error * 100.0);
      return false;
    }

    RCLCPP_INFO(
      get_logger(),
      "Step 2/6: IK succeeded. error=%.4f m, q=[%.4f, %.4f, %.4f, %.4f, %.4f]",
      final_pos_error, ik_q[0], ik_q[1], ik_q[2], ik_q[3], ik_q[4]);

    if (!moveArmToJointTarget(ik_q, "target", target_time_ms_))
    {
      return false;
    }

    Target final_target = target;
    std::vector<double> final_q = ik_q;
    if (enable_local_servo_)
    {
      if (!runLocalServoLoop(final_target, final_q))
      {
        RCLCPP_WARN(
          get_logger(),
          "Local servo did not finish. Continue grasp at current target instead of aborting.");
      }
    }

    if (!moveGripperToNamedTarget("close"))
    {
        return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(gripper_settle_ms_));

    if (!moveArmToJointTarget(postgrasp_q_, "postgrasp", postgrasp_time_ms_))
    {
      return false;
    }

    if (!moveGripperToNamedTarget("open"))
    {
        return false;
    }

    RCLCPP_INFO(get_logger(), "Pick-and-place cycle finished.");
    return true;
  }

  bool computeTool0PositionFromJointValues(const std::vector<double>& q, Target& tool0_target)
  {
    if (!robot_model_ || !jmg_)
    {
      return false;
    }

    moveit::core::RobotState state(robot_model_);
    state.setToDefaultValues();
    state.setJointGroupPositions(jmg_, q);
    state.enforceBounds(jmg_);
    state.update();

    const Eigen::Vector3d current = state.getGlobalLinkTransform("tool0").translation();
    tool0_target.x_m = current.x();
    tool0_target.y_m = current.y();
    tool0_target.z_m = current.z();
    return true;
  }

  bool getCurrentTool0Position(const std::vector<double>& fallback_q, Target& tool0_target)
  {
    if (move_group_)
    {
      auto current_state = move_group_->getCurrentState(1.0);
      if (current_state)
      {
        current_state->update();
        const Eigen::Vector3d current = current_state->getGlobalLinkTransform("tool0").translation();
        tool0_target.x_m = current.x();
        tool0_target.y_m = current.y();
        tool0_target.z_m = current.z();
        return true;
      }
    }

    return computeTool0PositionFromJointValues(fallback_q, tool0_target);
  }

  bool computeLocalServoStep(
    double error_x_m,
    double error_y_m,
    double& step_x_m,
    double& step_y_m)
  {
    if (local_servo_mode_ == "rule")
    {
      step_x_m = rule_servo_gain_ * error_x_m;
      step_y_m = rule_servo_gain_ * error_y_m;
    }
    else if (local_servo_mode_ == "ppo")
    {
      double delta_x_m = 0.0;
      double delta_y_m = 0.0;
      double delta_z_m = 0.0;
      double delta_age_s = 0.0;
      if (!getLatestPpoDelta(delta_x_m, delta_y_m, delta_z_m, delta_age_s))
      {
        if (!ppo_fallback_to_rule_)
        {
          RCLCPP_WARN(
            get_logger(),
            "No fresh PPO delta from '%s' within %.2f s.",
            ppo_delta_topic_.c_str(),
            local_servo_delta_timeout_s_);
          return false;
        }

        RCLCPP_WARN(
          get_logger(),
          "No fresh PPO delta from '%s' within %.2f s. Fallback to rule step.",
          ppo_delta_topic_.c_str(),
          local_servo_delta_timeout_s_);
        step_x_m = rule_servo_gain_ * error_x_m;
        step_y_m = rule_servo_gain_ * error_y_m;
        return true;
      }

      step_x_m = delta_x_m;
      step_y_m = delta_y_m;
    }
    else
    {
      RCLCPP_ERROR(
        get_logger(),
        "Unknown local_servo_mode='%s'. Use 'rule' or 'ppo'.",
        local_servo_mode_.c_str());
      return false;
    }

    return true;
  }

  bool runLocalServoLoop(Target& current_target, std::vector<double>& current_q)
  {
    for (int iter = 0; iter < local_servo_max_iters_; ++iter)
    {
      std::this_thread::sleep_for(std::chrono::milliseconds(local_servo_settle_ms_));

      double error_x_m = 0.0;
      double error_y_m = 0.0;
      double error_z_m = 0.0;
      double error_age_s = 0.0;
      if (!waitForFreshVisualServoError(error_x_m, error_y_m, error_z_m, error_age_s))
      {
        RCLCPP_WARN(
          get_logger(),
          "Local servo skipped: no fresh visual error from '%s' within %.2f s.",
          visual_servo_error_topic_.c_str(),
          local_servo_wait_timeout_s_);
        return true;
      }

      const double error_norm_m = std::sqrt(error_x_m * error_x_m + error_y_m * error_y_m);
      if (error_norm_m <= local_servo_stop_error_m_)
      {
        RCLCPP_INFO(
          get_logger(),
          "Local servo reached target: iter=%d error_xy=(%.1f, %.1f)mm norm=%.1fmm",
          iter,
          error_x_m * 1000.0,
          error_y_m * 1000.0,
          error_norm_m * 1000.0);
        return true;
      }

      double step_x_m = 0.0;
      double step_y_m = 0.0;
      if (!computeLocalServoStep(error_x_m, error_y_m, step_x_m, step_y_m))
      {
        RCLCPP_WARN(
          get_logger(),
          "Local servo step unavailable. Continue grasp without more local correction.");
        return true;
      }

      step_x_m = clampAbs(step_x_m, local_servo_max_step_m_);
      step_y_m = clampAbs(step_y_m, local_servo_max_step_m_);
      if (local_servo_limit_step_by_error_)
      {
        clampStepNorm(step_x_m, step_y_m, local_servo_step_limit_ratio_ * error_norm_m);
      }

      Target tool0_now;
      if (!getCurrentTool0Position(current_q, tool0_now))
      {
        RCLCPP_ERROR(get_logger(), "Failed to get current tool0 position for local servo.");
        return false;
      }

      Target next_target = tool0_now;
      next_target.x_m += local_servo_x_sign_ * step_x_m;
      next_target.y_m += local_servo_y_sign_ * step_y_m;
      next_target.z_m = current_target.z_m;
      publishLocalServoTargetBase(next_target);

      std::vector<double> next_q;
      double final_pos_error = std::numeric_limits<double>::infinity();
      if (!solvePositionIK(
            current_q,
            next_target.x_m,
            next_target.y_m,
            next_target.z_m,
            next_q,
            final_pos_error,
            local_servo_ik_position_tolerance_))
      {
        RCLCPP_ERROR(
          get_logger(),
          "Local servo IK failed at iter=%d target=(%.3f, %.3f, %.3f).",
          iter + 1,
          next_target.x_m,
          next_target.y_m,
          next_target.z_m);
        return false;
      }

      RCLCPP_INFO(
        get_logger(),
        "Local servo mode=%s iter=%d tool0_now=(%.3f, %.3f, %.3f) error_xy=(%.1f, %.1f)mm step_xy=(%.1f, %.1f)mm next=(%.3f, %.3f, %.3f)",
        local_servo_mode_.c_str(),
        iter + 1,
        tool0_now.x_m,
        tool0_now.y_m,
        tool0_now.z_m,
        error_x_m * 1000.0,
        error_y_m * 1000.0,
        step_x_m * 1000.0,
        step_y_m * 1000.0,
        next_target.x_m,
        next_target.y_m,
        next_target.z_m);

      if (!executeLocalServoJointTarget(current_q, next_q))
      {
        return false;
      }

      current_target = next_target;
      current_q = next_q;
    }

    RCLCPP_WARN(get_logger(), "Local servo stopped after max_iters=%d.", local_servo_max_iters_);
    return true;
  }

  bool executeLocalServoJointTarget(
    const std::vector<double>& current_q,
    const std::vector<double>& target_q)
  {
    RCLCPP_INFO(
      get_logger(),
      "Executing local servo through MoveIt plan. current=[%.4f, %.4f, %.4f, %.4f, %.4f] target=[%.4f, %.4f, %.4f, %.4f, %.4f]",
      current_q[0],
      current_q[1],
      current_q[2],
      current_q[3],
      current_q[4],
      target_q[0],
      target_q[1],
      target_q[2],
      target_q[3],
      target_q[4]);
    return moveArmToJointTarget(target_q, "local_servo", local_servo_time_ms_);
  }

  bool moveArmToJointTarget(
    const std::vector<double>& q,
    const std::string& label,
    int time_ms)
  {
    if (!move_group_)
    {
      RCLCPP_ERROR(get_logger(), "Arm MoveIt group is not initialized.");
      return false;
    }

    RCLCPP_INFO(
      get_logger(),
      "Planning arm %s q=[%.4f, %.4f, %.4f, %.4f, %.4f]",
      label.c_str(), q[0], q[1], q[2], q[3], q[4]);

    move_group_->setStartStateToCurrentState();
    if (!move_group_->setJointValueTarget(q))
    {
      RCLCPP_ERROR(get_logger(), "Failed to set arm %s joint target.", label.c_str());
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (!static_cast<bool>(move_group_->plan(plan)))
    {
      RCLCPP_ERROR(get_logger(), "Planning arm %s failed.", label.c_str());
      return false;
    }

    if (!retimeMoveItPlan(plan, label))
    {
      return false;
    }

    RCLCPP_INFO(get_logger(), "Executing arm %s...", label.c_str());
    if (move_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_ERROR(get_logger(), "Execution arm %s failed.", label.c_str());
      return false;
    }

    printStm32RadCommands(label, q, time_ms);
    return true;
  }

  bool retimeMoveItPlan(
    moveit::planning_interface::MoveGroupInterface::Plan& plan,
    const std::string& label)
  {
    if (moveit_retime_method_ == "none")
    {
      return true;
    }

    if (moveit_retime_method_ != "iptp" &&
      moveit_retime_method_ != "totg" &&
      moveit_retime_method_ != "ruckig")
    {
      RCLCPP_ERROR(
        get_logger(),
        "Unknown moveit_retime_method='%s'. Use 'none', 'iptp', 'totg', or 'ruckig'.",
        moveit_retime_method_.c_str());
      return false;
    }

    if (!robot_model_)
    {
      RCLCPP_ERROR(get_logger(), "Cannot retime trajectory: robot model is unavailable.");
      return false;
    }

    auto current_state = move_group_->getCurrentState(1.0);
    if (!current_state)
    {
      RCLCPP_ERROR(get_logger(), "Cannot retime trajectory: current state is unavailable.");
      return false;
    }

    robot_trajectory::RobotTrajectory trajectory(robot_model_, move_group_name_);
    trajectory.setRobotTrajectoryMsg(*current_state, plan.trajectory_);

    bool success = false;
    if (moveit_retime_method_ == "iptp")
    {
      trajectory_processing::IterativeParabolicTimeParameterization iptp;
      success = iptp.computeTimeStamps(
        trajectory,
        velocity_scaling_,
        acceleration_scaling_);
    }
    else if (moveit_retime_method_ == "totg")
    {
      trajectory_processing::TimeOptimalTrajectoryGeneration totg;
      success = totg.computeTimeStamps(
        trajectory,
        velocity_scaling_,
        acceleration_scaling_);
    }
    else if (moveit_retime_method_ == "ruckig")
    {
      trajectory_processing::TimeOptimalTrajectoryGeneration totg;
      success = totg.computeTimeStamps(
        trajectory,
        velocity_scaling_,
        acceleration_scaling_);

      if (success)
      {
        success = trajectory_processing::RuckigSmoothing::applySmoothing(
          trajectory,
          velocity_scaling_,
          acceleration_scaling_);
      }
    }

    if (!success)
    {
      RCLCPP_ERROR(
        get_logger(),
        "MoveIt retime failed for %s with method=%s.",
        label.c_str(),
        moveit_retime_method_.c_str());
      return false;
    }

    trajectory.getRobotTrajectoryMsg(plan.trajectory_);
    RCLCPP_INFO(
      get_logger(),
      "MoveIt retime success for %s: method=%s points=%zu",
      label.c_str(),
      moveit_retime_method_.c_str(),
      plan.trajectory_.joint_trajectory.points.size());
    return true;
  }

  bool moveGripperToNamedTarget(const std::string& state_name)
  {
      if (!gripper_group_)
      {
          RCLCPP_ERROR(get_logger(), "Gripper MoveIt group is not initialized.");
          return false;
      }

      gripper_group_->setStartStateToCurrentState();
      if (state_name == "open" || state_name == "close")
      {
          const double target = (state_name == "open") ? gripper_open_ : gripper_close_;
          RCLCPP_INFO(
            get_logger(),
            "Planning gripper numeric target: %s -> %.3f rad",
            state_name.c_str(),
            target);
          if (!gripper_group_->setJointValueTarget(std::vector<double>{target}))
          {
              RCLCPP_ERROR(
                get_logger(),
                "Failed to set gripper %s target to %.3f rad.",
                state_name.c_str(),
                target);
              return false;
          }
      }
      else
      {
          RCLCPP_INFO(get_logger(), "Planning gripper named target: %s", state_name.c_str());
          gripper_group_->setNamedTarget(state_name);
      }

      moveit::planning_interface::MoveGroupInterface::Plan plan;
      if (!static_cast<bool>(gripper_group_->plan(plan)))
      {
          RCLCPP_ERROR(get_logger(), "Planning gripper %s failed.", state_name.c_str());
          return false;
      }

      RCLCPP_INFO(get_logger(), "Executing gripper %s...", state_name.c_str());
      if (gripper_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS)
      {
          RCLCPP_ERROR(get_logger(), "Execution gripper %s failed.", state_name.c_str());
          return false;
      }

      return true;
  }

  bool solvePositionIK(
    const std::vector<double>& seed_q,
    double target_x,
    double target_y,
    double target_z,
    std::vector<double>& solution_q,
    double& final_pos_error,
    double position_tolerance = -1.0)
  {
    final_pos_error = std::numeric_limits<double>::infinity();

    if (!robot_model_ || !jmg_)
    {
      return false;
    }

    const auto& active_joint_models = jmg_->getActiveJointModels();
    if (active_joint_models.size() != seed_q.size())
    {
      return false;
    }

    std::vector<double> q = seed_q;
    const Eigen::Vector3d target(target_x, target_y, target_z);

    moveit::core::RobotState state(robot_model_);
    state.setToDefaultValues();

    for (int iter = 0; iter < ik_max_iters_; ++iter)
    {
      state.setJointGroupPositions(jmg_, q);
      state.enforceBounds(jmg_);
      state.update();

      const Eigen::Vector3d current = state.getGlobalLinkTransform("tool0").translation();
      const Eigen::Vector3d error = target - current;
      final_pos_error = error.norm();

      const double tolerance =
        (position_tolerance > 0.0) ? position_tolerance : ik_position_tolerance_;
      if (final_pos_error < tolerance)
      {
        state.copyJointGroupPositions(jmg_, q);
        solution_q = q;
        return true;
      }

      Eigen::MatrixXd jacobian;
      const Eigen::Vector3d reference_point_position(0.0, 0.0, 0.0);
      if (!state.getJacobian(jmg_, state.getLinkModel("tool0"), reference_point_position, jacobian))
      {
        return false;
      }

      if (jacobian.rows() < 3 || jacobian.cols() != static_cast<int>(q.size()))
      {
        return false;
      }

      const Eigen::MatrixXd J = jacobian.topRows(3);
      const Eigen::Matrix3d A =
        J * J.transpose() + (ik_damping_ * ik_damping_) * Eigen::Matrix3d::Identity();
      const Eigen::Vector3d y = A.ldlt().solve(error);
      const Eigen::VectorXd dq = J.transpose() * y;

      if (!dq.allFinite())
      {
        return false;
      }

      for (std::size_t i = 0; i < q.size(); ++i)
      {
        q[i] += ik_step_scale_ * dq(static_cast<int>(i));
      }

      state.setJointGroupPositions(jmg_, q);
      state.enforceBounds(jmg_);
      state.copyJointGroupPositions(jmg_, q);
    }

    solution_q = q;
    return false;
  }

  void printStm32RadCommands(const std::string& label, const std::vector<double>& q, int time_ms)
  {
    RCLCPP_INFO(get_logger(), "STM32 %s commands:", label.c_str());
    for (std::size_t i = 0; i < q.size(); ++i)
    {
      RCLCPP_INFO(get_logger(), "joint %zu %d %d", i, static_cast<int>(q[i] * 1000.0), time_ms);
    }
  }

private:
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr selected_block_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr visual_servo_error_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr ppo_delta_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr locked_target_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr local_servo_target_pub_;
  std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::unique_ptr<moveit::planning_interface::MoveGroupInterface> gripper_group_;
  moveit::core::RobotModelConstPtr robot_model_;
  const moveit::core::JointModelGroup* jmg_{nullptr};

  int sample_window_size_{6};
  double sample_spread_threshold_m_{0.010};
  double target_scale_x_{1.0};
  double target_scale_y_{1.0};
  double target_scale_z_{1.0};
  double target_offset_x_m_{0.0};
  double target_offset_y_m_{0.0};
  double target_offset_z_m_{0.0};
  bool auto_execute_{true};
  bool enable_local_servo_{true};
  std::string local_servo_mode_{"rule"};
  std::string visual_servo_error_topic_{"/visual_servo_error_base"};
  std::string ppo_delta_topic_{"/local_servo_delta_base"};
  double rule_servo_gain_{0.6};
  bool ppo_fallback_to_rule_{true};
  double local_servo_x_sign_{1.0};
  double local_servo_y_sign_{1.0};
  double local_servo_max_step_m_{0.004};
  double local_servo_stop_error_m_{0.002};
  int local_servo_max_iters_{8};
  double local_servo_error_timeout_s_{0.5};
  double local_servo_delta_timeout_s_{0.5};
  double local_servo_wait_timeout_s_{3.0};
  bool local_servo_limit_step_by_error_{true};
  double local_servo_step_limit_ratio_{1.0};
  double local_servo_ik_position_tolerance_{0.0005};
  int local_servo_settle_ms_{300};
  int local_servo_time_ms_{700};

  std::string selected_block_topic_{"/selected_block_base"};
  std::string locked_target_topic_{"/locked_selected_block_base"};
  std::string move_group_name_{"arm"};
  std::string gripper_group_name_{"gripper"};
  std::string reference_frame_{"base_link"};
  double planning_time_s_{10.0};
  int num_planning_attempts_{20};
  double velocity_scaling_{0.10};
  double acceleration_scaling_{0.10};
  std::string moveit_retime_method_{"totg"};

  std::vector<double> pregrasp_q_;
  std::vector<double> postgrasp_q_;
  double gripper_close_{0.488};
  double gripper_open_{0.0};
  int gripper_settle_ms_{500};
  int pregrasp_time_ms_{2000};
  int target_time_ms_{2500};
  int postgrasp_time_ms_{2500};

  int ik_max_iters_{120};
  double ik_position_tolerance_{0.005};
  double ik_damping_{0.05};
  double ik_step_scale_{0.5};

  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<Target> sample_buffer_;
  std::optional<Target> pending_target_;
  double latest_error_x_m_{0.0};
  double latest_error_y_m_{0.0};
  double latest_error_z_m_{0.0};
  double latest_delta_x_m_{0.0};
  double latest_delta_y_m_{0.0};
  double latest_delta_z_m_{0.0};
  std::chrono::steady_clock::time_point latest_error_time_{};
  std::chrono::steady_clock::time_point latest_delta_time_{};
  bool has_latest_error_{false};
  bool has_latest_delta_{false};
  bool pending_ready_{false};
  bool busy_{false};
  bool worker_running_{false};
  std::thread worker_thread_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<SelectedBlockBusyPositionIkNode>();
  node->initMoveIt();
  node->startWorker();
  rclcpp::spin(node);
  node->stopWorker();
  rclcpp::shutdown();
  return 0;
}
