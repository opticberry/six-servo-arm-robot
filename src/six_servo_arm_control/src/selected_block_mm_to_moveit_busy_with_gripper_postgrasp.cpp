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
  : Node("selected_block_mm_busy_moveit_node")
  {
    declare_parameter("a4_origin_x_m", 0.10);
    declare_parameter("a4_origin_y_m", 0.00);
    declare_parameter("a4_x_sign", 1.0);
    declare_parameter("a4_y_sign", 1.0);
    declare_parameter("table_z_m", 0.02);

    declare_parameter("sample_window_size", 6);
    declare_parameter("sample_spread_threshold_m", 0.010);
    declare_parameter("auto_execute", true);

    declare_parameter("move_group_name", "arm");
    declare_parameter("gripper_group_name", "gripper");
    declare_parameter("reference_frame", "base_link");
    declare_parameter("planning_time_s", 10.0);
    declare_parameter("num_planning_attempts", 20);
    declare_parameter("velocity_scaling", 0.10);
    declare_parameter("acceleration_scaling", 0.10);

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

    declare_parameter("gripper_close", 0.8);
    declare_parameter("gripper_open", 0.0);
    declare_parameter("gripper_settle_ms", 500);

    declare_parameter("pregrasp_time_ms", 2000);
    declare_parameter("target_time_ms", 2500);
    declare_parameter("postgrasp_time_ms", 2500);

    declare_parameter("ik_max_iters", 120);
    declare_parameter("ik_position_tolerance", 0.005);
    declare_parameter("ik_damping", 0.05);
    declare_parameter("ik_step_scale", 0.5);

    a4_origin_x_m_ = get_parameter("a4_origin_x_m").as_double();
    a4_origin_y_m_ = get_parameter("a4_origin_y_m").as_double();
    a4_x_sign_ = get_parameter("a4_x_sign").as_double();
    a4_y_sign_ = get_parameter("a4_y_sign").as_double();
    table_z_m_ = get_parameter("table_z_m").as_double();

    sample_window_size_ = get_parameter("sample_window_size").as_int();
    sample_spread_threshold_m_ = get_parameter("sample_spread_threshold_m").as_double();
    auto_execute_ = get_parameter("auto_execute").as_bool();

    move_group_name_ = get_parameter("move_group_name").as_string();
    gripper_group_name_ = get_parameter("gripper_group_name").as_string();
    reference_frame_ = get_parameter("reference_frame").as_string();
    planning_time_s_ = get_parameter("planning_time_s").as_double();
    num_planning_attempts_ = get_parameter("num_planning_attempts").as_int();
    velocity_scaling_ = get_parameter("velocity_scaling").as_double();
    acceleration_scaling_ = get_parameter("acceleration_scaling").as_double();

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
      "/selected_block_mm",
      10,
      std::bind(&SelectedBlockBusyPositionIkNode::onSelectedBlockMm, this, std::placeholders::_1));

    selected_block_base_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
      "/selected_block_base", 10);

    RCLCPP_INFO(get_logger(), "Busy position-IK node with gripper/postgrasp ready.");
    RCLCPP_INFO(
      get_logger(),
      "pregrasp=[%.3f, %.3f, %.3f, %.3f, %.3f], postgrasp=[%.3f, %.3f, %.3f, %.3f, %.3f]",
      pregrasp_q_[0], pregrasp_q_[1], pregrasp_q_[2], pregrasp_q_[3], pregrasp_q_[4],
      postgrasp_q_[0], postgrasp_q_[1], postgrasp_q_[2], postgrasp_q_[3], postgrasp_q_[4]);
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

  void onSelectedBlockMm(const geometry_msgs::msg::PointStamped::SharedPtr msg)
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

    const double x_base = a4_origin_x_m_ + a4_x_sign_ * (msg->point.x * 0.001);
    const double y_base = a4_origin_y_m_ + a4_y_sign_ * (msg->point.y * 0.001);
    const double z_base = table_z_m_;

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

    geometry_msgs::msg::PointStamped base_msg;
    base_msg.header.stamp = now();
    base_msg.header.frame_id = reference_frame_;
    base_msg.point.x = mean.x_m;
    base_msg.point.y = mean.y_m;
    base_msg.point.z = mean.z_m;
    selected_block_base_pub_->publish(base_msg);

    pending_target_ = mean;
    pending_ready_ = true;
    busy_ = true;
    sample_buffer_.clear();
    lock.unlock();
    cv_.notify_one();

    RCLCPP_INFO(
      get_logger(),
      "Locked target and published once: base=(%.3f, %.3f, %.3f), spread=%.4f m",
      mean.x_m, mean.y_m, mean.z_m, spread);
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

    RCLCPP_INFO(get_logger(), "Executing arm %s...", label.c_str());
    if (move_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS)
    {
      RCLCPP_ERROR(get_logger(), "Execution arm %s failed.", label.c_str());
      return false;
    }

    printStm32RadCommands(label, q, time_ms);
    return true;
  }

  bool moveGripperToNamedTarget(const std::string& state_name)
  {
      if (!gripper_group_)
      {
          RCLCPP_ERROR(get_logger(), "Gripper MoveIt group is not initialized.");
          return false;
      }

      RCLCPP_INFO(get_logger(), "Planning gripper named target: %s", state_name.c_str());

      gripper_group_->setStartStateToCurrentState();
      gripper_group_->setNamedTarget(state_name);

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
    double& final_pos_error)
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

      if (final_pos_error < ik_position_tolerance_)
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
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr selected_block_base_pub_;
  std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::unique_ptr<moveit::planning_interface::MoveGroupInterface> gripper_group_;
  moveit::core::RobotModelConstPtr robot_model_;
  const moveit::core::JointModelGroup* jmg_{nullptr};

  double a4_origin_x_m_{0.10};
  double a4_origin_y_m_{0.00};
  double a4_x_sign_{1.0};
  double a4_y_sign_{1.0};
  double table_z_m_{0.02};

  int sample_window_size_{6};
  double sample_spread_threshold_m_{0.010};
  bool auto_execute_{true};

  std::string move_group_name_{"arm"};
  std::string gripper_group_name_{"gripper"};
  std::string reference_frame_{"base_link"};
  double planning_time_s_{10.0};
  int num_planning_attempts_{20};
  double velocity_scaling_{0.10};
  double acceleration_scaling_{0.10};

  std::vector<double> pregrasp_q_;
  std::vector<double> postgrasp_q_;
  double gripper_close_{0.8};
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
