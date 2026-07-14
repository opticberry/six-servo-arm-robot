# URDF Learning Notes For Six-Servo Arm

This note summarizes the key URDF questions we discussed while preparing the 6-servo robotic arm model for MoveIt.

## 1. What are `link` and `joint`

- `link`: a rigid body that does not move relative to itself
- `joint`: a movable connection between two links

For this arm:

- servo `0` to `4` are the main rotational joints
- servo `5` drives the gripper opening/closing
- the rigid brackets between servos are grouped into links

## 2. Common names like `base_link`, `wrist`, `tool0`

- `base_link`: the fixed base of the robot, usually the root of the whole model
- `link_1`, `link_2`, ...: rigid arm sections between joints
- `wrist`: usually means the last few joints near the end of the arm
- `tool0`: an end-effector reference frame, usually placed at the useful working point of the gripper

These names are conventions, not mandatory syntax.

## 3. What `visual`, `collision`, and `inertial` mean

- `visual`: how the link looks in RViz or simulators
- `collision`: simplified geometry used for collision checking
- `inertial`: mass, center of mass, and inertia used by physics simulation

For early MoveIt work:

- `visual` is needed
- `collision` is needed
- `inertial` is not the highest priority if the current goal is only planning and display

## 4. Why each of them has its own `origin`

They all describe different things relative to the same link frame:

- `visual/origin`: where the display mesh or primitive sits
- `collision/origin`: where the collision geometry sits
- `inertial/origin`: where the center of mass and inertia frame sit

These positions do not have to be identical.

## 5. What `rpy="0 0 0"` means

`rpy` means:

- roll: rotation about X
- pitch: rotation about Y
- yaw: rotation about Z

So:

```xml
rpy="0 0 0"
```

means no rotation relative to the parent frame.

## 6. Is `link` origin the geometry center or the joint position

Usually, the link frame is placed at the joint connection location, not at the geometry center.

Important distinction:

- `joint/origin`: places the child link frame relative to the parent link frame
- `visual/collision/origin`: places geometry relative to the link frame
- `inertial/origin`: places the center of mass relative to the link frame

So if a link is `0.2 m` long and its frame is at one end:

- geometry center is often at `0.1 m`
- the next joint may be at `0.2 m`

That is why you may see:

```xml
<visual>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</visual>

<joint name="next_joint" type="revolute">
  <origin xyz="0 0 0.2" rpy="0 0 0"/>
</joint>
```

Both are correct because they refer to different things.

## 7. Is `visual xyz="0.1"` the link center of mass

No.

Only:

```xml
<inertial>
  <origin xyz="..." rpy="..."/>
</inertial>
```

is normally related to the center of mass.

`visual` only controls appearance placement.

## 8. Why `left_finger` and `right_finger` are often modeled separately

This is a common gripper modeling approach, not a strict rule.

Benefits:

- better collision checking
- clearer opening width representation
- easier grasp planning

A common simplification is:

- one active finger joint
- one `mimic` joint following it in the opposite direction

If only the main arm chain is needed at first, the gripper can be simplified temporarily.

## 9. Recommended frame placement strategy for this arm

For this 6-servo arm, a practical convention is:

- place each link frame at the servo axis that starts that rigid section
- place the link geometry roughly in the middle of that rigid section
- place the next joint at the far end of that rigid section
- place `tool0` between the two fingers

This convention makes SolidWorks measurements easier to map into URDF.

## 10. Practical priority for current stage

If the current goal is MoveIt visualization and planning:

1. get joint parent-child relationships correct
2. get joint axis directions correct
3. get link/joint positions correct
4. get collision geometry roughly correct
5. refine visuals
6. add realistic inertial values later if needed
