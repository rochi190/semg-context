# XArmAPI 完整函式與屬性參考文件

> 基於 `xarm_api.py` 原始碼逐行對照整理，涵蓋所有 `@property`、method、以及 alias 別名。  
> 傳回值中 `code == 0` 表示執行成功；非零值請參閱官方 API Code 文件。  
> 角度單位：預設為度（°）；若建構時傳入 `is_radian=True`，則全面使用弧度（rad）。

---

## 目錄

1. [建構子 `__init__`](#1-建構子)
2. [連線管理](#2-連線管理)
3. [狀態與模式屬性](#3-狀態與模式屬性)
4. [位置與角度屬性](#4-位置與角度屬性)
5. [速度、加速度、限制屬性](#5-速度加速度限制屬性)
6. [感測器與診斷屬性](#6-感測器與診斷屬性)
7. [安全模式屬性](#7-安全模式屬性)
8. [運動控制方法](#8-運動控制方法)
9. [速度控制方法（Velocity Control）](#9-速度控制方法)
10. [參數設定方法](#10-參數設定方法)
11. [狀態查詢方法](#11-狀態查詢方法)
12. [運動學工具方法](#12-運動學工具方法)
13. [末端夾爪 — xArm Gripper](#13-末端夾爪--xarm-gripper)
14. [末端夾爪 — Vacuum Gripper](#14-末端夾爪--vacuum-gripper)
15. [末端夾爪 — Robotiq Gripper](#15-末端夾爪--robotiq-gripper)
16. [末端夾爪 — BIO Gripper](#16-末端夾爪--bio-gripper)
17. [末端夾爪 — 其他夾爪](#17-末端夾爪--其他夾爪)
18. [Tool GPIO（tgpio）](#18-tool-gpiotgpio)
19. [Controller GPIO（cgpio）](#19-controller-gpiocgpio)
20. [六軸力矩感測器（FT Sensor）](#20-六軸力矩感測器ft-sensor)
21. [Linear Motor（線性馬達）](#21-linear-motor線性馬達)
22. [Callback 回呼管理](#22-callback-回呼管理)
23. [軌跡錄製與回放](#23-軌跡錄製與回放)
24. [校正工具方法](#24-校正工具方法)
25. [RS485 / Modbus RTU](#25-rs485--modbus-rtu)
26. [Modbus TCP（標準 Modbus）](#26-modbus-tcp標準-modbus)
27. [縮減模式與安全邊界](#27-縮減模式與安全邊界)
28. [進階診斷與錯誤資訊](#28-進階診斷與錯誤資訊)
29. [DH 參數、回饋、其他進階功能](#29-dh-參數回饋其他進階功能)
30. [Lite6 專屬夾爪](#30-lite6-專屬夾爪)
31. [偵錯與版本查詢](#31-偵錯與版本查詢)
32. [屬性別名（Alias）](#32-屬性別名alias)

---

## 1. 建構子

### `__init__(port=None, is_radian=False, do_not_open=False, **kwargs)`

建立 `XArmAPI` 實例並（預設）自動連線。

| 參數 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `port` | str | None | 機械臂 IP 位址，如 `'192.168.1.185'`；`do_not_open=False` 時必須填寫 |
| `is_radian` | bool | False | 全域角度單位。`False`=度（°），`True`=弧度（rad）。影響所有帶角度參數的 method 與 property |
| `do_not_open` | bool | False | `True` 時不自動連線，需手動呼叫 `connect()` |
| `axis` | int（kwargs） | 7 | 軸數，序列埠連線時才需要指定 |
| `check_tcp_limit` | bool（kwargs） | False | 是否檢查 `set_position` / `move_arc_lines` 中 roll/pitch/yaw 超限 |
| `check_joint_limit` | bool（kwargs） | True | 是否檢查 `set_servo_angle` / `set_servo_angle_j` 超限 |
| `check_cmdnum_limit` | bool（kwargs） | True | 是否檢查指令佇列數量超限 |
| `max_cmdnum` | int（kwargs） | 512 | 最大佇列指令數，僅在 `check_cmdnum_limit=True` 時有效 |
| `check_is_ready` | bool（kwargs） | True | 是否在動作前確認機械臂就緒，僅在 firmware < 1.5.20 時有效 |

> **角度單位換算：** 1 rad = 57.296°；1° = 0.01745 rad

---

## 2. 連線管理

### `connect(port=None, baudrate=None, timeout=None, axis=None, **kwargs)`
連線至 xArm 控制器。

| 參數 | 說明 |
|------|------|
| `port` | IP 位址，預設使用建構時的值 |
| `baudrate` | 串列埠鮑率（序列埠連線使用，通常不需要設定） |
| `timeout` | 串列埠逾時（通常不需要設定） |
| `axis` | 軸數，序列埠連線時才需要 |

---

### `disconnect()`
斷開與 xArm 的連線。無參數，無傳回值。

---

### `send_cmd_sync(command=None)`
發送文字指令並等待回應（不等待動作完成）。

| 參數 | 說明 |
|------|------|
| `command` | G-code 或 H-code 字串，例如 `'G1 X300 Y0 Z200 F100'` |

支援的指令格式（節錄）：

| 指令 | 對應 API |
|------|---------|
| `G1 X{x} Y{y} Z{z} A{roll} B{pitch} C{yaw} F{speed} Q{acc}` | `set_position` (MoveLine) |
| `G7 I{J1} J{J2} ... F{speed}` | `set_servo_angle` |
| `G8 F{speed}` | `move_gohome` |
| `H11 I{servo_id} V{enable}` | `motion_enable` |
| `H12 V{state}` | `set_state` |
| `H19 V{mode}` | `set_mode` |

---

## 3. 狀態與模式屬性

| 屬性名稱 | 型別 | 說明 |
|----------|------|------|
| `connected` | bool | 是否已連線 |
| `default_is_radian` | bool | 建構時設定的角度單位（`is_radian` 值） |
| `state` | int | 機械臂狀態：1=運動中、2=睡眠、3=暫停、4=停止中 |
| `mode` | int | 控制模式（見下表），需 socket 連線且 enable_report=True |
| `is_simulation_robot` | bool | 是否為模擬機器人 |
| `has_err_warn` | bool | 是否有錯誤或警告 |
| `has_error` | bool | 是否有錯誤 |
| `has_warn` | bool | 是否有警告 |
| `error_code` | int | 控制器錯誤碼 |
| `warn_code` | int | 控制器警告碼 |
| `cmd_num` | int | 控制器指令快取數量 |
| `device_type` | int | 裝置類型（需 rich report） |
| `axis` | int | 軸數（需 rich report） |
| `master_id` | int | Master ID（需 rich report） |
| `slave_id` | int | Slave ID（需 rich report） |
| `is_reduced_mode` | bool | 是否處於縮減模式 |
| `is_fence_mode` | bool | 是否處於安全邊界模式 |
| `is_report_current` | bool | 是否回報電流 |
| `is_approx_motion` | bool | 是否允許近似運動 |
| `is_cart_continuous` | bool | 笛卡爾運動是否連續 |
| `is_collision_rebound` | bool | 碰撞反彈是否啟用 |

**mode 值說明：**

| 值 | 模式 |
|----|------|
| 0 | 位置控制模式（預設） |
| 1 | 伺服運動模式（需搭配 `set_servo_angle_j` / `set_servo_cartesian`） |
| 2 | 關節示教模式 |
| 3 | 笛卡爾示教模式（無效） |
| 4 | 關節速度控制模式 |
| 5 | 笛卡爾速度控制模式 |
| 6 | 關節線上軌跡規劃模式 |
| 7 | 笛卡爾線上軌跡規劃模式 |

---

## 4. 位置與角度屬性

| 屬性名稱 | 型別 | 說明 |
|----------|------|------|
| `position` | list | 笛卡爾位置 `[x(mm), y(mm), z(mm), roll, pitch, yaw]`，角度依 `default_is_radian` |
| `position_aa` | list | 軸角表示位姿 `[x(mm), y(mm), z(mm), rx, ry, rz]` |
| `angles` | list | 各關節角度 `[J1, J2, ..., J7]`，依 `default_is_radian` |
| `last_used_position` | list | 上次 `set_position` 使用的位姿，作為下次的預設值 |
| `last_used_angles` | list | 上次 `set_servo_angle` 使用的角度，作為下次的預設值 |
| `tcp_offset` | list | TCP 偏移 `[x, y, z, roll, pitch, yaw]`（需 socket + enable_report） |
| `world_offset` | list | 世界座標偏移 `[x, y, z, roll, pitch, yaw]`（需 firmware > 1.2.11） |
| `gravity_direction` | list | 重力方向（需 rich report） |
| `ft_ext_force` | list | 六軸力矩感測器外力偵測值（補償後） |
| `ft_raw_force` | list | 六軸力矩感測器原始讀值（未補償） |

---

## 5. 速度、加速度、限制屬性

| 屬性名稱 | 型別 | 說明 |
|----------|------|------|
| `tcp_jerk` | float | TCP jerk（mm/s³） |
| `tcp_speed_limit` | list | TCP 速度限制 `[min(mm/s), max(mm/s)]`（需 rich report） |
| `tcp_acc_limit` | list | TCP 加速度限制 `[min(mm/s²), max(mm/s²)]`（需 rich report） |
| `last_used_tcp_speed` | float | 上次 `set_position` 使用的速度（mm/s） |
| `last_used_tcp_acc` | float | 上次 `set_position` 使用的加速度（mm/s²） |
| `joint_jerk` | float | 關節 jerk（°/s³ 或 rad/s³） |
| `joint_speed_limit` | list | 關節速度限制（需 rich report） |
| `joint_acc_limit` | list | 關節加速度限制（需 rich report） |
| `last_used_joint_speed` | float | 上次 `set_servo_angle` 使用的速度 |
| `last_used_joint_acc` | float | 上次 `set_servo_angle` 使用的加速度 |
| `reduced_max_tcp_speed` | float | 縮減模式下的最大 TCP 速度（mm/s） |
| `reduced_max_joint_speed` | float | 縮減模式下的最大關節速度 |
| `reduced_tcp_boundary` | list | 縮減模式的 TCP 邊界 `[x_max, x_min, y_max, y_min, z_max, z_min]` |
| `reduced_joint_limits` | list | 縮減模式的關節限位 `[[J1_min, J1_max], ..., [J7_min, J7_max]]` |

---

## 6. 感測器與診斷屬性

| 屬性名稱 | 型別 | 說明 |
|----------|------|------|
| `version` | str | xArm 韌體版本 |
| `version_number` | tuple | `(major, minor, revision)` |
| `sn` | str | xArm 序號 |
| `control_box_sn` | str | 控制箱序號 |
| `temperatures` | list | 各伺服馬達溫度 `[T1, ..., T7]`（需 firmware > 1.2.11） |
| `voltages` | list | 各伺服馬達電壓 `[V1, ..., V7]` |
| `currents` | list | 各伺服馬達電流 `[I1, ..., I7]` |
| `joints_torque` | list | 各關節力矩（需 rich report） |
| `tcp_load` | list | TCP 負載 `[weight(kg), [x(mm), y(mm), z(mm)]]`（需 rich report） |
| `collision_sensitivity` | int | 碰撞靈敏度 0~5（需 rich report） |
| `teach_sensitivity` | int | 示教靈敏度 1~5（需 rich report） |
| `motor_brake_states` | list | 馬達剎車狀態（0=啟用，1=停用），需 rich report |
| `motor_enable_states` | list | 馬達啟用狀態（0=停用，1=啟用），需 rich report |
| `servo_codes` | list | 伺服狀態與錯誤碼 `[[status, code], ...]`（含 tool GPIO） |
| `realtime_tcp_speed` | float | TCP 即時速度（mm/s），需 firmware > 1.2.11 |
| `realtime_joint_speeds` | list | 各關節即時速度（需 firmware > 1.2.11） |
| `count` | int | 計數器數值 |
| `cgpio_states` | list | 控制器 GPIO 狀態（詳見 `get_cgpio_state`） |
| `self_collision_params` | list | 自碰撞參數 `[detection_on, tool_type, model_params]` |
| `gpio_reset_config` | list | GPIO 重置設定 `[cgpio_reset_enable, tgpio_reset_enable]` |

---

## 7. 安全模式屬性

| 屬性名稱 | 說明 |
|----------|------|
| `only_check_result` | 僅檢查結果（內部用） |
| `report_data` | 原始 report 資料 |
| `arm` | XArm 內部實作實例（不建議直接使用） |
| `core` | 核心層 API，進階開發者用（如 `self.core.move_line()`） |

---

## 8. 運動控制方法

### `get_position(is_radian=None) → (code, [x, y, z, roll, pitch, yaw])`
取得目前笛卡爾位姿。

| 參數 | 說明 |
|------|------|
| `is_radian` | 回傳 roll/pitch/yaw 是否用弧度，預設 `default_is_radian` |

---

### `set_position(x, y, z, roll, pitch, yaw, radius=None, speed=None, mvacc=None, mvtime=None, relative=False, is_radian=None, wait=False, timeout=None, **kwargs) → code`
設定笛卡爾目標位姿（基座座標系）。未填參數使用 `last_used_position` 的對應值。

| 參數 | 說明 |
|------|------|
| `x/y/z` | 目標位置（mm） |
| `roll/pitch/yaw` | 姿態角（°或rad），分別繞 X/Y/Z 軸旋轉 |
| `radius` | 軌跡半徑：`None` 或 < 0 → MoveLine（直線）；`>= 0` → MoveArcLine（弧線插補） |
| `speed` | 移動速度（mm/s），預設 `last_used_tcp_speed` |
| `mvacc` | 加速度（mm/s²），預設 `last_used_tcp_acc` |
| `mvtime` | 保留，填 0 |
| `relative` | 是否為相對移動 |
| `is_radian` | roll/pitch/yaw 是否為弧度 |
| `wait` | 是否等待動作完成後才返回 |
| `timeout` | 最大等待秒數（僅 wait=True 時有效） |

---

### `set_tool_position(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, speed=None, mvacc=None, mvtime=None, is_radian=None, wait=False, timeout=None, radius=None, **kwargs) → code`
相對**工具座標系**移動（座標系隨末端位姿變化）。

| 參數 | 說明 |
|------|------|
| `x/y/z` | 相對工具座標系的位移量（mm），預設 0 |
| `roll/pitch/yaw` | 相對工具座標系的旋轉量，預設 0 |
| `radius` | 同 `set_position`，需 firmware >= 1.11.100 |
| `speed/mvacc/wait/timeout` | 同 `set_position` |

---

### `get_servo_angle(servo_id=None, is_radian=None, is_real=False) → (code, angles 或 angle)`
取得關節角度。

| 參數 | 說明 |
|------|------|
| `servo_id` | 1~軸數：取單一關節角度；None 或 8：取全部關節角度 |
| `is_radian` | 回傳是否用弧度 |
| `is_real` | 是否取即時值（非緩衝值） |

---

### `set_servo_angle(servo_id=None, angle=None, speed=None, mvacc=None, mvtime=None, relative=False, is_radian=None, wait=False, timeout=None, radius=None, **kwargs) → code`
設定關節角度目標。

| 參數 | 說明 |
|------|------|
| `servo_id` | 1~軸數：控制單一關節，`angle` 為數值；None/8：全部關節，`angle` 為 list |
| `angle` | 目標角度（°或rad）或角度 list |
| `speed` | 關節速度（°/s 或 rad/s） |
| `mvacc` | 關節加速度（°/s² 或 rad/s²） |
| `relative` | 相對移動 |
| `radius` | 混合半徑（需 firmware > 1.5.20），不可大於軌跡長度 |
| `wait/timeout` | 同 `set_position` |

---

### `set_servo_angle_j(angles, speed=None, mvacc=None, mvtime=None, is_radian=None, **kwargs) → code`
伺服模式下設定關節角度（**只執行最後一筆指令**）。  
⚠️ 必須先呼叫 `set_mode(1)` 切換至伺服運動模式。  
不修改 `last_used_angles` 等快取值。

| 參數 | 說明 |
|------|------|
| `angles` | 角度 list `[J1, J2, ..., J7]`（°或rad） |

---

### `set_servo_cartesian(mvpose, speed=None, mvacc=None, mvtime=0, is_radian=None, is_tool_coord=False, **kwargs) → code`
伺服模式下設定笛卡爾位姿（**只執行最後一筆指令**）。  
⚠️ 必須先呼叫 `set_mode(1)`。

| 參數 | 說明 |
|------|------|
| `mvpose` | 目標位姿 `[x, y, z, roll, pitch, yaw]` |
| `is_tool_coord` | 是否使用工具座標系 |

---

### `set_servo_cartesian_aa(axis_angle_pose, speed=None, mvacc=None, is_radian=None, is_tool_coord=False, relative=False, **kwargs) → code`
伺服模式下，以**軸角（axis-angle）**表示法設定笛卡爾位姿。  
需 firmware >= 1.4.7，且先呼叫 `set_mode(1)`。

| 參數 | 說明 |
|------|------|
| `axis_angle_pose` | `[x(mm), y(mm), z(mm), rx, ry, rz]` |
| `is_tool_coord` | 是否使用工具座標系 |
| `relative` | 相對移動 |

---

### `move_circle(pose1, pose2, percent, speed=None, mvacc=None, mvtime=None, is_radian=None, wait=False, timeout=None, is_tool_coord=False, is_axis_angle=False, **kwargs) → code`
三點圓弧運動（起點=當前位置，終點 1=pose1，終點 2=pose2）。

| 參數 | 說明 |
|------|------|
| `pose1` | 圓弧第二點 `[x, y, z, roll, pitch, yaw]` |
| `pose2` | 圓弧第三點 `[x, y, z, roll, pitch, yaw]` |
| `percent` | 弧長佔整圓周的百分比（0~100） |
| `is_tool_coord` | 工具座標系（需 firmware >= 1.11.100） |
| `is_axis_angle` | 使用軸角表示（需 firmware >= 1.11.100） |

---

### `move_gohome(speed=None, mvacc=None, mvtime=None, is_radian=None, wait=False, timeout=None, **kwargs) → code`
回到零點（Home 位置）。⚠️ 不帶限位偵測。

| 參數 | 說明 |
|------|------|
| `speed` | 速度，預設 50°/s |
| `mvacc` | 加速度，預設 5000°/s² |

---

### `move_arc_lines(paths, is_radian=None, times=1, first_pause_time=0.1, repeat_pause_time=0, automatic_calibration=True, speed=None, mvacc=None, mvtime=None, wait=False)`
連續直線插補運動（支援弧線過渡）。

| 參數 | 說明 |
|------|------|
| `paths` | 路徑點 list，每點格式 `[x, y, z, roll, pitch, yaw]` 或 `[x, y, z, roll, pitch, yaw, radius]` |
| `times` | 重複次數，0=無限迴圈 |
| `first_pause_time` | 初始等待時間（s），讓指令快取以利連續運動規劃（預設 0.1s） |
| `repeat_pause_time` | 每次重複間隔（s） |
| `automatic_calibration` | 自動校準（預設 True） |

---

### `set_position_aa(axis_angle_pose, speed=None, mvacc=None, mvtime=None, is_radian=None, is_tool_coord=False, relative=False, wait=False, timeout=None, radius=None, **kwargs) → code`
以**軸角（axis-angle）**表示法設定目標位姿。

| 參數 | 說明 |
|------|------|
| `axis_angle_pose` | `[x(mm), y(mm), z(mm), rx, ry, rz]` |
| `is_tool_coord` | 若為 True，`relative` 參數無效 |
| `radius` | 需 firmware >= 1.11.100；MoveLineAA（None/< 0）或 MoveArcLineAA（>= 0） |
| `motion_type`（kwargs） | 0=線性規劃；1=優先線性，不可則關節規劃；2=直接關節規劃（需 firmware >= 1.11.100） |

---

### `get_position_aa(is_radian=None) → (code, [x, y, z, rx, ry, rz])`
取得以軸角表示的目前位姿。

---

### `get_pose_offset(pose1, pose2, orient_type_in=0, orient_type_out=0, is_radian=None) → (code, pose)`
計算兩個位姿之間的偏移量。

| 參數 | 說明 |
|------|------|
| `pose1/pose2` | `[x, y, z, roll/rx, pitch/ry, yaw/rz]` |
| `orient_type_in` | 輸入姿態表示：0=RPY，1=軸角 |
| `orient_type_out` | 輸出姿態表示：0=RPY，1=軸角 |

---

### `set_servo_attach(servo_id=None) → code`
鎖定（Attach）伺服馬達。

| 參數 | 說明 |
|------|------|
| `servo_id` | 1~軸數（鎖定單一關節）或 8（鎖定全部） |

---

### `set_servo_detach(servo_id=None) → code`
解鎖（Detach）伺服馬達。⚠️ 解鎖前確保做好保護措施，避免意外掉臂造成傷害。

| 參數 | 說明 |
|------|------|
| `servo_id` | 1~軸數（解鎖單一關節）或 8（解鎖全部） |

---

### `emergency_stop()`
緊急停止：依序執行 `set_state(4)` → `motion_enable(True)` → `set_state(0)`。  
⚠️ **不會自動清除錯誤**，若有錯誤需依錯誤碼另行處理。

---

### `reset(speed=None, mvacc=None, mvtime=None, is_radian=None, wait=False, timeout=None)`
重置機械臂：自動清除警告與錯誤，並在需要時啟用運動與設定狀態。  
⚠️ 不帶限位偵測，不修改 `last_used_*` 快取值。

---

## 9. 速度控制方法

### `vc_set_joint_velocity(speeds, is_radian=None, is_sync=True, duration=-1, **kwargs) → code`
關節速度控制。⚠️ 需先呼叫 `set_mode(4)`（需 firmware >= 1.6.9）。

| 參數 | 說明 |
|------|------|
| `speeds` | `[spd_J1, ..., spd_J7]`（°/s 或 rad/s） |
| `is_sync` | 各關節是否同步加減速，預設 True |
| `duration` | 速度指令持續時間（s）：> 0=N 秒後自動停止；0=永久生效；< 0=相容舊協議（等同 0）。需 firmware >= 1.8.0 |

---

### `vc_set_cartesian_velocity(speeds, is_radian=None, is_tool_coord=False, duration=-1, **kwargs) → code`
笛卡爾速度控制。⚠️ 需先呼叫 `set_mode(5)`（需 firmware >= 1.6.9）。

| 參數 | 說明 |
|------|------|
| `speeds` | `[spd_x(mm/s), spd_y, spd_z, spd_rx, spd_ry, spd_rz]` |
| `is_tool_coord` | 是否使用工具座標系 |
| `duration` | 同 `vc_set_joint_velocity`，需 firmware >= 1.8.0 |

---

## 10. 參數設定方法

### `set_state(state=0) → code`
設定機械臂狀態。

| `state` 值 | 說明 |
|------------|------|
| 0 | 運動狀態（正常工作） |
| 3 | 暫停 |
| 4 | 停止 |
| 6 | 減速停止 |

---

### `set_mode(mode=0, detection_param=0) → code`
設定控制模式（參見模式表格）。

| 參數 | 說明 |
|------|------|
| `mode` | 見「狀態與模式屬性」中的 mode 值說明 |
| `detection_param` | 示教偵測：0=開啟動作偵測（預設）；1=關閉。需 firmware >= 1.10.1，且 mode=2 時才有效 |

---

### `motion_enable(enable=True, servo_id=None) → code`
啟用或停用馬達。

| 參數 | 說明 |
|------|------|
| `enable` | True=啟用，False=停用 |
| `servo_id` | 1~軸數（單一馬達）或 None/8（全部） |

---

### `set_tcp_offset(offset, is_radian=None, wait=True, **kwargs) → code`
設定末端工具座標偏移。若要清除，傳入 `[0, 0, 0, 0, 0, 0]`。

| 參數 | 說明 |
|------|------|
| `offset` | `[x(mm), y(mm), z(mm), roll, pitch, yaw]` |
| `wait` | 等待機械臂停止後才設定 |

> 未呼叫 `save_conf()` 則重開機後失效。

---

### `set_tcp_jerk(jerk) → code`
設定笛卡爾空間的平移 jerk（mm/s³）。

---

### `set_tcp_maxacc(acc) → code`
設定笛卡爾空間最大加速度（mm/s²）。

---

### `set_joint_jerk(jerk, is_radian=None) → code`
設定關節空間 jerk（°/s³ 或 rad/s³）。

---

### `set_joint_maxacc(acc, is_radian=None) → code`
設定關節空間最大加速度（°/s² 或 rad/s²）。

---

### `set_tcp_load(weight, center_of_gravity, wait=False, **kwargs) → code`
設定末端負載。

| 參數 | 說明 |
|------|------|
| `weight` | 負載重量（kg） |
| `center_of_gravity` | 重心位置 `[x(mm), y(mm), z(mm)]` |

---

### `set_collision_sensitivity(value, wait=True) → code`
設定碰撞靈敏度（0~5，5 最靈敏）。

---

### `set_teach_sensitivity(value, wait=True) → code`
設定拖動示教靈敏度（1~5）。

---

### `set_gravity_direction(direction, wait=True) → code`
設定重力方向向量（用於力矩補償與碰撞偵測）。

| 參數 | 說明 |
|------|------|
| `direction` | `[x, y, z]`，例如落地安裝 → `[0, 0, -1]` |

---

### `set_mount_direction(base_tilt_deg, rotation_deg, is_radian=None) → code`
設定機械臂安裝方式（傾斜角、旋轉角）。

---

### `set_world_offset(offset, is_radian=None, wait=True) → code`
設定基座座標偏移（需 firmware >= 1.2.11）。

| 參數 | 說明 |
|------|------|
| `offset` | `[x, y, z, roll, pitch, yaw]` |

---

### `set_timeout(timeout)`
設定指令回應逾時時間（秒）。

---

### `clean_conf() → code`
清除目前設定，恢復系統預設值。

---

### `save_conf() → code`
儲存目前設定（重開機後仍有效）。

---

### `set_pause_time(sltime, wait=False) → code`
讓機械臂暫停指定秒數。

| 參數 | 說明 |
|------|------|
| `sltime` | 暫停時間（秒） |

---

### `set_simulation_robot(on_off) → code`
設定是否為模擬機器人。

---

### `system_control(value=1) → code`
控制控制箱系統。`value=1`：關機；`value=2`：重開機。

---

### `set_counter_reset() → code`
重設計數器為 0。

---

### `set_counter_increase(val=1) → code`
計數器遞增（目前只支援 +1）。

---

### `set_only_check_type(only_check_type=0)`
設定運動過程的偵測類型（全域配置，影響本 SDK 實例的所有運動介面）。需 firmware >= 1.11.100。

---

## 11. 狀態查詢方法

### `get_state() → (code, state)`
取得機械臂狀態（1=運動中、2=睡眠、3=暫停、4=停止中）。

---

### `get_is_moving() → bool`
快速查詢是否正在運動。

---

### `get_cmdnum() → (code, cmd_num)`
取得控制器指令快取數量。

---

### `get_err_warn_code(show=False, lang='en') → (code, [error_code, warn_code])`
取得控制器錯誤碼與警告碼。

| 參數 | 說明 |
|------|------|
| `show` | 是否印出詳細說明 |
| `lang` | 顯示語言：`'en'` 或 `'cn'` |

---

### `clean_error() → code`
清除錯誤。清除後需手動呼叫 `motion_enable(True)` 及 `set_state(0)`。

---

### `clean_warn() → code`
清除警告。

---

### `get_version() → (code, version)`
取得韌體版本字串。

---

### `get_robot_sn() → (code, sn)`
取得機械臂序號。

---

### `check_verification() → (code, status)`
驗證授權狀態（0=已驗證）。

---

### `get_joint_states(is_radian=None, num=3) → (code, [position, velocity, effort])`
取得關節狀態（需 firmware >= 1.9.0）。

| 參數 | 說明 |
|------|------|
| `num` | 回傳資料筆數：1=position；2=position+velocity；3=position+velocity+effort |

---

## 12. 運動學工具方法

### `get_inverse_kinematics(pose, input_is_radian=None, return_is_radian=None, limited=True, ref_angles=None) → (code, angles)`
逆向運動學（IK）：笛卡爾位姿 → 關節角度。

| 參數 | 說明 |
|------|------|
| `pose` | `[x, y, z, roll, pitch, yaw]` |
| `limited` | 結果是否限制在 ±180° 以內（需 firmware >= 2.7.103） |
| `ref_angles` | 參考關節角度（需 firmware >= 2.7.103） |

---

### `get_forward_kinematics(angles, input_is_radian=None, return_is_radian=None) → (code, pose)`
正向運動學（FK）：關節角度 → 笛卡爾位姿。

| 參數 | 說明 |
|------|------|
| `angles` | `[J1, J2, ..., Jn]` |

---

### `is_tcp_limit(pose, is_radian=None) → (code, limit)`
檢查笛卡爾位姿是否超出限位（True=超限，False=未超限，None=失敗）。

---

### `is_joint_limit(joint, is_radian=None) → (code, limit)`
檢查關節角度是否超出限位（True=超限，False=未超限，None=失敗）。

---

### `get_joints_torque() → (code, joints_torque)`
取得各關節力矩（即時值）。

---

### `set_xarm7_ik_redundancy(jnt_ref, punish_coeff) → code`
設定 xArm7 冗餘自由度 IK 解參數（需 firmware >= 2.7.107）。

| 參數 | 說明 |
|------|------|
| `jnt_ref` | 參考關節角度，解將盡量靠近此值 |
| `punish_coeff` | 懲罰係數 |

---

### `get_xarm7_ik_redundancy() → (code, params)`
取得 xArm7 冗餘 IK 解參數（需 firmware >= 2.7.107）。

---

## 13. 末端夾爪 — xArm Gripper

### `set_gripper_enable(enable, **kwargs) → code`
啟用或停用夾爪。`True`=啟用。

---

### `set_gripper_mode(mode, **kwargs) → code`
設定夾爪模式。`mode=0`：位置控制模式。

---

### `get_gripper_position(**kwargs) → (code, pos)`
取得夾爪位置（pulse 單位）。

---

### `get_gripper_g2_position(**kwargs) → (code, pos)`
取得 xArm Gripper G2 位置（mm）。

---

### `set_gripper_position(pos, wait=False, speed=None, auto_enable=False, timeout=None, **kwargs) → code`
設定夾爪位置。

| 參數 | 說明 |
|------|------|
| `pos` | 目標位置（pulse） |
| `speed` | 速度（r/min） |
| `auto_enable` | 自動啟用（預設 False） |
| `timeout` | 等待逾時（s），預設 10s |

---

### `set_gripper_g2_position(pos, speed=100, force=50, wait=False, timeout=None, **kwargs) → code`
設定 xArm Gripper G2 位置。

| 參數 | 說明 |
|------|------|
| `pos` | 0~84 mm |
| `speed` | 15~225 mm/s，預設 100 |
| `force` | 1~100，預設 50 |

---

### `set_gripper_speed(speed, **kwargs) → code`
設定夾爪速度（r/min）。

---

### `get_gripper_status() → (code, status)`
取得夾爪狀態（需夾爪 firmware >= 3.4.3）。

| `status & 0x03` | 說明 |
|-----------------|------|
| 0 | 停止 |
| 1 | 運動中 |
| 2 | 已夾持 |

---

### `get_gripper_err_code(**kwargs) → (code, err_code)`
取得夾爪錯誤碼。

---

### `clean_gripper_error(**kwargs) → code`
清除夾爪錯誤。

---

## 14. 末端夾爪 — Vacuum Gripper

### `get_vacuum_gripper(hardware_version=1) → (code, state)`
取得真空吸盤狀態。

| `state` | 說明 |
|---------|------|
| -1 | 吸盤已關閉 |
| 0 | 吸盤開啟但未吸取物件 |
| 1 | 已吸取物件 |

| `hardware_version` | 說明 |
|--------------------|------|
| 1 | 插件式連接（預設） |
| 2 | 接觸式連接 |

---

### `set_vacuum_gripper(on, wait=False, timeout=3, delay_sec=None, sync=True, hardware_version=1) → code`
控制真空吸盤開/關。

| 參數 | 說明 |
|------|------|
| `on` | True=開（等同 tgpio[0]=1, tgpio[1]=0）；False=關 |
| `wait` | 等待吸取成功 |
| `delay_sec` | 延遲生效時間（s），None=立即生效 |
| `sync` | True=在運動佇列中執行；False=立即執行（需 firmware >= 2.4.101，且 delay_sec <= 0） |

---

## 15. 末端夾爪 — Robotiq Gripper

### `robotiq_reset(**kwargs) → (code, robotiq_response)`
重置 Robotiq 夾爪（清除先前的啟用狀態）。

---

### `robotiq_set_activate(wait=True, timeout=3, **kwargs) → (code, robotiq_response)`
啟動 Robotiq 夾爪（若尚未啟動）。

---

### `robotiq_set_position(pos, speed=0xFF, force=0xFF, wait=True, timeout=5, **kwargs) → (code, robotiq_response)`
設定 Robotiq 夾爪位置。

| 參數 | 說明 |
|------|------|
| `pos` | 0（全開）~ 255（全關） |
| `speed` | 0~255 |
| `force` | 0~255 |

---

### `robotiq_open(speed=0xFF, force=0xFF, wait=True, timeout=5, **kwargs) → (code, robotiq_response)`
開啟 Robotiq 夾爪（pos=0）。

---

### `robotiq_close(speed=0xFF, force=0xFF, wait=True, timeout=5, **kwargs) → (code, robotiq_response)`
關閉 Robotiq 夾爪（pos=255）。

---

### `robotiq_get_status(number_of_registers=3, **kwargs) → (code, robotiq_response)`
讀取 Robotiq 狀態暫存器。

| `number_of_registers` | 說明 |
|-----------------------|------|
| 1 | 只讀 0x07D0（GRIPPER STATUS） |
| 2 | 讀 0x07D0、0x07D1（+ FAULT/POSITION_REQUEST_ECHO） |
| 3 | 讀 0x07D0~0x07D2（+ POSITION/CURRENT） |

---

### `robotiq_status`（屬性）
最後一次取得的 Robotiq 狀態 dict：

```python
{
    'gOBJ': 0,  # 物件偵測狀態
    'gSTA': 0,  # 夾爪狀態與運動
    'gGTO': 0,  # 動作狀態（go-to bit）
    'gACT': 0,  # 啟用狀態
    'kFLT': 0,  # 位置請求 echo
    'gFLT': 0,  # 故障狀態
    'gPR': 0,   # 請求位置 echo
    'gPO': 0,   # 實際位置（encoder）
    'gCU': 0,   # 即時電流
}
```
> -1 表示從未更新。

---

## 16. 末端夾爪 — BIO Gripper

### `set_bio_gripper_enable(enable=True, wait=True, timeout=3, **kwargs) → code`
啟用 BIO 夾爪（若尚未啟用）。

---

### `set_bio_gripper_speed(speed, **kwargs) → code`
設定 BIO 夾爪速度。

---

### `set_bio_gripper_control_mode(mode, **kwargs) → code`
設定 BIO 夾爪控制模式（僅新版 BIO Gripper 支援）。`mode=0`：開合模式；`mode=1`：位置迴路模式。

---

### `set_bio_gripper_force(force, **kwargs) → code`
設定 BIO 夾爪夾力（10~100，僅新版支援）。

---

### `get_bio_gripper_g2_position(**kwargs) → (code, pos)`
取得 BIO Gripper G2 位置（mm）。

---

### `set_bio_gripper_g2_position(pos, speed=2000, force=100, wait=True, timeout=5, **kwargs) → (code, response)`
設定 BIO Gripper G2 位置。

| 參數 | 說明 |
|------|------|
| `pos` | 71~150 mm |
| `speed` | 500~4500 pulse/s，預設 2000 |
| `force` | 1~100，預設 100 |

---

### `open_bio_gripper(speed=0, wait=True, timeout=5, **kwargs) → code`
開啟 BIO 夾爪（`speed=0` 表示不設定速度）。

---

### `close_bio_gripper(speed=0, wait=True, timeout=5, **kwargs) → code`
關閉 BIO 夾爪。

---

### `get_bio_gripper_status() → (code, status)`
取得 BIO 夾爪狀態。

| `status & 0x03` | 說明 |
|-----------------|------|
| 0 | 停止 |
| 1 | 運動中 |
| 2 | 已夾持 |
| 3 | 錯誤 |

| `(status >> 2) & 0x03` | 說明 |
|------------------------|------|
| 0 | 未啟用 |
| 1 | 啟用中 |
| 2 | 已啟用 |

---

### `get_bio_gripper_error() → (code, error_code)`
取得 BIO 夾爪錯誤碼。

---

### `clean_bio_gripper_error() → code`
清除 BIO 夾爪錯誤。

---

## 17. 末端夾爪 — 其他夾爪

### DH-PGC-140-50 夾爪

#### `set_dhpgc_gripper_activate(wait=True, timeout=3, **kwargs) → code`
啟動 DH-PGC-140-50 夾爪。

#### `set_dhpgc_gripper_position(pos, speed=50, force=50, wait=True, timeout=5, **kwargs) → code`
| 參數 | 說明 |
|------|------|
| `pos` | 0~1000 |
| `speed` | 1~100 |
| `force` | 20~100 |

---

### INSPIRE-ROBOTS RH56DFX 手指

#### `set_rh56_finger_position(finger_id, pos, speed=500, force=500, wait=False, timeout=None, **kwargs) → code`
| 參數 | 說明 |
|------|------|
| `finger_id` | 1~6（手指編號） |
| `pos` | 0~1000 |
| `speed` | 0~1000，預設 500 |
| `force` | 0~1000，預設 500 |

---

## 18. Tool GPIO（tgpio）

### `get_tgpio_digital(ionum=None) → (code, value 或 list)`
讀取工具端數位 GPIO 輸入。`ionum=0/1` 讀取單一；`None`=同時讀取 0 和 1。

---

### `get_tgpio_output_digital(ionum=None) → (code, value 或 list)`
讀取工具端數位 GPIO **輸出**值。`ionum=0/1/2/3/4/None`。

---

### `get_tool_digital_input(ionum=None) → (code, value 或 list)`
讀取工具端數位輸入（與 `get_tgpio_digital` 差異：不傳 ionum 時取得 TI2 值）。`ionum=0~4`。

---

### `set_tgpio_digital(ionum, value, delay_sec=None, sync=True) → code`
設定工具端數位 GPIO 輸出。

| 參數 | 說明 |
|------|------|
| `ionum` | 0 或 1 |
| `value` | 輸出值 |
| `delay_sec` | 延遲生效秒數，None=立即 |
| `sync` | True=在運動佇列執行；False=立即執行（需 firmware >= 2.4.101 且 delay_sec <= 0） |

---

### `get_tgpio_analog(ionum=None) → (code, value 或 list)`
讀取工具端類比 GPIO。`ionum=0/1/None`。

---

### `set_tgpio_digital_with_xyz(ionum, value, xyz, fault_tolerance_radius) → code`
當機械臂到達指定 XYZ 位置時，設定工具端數位 GPIO。

| 參數 | 說明 |
|------|------|
| `ionum` | 0 或 1 |
| `xyz` | 觸發位置 `[x, y, z]` |
| `fault_tolerance_radius` | 容忍半徑（mm） |

---

### `config_tgpio_reset_when_stop(on_off) → code`
設定機械臂進入停止狀態時，是否自動重置工具端數位輸出。

---

### `get_tgpio_version() → (code, version)`
取得工具端 GPIO 版本（偵錯用）。

---

### `set_tgpio_monitor_params(io_type, frequency) → code`
設定 TGPIO 監控參數（需 firmware >= 2.7.101）。

| 參數 | 說明 |
|------|------|
| `io_type` | 0=關閉監控；其他值見文件 |
| `frequency` | 回報頻率 |

---

### `get_tgpio_monitor_params() → (code, params)`
取得 TGPIO 監控參數（需 firmware >= 2.7.101）。

---

## 19. Controller GPIO（cgpio）

### `get_cgpio_digital(ionum=None) → (code, value 或 list)`
讀取控制箱數位 GPIO 輸入。`ionum=0~7`（CI0~CI7）或 `8~15`（DI0~DI7）或 `None`（全部）。

---

### `get_cgpio_analog(ionum=None) → (code, value 或 list)`
讀取控制箱類比 GPIO。`ionum=0/1/None`。

---

### `set_cgpio_digital(ionum, value, delay_sec=None, sync=True) → code`
設定控制箱數位 GPIO 輸出。`ionum=0~7`（CO0~CO7）或 `8~15`（DO0~DO7）。

---

### `set_cgpio_analog(ionum, value, sync=True) → code`
設定控制箱類比 GPIO 輸出。`ionum=0 或 1`。

---

### `set_cgpio_digital_input_function(ionum, fun) → code`
設定控制箱數位**輸入**功能模式。

| `fun` | 說明 |
|-------|------|
| 0 | 通用輸入 |
| 1 | 外部緊急停止 |
| 2 | 保護重置 |
| 11 | 離線任務 |
| 12 | 示教模式 |
| 13 | 縮減模式 |
| 14 | 啟用手臂 |

---

### `set_cgpio_digital_output_function(ionum, fun) → code`
設定控制箱數位**輸出**功能模式。

| `fun` | 說明 |
|-------|------|
| 0 | 通用輸出 |
| 1 | 緊急停止 |
| 2 | 運動中 |
| 11 | 有錯誤 |
| 12 | 有警告 |
| 13 | 碰撞中 |
| 14 | 示教中 |
| 15 | 離線任務中 |
| 16 | 縮減模式中 |
| 17 | 已啟用 |
| 18 | 急停已按下 |

---

### `get_cgpio_state() → (code, states)`
取得控制箱 GPIO 完整狀態。

| 索引 | 說明 |
|------|------|
| `[0]` | GPIO 模組狀態（0=正常，1=錯誤，6=通訊失敗） |
| `[1]` | GPIO 模組錯誤碼 |
| `[2]` | 數位輸入功能狀態（bit mask） |
| `[3]` | 數位輸入配置狀態（bit mask） |
| `[4]` | 數位輸出功能狀態（bit mask） |
| `[5]` | 數位輸出配置狀態（bit mask） |
| `[6]` | 類比輸入 0 |
| `[7]` | 類比輸入 1 |
| `[8]` | 類比輸出 0 |
| `[9]` | 類比輸出 1 |
| `[10]` | 數位輸入功能模式 list |
| `[11]` | 數位輸出功能模式 list |

---

### `set_cgpio_digital_with_xyz(ionum, value, xyz, fault_tolerance_radius) → code`
當機械臂到達指定 XYZ 位置時，設定控制箱數位 GPIO。`ionum=0~15`。

---

### `set_cgpio_analog_with_xyz(ionum, value, xyz, fault_tolerance_radius) → code`
當機械臂到達指定 XYZ 位置時，設定控制箱類比 GPIO。`ionum=0~1`。

---

### `config_cgpio_reset_when_stop(on_off) → code`
設定機械臂停止時，是否自動重置控制箱數位輸出。

---

## 20. 六軸力矩感測器（FT Sensor）

> 以下功能需 firmware >= 1.8.3 且安裝 UFACTORY 六軸力矩感測器（不支援第三方）。

### `set_ft_sensor_zero() → code`
將當前狀態設為力矩感測器零點。

---

### `set_ft_sensor_enable(on_off) → code`
啟用/停用力矩感測器資料採樣。

---

### `set_ft_sensor_mode(mode, **kwargs) → code`
設定力控模式。`mode=0`：非力控模式；`1`：導納控制；`2`：力控模式。

---

### `get_ft_sensor_mode() → (code, mode)`
取得當前力控模式。

---

### `get_ft_sensor_data(is_raw=False) → (code, ft_data)`
取得力矩感測器資料（補償後外力值）。`is_raw=True` 回傳原始值（需 firmware >= 2.6.109）。

---

### `get_ft_sensor_config() → (code, config)`
取得力矩感測器完整設定（模式、負載、零點偏移、導納參數等）。

---

### `get_ft_sensor_error() → (code, error)`
取得力矩感測器錯誤碼。

---

### `iden_ft_sensor_load_offset() → (code, load_offset)`
用力矩感測器自動識別 TCP 負載與偏移。

回傳：`[mass(kg), x_centroid(mm), y_centroid(mm), z_centroid(mm), Fx_offset, Fy_offset, Fz_offset, Tx_offset, Ty_offset, Tz_offset]`

---

### `set_ft_sensor_load_offset(iden_result_list, association_setting_tcp_load=False, **kwargs) → code`
寫入力矩感測器識別到的負載與偏移參數。

| 參數 | 說明 |
|------|------|
| `association_setting_tcp_load` | True=同時更新 TCP 負載設定 |

---

### `set_ft_sensor_admittance_parameters(coord=None, c_axis=None, M=None, K=None, B=None, **kwargs) → code`
設定導納控制參數（用於柔順控制）。

| 參數 | 說明 |
|------|------|
| `coord` | 任務座標系：0=基座座標，1=工具座標 |
| `c_axis` | 6D 向量（0/1），1 表示該軸進行導納控制 |
| `M` | 6D 質量向量（kg） |
| `K` | 6D 剛度係數 |
| `B` | 6D 阻尼係數（控制器內部設為 `2*sqrt(M*K)`） |

---

### `set_ft_sensor_force_parameters(coord=None, c_axis=None, f_ref=None, limits=None, kp=None, ki=None, kd=None, xe_limit=None, **kwargs) → code`
設定力控模式參數。

| 參數 | 說明 |
|------|------|
| `f_ref` | 6D 向量，目標力/力矩（機械臂調整位置以達到此力） |
| `limits` | 6D 向量，柔順軸的最大 TCP 速度 |
| `kp/ki/kd` | 6D PID 增益 |
| `xe_limit` | 6D 最大允許 TCP 速度（mm/s） |

---

### `set_ft_collision_detection(on_off) → code`
啟用力矩感測器碰撞偵測（需 firmware >= 2.6.103）。

---

### `set_ft_collision_rebound(on_off) → code`
啟用力矩感測器碰撞反彈（需 firmware >= 2.6.103）。

---

### `set_ft_collision_threshold(thresholds) → code`
設定碰撞偵測閾值。`thresholds=[x(N), y(N), z(N), Rx(Nm), Ry(Nm), Rz(Nm)]`，x 範圍：5~200N。

---

### `set_ft_collision_reb_distance(distances, is_radian=None) → code`
設定碰撞反彈距離。`distances=[x(mm), y(mm), z(mm), Rx, Ry, Rz]`。

---

### `set_ft_admittance_ctrl_threshold(thresholds) → code`
設定導納控制各方向反應閾值（需 firmware >= 2.6.110）。

---

### `get_ft_collision_detection() → (code, on_off)`
查詢碰撞偵測是否啟用。

---

### `get_ft_collision_rebound() → (code, on_off)`
查詢碰撞反彈是否啟用。

---

### `get_ft_collision_threshold() → (code, thresholds)`
取得碰撞偵測閾值。

---

### `get_ft_collision_reb_distance(is_radian=None) → (code, distances)`
取得碰撞反彈距離。

---

### `get_ft_admittance_ctrl_threshold() → (code, thresholds)`
取得導納控制閾值（需 firmware >= 2.6.110）。

---

### `iden_tcp_load(estimated_mass=0) → (code, load)`
用電流識別 TCP 負載（需 firmware >= 1.8.0）。  
⚠️ Lite6 機型**必須**指定 `estimated_mass`。

回傳：`[mass, x_centroid, y_centroid, z_centroid]`

---

## 21. Linear Motor（線性馬達）

> 以下功能需 firmware >= 1.8.0。

### `get_linear_motor_registers(**kwargs) → (code, status_dict)`
取得線性馬達完整狀態（position、status、error、is_enabled、on_zero、sci、sco）。

---

### `get_linear_motor_pos() → (code, position)`
取得線性馬達位置。

---

### `get_linear_motor_status() → (code, status)`
取得線性馬達狀態（0=完成，1=運動中，2=已停止）。

---

### `get_linear_motor_error() → (code, error)`
取得線性馬達錯誤碼。

---

### `get_linear_motor_is_enabled() → (code, status)`
查詢是否已啟用（0=未啟用，1=啟用）。

---

### `get_linear_motor_on_zero() → (code, status)`
查詢是否在零點（0=否，1=是）。

---

### `get_linear_motor_sci() → (code, sci1)` / `get_linear_motor_sco() → (code, sco)`
取得 SCI/SCO 狀態。

---

### `clean_linear_motor_error() → code`
清除線性馬達錯誤。

---

### `set_linear_motor_enable(enable) → code`
啟用/停用線性馬達。

---

### `set_linear_motor_speed(speed) → code`
設定線性馬達速度（1~1000 mm/s）。

---

### `set_linear_motor_back_origin(wait=True, **kwargs) → code`
使線性馬達回零點（首次上電必須執行）。

---

### `set_linear_motor_pos(pos, speed=None, wait=True, timeout=100, **kwargs) → code`
設定線性馬達目標位置。

| 參數 | 說明 |
|------|------|
| `pos` | 0~700/1000/1500 mm（依 SN 型號而定；AL1300=0~700mm，AL1301=另有範圍） |

---

### `set_linear_motor_stop() → code`
停止線性馬達。

---

## 22. Callback 回呼管理

### `register_report_callback(callback, report_cartesian=True, report_joints=True, report_state=True, report_error_code=True, report_warn_code=True, report_mtable=True, report_mtbrake=True, report_cmd_num=True) → bool`
註冊通用回報 callback（需 enable_report=True）。

callback 資料格式：
```python
{
    'cartesian': [x, y, z, roll, pitch, yaw],  # 若 report_cartesian=True
    'joints': [J1, ..., J7],                    # 若 report_joints=True
    'error_code': 0,
    'warn_code': 0,
    'state': state,
    'mtbrake': [...],
    'mtable': [...],
    'cmdnum': 0,
}
```

---

### `register_report_location_callback(callback, report_cartesian=True, report_joints=True) → bool`
只回報位置的 callback。

```python
{'cartesian': [x, y, z, roll, pitch, yaw], 'joints': [J1, ..., J7]}
```

---

### `register_connect_changed_callback(callback) → bool`
連線狀態變更時觸發。`{'connected': bool, 'reported': bool}`

---

### `register_state_changed_callback(callback) → bool`
機械臂狀態變更時觸發。`{'state': int}`

---

### `register_mode_changed_callback(callback) → bool`
控制模式變更時觸發（需 socket + enable_report）。`{'mode': int}`

---

### `register_mtable_mtbrake_changed_callback(callback) → bool`
馬達啟用/剎車狀態變更時觸發。`{'mtable': [...], 'mtbrake': [...]}`

---

### `register_error_warn_changed_callback(callback) → bool`
錯誤或警告碼變更時觸發。`{'error_code': int, 'warn_code': int}`

---

### `register_cmdnum_changed_callback(callback) → bool`
指令佇列數量變更時觸發。`{'cmdnum': int}`

---

### `register_temperature_changed_callback(callback) → bool`
溫度變更時觸發。`{'temperatures': [T1, ..., T7]}`

---

### `register_count_changed_callback(callback) → bool`
計數器變更時觸發。`{'count': int}`

---

### `register_iden_progress_changed_callback(callback) → bool`
識別進度變更時觸發。`{'progress': int}`

---

### `register_feedback_callback(callback=None) → bool`
Feedback callback（需 firmware >= 2.1.0）。資料為 bytes。

---

### Release 系列（每個 register 對應一個 release）

| 方法 | 說明 |
|------|------|
| `release_report_callback(callback)` | 解除通用回報 callback |
| `release_report_location_callback(callback)` | 解除位置回報 callback |
| `release_connect_changed_callback(callback)` | 解除連線狀態 callback |
| `release_state_changed_callback(callback)` | 解除狀態 callback |
| `release_mode_changed_callback(callback)` | 解除模式 callback |
| `release_mtable_mtbrake_changed_callback(callback)` | 解除馬達狀態 callback |
| `release_error_warn_changed_callback(callback)` | 解除錯誤警告 callback |
| `release_cmdnum_changed_callback(callback)` | 解除佇列數量 callback |
| `release_temperature_changed_callback(callback)` | 解除溫度 callback |
| `release_count_changed_callback(callback)` | 解除計數器 callback |
| `release_iden_progress_changed_callback(callback)` | 解除識別進度 callback |
| `release_feedback_callback(callback)` | 解除 feedback callback |

---

## 23. 軌跡錄製與回放

> 需 firmware >= 1.2.0，搭配 xArmStudio >= 1.2.0。

### `get_trajectories() → (code, trajectories)`
取得已存軌跡清單。每筆：`{'name': str, 'duration': float(秒)}`。

---

### `start_record_trajectory() → code`
開始錄製軌跡（需先呼叫 `set_mode(2)` 和 `set_state(0)` 進入示教模式）。

---

### `stop_record_trajectory(filename=None, **kwargs) → code`
停止錄製。

| 參數 | 說明 |
|------|------|
| `filename` | 儲存名稱（僅英文或數字，最長 50 字元）；None=僅停止不儲存 |

---

### `get_record_seconds() → (code, seconds)`
取得錄製秒數（需 firmware >= 2.4.0，錄製中或錄製完成但尚未儲存時有效）。

---

### `save_record_trajectory(filename, wait=True, timeout=5, **kwargs) → code`
儲存當前錄製的軌跡至控制箱。⚠️ 重複呼叫會以空軌跡覆蓋。

---

### `load_trajectory(filename, wait=True, timeout=None, **kwargs) → code`
載入指定軌跡至記憶體。

---

### `playback_trajectory(times=1, filename=None, wait=True, double_speed=1, **kwargs) → code`
播放軌跡。

| 參數 | 說明 |
|------|------|
| `times` | 重複次數（僅在當前位置=軌跡結束位置時有效，否則只播放一次） |
| `filename` | 若為 None，需先呼叫 `load_trajectory` |
| `double_speed` | 倍速播放：1/2/4（需 firmware > 1.2.11） |

---

### `get_trajectory_rw_status() → (code, status)`
取得軌跡讀寫狀態（0=無、1=載入中、2=載入成功、3=載入失敗、4=儲存中、5=儲存成功、6=儲存失敗）。

---

### `get_record_seconds() → (code, seconds)`
取得實際錄製時長（需 firmware >= 2.4.0）。

---

### `delete_trajectory(name) → code`
刪除指定軌跡。

---

### `get_traj_speeding(rate) → (code, speed_info)`
取得軌跡錄製中超速的關節與速度資訊。`rate=1/2/4`。

---

## 24. 校正工具方法

> 需 firmware >= 1.6.9。

### `calibrate_tcp_coordinate_offset(four_points, is_radian=None) → (code, [x, y, z])`
四點法校正 TCP 位置偏移。  
`four_points`：4 個示教位置 list，每點 `[x, y, z, roll, pitch, yaw]`。

---

### `calibrate_tcp_orientation_offset(rpy_be, rpy_bt, input_is_radian=None, return_is_radian=None) → (code, [roll, pitch, yaw])`
額外示教點校正 TCP 姿態偏移。

| 參數 | 說明 |
|------|------|
| `rpy_be` | 無 TCP 偏移時的示教點 RPY |
| `rpy_bt` | 有 TCP 偏移時的示教點 RPY |

---

### `calibrate_user_orientation_offset(three_points, mode=0, trust_ind=0, ...) → (code, [roll, pitch, yaw])`
三點法示教使用者座標系姿態偏移。

點位選取規則：第一點為原點，第二點沿目標座標系 X+ 方向，第三點沿 Y+ 方向（允許不完全垂直，計算時自動修正）。

---

### `calibrate_user_coordinate_offset(rpy_ub, pos_b_uorg, is_radian=None) → (code, [x, y, z])`
確定使用者座標系位置偏移（一個額外示教點）。

| 參數 | 說明 |
|------|------|
| `rpy_ub` | `calibrate_user_orientation_offset` 的結果 |
| `pos_b_uorg` | 示教點在基座座標系的位置（若機械臂無法到達，可手動輸入） |

---

## 25. RS485 / Modbus RTU

### `set_rs485_baudrate(baud, target='robot', **kwargs) → (code, baudrate)`
設定 RS485 鮑率。

| 參數 | 說明 |
|------|------|
| `baud` | 支援：4800/9600/19200/38400/57600/115200/230400/460800/921600/1000000/1500000/2000000/2500000 |
| `target` | `'robot'`（機器人 RS485）或 `'control_box'`（控制箱 RS485） |

---

### `get_rs485_baudrate(target='robot', **kwargs) → (code, baudrate)`
取得 RS485 鮑率。

---

### `set_rs485_timeout(timeout, target='robot', protocol='modbus_rtu', **kwargs) → code`
設定 RS485 逾時（毫秒）。`protocol='modbus_rtu'` 或 `'transparent'`。

---

### `get_rs485_timeout(target='robot', protocol='modbus_rtu', **kwargs) → (code, timeout)`
取得 RS485 逾時（毫秒）。

---

### `set_rs485_data(datas, min_res_len=0, target='robot', protocol='modbus_rtu', use_503_port=False, **kwargs) → (code, modbus_response)`
發送 Modbus 資料至 RS485。

| 參數 | 說明 |
|------|------|
| `datas` | 資料 list |
| `min_res_len` | 最小回應長度（0=不檢查） |
| `use_503_port` | 使用 port 503 通訊（透明傳輸用，需 firmware >= 1.11.0） |

---

### `set_modbusrtu_params(slave_id, baudrate, stopbits=1, parity=0) → code`
設定 Modbus RTU 參數。

| 參數 | 說明 |
|------|------|
| `slave_id` | 1~247 |
| `stopbits` | 1 或 2 |
| `parity` | 同位元 |

---

### `get_modbusrtu_params() → (code, [slave_id, baudrate, stopbits, parity])`
取得 Modbus RTU 參數。

---

### `getset_tgpio_modbus_data(datas, min_res_len=0, ...) → (code, response)`
發送 Modbus 資料至 RS485（舊版介面，建議改用 `set_rs485_data`）。

---

### `set_baud_checkset_enable(enable) → code`
啟用/停用末端 IO 板自動鮑率校驗（影響 gripper/bio/robotiq/linear_motor）。

---

### `set_checkset_default_baud(type_, baud) → code`
設定校驗鮑率值。`type_=1`（xArm Gripper）/`2`（BIO）/`3`（Robotiq）/`4`（Linear Motor）；`baud <= 0` 停用校驗。

---

### `get_checkset_default_baud(type_) → (code, baud)`
取得校驗鮑率值。

---

## 26. Modbus TCP（標準 Modbus）

### `read_coil_bits(addr, quantity) → (code, bits)`
讀取線圈（Function Code 0x01）。

---

### `read_input_bits(addr, quantity) → (code, bits)`
讀取離散輸入（Function Code 0x02）。

---

### `read_holding_registers(addr, quantity, is_signed=False) → (code, regs)`
讀取保持暫存器（Function Code 0x03）。

---

### `read_input_registers(addr, quantity, is_signed=False) → (code, regs)`
讀取輸入暫存器（Function Code 0x04）。

---

### `write_single_coil_bit(addr, bit_val) → code`
寫入單一線圈（Function Code 0x05）。`bit_val=0/1`。

---

### `write_single_holding_register(addr, reg_val) → code`
寫入單一保持暫存器（Function Code 0x06）。

---

### `write_multiple_coil_bits(addr, bits) → code`
寫入多個線圈（Function Code 0x0F）。

---

### `write_multiple_holding_registers(addr, regs) → code`
寫入多個保持暫存器（Function Code 0x10）。

---

### `mask_write_holding_register(addr, and_mask, or_mask) → code`
遮罩寫入保持暫存器（Function Code 0x16）。

---

### `write_and_read_holding_registers(r_addr, r_quantity, w_addr, w_regs, is_signed=False) → (code, regs)`
寫入並讀取保持暫存器（Function Code 0x17）。

---

### `send_hex_cmd(datas, **kwargs) → code 或 hex_list`
發送十六進位協議指令。`datas`：十六進位資料 list。

---

## 27. 縮減模式與安全邊界

### `get_reduced_mode() → (code, mode)`
取得縮減模式狀態（0=關閉，1=開啟）。

---

### `set_reduced_mode(on) → code`
開啟/關閉縮減模式（需 firmware >= 1.2.0）。

---

### `get_reduced_states(is_radian=None) → (code, states)`
取得縮減模式的完整設定狀態（需 firmware >= 1.2.0）。

若 firmware > 1.2.11，states 包含：`[reduced_on, [x_max, x_min, y_max, y_min, z_max, z_min], max_tcp_speed, max_joint_speed, joint_ranges, fence_on, collision_rebound_on]`

---

### `set_reduced_max_tcp_speed(speed) → code`
設定縮減模式最大 TCP 速度（mm/s）。需重新啟動縮減模式生效。

---

### `set_reduced_max_joint_speed(speed, is_radian=None) → code`
設定縮減模式最大關節速度。需重新啟動縮減模式生效。

---

### `set_reduced_tcp_boundary(boundary) → code`
設定安全邊界。`boundary=[x_max, x_min, y_max, y_min, z_max, z_min]`（mm）。

---

### `set_reduced_joint_range(joint_range, is_radian=None) → code`
設定縮減模式下的關節限位（需 firmware >= 1.2.11）。  
`joint_range=[J1_min, J1_max, ..., J7_min, J7_max]`

---

### `set_fence_mode(on) → code`
開啟/關閉安全邊界模式（需 firmware >= 1.2.11）。

---

### `set_collision_rebound(on) → code`
開啟/關閉碰撞反彈（需 firmware >= 1.2.11）。

---

### `set_self_collision_detection(on_off) → code`
開啟/關閉自碰撞偵測。

---

### `set_collision_tool_model(tool_type, *args, **kwargs) → code`
設定末端工具幾何模型（用於自碰撞偵測）。

| `tool_type` | 工具 |
|-------------|------|
| 0 | 無末端工具 |
| 1 | xArm Gripper |
| 2 | xArm Vacuum Gripper |
| 3 | xArm BIO Gripper |
| 4 | Robotiq-2F-85 |
| 5 | Robotiq-2F-140 |
| 7 | Lite Gripper |
| 8 | Lite Vacuum Gripper |
| 9 | xArm Gripper G2 |
| 10 | DH-ROBOTICS PGC-140-50 |
| 11 | INSPIRE-ROBOTS RH56DFX-2L |
| 12 | INSPIRE-ROBOTS RH56DFX-2R |
| 13 | xArm BIO Gripper G2 |
| 21 | 圓柱體（需指定 `radius(mm)`, `height(mm)`, `x_offset`, `y_offset`, `z_offset`） |
| 22 | 長方體（需指定 `x(mm)`, `y(mm)`, `z(mm)`, `x_offset`, `y_offset`, `z_offset`） |

---

### `set_report_tau_or_i(tau_or_i=0) → code`
設定回報力矩（0）或電流（1）。

---

### `get_report_tau_or_i() → (code, tau_or_i)`
取得目前回報類型。

---

## 28. 進階診斷與錯誤資訊

### `get_c31_error_info() → (code, err_info)`
取得碰撞錯誤（C31）詳細資訊（需 firmware >= 2.3.0）。

---

### `get_c37_error_info(is_radian=None) → (code, err_info)`
取得負載錯誤（C37）詳細資訊。

---

### `get_c23_error_info(is_radian=None) → (code, err_info)`
取得關節角度超限錯誤（C23）詳細資訊。

---

### `get_c24_error_info(is_radian=None) → (code, err_info)`
取得關節速度超限錯誤（C24）詳細資訊。

---

### `get_c60_error_info() → (code, err_info)`
取得線速度超限錯誤（C60）詳細資訊（需 firmware >= 2.3.0，mode 1）。

---

### `get_c38_error_info(is_radian=None) → (code, err_info)`
取得關節硬體角度超限錯誤（C38）詳細資訊（需 firmware >= 2.4.0）。

---

### `get_c54_error_info() → (code, err_info)`
取得力矩感測器碰撞錯誤（C54）詳細資訊（需 firmware >= 2.6.103）。

---

### `get_poe_status() → (code, status)`
取得 PoE 狀態（需 firmware >= 2.3.0）。

---

### `get_iden_status() → (code, status)`
取得識別狀態（需 firmware >= 2.3.0）。

---

### `get_servo_debug_msg(show=False, lang='en') → (code, servo_info_list)`
取得伺服偵錯訊息（僅供偵錯用）。

---

## 29. DH 參數、回饋、其他進階功能

### `get_dh_params() → (code, dh_params)`
取得 DH 參數（需 firmware >= 2.0.0）。

---

### `set_dh_params(dh_params, flag=0) → code`
設定 DH 參數（需 firmware >= 2.0.0）。⚠️ 僅供需要外部 DH 參數的使用者，一般使用者不應修改。

---

### `set_feedback_type(feedback_type) → code`
設定回饋類型（需 firmware >= 2.1.0，僅在 position mode 有效，只影響後續任務）。

---

### `set_cartesian_velo_continuous(on_off) → code`
設定笛卡爾運動速度是否連續（需 firmware >= 1.9.0）。

---

### `set_allow_approx_motion(on_off) → code`
允許在某些奇異點附近使用近似解以避免超速（需 firmware >= 1.9.0）。

---

### `get_allow_approx_motion() → code`
取得是否啟用近似解（需 firmware >= 1.9.0）。

---

### `iden_joint_friction(sn=None) → (code, result)`
識別關節摩擦力（需 firmware >= 1.9.0）。

---

### `set_linear_spd_limit_factor(factor) → code`
設定線速度限制係數（預設 1.2，需 firmware >= 2.3.0，mode 1 才有效）。

---

### `get_linear_spd_limit_factor() → (code, factor)`
取得線速度限制係數。

---

### `set_cmd_mat_history_num(num) → code`
設定指令矩陣歷史數量（需 firmware >= 2.3.0）。

---

### `get_cmd_mat_history_num() → (code, num)`
取得指令矩陣歷史數量。

---

### `set_fdb_mat_history_num(num) → code`
設定回饋矩陣歷史數量。

---

### `get_fdb_mat_history_num() → (code, num)`
取得回饋矩陣歷史數量。

---

### `get_base_board_version(board_id=10) → (code, version)`
取得底板版本。

---

### `get_initial_point() → (code, point)`
從 Studio 取得初始點。`point=[J1, J2, ..., J7]`。

---

### `set_initial_point(point) → code`
設定初始點。

---

### `get_mount_direction() → (code, [tilt_angle, rotate_angle])`
取得安裝方向角度。

---

### `set_external_device_monitor_params(dev_type, frequency) → code`
設定外部裝置監控參數（需 firmware >= 2.7.100）。啟動後外部裝置的位置/速度/電流資訊將透過 port 30000 回報。

---

### `get_external_device_monitor_params() → (code, params)`
取得外部裝置監控參數。

---

### `run_blockly_app(path, **kwargs)`
執行 xArmStudio 生成的 Blockly 應用程式。

---

### `run_gcode_file(path, **kwargs)`
執行 G-code 檔案。

---

### `run_gcode_app(path, **kwargs) → code`
執行 xArmStudio G-code 專案檔。

---

## 30. Lite6 專屬夾爪

> 需 firmware >= 1.10.0，**僅適用於 Lite6 系列機械臂**。

### `open_lite6_gripper(sync=True) → code`
開啟 Lite6 夾爪。`sync=True` 表示在運動佇列中執行；`False` 立即執行。

---

### `close_lite6_gripper(sync=True) → code`
關閉 Lite6 夾爪。

---

### `stop_lite6_gripper(sync=True) → code`
停止 Lite6 夾爪。

---

## 31. 偵錯與版本查詢

| 方法 | 說明 |
|------|------|
| `get_gripper_version() → (code, version)` | 取得夾爪版本（偵錯用） |
| `get_servo_version(servo_id=1) → (code, version)` | 取得指定伺服版本（servo_id=1~7） |
| `get_harmonic_type(servo_id=1) → (code, type)` | 取得諧波類型（偵錯用） |
| `get_hd_types() → (code, types)` | 取得所有諧波類型（偵錯用） |
| `set_rs485_use_503_port(use_503_port=True)` | 設定是否使用 port 503 進行 RS485 通訊 |
| `set_tgpio_modbus_timeout(...)` | 舊版介面，請改用 `set_rs485_timeout` |
| `get_tgpio_modbus_timeout(...)` | 舊版介面，請改用 `get_rs485_timeout` |
| `set_tgpio_modbus_baudrate(baud)` | 舊版介面，請改用 `set_rs485_baudrate` |
| `get_tgpio_modbus_baudrate()` | 舊版介面，請改用 `get_rs485_baudrate` |

---

## 32. 屬性別名（Alias）

以下別名在 `__getattr__` 中對應到實際方法，可互換使用：

| 別名 | 實際方法/屬性 |
|------|--------------|
| `get_ik` | `get_inverse_kinematics` |
| `get_fk` | `get_forward_kinematics` |
| `set_sleep_time` | `set_pause_time` |
| `register_maable_mtbrake_changed_callback` | `register_mtable_mtbrake_changed_callback` |
| `release_maable_mtbrake_changed_callback` | `release_mtable_mtbrake_changed_callback` |
| `position_offset` | `tcp_offset`（屬性） |
| `get_gpio_digital` | `get_tgpio_digital` |
| `set_gpio_digital` | `set_tgpio_digital` |

---

## 常見使用模式（針對本專案 sEMG → Lite6 控制）

```python
from xarm.wrapper import XArmAPI

arm = XArmAPI('192.168.1.x', is_radian=False)
arm.motion_enable(True)
arm.set_mode(0)       # 位置控制模式
arm.set_state(0)      # 運動狀態

# 移動到目標位置（等待完成）
code = arm.set_position(x=300, y=0, z=200, roll=180, pitch=0, yaw=0, wait=True)

# 控制 Lite6 夾爪（sEMG 觸發）
arm.open_lite6_gripper(sync=True)   # 對應 ED（伸指）動作
arm.close_lite6_gripper(sync=True)  # 對應 FDS（屈指）動作

# 清除錯誤並恢復
arm.clean_error()
arm.motion_enable(True)
arm.set_state(0)

arm.disconnect()
```

---

*本文件由 `xarm_api.py`（Ufactory xArm Python SDK）原始碼完整解析生成，2026/07/26。*
