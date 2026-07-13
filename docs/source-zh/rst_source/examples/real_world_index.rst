真实机器人强化学习
========================================

当你的出发点是真实机器人硬件时，请使用本节。如果你使用 Franka 机械臂或 Franka-based rig，请从 Franka 开始；GimArm、XSquare Turtle2 和 Dexmal DOS-W1 请进入对应机器人页面。

每个章节都给出遥操作、数据采集、Sim-to-Real 迁移、部署或在线 RL 所需的配置路径。

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/franka.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka.html" style="text-decoration: underline; color: blue;">
           <b>Franka</b>
         </a><br>
         进入 Franka 章节，查看基础真机 RL、reward model、ZED + Robotiq、GELLO、VR / PICO、双臂、灵巧手、Pi0 SFT 和 HG-DAgger
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/blob/main/pic/franka_reward_model.jpg?raw=true"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_reward_model.html" style="text-decoration: underline; color: blue;">
           <b>Franka真机强化学习（基于 Reward Model ）</b>
         </a><br>
         使用 reward model 辅助完成机器人操作任务
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/robotiq_zed.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_zed_robotiq.html" style="text-decoration: underline; color: blue;">
           <b>Franka 真机使用 ZED 相机与 Robotiq 夹爪</b>
         </a><br>
         Franka 真机中 ZED 相机、Robotiq 夹爪安装与数据采集配置
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/gello.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_gello.html" style="text-decoration: underline; color: blue;">
           <b>Franka 真机使用 GELLO 遥操作设备</b>
         </a><br>
         Franka 真机中 GELLO 遥操作设备安装、配置与验证流程
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/dual.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dual_franka.html" style="text-decoration: underline; color: blue;">
           <b>双 Franka 真机：GELLO 采集 + π₀.₅ SFT</b>
         </a><br>
         双节点双臂 Franka rig：GELLO 关节空间采集、rot6d SFT、脚踏部署
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/dexhand.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_dexhand.html" style="text-decoration: underline; color: blue;">
           <b>Franka 机械臂与灵巧手真机强化学习</b>
         </a><br>
         Franka 机械臂 + 睿研五指灵巧手真机强化学习
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_pi0_sft_deploy.html" style="text-decoration: underline; color: blue;">
           <b>Franka真机Pi0监督微调与部署全流程</b>
         </a><br>
         数据采集 + Pi0 SFT + 真机部署的完整端到端演示
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/hg-dagger.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/hg-dagger.html" style="text-decoration: underline; color: blue;">
           <b>Franka 机械臂上的 HG-DAgger</b>
         </a><br>
         Human-Gated 真机 DAgger 流程：数据采集、SFT 与在线干预训练
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/gim-arm.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/gim_arm.html" style="text-decoration: underline; color: blue;">
           <b>GimArm</b>
         </a><br>
         在 GimArm 六自由度机械臂上通过 SocketCAN 与 Pinocchio FK 训练 peg-insertion 任务
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/xsquare_turtle2.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/xsquare_turtle2_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/xsquare_turtle2.html" style="text-decoration: underline; color: blue;">
           <b>XSquare Turtle2</b>
         </a><br>
         在 XSquare Turtle2 双臂机器人上运行 SAC + CNN 策略
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/dosw1.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/dos-w1.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dosw1.html" style="text-decoration: underline; color: blue;">
           <b>Dexmal DOS-W1</b>
         </a><br>
         在 Dexmal DOS-W1 双臂机器人上训练 Flow Matching + SAC 抓取任务
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/so101_arm.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/so101.html" style="text-decoration: underline; color: blue;">
           <b>SO101 双臂真机强化学习</b>
         </a><br>
         基于 pi0.5 + PPO 的 SO101 双臂 follower 臂平台真机训练
       </p>
     </div>

   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   embodied/franka
   embodied/franka_reward_model
   embodied/franka_zed_robotiq
   embodied/franka_gello
   embodied/dual_franka
   embodied/franka_dexhand
   embodied/franka_pi0_sft_deploy
   embodied/hg-dagger
   embodied/gim_arm
   embodied/xsquare_turtle2
   embodied/dosw1
   embodied/so101
