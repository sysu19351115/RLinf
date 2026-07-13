RL with Real-World Robots
=========================

Use this section when your starting point is physical robot hardware. Start with Franka if you use a Franka arm or a Franka-based rig; use the other robot pages for GimArm, XSquare Turtle2, and Dexmal DOS-W1.

Each section gives the setup path for teleoperation, data collection, sim-to-real transfer, deployment, or online RL.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; align-items: flex-start; justify-items: center; max-width: 980px; margin: 0 auto;">

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/franka.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/franka_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka.html" style="text-decoration: underline; color: blue;">
           <b>Franka</b>
         </a><br>
         Use the Franka section for base real-world RL, reward models, ZED + Robotiq, GELLO, VR / PICO, dual-arm rigs, dexterous hands, Pi0 SFT, and HG-DAgger
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/blob/main/pic/franka_reward_model.jpg?raw=true"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_reward_model.html" style="text-decoration: underline; color: blue;">
           <b>Real-World RL with Franka (Reward Model)</b>
         </a><br>
         Use a reward model to assist robotic manipulation tasks
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/robotiq_zed.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_zed_robotiq.html" style="text-decoration: underline; color: blue;">
           <b>Real-World Franka with ZED Cameras and Robotiq Gripper</b>
         </a><br>
         ZED camera, Robotiq gripper, and data-collection setup for Franka
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/gello.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_gello.html" style="text-decoration: underline; color: blue;">
           <b>Real-World Franka with GELLO Teleoperation</b>
         </a><br>
         GELLO teleoperation setup, configuration, and verification for Franka
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/dual.jpeg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dual_franka.html" style="text-decoration: underline; color: blue;">
           <b>Real-World Dual-Franka: GELLO + π₀.₅ SFT</b>
         </a><br>
         Two-node dual-arm Franka rig: GELLO joint collection, rot6d SFT, foot-pedal eval
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/dexhand.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_dexhand.html" style="text-decoration: underline; color: blue;">
           <b>Real-World RL with Franka and Dexterous Hand</b>
         </a><br>
         Franka arm + Ruiyan five-finger dexterous hand real-world RL
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/pi0_icon.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/franka_pi0_sft_deploy.html" style="text-decoration: underline; color: blue;">
           <b>Franka Pi0 SFT and Deployment</b>
         </a><br>
         Data collection + Pi0 SFT + real-world deployment demo
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/hg-dagger.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/hg-dagger.html" style="text-decoration: underline; color: blue;">
           <b>HG-DAgger on a Franka arm</b>
         </a><br>
         Human-Gated real-world DAgger pipeline: collection, SFT, and online intervention training
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/gim-arm.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/gim_arm.html" style="text-decoration: underline; color: blue;">
           <b>GimArm</b>
         </a><br>
         Train a 6-DOF GimArm peg-insertion task over SocketCAN with Pinocchio-based FK
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/xsquare_turtle2.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/xsquare_turtle2_arm_small.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/xsquare_turtle2.html" style="text-decoration: underline; color: blue;">
           <b>XSquare Turtle2</b>
         </a><br>
         Run SAC with a CNN policy on the XSquare Turtle2 dual-arm robot
       </p>
     </div>
     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <a href="embodied/dosw1.html" style="display: block;"><img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/dos-w1.png"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" /></a>
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/dosw1.html" style="text-decoration: underline; color: blue;">
           <b>Dexmal DOS-W1</b>
         </a><br>
         Train a flow-matching + SAC pick task on the Dexmal DOS-W1 dual-arm robot
       </p>
     </div>

     <div style="flex: 1 1 30%; max-width: 300px; text-align: center;">
       <img src="https://github.com/RLinf/misc/raw/main/pic/so101_arm.jpg"
            style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.15);" />
       <p style="margin-top: 8px; font-size: 14px; line-height: 1.4;">
         <a href="embodied/so101.html" style="text-decoration: underline; color: blue;">
           <b>Real-World RL with SO101 Bimanual Arms</b>
         </a><br>
           pi0.5 + PPO on the SO101 bimanual follower-arm platform
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
