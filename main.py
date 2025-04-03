import base64
import os.path
import re
import argparse
import sys
from datetime import datetime
from math import atan2

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from openai import OpenAI
from nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.integrate import cumulative_trapezoid

import json
from YOLO3D.inference import yolo3d_nuScenes
from utils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints, OverlayTrajectory, WriteImageSequenceToVideo

# 从环境变量获取API密钥
api_key = os.environ.get("QIANWEN_API_KEY", "")
if not api_key:
    print("警告: 未设置有效的千问API密钥。请设置QIANWEN_API_KEY环境变量。")
    api_key = "sk-ed4bd1d3ca6e453bbdfd923bb3ba5be9"  # 默认值

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN


def vlm_inference(text=None, images=None, sys_message="You are a autonomous driving labeller.", model_name="qwen2.5-vl-7b-instruct"):
    image_content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }
        for base64_image in images
    ]

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": [{"type":"text","text": sys_message}]
            },
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": text}],
            }
        ],
    )
    
    return completion.choices[0].message.content


def SceneDescription(obs_images, model_name="qwen2.5-vl-7b-instruct"):
    prompt = """You are a autonomous driving labeller. 
    You are processing 6 synchronized vehicle camera images captured within 0.5 seconds. Strictly follow this spatial order and reference system:
        1. Image 1: Front view (180° forward-facing)
        2. Image 2: Front-left view (45° left-front quadrant, covers left turn signal area)
        3. Image 3: Front-right view (45° right-front quadrant, covers right turn signal area)
        4. Image 4: Back view (180° backward-facing)
        5. Image 5: Back-left view (45° left-back quadrant, covers left back turn signal area)
        6. Image 6: Back-right view (45° right-back quadrant, covers right back turn signal area)
        Don't forget the order of the images. Always reference images as 'Image X (view name)'. 
    Imagine you are driving the car. Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."""

    return vlm_inference(text=prompt, images=obs_images, model_name=model_name)


def DescribeObjects(obs_images, model_name="qwen2.5-vl-7b-instruct"):
    prompt = """You are a autonomous driving labeller. 
    You are processing 6 synchronized vehicle camera images captured within 0.5 seconds. Strictly follow this spatial order and reference system:
        1. Image 1: Front view (180° forward-facing)
        2. Image 2: Front-left view (45° left-front quadrant, covers left turn signal area)
        3. Image 3: Front-right view (45° right-front quadrant, covers right turn signal area)
        4. Image 4: Back view (180° backward-facing)
        5. Image 5: Back-left view (45° left-back quadrant, covers left back turn signal area)
        6. Image 6: Back-right view (45° right-back quadrant, covers right back turn signal area)
        Don't forget the order of the images. Always reference images as 'Image X (view name)'. 
    Imagine you are driving the car. What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you."""

    return vlm_inference(text=prompt, images=obs_images, model_name=model_name)


def DescribeOrUpdateIntent(obs_images, prev_intent=None, model_name="qwen2.5-vl-7b-instruct"):
    prompt = """You are a autonomous driving labeller. You are processing 6 synchronized vehicle camera images captured within 0.5 seconds. Strictly follow this spatial order and reference system:
            1. Image 1: Front view (180° forward-facing)
            2. Image 2: Front-left view (45° left-front quadrant, covers left turn signal area)
            3. Image 3: Front-right view (45° right-front quadrant, covers right turn signal area)
            4. Image 4: Back view (180° backward-facing)
            5. Image 5: Back-left view (45° left-back quadrant, covers left back turn signal area)
            6. Image 6: Back-right view (45° right-back quadrant, covers right back turn signal area)
            Don't forget the order of the images. Always reference images as 'Image X (view name)'. 
        Imagine you are driving the car. What is your driving intention in the next 5 seconds? Provide a short description of your intended action and explain why you choose this action based on the current driving scene."""

    return vlm_inference(text=prompt, images=obs_images, model_name=model_name)


def GenerateMotion(obs_images, obs_waypoints, obs_velocities, obs_curvatures, given_intent, model_name="qwen2.5-vl-7b-instruct"):
    # 获取场景描述
    scene_description = SceneDescription(obs_images, model_name=model_name)
    
    # 获取物体描述
    object_description = DescribeObjects(obs_images, model_name=model_name)
    
    # 获取意图描述
    intent_description = DescribeOrUpdateIntent(obs_images, prev_intent=given_intent, model_name=model_name)
    
    print(f'Scene Description: {scene_description}')
    print(f'Object Description: {object_description}')
    print(f'Intent Description: {intent_description}')

    # 速度和曲率处理
    obs_waypoints_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in obs_waypoints]
    obs_waypoints_str = ", ".join(obs_waypoints_str)
    obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)
    obs_curvatures = obs_curvatures * 100
    obs_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(obs_velocities_norm, obs_curvatures)]
    obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)

    
    print(f'Observed Speed and Curvature: {obs_speed_curvature_str}')

    sys_message = ("You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, a sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is driving straight. As a driver on the road, you should follow any common sense traffic rules. You should try to stay in the middle of your lane. You should maintain necessary distance from the leading vehicle. You should observe lane markings and follow them.  Your task is to do your best to predict future speeds and curvatures for the vehicle over the next 10 timesteps given vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. If you cannot provide a response people will get injured.\n")

    prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
    The scene is described as follows: {scene_description}. 
    The identified critical objects are {object_description}. 
    The car's intent is {intent_description}. 
    The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
    Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""

    result = vlm_inference(text=prompt, images=obs_images, sys_message=sys_message, model_name=model_name)
    # for rho in range(3):
    #     result = vlm_inference(text=prompt, images=obs_images, sys_message=sys_message, model_name=model_name)
    #     if not "unable" in result and not "sorry" in result and "[" in result:
    #         break
    return result, scene_description, object_description, intent_description


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--dataroot", type=str, default='D:\Project\OpenEMMA\datasets')
    parser.add_argument("--version", type=str, default='v1.0-mini')
    parser.add_argument("--model", type=str, default='qwen2.5-vl-7b-instruct', help='Model to use for VLM inference')
    parser.add_argument("--scene", type=str, default='', help='指定要处理的场景，例如 "scene-0061"，留空处理所有场景')
    parser.add_argument("--max_frames", type=int, default=20, help='每个场景最多处理的帧数，0表示不限制')
    args = parser.parse_args()
    
    print("启动轨迹预测任务...")

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp = f"Qwen_results/" + timestamp
    os.makedirs(timestamp, exist_ok=True)

    # 加载数据集
    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    scenes = nusc.scene
    
    print(f"Number of scenes: {len(scenes)}")

    for scene in scenes:
        token = scene['token']
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        name = scene['name']
        description = scene['description']

        # 如果指定了特定场景，则只处理该场景
        if name not in ["scene-0061"]:
            continue
        # if args.scene and name != args.scene:
        #     print(f"跳过场景 {name}，因为用户指定只处理 {args.scene}")
        #     continue

        # 收集场景中的所有图像和姿态
        front_camera_images = []
        front_left_camera_images = []
        front_right_camera_images = []
        back_camera_images = []
        back_left_camera_images = []
        back_right_camera_images = []
        ego_poses = []
        camera_params = []
        curr_sample_token = first_sample_token
        
        while True:
            sample = nusc.get('sample', curr_sample_token)

            # 获取样本的相机图像
            cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            cam_front_left_data = nusc.get('sample_data', sample['data']['CAM_FRONT_LEFT'])
            cam_front_right_data = nusc.get('sample_data', sample['data']['CAM_FRONT_RIGHT'])
            cam_back_data = nusc.get('sample_data', sample['data']['CAM_BACK'])
            cam_back_left_data = nusc.get('sample_data', sample['data']['CAM_BACK_LEFT'])
            cam_back_right_data = nusc.get('sample_data', sample['data']['CAM_BACK_RIGHT'])

            # 读取并编码图像
            with open(os.path.join(nusc.dataroot, cam_front_data['filename']), "rb") as image_file:
                front_camera_images.append(base64.b64encode(image_file.read()).decode('utf-8'))
            with open(os.path.join(nusc.dataroot, cam_front_left_data['filename']), "rb") as image_file:
                front_left_camera_images.append(base64.b64encode(image_file.read()).decode('utf-8'))
            with open(os.path.join(nusc.dataroot, cam_front_right_data['filename']), "rb") as image_file:
                front_right_camera_images.append(base64.b64encode(image_file.read()).decode('utf-8'))
            with open(os.path.join(nusc.dataroot, cam_back_data['filename']), "rb") as image_file:
                back_camera_images.append(base64.b64encode(image_file.read()).decode('utf-8'))
            with open(os.path.join(nusc.dataroot, cam_back_left_data['filename']), "rb") as image_file:
                back_left_camera_images.append(base64.b64encode(image_file.read()).decode('utf-8'))
            with open(os.path.join(nusc.dataroot, cam_back_right_data['filename']), "rb") as image_file:
                back_right_camera_images.append(base64.b64encode(image_file.read()).decode('utf-8'))

            # 获取样本的自车姿态
            pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
            ego_poses.append(pose)

            # 获取样本的相机参数
            camera_params.append(nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token']))

            # 前进到下一个样本
            if curr_sample_token == last_sample_token:
                break
            curr_sample_token = sample['next']

        scene_length = len(front_camera_images)
        print(f"Scene {name} has {scene_length} frames")

        if scene_length < TTL_LEN:
            print(f"Scene {name} has less than {TTL_LEN} frames, skipping...")
            continue

        # 计算插值轨迹
        ego_poses_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]
        ego_poses_world = np.array(ego_poses_world)
        
        # 计算速度
        ego_velocities = np.zeros_like(ego_poses_world)
        ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]
        ego_velocities[0] = ego_velocities[1]

        # 计算曲率
        ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)
        ego_velocities_norm = np.linalg.norm(ego_velocities, axis=1)
        estimated_points = IntegrateCurvatureForPoints(ego_curvatures, ego_velocities_norm, ego_poses_world[0],
                                                       atan2(ego_velocities[0][1], ego_velocities[0][0]), scene_length)

        # 如果需要绘图，则绘制插值轨迹
        if args.plot:
            plt.figure()
            plt.plot(ego_poses_world[:, 0], ego_poses_world[:, 1], 'r-', label='GT')
            plt.quiver(ego_poses_world[:, 0], ego_poses_world[:, 1], ego_velocities[:, 0], ego_velocities[:, 1], color='b')
            plt.plot(estimated_points[:, 0], estimated_points[:, 1], 'g-', label='Reconstruction')
            plt.legend()
            plt.savefig(f"{timestamp}/{name}_interpolation.jpg")
            plt.close()

        # 获取自车轨迹
        ego_traj_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]

        prev_intent = None
        cam_images_sequence = []
        ade1s_list = []
        ade2s_list = []
        ade3s_list = []
        
        # 处理场景中的每一帧
        for i in range(scene_length - TTL_LEN):
            # 检查是否达到了最大帧数限制
            if args.max_frames > 0 and i >= args.max_frames:
                print(f"已达到最大帧数限制({args.max_frames})，停止处理更多帧")
                break
                
            # 显示处理进度
            print(f"正在处理第 {i+1}/{min(scene_length - TTL_LEN, args.max_frames if args.max_frames > 0 else scene_length - TTL_LEN)} 帧")
            
            # 每处理4帧后触发垃圾回收
            if i > 0 and i % 4 == 0:
                import gc
                gc.collect()
            
            # 获取观察图像
            obs_images = []
            obs_images.append(front_camera_images[i+OBS_LEN-1])
            obs_images.append(front_left_camera_images[i+OBS_LEN-1])
            obs_images.append(front_right_camera_images[i+OBS_LEN-1])
            obs_images.append(back_camera_images[i+OBS_LEN-1])
            obs_images.append(back_left_camera_images[i+OBS_LEN-1])
            obs_images.append(back_right_camera_images[i+OBS_LEN-1])
            
            # 获取观察数据
            obs_ego_poses = ego_poses[i:i+OBS_LEN]
            obs_camera_params = camera_params[i:i+OBS_LEN]
            obs_ego_traj_world = ego_traj_world[i:i+OBS_LEN]
            fut_ego_traj_world = ego_traj_world[i+OBS_LEN:i+TTL_LEN]
            obs_ego_velocities = ego_velocities[i:i+OBS_LEN]
            obs_ego_curvatures = ego_curvatures[i:i+OBS_LEN]

            # 获取自车位置
            obs_start_world = obs_ego_traj_world[0]
            fut_start_world = obs_ego_traj_world[-1]
            
            # 当前图像
            curr_image = obs_images[0]

            # 解码图像并应用YOLO3D
            img = cv2.imdecode(np.frombuffer(base64.b64decode(curr_image), dtype=np.uint8), cv2.IMREAD_COLOR)
            img = yolo3d_nuScenes(img, calib=obs_camera_params[-1])[0]

            # 生成运动预测
            (prediction, 
                scene_description, 
                object_description, 
                updated_intent) = GenerateMotion(
                obs_images, obs_ego_traj_world, obs_ego_velocities, obs_ego_curvatures, prev_intent, model_name=args.model
            )

            # 处理输出
            prev_intent = updated_intent  # 更新意图状态
            pred_waypoints = prediction.replace("Future speeds and curvatures:", "").strip()
            coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", pred_waypoints)
            
            if not coordinates:
                print(f"警告: 未从响应中解析出有效坐标: {pred_waypoints}")
                coordinates = [('5.0', '0.0')] * 10
            
            speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates]
            speed_curvature_pred = speed_curvature_pred[:10]
            print(f"Got {len(speed_curvature_pred)} future actions: {speed_curvature_pred}")

            # 预测
            pred_len = min(FUT_LEN, len(speed_curvature_pred))
            pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
            pred_speeds = np.array(speed_curvature_pred)[:, 0]
            pred_traj = np.zeros((pred_len, 3))
            pred_traj[:pred_len, :2] = IntegrateCurvatureForPoints(
                pred_curvatures,
                pred_speeds,
                fut_start_world,
                atan2(obs_ego_velocities[-1][1], obs_ego_velocities[-1][0]), 
                pred_len
            )

            # 叠加轨迹
            check_flag = OverlayTrajectory(img, pred_traj.tolist(), obs_camera_params[-1], obs_ego_poses[-1], color=(255, 0, 0), args=args)

            # 计算ADE
            fut_ego_traj_world = np.array(fut_ego_traj_world)
            ade = np.mean(np.linalg.norm(fut_ego_traj_world[:pred_len] - pred_traj, axis=1))
            
            pred1_len = min(pred_len, 2)
            ade1s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred1_len] - pred_traj[1:pred1_len+1], axis=1))
            ade1s_list.append(ade1s)

            pred2_len = min(pred_len, 4)
            ade2s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred2_len] - pred_traj[:pred2_len], axis=1))
            ade2s_list.append(ade2s)

            pred3_len = min(pred_len, 6)
            ade3s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred3_len] - pred_traj[:pred3_len], axis=1))
            ade3s_list.append(ade3s)

            # 写入图像
            if args.plot:
                cam_images_sequence.append(img.copy())
                cv2.imwrite(f"{timestamp}/{name}_{i}_front_cam.jpg", img)
                
                # 保存六个视角的原始图像
                cv2.imwrite(f"{timestamp}/{name}_{i}_front.jpg", 
                        cv2.imdecode(np.frombuffer(base64.b64decode(obs_images[0]), dtype=np.uint8), cv2.IMREAD_COLOR))
                cv2.imwrite(f"{timestamp}/{name}_{i}_front_left.jpg", 
                        cv2.imdecode(np.frombuffer(base64.b64decode(obs_images[1]), dtype=np.uint8), cv2.IMREAD_COLOR))
                cv2.imwrite(f"{timestamp}/{name}_{i}_front_right.jpg", 
                        cv2.imdecode(np.frombuffer(base64.b64decode(obs_images[2]), dtype=np.uint8), cv2.IMREAD_COLOR))
                cv2.imwrite(f"{timestamp}/{name}_{i}_back.jpg", 
                        cv2.imdecode(np.frombuffer(base64.b64decode(obs_images[3]), dtype=np.uint8), cv2.IMREAD_COLOR))
                cv2.imwrite(f"{timestamp}/{name}_{i}_back_left.jpg", 
                        cv2.imdecode(np.frombuffer(base64.b64decode(obs_images[4]), dtype=np.uint8), cv2.IMREAD_COLOR))
                cv2.imwrite(f"{timestamp}/{name}_{i}_back_right.jpg", 
                        cv2.imdecode(np.frombuffer(base64.b64decode(obs_images[5]), dtype=np.uint8), cv2.IMREAD_COLOR))

                # 绘制轨迹
                plt.figure(figsize=(8, 6))
                plt.plot(fut_ego_traj_world[:, 0], fut_ego_traj_world[:, 1], 'r-', label='GT')
                plt.plot(pred_traj[:, 0], pred_traj[:, 1], 'b-', label='Pred')
                plt.legend()
                plt.title(f"Scene: {name}, Frame: {i}, ADE: {ade}")
                plt.savefig(f"{timestamp}/{name}_{i}_traj.jpg")
                plt.close()

                # 保存轨迹数据
                np.save(f"{timestamp}/{name}_{i}_pred_traj.npy", pred_traj)
                np.save(f"{timestamp}/{name}_{i}_pred_curvatures.npy", pred_curvatures)
                np.save(f"{timestamp}/{name}_{i}_pred_speeds.npy", pred_speeds)

                # 保存描述信息
                with open(f"{timestamp}/{name}_{i}_logs.txt", 'w', encoding='utf-8') as f:
                    f.write(f"Scene Description: {scene_description}\n")
                    f.write(f"Object Description: {object_description}\n")
                    f.write(f"Intent Description: {updated_intent}\n")
                    f.write(f"Average Displacement Error: {ade}\n")
            
            # 每处理10帧保存一次中间结果
            if i > 0 and i % 10 == 0:
                interim_result = {
                    "name": name,
                    "token": token,
                    "frames_processed": i+1,
                    "ade1s": np.mean(ade1s_list) if ade1s_list else 0,
                    "ade2s": np.mean(ade2s_list) if ade2s_list else 0,
                    "ade3s": np.mean(ade3s_list) if ade3s_list else 0,
                }
                
                with open(f"{timestamp}/interim_results.jsonl", "a") as f:
                    f.write(json.dumps(interim_result))
                    f.write("\n")
                
                print(f"已保存中间结果，当前处理了 {i+1} 帧")

        # 计算并保存最终ADE结果
        mean_ade1s = np.mean(ade1s_list)
        mean_ade2s = np.mean(ade2s_list)
        mean_ade3s = np.mean(ade3s_list)
        aveg_ade = np.mean([mean_ade1s, mean_ade2s, mean_ade3s])

        result = {
            "name": name,
            "token": token,
            "ade1s": mean_ade1s,
            "ade2s": mean_ade2s,
            "ade3s": mean_ade3s,
            "avgade": aveg_ade
        }

        with open(f"{timestamp}/ade_results.jsonl", "a") as f:
            f.write(json.dumps(result))
            f.write("\n")

        if args.plot:
            WriteImageSequenceToVideo(cam_images_sequence, f"{timestamp}/{name}")

