# AutoChoreographer

AutoChoreographer 是一个基于计算机视觉和深度学习的自动驾驶场景理解系统，集成了3D物体检测、场景理解和轨迹预测功能，并提供了基于千问大模型的视觉语言理解能力。

## 功能特点

- **3D 物体检测**：基于 YOLO3D 的高精度 3D 物体检测
- **场景理解**：利用千问大模型进行场景语义理解
- **轨迹预测**：基于历史轨迹和场景理解的未来轨迹预测
- **Web 前端**：基于 Streamlit 的可视化界面

## 系统要求

- Python 3.8 或更高版本
- CUDA 支持的 GPU（推荐 NVIDIA GPU 8GB+ 显存）
- 至少 16GB 系统内存
- 至少 20GB 可用磁盘空间

## 快速开始

### 安装

1. **克隆代码仓库**

```bash
git clone https://github.com/obsidian368/AutoChoreographer.git
cd AutoChoreographer
```

2. **创建并激活虚拟环境**

```bash
# 使用 conda
conda create -n AutoChoreographer python=3.8
conda activate AutoChoreographer

# 或使用 venv
python -m venv AutoChoreographer_env
# Windows
AutoChoreographer_env\Scripts\activate
# Linux/Mac
source AutoChoreographer_env/bin/activate
```

3. **安装依赖**

```bash
# 安装后端依赖
pip install -r requirements.txt

# 安装 YOLO3D 依赖
cd YOLO3D
pip install -r requirements.txt
cd ..

# 安装前端依赖
cd frontend
pip install -r requirements.txt
cd ..
```

4. **配置 API 密钥**

设置千问 API 密钥环境变量：

```bash
# Windows
set QIANWEN_API_KEY=your_api_key_here

# Linux/Mac
export QIANWEN_API_KEY=your_api_key_here
```

5. **下载预训练模型**

确保 YOLO3D 目录中包含以下预训练模型文件：
- `YOLO3D/yolov5s.pt`
- `YOLO3D/yolo11n_nuimages.pt`

### 运行

1. **启动后端服务**

```bash
python main.py --dataroot /path/to/nuscenes/dataset --version v1.0-mini
```

2. **启动前端服务**

```bash
cd frontend
streamlit run app.py
```

3. **访问 Web 界面**

打开浏览器访问 `http://localhost:8501`

## 命令行参数

### 主程序参数

- `--dataroot`: NuScenes 数据集路径
- `--version`: 数据集版本
- `--model`: 使用的 VLM 模型名称（默认为 "qwen2.5-vl-7b-instruct"）
- `--scene`: 指定要处理的场景（可选）
- `--max_frames`: 每个场景最多处理的帧数（默认为 20）
- `--plot`: 是否生成可视化结果（默认为 True）

## Docker 部署

### 构建 Docker 镜像

```bash
cd YOLO3D
docker build -t AutoChoreographer-yolo3d .
cd ..
```

### 运行 Docker 容器

```bash
# 运行 YOLO3D 容器
cd YOLO3D
./runDocker.sh
cd ..

# 运行主应用
docker run -it --gpus all -v $(pwd):/app -e QIANWEN_API_KEY=your_api_key_here AutoChoreographer-yolo3d python main.py --dataroot /app/data/nuscenes --version v1.0-mini
```

## 项目结构

```
AutoChoreographer/
├── main.py                 # 主程序入口
├── utils.py                # 工具函数
├── requirements.txt        # 后端依赖
├── YOLO3D/                 # 3D 物体检测模块
│   ├── inference.py        # 推理代码
│   ├── requirements.txt    # YOLO3D 依赖
│   └── weights/            # 预训练模型
├── frontend/               # 前端界面
│   ├── app.py              # Streamlit 应用
│   └── requirements.txt    # 前端依赖
└── assets/                 # 资源文件
```

## 高级配置

### 自定义模型参数

可以在 `main.py` 中修改以下参数：
- `OBS_LEN`: 观察序列长度
- `FUT_LEN`: 预测序列长度
- `TTL_LEN`: 总序列长度

### 自定义 YOLO3D 配置

YOLO3D 的配置可以在 `YOLO3D/inference.py` 中修改。

