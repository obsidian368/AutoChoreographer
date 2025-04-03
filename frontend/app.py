import streamlit as st
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import time
import subprocess
import glob
from datetime import datetime

# Add the parent directory to the path to import from main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 解决Windows中文编码问题
if sys.platform == 'win32':
    # 确保文件系统操作使用UTF-8编码
    import locale
    # 设置Python运行时的默认编码为UTF-8
    if locale.getpreferredencoding() != 'utf-8':
        import subprocess
        try:
            # 设置控制台代码页
            subprocess.run(['chcp', '65001'], shell=True, check=False)
            print("已设置控制台为UTF-8编码")
        except:
            print("警告: 无法设置控制台编码，可能导致中文路径问题")
    # 设置环境变量强制Python使用UTF-8
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Set page configuration
st.set_page_config(
    page_title="AutoChoreographer自动驾驶框架",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 检查是否有正在运行的分析进程
if 'analysis_process' in st.session_state and st.session_state.get('is_running', False):
    # 获取进程对象
    process = st.session_state['analysis_process']
    
    # 非阻塞式检查进程状态
    if process.poll() is not None:
        # 进程已结束
        returncode = process.returncode
        stdout = process.stdout.read() if process.stdout else ""
        stderr = process.stderr.read() if process.stderr else ""
        
        # 重置状态
        st.session_state['is_running'] = False
        
        # 显示结果
        if returncode != 0:
            st.error(f"处理过程中发生错误:\n{stderr}")
        else:
            # 美化成功消息
            st.markdown(f"""
            <div style='background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;'>
                <h3 style='color: #2E7D32; margin-top: 0;'>✅ 分析完成!</h3>
                <p style='margin-bottom: 0;'>结果已保存到</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 更新结果文件夹列表
            result_folders = glob.glob("Qwen_results/*")
            result_folders.sort(reverse=True)
            if result_folders:
                st.session_state['selected_result'] = result_folders[0]

# 设置自定义主题和CSS
st.markdown("""
<style>
    /* 全局字体和颜色 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* 主要颜色变量 */
    :root {
        --primary-color: #2962FF;
        --primary-light: #5B8DEF;
        --primary-dark: #0039CB;
        --secondary-color: #26A69A;
        --text-color: #212121;
        --text-light: #666666;
        --bg-light: #F8F9FA;
        --card-shadow: rgba(0, 0, 0, 0.1);
    }
    
    /* 页脚固定在底部 */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
        padding: 20px 0;
        z-index: 999;
    }
    
    /* 为页脚腾出空间 */
    .main-content {
        margin-bottom: 120px; /* 根据页脚高度调整 */
    }
    
    /* 卡片样式 */
    .stCard {
        border-radius: 16px;
        box-shadow: 0 8px 16px var(--card-shadow);
        padding: 24px;
        margin: 20px 0;
        background-color: white;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .stCard:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    /* 标题样式 */
    h1 {
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid var(--primary-light);
    }
    
    h2 {
        font-weight: 600;
        color: var(--primary-color);
        margin-top: 30px;
    }
    
    h3 {
        font-weight: 500;
        color: var(--primary-dark);
    }
    
    /* 按钮样式 */
    .stButton>button {
        border-radius: 12px;
        font-weight: 500;
        padding: 12px 20px;
        transition: all 0.3s ease;
        border: none !important;
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark)) !important;
        color: white !important;
    }
    
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    /* 分割线样式 */
    hr {
        margin: 24px 0;
        border-top: 1px solid #E3F2FD;
    }
    
    /* 输入框样式 */
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 1px solid #E0E0E0;
        padding: 10px 15px;
        transition: all 0.2s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(41, 98, 255, 0.1);
    }
    
    /* 信息框样式 */
    .stAlert {
        border-radius: 12px;
        border-left-width: 10px;
    }
    
    /* 侧边栏样式 */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid rgba(0,0,0,0.05);
    }
    
    /* 分析结果卡片 */
    .analysis-card {
        background-color: var(--bg-light);
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 18px;
        border-left: 6px solid var(--primary-color);
        transition: transform 0.2s ease;
    }
    
    .analysis-card:hover {
        transform: translateX(5px);
    }
    
    /* 内容区块 */
    .content-block {
        padding: 16px;
        margin-bottom: 18px;
        border-radius: 12px;
        background-color: var(--bg-light);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* 导航按钮 */
    .nav-button {
        width: 100%;
        margin: 8px 0;
        border-radius: 12px !important;
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark)) !important;
    }
    
    /* 标签页 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #f0f2f6;
        padding: 8px;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 18px;
        font-weight: 500;
        background-color: #F0F2F6;
        transition: all 0.2s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark)) !important;
        color: white !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* 图片容器 */
    .image-container {
        overflow: hidden;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    
    .image-container:hover {
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        transform: scale(1.02);
    }
    
    .image-container img {
        border-radius: 12px;
        width: 100%;
        height: auto;
    }
    
    /* 进度条 */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary-color), var(--primary-dark));
        border-radius: 10px;
    }
    
    /* 下拉选择框 */
    .stSelectbox {
        border-radius: 12px;
    }
    
    .stSelectbox > div > div > div {
        background-color: white;
        border-radius: 12px;
    }
    
    /* 美化文本区域 */
    .text-display {
        white-space: pre-wrap; 
        word-wrap: break-word; 
        max-height: 200px; 
        overflow-y: auto; 
        padding: 15px; 
        background-color: #F8F9FA; 
        border-radius: 12px;
        border: 1px solid #E0E0E0;
        font-size: 0.95em;
    }
    
    /* 滚动条美化 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 0, 0, 0.3);
    }
    
    /* 加载动画 */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(41, 98, 255, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(41, 98, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(41, 98, 255, 0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* 全局淡入效果 */
    .stApp {
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* 元素悬浮效果 */
    .hover-effect {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .hover-effect:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    
    /* 表格美化 */
    table {
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    thead tr th {
        background-color: #E3F2FD !important;
        color: #0D47A1 !important;
        font-weight: 600 !important;
        padding: 12px 15px !important;
    }
    
    tbody tr:nth-child(even) {
        background-color: #F5F9FF !important;
    }
    
    tbody tr:hover {
        background-color: #E8F5E9 !important;
    }
    
    /* 表单元素美化 */
    input, select, textarea, .stSelectbox > div > div > div {
        border-radius: 10px !important;
        border: 1px solid #E0E0E0 !important;
        padding: 8px 12px !important;
        transition: all 0.3s ease !important;
    }
    
    input:focus, select:focus, textarea:focus, .stSelectbox > div > div > div:focus {
        border-color: #2962FF !important;
        box-shadow: 0 0 0 2px rgba(41, 98, 255, 0.2) !important;
        outline: none !important;
    }
    
    /* 美化工具提示 */
    .stTooltipIcon {
        color: #2962FF !important;
    }
    
    /* 美化选择框 */
    .stCheckbox > div > label {
        display: flex !important;
        align-items: center !important;
    }
    
    .stCheckbox > div > label > div {
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }
    
    .stCheckbox > div > label > div[data-testid="stCheckbox"] {
        border-color: #2962FF !important;
    }
    
    /* 美化通知 */
    div[data-testid="stNotificationContent"] {
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# 设置页面标题和徽标
col1, col2 = st.columns([1, 4])
with col1:
    # 使用更现代的SVG车辆图标
    st.markdown("""
    <div style="display: flex; justify-content: center; margin-bottom: 20px;">
        <svg width="120px" height="120px" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="carGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" style="stop-color:#2962FF" />
                    <stop offset="100%" style="stop-color:#0039CB" />
                </linearGradient>
            </defs>
            <g stroke="none" stroke-width="1" fill="none" fill-rule="evenodd">
                <path d="M19,14 L19,16 C19,16.5522847 18.5522847,17 18,17 L16,17 C15.4477153,17 15,16.5522847 15,16 L15,14 L9,14 L9,16 C9,16.5522847 8.55228475,17 8,17 L6,17 C5.44771525,17 5,16.5522847 5,16 L5,14 L5,13 L5,10.5 C5,9.67157288 5.67157288,9 6.5,9 L17.5,9 C18.3284271,9 19,9.67157288 19,10.5 L19,13 L19,14 Z" fill="url(#carGradient)"></path>
                <path d="M6.5,9 L17.5,9 C18.3284271,9 19,9.67157288 19,10.5 L19,11 L5,11 L5,10.5 C5,9.67157288 5.67157288,9 6.5,9 Z" fill="#0039CB"></path>
                <rect stroke="url(#carGradient)" stroke-width="2" x="3" y="11" width="18" height="5" rx="1"></rect>
                <rect fill="url(#carGradient)" x="7" y="14" width="2" height="3" rx="1"></rect>
                <rect fill="url(#carGradient)" x="15" y="14" width="2" height="3" rx="1"></rect>
                <path d="M5,11 L5,7 C5,5.34314575 6.34314575,4 8,4 L16,4 C17.6568542,4 19,5.34314575 19,7 L19,11" stroke="url(#carGradient)" stroke-width="2"></path>
                <rect fill="#2962FF" x="5" y="4" width="14" height="2" rx="1"></rect>
            </g>
        </svg>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style="animation: fadeIn 1s ease-out; padding: 20px;">
        <h1 style="color: #1E88E5; font-size: 2.2em; margin-bottom: 10px;">AutoChoreographer</h1>
        <p style="font-size: 1.2em; color: #666;">基于千问大模型和计算机视觉的端到端自动驾驶决策系统</p>
    </div>
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
    """, unsafe_allow_html=True)

# 创建信息卡片
st.markdown("""
<div class="stCard" style="animation: slideIn 0.8s ease-out;">
    <h3>🚀 欢迎使用AutoChoreographer自动驾驶框架</h3>
    <p>本应用通过处理nuScenes数据集图像，执行YOLO3D目标检测，并利用千问模型生成轨迹。</p>
    <p>请在侧边栏配置参数，然后点击"运行分析"按钮开始处理。</p>
    <div style="margin-top: 15px; padding: 10px; background-color: #E3F2FD; border-radius: 8px; border-left: 4px solid #2962FF;">
        <p style="margin: 0; font-size: 0.9em;"><strong>💡 提示:</strong> 选择更大的模型会提高分析质量，但可能需要更长处理时间。</p>
    </div>
</div>
<style>
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid rgba(0,0,0,0.1);">
        <h2 style="color: #2962FF; font-size: 1.5em;">
            <svg width="24" height="24" viewBox="0 0 24 24" style="vertical-align: middle; margin-right: 10px;">
                <path fill="#2962FF" d="M19.14,12.94c0.04-0.3,0.06-0.61,0.06-0.94c0-0.32-0.02-0.64-0.07-0.94l2.03-1.58c0.18-0.14,0.23-0.41,0.12-0.61 l-1.92-3.32c-0.12-0.22-0.37-0.29-0.59-0.22l-2.39,0.96c-0.5-0.38-1.03-0.7-1.62-0.94L14.4,2.81c-0.04-0.24-0.24-0.41-0.48-0.41 h-3.84c-0.24,0-0.43,0.17-0.47,0.41L9.25,5.35C8.66,5.59,8.12,5.92,7.63,6.29L5.24,5.33c-0.22-0.08-0.47,0-0.59,0.22L2.74,8.87 C2.62,9.08,2.66,9.34,2.86,9.48l2.03,1.58C4.84,11.36,4.8,11.69,4.8,12s0.02,0.64,0.07,0.94l-2.03,1.58 c-0.18,0.14-0.23,0.41-0.12,0.61l1.92,3.32c0.12,0.22,0.37,0.29,0.59,0.22l2.39-0.96c0.5,0.38,1.03,0.7,1.62,0.94l0.36,2.54 c0.05,0.24,0.24,0.41,0.48,0.41h3.84c0.24,0,0.44-0.17,0.47-0.41l0.36-2.54c0.59-0.24,1.13-0.56,1.62-0.94l2.39,0.96 c0.22,0.08,0.47,0,0.59-0.22l1.92-3.32c0.12-0.22,0.07-0.47-0.12-0.61L19.14,12.94z M12,15.6c-1.98,0-3.6-1.62-3.6-3.6 s1.62-3.6,3.6-3.6s3.6,1.62,3.6,3.6S13.98,15.6,12,15.6z"></path>
            </svg>
            系统配置
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div style="background: linear-gradient(135deg, #E3F2FD, #BBDEFB); padding: 15px; border-radius: 12px; margin-bottom: 20px;">
            <h3 style="font-size: 1.2em; color: #0D47A1; margin-top: 0;">
                <svg width="20" height="20" viewBox="0 0 24 24" style="vertical-align: middle; margin-right: 8px;">
                    <path fill="#0D47A1" d="M20,6h-8l-2-2H4C2.9,4,2,4.9,2,6v12c0,1.1,0.9,2,2,2h16c1.1,0,2-0.9,2-2V8C22,6.9,21.1,6,20,6z M20,18H4V6h5.17l2,2H20V18z"/>
                </svg>
                数据集设置
            </h3>
        """, unsafe_allow_html=True)
        
        # Dataset path input with icon
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <svg width="18" height="18" viewBox="0 0 24 24" style="margin-right: 10px;">
                <path fill="#2962FF" d="M20,6h-8l-2-2H4C2.9,4,2,4.9,2,6v12c0,1.1,0.9,2,2,2h16c1.1,0,2-0.9,2-2V8C22,6.9,21.1,6,20,6z M20,18H4V6h5.17l2,2H20V18z"/>
            </svg>
            <span style="font-weight: 500;">数据集路径</span>
        </div>
        """, unsafe_allow_html=True)
        
        dataset_path = st.text_input("", value="/root/workspace/", 
                                     label_visibility="collapsed",
                                     help="指定nuScenes数据集的本地路径")
    
        # Dataset version selection
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 8px; margin-top: 15px;">
            <svg width="18" height="18" viewBox="0 0 24 24" style="margin-right: 10px;">
                <path fill="#2962FF" d="M19,3H5C3.9,3,3,3.9,3,5v14c0,1.1,0.9,2,2,2h14c1.1,0,2-0.9,2-2V5C21,3.9,20.1,3,19,3z M19,5v3H5V5H19z M5,19V10h14v9H5z"/>
                <rect fill="#2962FF" x="7" y="12" width="2" height="5"/>
                <rect fill="#2962FF" x="11" y="12" width="2" height="5"/>
                <rect fill="#2962FF" x="15" y="12" width="2" height="5"/>
            </svg>
            <span style="font-weight: 500;">数据集版本</span>
        </div>
        """, unsafe_allow_html=True)
        
        dataset_version = st.selectbox("", ["v1.0-mini", "v1.0-trainval", "v1.0-test"], 
                                      label_visibility="collapsed",
                                      help="选择要使用的nuScenes数据集版本")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 15px 0; opacity: 0.3;'>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div style="background: linear-gradient(135deg, #E8F5E9, #C8E6C9); padding: 15px; border-radius: 12px; margin-bottom: 20px;">
            <h3 style="font-size: 1.2em; color: #2E7D32; margin-top: 0;">
                <svg width="20" height="20" viewBox="0 0 24 24" style="vertical-align: middle; margin-right: 8px;">
                    <path fill="#2E7D32" d="M9,4c0-1.1,0.9-2,2-2s2,0.9,2,2v2H9V4z M16,6h3c0.6,0,1,0.4,1,1v12c0,0.6-0.4,1-1,1H5c-0.6,0-1-0.4-1-1V7 c0-0.6,0.4-1,1-1h3v2H6v10h12V8h-2V6z M12,9c1.1,0,2,0.9,2,2s-0.9,2-2,2s-2-0.9-2-2S10.9,9,12,9z"/>
                </svg>
                模型设置
            </h3>
        """, unsafe_allow_html=True)
        
        # Model selection with icon
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <svg width="18" height="18" viewBox="0 0 24 24" style="margin-right: 10px;">
                <path fill="#2962FF" d="M12,3L1,9l4,2.18v6L12,21l7-3.82v-6l2-1.09V17h2V9L12,3z M18.82,9L12,12.72L5.18,9L12,5.28L18.82,9z M17,15.99l-5,2.73l-5-2.73v-3.72L12,15l5-2.73V15.99z"/>
            </svg>
            <span style="font-weight: 500;">模型选择</span>
        </div>
        """, unsafe_allow_html=True)
        
        model_options = {
            "qwen2.5-vl-3b-instruct": "千问2.5-VL-3B (推理速度快)",
            "qwen2.5-vl-7b-instruct": "千问2.5-VL-7B (平衡性能与效率)",
            "qwen2.5-vl-72b-instruct": "千问2.5-VL-72B (性能最强)",
        }
        model_name = st.selectbox(
            "",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            label_visibility="collapsed",
            help="选择要使用的模型版本，模型越大性能越好但速度越慢"
        )
        
        # Get API key input with icon
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 8px; margin-top: 15px;">
            <svg width="18" height="18" viewBox="0 0 24 24" style="margin-right: 10px;">
                <path fill="#2962FF" d="M12.65,10C11.83,7.67,9.61,6,7,6c-3.31,0-6,2.69-6,6s2.69,6,6,6c2.61,0,4.83-1.67,5.65-4H17v4h4v-4h2v-4H12.65z M7,14c-1.1,0-2-0.9-2-2s0.9-2,2-2s2,0.9,2,2S8.1,14,7,14z"/>
            </svg>
            <span style="font-weight: 500;">千问API密钥</span>
        </div>
        """, unsafe_allow_html=True)
        
        api_key = st.text_input(
            "", 
            type="password", 
            key="api_key",
            label_visibility="collapsed",
            help="从阿里云获取您的千问API密钥，必须是有效密钥才能运行分析"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 15px 0; opacity: 0.3;'>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div style="background: linear-gradient(135deg, #E1F5FE, #B3E5FC); padding: 15px; border-radius: 12px; margin-bottom: 20px;">
            <h3 style="font-size: 1.2em; color: #0277BD; margin-top: 0;">
                <svg width="20" height="20" viewBox="0 0 24 24" style="vertical-align: middle; margin-right: 8px;">
                    <path fill="#0277BD" d="M15,5l-1.41,1.41L18.17,11H2V13h16.17l-4.59,4.59L15,19l7-7L15,5z"/>
                </svg>
                运行参数
            </h3>
        """, unsafe_allow_html=True)
        
        # 选择特定场景
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <svg width="18" height="18" viewBox="0 0 24 24" style="margin-right: 10px;">
                <path fill="#2962FF" d="M15,3l2.3,2.3l-2.89,2.87l1.42,1.42L18.7,6.7L21,9V3H15z M3,9l2.3-2.3l2.87,2.89l1.42-1.42L6.7,5.3L9,3H3V9z M9,21 l-2.3-2.3l2.89-2.87l-1.42-1.42L5.3,17.3L3,15v6H9z M21,15l-2.3,2.3l-2.87-2.89l-1.42,1.42l2.89,2.87L15,21h6V15z"/>
            </svg>
            <span style="font-weight: 500;">指定场景(例如: scene-0757"，留空处理所有场景)</span>
        </div>
        """, unsafe_allow_html=True)
        
        specific_scene = st.text_input(
            "", 
            value="", 
            label_visibility="collapsed",
            help="留空处理所有场景，例如: scene-0757"
        )
        
        # 限制处理帧数
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 8px; margin-top: 15px;">
            <svg width="18" height="18" viewBox="0 0 24 24" style="margin-right: 10px;">
                <path fill="#2962FF" d="M7,15h3c0.55,0,1-0.45,1-1v-1H9v1H7v-3h1v1h2v-1c0-0.55-0.45-1-1-1H7c-0.55,0-1,0.45-1,1v3C6,14.55,6.45,15,7,15z M14,15h2c0.55,0,1-0.45,1-1v-3c0-0.55-0.45-1-1-1h-2c-0.55,0-1,0.45-1,1v3C13,14.55,13.45,15,14,15z M14,11h2v3h-2V11z M19,3H5 C3.9,3,3,3.9,3,5v14c0,1.1,0.9,2,2,2h14c1.1,0,2-0.9,2-2V5C21,3.9,20.1,3,19,3z M19,19H5V5h14V19z"/>
            </svg>
            <span style="font-weight: 500;">每个场景最大处理帧数</span>
        </div>
        """, unsafe_allow_html=True)
        
        max_frames = st.number_input(
            "", 
            min_value=1, 
            max_value=30, 
            value=15, 
            label_visibility="collapsed",
            help="控制每个场景处理的最大帧数，较小的值可以加快处理速度"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 20px 0 25px 0; opacity: 0.3;'>", unsafe_allow_html=True)
    
    # Run button with animation
    st.markdown("""
    <style>
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(41, 98, 255, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(41, 98, 255, 0); }
            100% { box-shadow: 0 0 0 0 rgba(41, 98, 255, 0); }
        }
        
        .run-button {
            background: linear-gradient(135deg, #2962FF, #0039CB);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 15px 0;
            font-size: 16px;
            font-weight: 600;
            width: 100%;
            text-align: center;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(41, 98, 255, 0.3);
            transition: all 0.3s ease;
            animation: pulse 2s infinite;
        }
        
        .run-button:hover {
            box-shadow: 0 6px 12px rgba(41, 98, 255, 0.4);
            transform: translateY(-2px);
        }
        
        .stButton button {
            background: linear-gradient(135deg, #2962FF, #0039CB);
            color: white;
            font-weight: 600;
            border: none;
            padding: 12px 15px;
            animation: pulse 2s infinite;
        }
    </style>
    """, unsafe_allow_html=True)
    
    run_button = st.button(
        "🚀 运行分析", 
        key="run_analysis", 
        use_container_width=True
    )
    
    st.markdown("""
    <div style="margin-top: 15px; padding: 10px; background-color: #EDE7F6; border-radius: 8px; border-left: 4px solid #673AB7;">
        <p style="margin: 0; font-size: 0.9em;"><strong>⏱️ 处理时间:</strong> 根据选择的数据集大小和模型，分析可能需要几分钟到几小时不等。</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Initialize variables
    selected_result = None
    selected_scene = None
    selected_frame = None
    load_button = False
    
    # Results folder selection (only show after first run)
    result_folders = glob.glob("Qwen_results/*")
    if result_folders:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #F3E5F5, #E1BEE7); padding: 15px; border-radius: 12px; margin: 20px 0;">
            <h3 style="font-size: 1.2em; color: #6A1B9A; margin-top: 0;">
                <svg width="20" height="20" viewBox="0 0 24 24" style="vertical-align: middle; margin-right: 8px;">
                    <path fill="#6A1B9A" d="M20,6h-8l-2-2H4C2.9,4,2,4.9,2,6v12c0,1.1,0.9,2,2,2h16c1.1,0,2-0.9,2-2V8C22,6.9,21.1,6,20,6z M20,18H4V8h16V18z M13,17h2v-3h3v-2h-3V9h-2v3H8v2h5V17z"/>
                </svg>
                查看历史结果
            </h3>
        """, unsafe_allow_html=True)
        
        result_folders.sort(reverse=True)  # Show newest first
        
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 8px; margin-top: 10px;">
            <svg width="18" height="18" viewBox="0 0 24 24" style="margin-right: 10px;">
                <path fill="#6A1B9A" d="M20,6h-8l-2-2H4C2.9,4,2,4.9,2,6v12c0,1.1,0.9,2,2,2h16c1.1,0,2-0.9,2-2V8C22,6.9,21.1,6,20,6z M20,18H4V8h16V18z"/>
            </svg>
            <span style="font-weight: 500;">选择结果文件夹</span>
        </div>
        """, unsafe_allow_html=True)
        
        selected_result = st.selectbox("", result_folders, label_visibility="collapsed", format_func=lambda x: x.split('/')[-1] + " (" + datetime.fromtimestamp(os.path.getctime(x)).strftime('%Y-%m-%d %H:%M') + ")")
        
        # Get available scenes in the selected result folder
        scene_files = glob.glob(f"{selected_result}/*_logs.txt")
        scenes = sorted(list(set([os.path.basename(f).split('_')[0] for f in scene_files])))
        
        if scenes:
            st.markdown("""
            <div style="display: flex; align-items: center; margin-bottom: 8px; margin-top: 15px;">
                <svg width="18" height="18" viewBox="0 0 24 24" style="margin-right: 10px;">
                    <path fill="#6A1B9A" d="M3,5v14h18V5H3z M7,7v2H5V7H7z M5,13v-2h2v2H5z M5,15h2v2H5z M19,17H9v-2h10V17z M19,13H9v-2h10V13z M19,9H9V7h10V9z"/>
                </svg>
                <span style="font-weight: 500;">选择场景</span>
            </div>
            """, unsafe_allow_html=True)
            
            selected_scene = st.selectbox("", scenes, label_visibility="collapsed")
            
            # Get all frame indices for the selected scene
            frame_files = glob.glob(f"{selected_result}/{selected_scene}_*_logs.txt")
            frames = sorted([int(os.path.basename(f).split('_')[1]) for f in frame_files])
            
            if frames:
                st.markdown("""
                <div style="display: flex; align-items: center; margin-bottom: 8px; margin-top: 15px;">
                    <svg width="18" height="18" viewBox="0 0 24 24" style="margin-right: 10px;">
                        <path fill="#6A1B9A" d="M9,3L7.17,5H4C2.9,5,2,5.9,2,7v12c0,1.1,0.9,2,2,2h16c1.1,0,2-0.9,2-2V7c0-1.1-0.9-2-2-2h-3.17L15,3H9z M12,18 c-2.76,0-5-2.24-5-5s2.24-5,5-5s5,2.24,5,5S14.76,18,12,18z"/>
                        <path fill="#6A1B9A" d="M12,9c-2.21,0-4,1.79-4,4s1.79,4,4,4s4-1.79,4-4S14.21,9,12,9z"/>
                    </svg>
                    <span style="font-weight: 500;">选择帧</span>
                </div>
                """, unsafe_allow_html=True)
                
                selected_frame = st.selectbox("", frames, label_visibility="collapsed")
                
                st.markdown("""
                <style>
                    .load-button {
                        background: linear-gradient(135deg, #6A1B9A, #8E24AA);
                        color: white;
                        border: none;
                        border-radius: 12px;
                        padding: 10px 0;
                        font-weight: 600;
                        width: 100%;
                        text-align: center;
                        margin-top: 10px;
                        transition: all 0.3s ease;
                    }
                    
                    .load-button:hover {
                        box-shadow: 0 4px 8px rgba(106, 27, 154, 0.3);
                        transform: translateY(-2px);
                    }
                </style>
                """, unsafe_allow_html=True)
                
                load_button = st.button("🔍 加载可视化", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# 修改安全可视化函数以添加更好的错误处理和动画效果
def safe_visualize_frame(result_folder, scene, frame):
    """
    安全的帧可视化包装函数，处理可能出现的各种异常
    """
    try:
        # 添加加载动画
        with st.spinner('正在加载可视化数据...'):
            # 检查参数合法性
            if not result_folder or not scene or frame is None:
                st.warning("⚠️ 缺少必要的参数来加载可视化")
                return False
            
            # 检查文件存在性
            front_cam_path = f"{result_folder}/{scene}_{frame}_front_cam.jpg"
            logs_path = f"{result_folder}/{scene}_{frame}_logs.txt"
            
            if not os.path.exists(front_cam_path) or not os.path.exists(logs_path):
                st.warning(f"⚠️ 帧数据缺失: 场景 {scene}, 帧 {frame}")
                return False
            
            # 实际调用可视化功能
            visualize_frame(result_folder, scene, frame)
            return True
    except Exception as e:
        # 捕获并显示可能的异常
        st.error(f"❌ 可视化处理出错: {str(e)}")
        # 出错时重置播放状态
        st.session_state.is_playing = False
        return False

# Function to load and display a specific frame
def visualize_frame(result_folder, scene, frame):
    # Create columns for the grid layout
    # col2 = st.columns([1])  # 修改为两列布局
    
    # Path to files
    front_cam_path = f"{result_folder}/{scene}_{frame}_front_cam.jpg"
    traj_path = f"{result_folder}/{scene}_{frame}_traj.jpg"
    logs_path = f"{result_folder}/{scene}_{frame}_logs.txt"
    
    # Check if files exist
    if not os.path.exists(front_cam_path) or not os.path.exists(traj_path) or not os.path.exists(logs_path):
        st.error(f"⚠️ 未找到场景 {scene}，帧 {frame} 的文件!")
        return
    
    # Load the log information
    with open(logs_path, 'r', encoding='utf-8') as f:
        logs = f.read()
    
    scene_description = ""
    object_description = ""
    intent_description = ""
    
    # 修改文本提取逻辑，确保获取完整的描述
    current_desc = ""
    for line in logs.split('\n'):
        if line.startswith("Scene Description:"):
            if current_desc:
                intent_description = current_desc.strip()
            current_desc = line.replace("Scene Description:", "").strip()
        elif line.startswith("Object Description:"):
            scene_description = current_desc
            current_desc = line.replace("Object Description:", "").strip()
        elif line.startswith("Intent Description:"):
            object_description = current_desc
            current_desc = line.replace("Intent Description:", "").strip()
        else:
            current_desc += "\n" + line.strip()
    
    # 处理最后一个描述
    if current_desc:
        intent_description = current_desc.strip()
    
    # Get all camera images
    camera_views = {
        "front": f"{result_folder}/{scene}_{frame}_front.jpg",
        "front_left": f"{result_folder}/{scene}_{frame}_front_left.jpg",
        "front_right": f"{result_folder}/{scene}_{frame}_front_right.jpg",
        "back": f"{result_folder}/{scene}_{frame}_back.jpg",
        "back_left": f"{result_folder}/{scene}_{frame}_back_left.jpg",
        "back_right": f"{result_folder}/{scene}_{frame}_back_right.jpg",
        "front_cam": front_cam_path,  # The processed image with detections
        "trajectory": traj_path
    }
    
    # Create a grid for all camera views
    # with col2:
    st.markdown("""
    <h2 style='color: #2962FF; font-size: 1.5em; display: flex; align-items: center;'>
        <svg width="24" height="24" viewBox="0 0 24 24" style="margin-right: 10px;">
            <path fill="#2962FF" d="M17,10.5V7c0-0.55-0.45-1-1-1H4C3.45,6,3,6.45,3,7v10c0,0.55,0.45,1,1,1h12c0.55,0,1-0.45,1-1v-3.5l4,4v-11L17,10.5z"/>
        </svg>
        摄像头视图与轨迹
    </h2>
    """, unsafe_allow_html=True)
    
    # 使用标签页组织不同的视图
    tabs = st.tabs(["🚗 全部视图", "📊 轨迹分析", "🧠 AI分析"])
    
    with tabs[0]:
        # Create 3x3 grid for views
        grid1, grid2, grid3 = st.columns(3)
        grid4, grid5, grid6 = st.columns(3)
        grid7, grid8, grid9 = st.columns(3)
        
        # 封装图像显示函数，增加一致的样式
        def display_image(col, image_path, caption, fallback_text="图像不可用"):
            if os.path.exists(image_path):
                with col:
                    st.markdown(f"<div class='image-container'>", unsafe_allow_html=True)
                    img = Image.open(image_path)
                    st.image(img, caption=caption, use_column_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                with col:
                    st.info(fallback_text)
        
        # 按布局显示所有摄像头视角，添加动画效果
        st.markdown("""
        <style>
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            .fadeIn { animation: fadeIn 0.5s ease-out; }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("<div class='fadeIn'>", unsafe_allow_html=True)
        display_image(grid1, camera_views["front_left"], "📷 前左视角")
        display_image(grid2, camera_views["front"], "📷 前视角")
        display_image(grid3, camera_views["front_right"], "📷 前右视角")
        
        # 左侧空格
        with grid4:
            st.empty()
        
        # 中间显示带轨迹的前视图
        display_image(grid5, camera_views["front_cam"], "📷 带轨迹前视图")
        
        # 右侧空格
        with grid6:
            st.empty()
        
        display_image(grid7, camera_views["back_left"], "📷 后左视角")
        display_image(grid8, camera_views["back"], "📷 后视角")
        display_image(grid9, camera_views["back_right"], "📷 后右视角")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[1]:
        # 轨迹分析标签页内容
        st.markdown("""
        <h3 style='color: #0D47A1; display: flex; align-items: center;'>
            <svg width="20" height="20" viewBox="0 0 24 24" style="margin-right: 8px;">
                <path fill="#0D47A1" d="M3,17.25V21h3.75L17.81,9.94l-3.75-3.75L3,17.25z M20.71,7.04c0.39-0.39,0.39-1.02,0-1.41l-2.34-2.34 c-0.39-0.39-1.02-0.39-1.41,0l-1.83,1.83l3.75,3.75L20.71,7.04z"/>
            </svg>
            预测轨迹
        </h3>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns([2, 1])
        
        with col_a:
            if os.path.exists(camera_views["trajectory"]):
                st.markdown("<div class='image-container'>", unsafe_allow_html=True)
                traj_img = Image.open(camera_views["trajectory"])
                st.image(traj_img, caption="车辆轨迹分析", use_column_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.warning("轨迹图像不可用")
        
        with col_b:
            st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
            
            # 提取ADE信息
            ade_value = "未知"
            for line in logs.split('\n'):
                if line.startswith("Average Displacement Error:"):
                    ade_value = line.replace("Average Displacement Error:", "").strip()
                    break
            
            st.markdown(f"""
            <h4 style='color: #0D47A1; display: flex; align-items: center;'>
                <svg width="16" height="16" viewBox="0 0 24 24" style="margin-right: 8px;">
                    <path fill="#0D47A1" d="M19,3H5C3.9,3,3,3.9,3,5v14c0,1.1,0.9,2,2,2h14c1.1,0,2-0.9,2-2V5C21,3.9,20.1,3,19,3z M9,17H7v-7h2V17z M13,17h-2V7h2V17z M17,17h-2v-4h2V17z"/>
                </svg>
                轨迹评估指标
            </h4>
            <p><strong>场景:</strong> {scene}</p>
            <p><strong>帧:</strong> {frame}</p>
            <p><strong>平均位移误差(ADE):</strong> {ade_value}</p>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        # AI分析标签页内容
        st.markdown("""
        <h3 style='color: #0D47A1; display: flex; align-items: center;'>
            <svg width="20" height="20" viewBox="0 0 24 24" style="margin-right: 8px;">
                <path fill="#0D47A1" d="M21,11c0,0.33-0.04,0.66-0.07,0.99c-0.24-0.11-0.49-0.19-0.76-0.19c-1.1,0-2,0.9-2,2 c0,0.08,0.01,0.15,0.02,0.22C17.14,14.62,16.26,15,15.3,15c-0.89,0-1.73-0.31-2.39-0.84l0.09-0.92c0.37,0.29,0.83,0.45,1.31,0.45 c1.1,0,2-0.9,2-2c0-1.1-0.9-2-2-2c-0.58,0-1.12,0.25-1.49,0.68c-0.26-0.65-0.69-1.24-1.27-1.65c0.37-0.36,0.86-0.59,1.41-0.59 c0.53,0,1.04,0.21,1.41,0.59c0.36-0.36,0.85-0.59,1.41-0.59c0.95,0,1.75,0.67,1.95,1.56c0.48-0.28,1.04-0.42,1.66-0.42 c1.75,0,3.2,1.38,3.27,3.11C20.93,12.18,21,11.61,21,11 M12,12c0-2.76-2.24-5-5-5c-1.77,0-3.31,0.92-4.21,2.3 C2.29,9.65,2,10.1,1.75,10.59C1.57,10.93,1.41,11.29,1.3,11.68c-0.11,0.4-0.18,0.82-0.18,1.26c0,0.41,0.06,0.8,0.15,1.18 c0.06,0.23,0.14,0.45,0.23,0.67c0.55,1.35,1.63,2.4,2.96,2.87c0.5,0.17,1.03,0.27,1.59,0.27c0.51,0,1-0.08,1.47-0.22 c0.35-0.11,0.69-0.25,1.01-0.42c0.88-0.5,1.58-1.23,2.06-2.08c0.27-0.48,0.46-1.01,0.57-1.56C11.94,13.15,12,12.58,12,12 M17.12,9.12L17.12,9.12l0,0c-0.06-1.09-0.73-2.05-1.75-2.51c-0.46-0.21-0.94-0.28-1.41-0.22c-0.01,0-0.03,0-0.04,0 c-0.22,0.03-0.44,0.1-0.65,0.19c-0.66-0.34-1.41-0.54-2.17-0.58c-0.03,0-0.06,0-0.09,0l-0.01-0.01 C10.54,5.9,10.16,5.89,9.76,5.9C8.24,5.96,6.96,6.69,6.16,7.83c-0.04,0.05-0.08,0.11-0.12,0.17c-0.04,0.06-0.08,0.11-0.11,0.17 c-0.06,0.09-0.11,0.18-0.16,0.28c-0.02,0.04-0.05,0.08-0.07,0.12C5.51,8.94,5.35,9.32,5.23,9.72c-0.05,0.18-0.09,0.37-0.11,0.56 c-0.01,0.06-0.02,0.13-0.02,0.19C5.08,10.66,5.07,10.85,5.08,11c0,0.09,0.01,0.17,0.01,0.25c0.01,0.12,0.03,0.24,0.05,0.35 c0.03,0.21,0.08,0.42,0.15,0.62c0.06,0.17,0.14,0.33,0.22,0.48c0.09,0.16,0.21,0.32,0.33,0.46c0.12,0.14,0.24,0.26,0.38,0.37 c0.12,0.09,0.25,0.17,0.39,0.24c0.07,0.03,0.13,0.06,0.19,0.08c0.18,0.07,0.37,0.11,0.57,0.12c0.09,0,0.18,0,0.26-0.01 c0.21-0.03,0.42-0.08,0.62-0.16c0.26-0.11,0.51-0.26,0.73-0.45c0.15-0.13,0.29-0.27,0.42-0.43c0.03-0.05,0.06-0.09,0.09-0.14 l0.01,0c0.12-0.19,0.22-0.4,0.29-0.61c0.24,0.12,0.51,0.19,0.79,0.19c0.05,0,0.1,0,0.15-0.01c0.36-0.03,0.69-0.17,0.96-0.37 c0.27,0.2,0.59,0.34,0.96,0.37c0.05,0,0.1,0.01,0.15,0.01c0.28,0,0.55-0.07,0.79-0.19c0.07,0.21,0.17,0.41,0.29,0.61l0.01,0 c0.03,0.05,0.06,0.09,0.09,0.14c0.13,0.16,0.27,0.3,0.42,0.43c0.22,0.19,0.47,0.34,0.73,0.45c0.2,0.08,0.41,0.13,0.62,0.16 c0.09,0.01,0.18,0.01,0.26,0.01c0.2-0.01,0.39-0.05,0.57-0.12c0.07-0.02,0.13-0.05,0.19-0.08c0.14-0.07,0.27-0.15,0.39-0.24 c0.14-0.11,0.26-0.23,0.38-0.37c0.12-0.14,0.24-0.3,0.33-0.46c0.08-0.15,0.16-0.31,0.22-0.48c0.07-0.2,0.12-0.41,0.15-0.62 c0.02-0.11,0.04-0.23,0.05-0.35C17.99,11.25,18,11.16,18,11C18.01,10.2,17.67,9.38,17.12,9.12z"/>
            </svg>
            AI场景理解
        </h3>
        """, unsafe_allow_html=True)
        
        # 使用卡片样式展示分析结果
        st.markdown("""
        <style>
            @keyframes slideInRight {
                from { opacity: 0; transform: translateX(30px); }
                to { opacity: 1; transform: translateX(0); }
            }
            
            .analysis-card {
                animation: slideInRight 0.5s ease-out;
                animation-fill-mode: both;
            }
            
            .analysis-card:nth-child(1) { animation-delay: 0.1s; }
            .analysis-card:nth-child(2) { animation-delay: 0.2s; }
            .analysis-card:nth-child(3) { animation-delay: 0.3s; }
            
            .text-display {
                transition: all 0.3s ease;
                border-left: 4px solid transparent;
            }
            
            .text-display:hover {
                border-left: 4px solid #2962FF;
                background-color: #F5F7FA;
            }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
            st.markdown("""
            <h4 style='color: #0D47A1; display: flex; align-items: center;'>
                <svg width="16" height="16" viewBox="0 0 24 24" style="margin-right: 8px;">
                    <path fill="#0D47A1" d="M19,3H5C3.9,3,3,3.9,3,5v14c0,1.1,0.9,2,2,2h14c1.1,0,2-0.9,2-2V5C21,3.9,20.1,3,19,3z M9,17H7v-7h2V17z M13,17h-2V7h2V17z M17,17h-2v-4h2V17z"/>
                </svg>
                场景描述
            </h4>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="text-display">
            {scene_description}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
            st.markdown("""
            <h4 style='color: #0D47A1; display: flex; align-items: center;'>
                <svg width="16" height="16" viewBox="0 0 24 24" style="margin-right: 8px;">
                    <path fill="#0D47A1" d="M12,4.5C7,4.5,2.73,7.61,1,12c1.73,4.39,6,7.5,11,7.5s9.27-3.11,11-7.5C21.27,7.61,17.53,4.5,12,4.5z M12,17 c-2.76,0-5-2.24-5-5s2.24-5,5-5s5,2.24,5,5S14.76,17,12,17z M12,9c-1.66,0-3,1.34-3,3s1.34,3,3,3s3-1.34,3-3S13.66,9,12,9z"/>
                </svg>
                物体识别
            </h4>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="text-display">
            {object_description}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
            st.markdown("""
            <h4 style='color: #0D47A1; display: flex; align-items: center;'>
                <svg width="16" height="16" viewBox="0 0 24 24" style="margin-right: 8px;">
                    <path fill="#0D47A1" d="M12,2C6.47,2,2,6.47,2,12c0,5.53,4.47,10,10,10s10-4.47,10-10C22,6.47,17.53,2,12,2z M12,20c-4.42,0-8-3.58-8-8 c0-4.42,3.58-8,8-8s8,3.58,8,8C20,16.42,16.42,20,12,20z"/>
                    <path fill="#0D47A1" d="M13,7h-2v6h6v-2h-4V7z"/>
                </svg>
                驾驶意图
            </h4>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="text-display">
            {intent_description}
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # 封装在卡片内
    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
    
    # Get all frame indices for the selected scene
    frame_files = glob.glob(f"{result_folder}/{scene}_*_logs.txt")
    frames = sorted([int(os.path.basename(f).split('_')[1]) for f in frame_files])
    
    current_index = frames.index(frame) if frame in frames else 0
    
    # 显示当前场景信息
    st.markdown(f"""
    <div style='text-align: center; margin-bottom: 15px;'>
        <h3 style='margin: 0; color: #0D47A1;'>场景: {scene}</h3>
        <p style='margin: 5px 0;'>当前帧: {frame} / {len(frames) - 1}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    if len(frames) > 1:
        progress = current_index / (len(frames) - 1) if len(frames) > 1 else 0
        st.progress(progress)
    else:
        st.progress(1.0)  # Show full progress if only one frame exists
    
    # Previous and Next buttons - 使用图标和更现代的样式
    col1a, col1b = st.columns(2)
    
    # Only show navigation buttons if there are multiple frames
    if len(frames) > 1:
        with col1a:
            if current_index > 0:
                if st.button("⬅️ 上一帧", use_container_width=True, key="prev_frame"):
                    try:
                        next_frame = frames[current_index - 1]
                        if 'last_nav_frame' not in st.session_state or st.session_state.last_nav_frame != next_frame:
                            st.session_state.last_nav_frame = next_frame
                            st.session_state.selected_frame = next_frame
                            st.session_state.is_playing = False  # 导航时停止播放
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"导航出错: {str(e)}")
            else:
                st.button("⬅️ 上一帧", use_container_width=True, key="prev_frame", disabled=True)
        
        with col1b:
            if current_index < len(frames) - 1:
                if st.button("➡️ 下一帧", use_container_width=True, key="next_frame"):
                    try:
                        next_frame = frames[current_index + 1]
                        if 'last_nav_frame' not in st.session_state or st.session_state.last_nav_frame != next_frame:
                            st.session_state.last_nav_frame = next_frame
                            st.session_state.selected_frame = next_frame
                            st.session_state.is_playing = False  # 导航时停止播放
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"导航出错: {str(e)}")
            else:
                st.button("➡️ 下一帧", use_container_width=True, key="next_frame", disabled=True)
    else:
        with col1a:
            st.button("⬅️ 上一帧", use_container_width=True, key="prev_frame", disabled=True)
        with col1b:
            st.button("➡️ 下一帧", use_container_width=True, key="next_frame", disabled=True)
    
    # Timeline slider
    st.markdown("<p style='margin: 15px 0 5px 0;'>时间轴滑块:</p>", unsafe_allow_html=True)
    
    # Check if frames list has enough items to create a slider
    if len(frames) > 1:
        selected_frame_index = st.select_slider(
            "",
            options=frames,
            value=frame,
            label_visibility="collapsed"
        )
        
        if selected_frame_index != frame:
            try:
                if 'last_slider_frame' not in st.session_state or st.session_state.last_slider_frame != selected_frame_index:
                    st.session_state.last_slider_frame = selected_frame_index
                    st.session_state.selected_frame = selected_frame_index
                    st.session_state.is_playing = False  # 滑动时停止播放
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"滑块导航出错: {str(e)}")
    else:
        st.info("无法创建时间轴滑块: 需要至少两帧数据")
    
    # 帧跳转输入框
    st.markdown("<p style='margin: 15px 0 5px 0;'>跳转到指定帧:</p>", unsafe_allow_html=True)
    jump_cols = st.columns([3, 1])
    
    # Ensure frames list has items to prevent min/max errors
    if len(frames) > 0:
        with jump_cols[0]:
            jump_frame = st.number_input("", min_value=min(frames), max_value=max(frames), value=frame, label_visibility="collapsed")
        with jump_cols[1]:
            if st.button("跳转", use_container_width=True, key="jump_button"):
                try:
                    if jump_frame in frames:
                        # 设置防止循环刷新的标记
                        if 'last_jump_frame' not in st.session_state or st.session_state.last_jump_frame != jump_frame:
                            st.session_state.last_jump_frame = jump_frame
                            st.session_state.selected_frame = jump_frame
                            st.session_state.is_playing = False  # 跳转时停止播放
                            st.experimental_rerun()
                    else:
                        st.error(f"帧 {jump_frame} 不存在")
                except Exception as e:
                    st.error(f"跳转出错: {str(e)}")
    else:
        with jump_cols[0]:
            st.info("无法跳转: 未找到帧数据")
        with jump_cols[1]:
            st.button("跳转", use_container_width=True, key="jump_button", disabled=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 播放控制面板
    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: #0D47A1; font-size: 1.2em;'>▶️ 播放控制</h3>", unsafe_allow_html=True)
    
    # 播放/暂停按钮和自动播放速度控制
    play_col1, play_col2 = st.columns(2)
    
    # Only enable play controls if there are multiple frames
    if len(frames) > 1:
        with play_col1:
            if st.session_state.is_playing:
                if st.button("⏸️ 暂停", use_container_width=True):
                    st.session_state.is_playing = False
                    st.experimental_rerun()
            else:
                if st.button("▶️ 播放", use_container_width=True):
                    st.session_state.is_playing = True
                    st.experimental_rerun()
        
        # with play_col2:
        #     # 重置按钮，回到第一帧
        #     if st.button("⏮️ 重置", use_container_width=True):
        #         if frames:  # 确保frames列表非空
        #             st.session_state.selected_frame = frames[0]
        #             st.session_state.is_playing = False
        #             st.experimental_rerun()
        #         else:
        #             st.error("无法重置：没有可用的帧数据")
        
        # 播放速度控制
        st.markdown("<p style='margin: 15px 0 5px 0;'>播放间隔 (秒):</p>", unsafe_allow_html=True)
        st.session_state.play_speed = st.slider(
            "",
            min_value=0.1,
            max_value=5.0,
            value=st.session_state.play_speed,
            step=0.1,
            label_visibility="collapsed"
        )
        
        # 自动播放功能
        if st.session_state.is_playing and current_index < len(frames) - 1:
            # 显示当前状态
            st.info(f"自动播放中... 当前帧: {frame}")
            
            # 添加进度条
            progress_bar = st.progress(current_index / (len(frames) - 1))
            
            try:
                # 等待设定的时间间隔
                time.sleep(st.session_state.play_speed)
                
                # 前进到下一帧
                next_frame = frames[current_index + 1]
                if 'last_auto_frame' not in st.session_state or st.session_state.last_auto_frame != next_frame:
                    st.session_state.last_auto_frame = next_frame
                    st.session_state.selected_frame = next_frame
                    st.experimental_rerun()
                else:
                    # 如果检测到可能的循环，停止播放
                    st.session_state.is_playing = False
                    st.warning("检测到播放异常，已自动停止")
            except Exception as e:
                st.error(f"自动播放出错: {str(e)}")
                st.session_state.is_playing = False
    else:
        # Disable play controls when not enough frames
        with play_col1:
            st.button("▶️ 播放", use_container_width=True, disabled=True)
        with play_col2:
            st.button("⏮️ 重置", use_container_width=True, disabled=True)
        st.info("播放控制不可用: 需要至少两帧数据")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # 底部添加场景信息框
    st.markdown("<hr style='margin: 30px 0 20px 0;'>", unsafe_allow_html=True)
    
    with st.expander("📊 场景统计信息", expanded=False):
        info_cols = st.columns(4)
        
        with info_cols[0]:
            st.metric("总帧数", len(frames))
        
        with info_cols[1]:
            st.metric("当前帧", f"{current_index}/{len(frames) - 1}")
        
        with info_cols[2]:
            st.metric("场景ID", scene)
        
        # 尝试计算平均ADE
        try:
            ade_sum = 0
            ade_count = 0
            for f in frames:
                log_path = f"{result_folder}/{scene}_{f}_logs.txt"
                if os.path.exists(log_path):
                    with open(log_path, 'r', encoding='utf-8') as log_file:
                        log_content = log_file.read()
                        for line in log_content.split('\n'):
                            if line.startswith("Average Displacement Error:"):
                                ade_value = float(line.replace("Average Displacement Error:", "").strip())
                                ade_sum += ade_value
                                ade_count += 1
                                break
            
            with info_cols[3]:
                if ade_count > 0:
                    st.metric("平均ADE", f"{ade_sum/ade_count:.4f}")
                else:
                    st.metric("平均ADE", "未知")
        except:
            with info_cols[3]:
                st.metric("平均ADE", "计算出错")

# Initialize session state
if 'selected_frame' not in st.session_state:
    st.session_state.selected_frame = None  # 改回None，让加载按钮来设置初始帧
    # 添加自动播放相关的状态
    st.session_state.is_playing = False
    st.session_state.play_speed = 1.0  # 播放速度，单位为秒
    # 添加防止无限循环的状态变量
    st.session_state.last_nav_frame = None
    st.session_state.last_jump_frame = None
    st.session_state.last_slider_frame = None
    st.session_state.last_auto_frame = None


# Run the analysis if button is clicked
if run_button:
    # 验证API密钥是否提供
    if not api_key or api_key == "sk-93a03d4f21934f1191afa00c21455346":
        st.error("⚠️ 请提供有效的千问API密钥！没有有效的API密钥，分析将无法进行。")
        st.warning("您可以从阿里云获取密钥: https://dashscope.aliyun.com/")
    else:
        # 显示动画加载指示器
        st.markdown("""
        <style>
            .loading-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                margin: 30px 0;
            }
            
            .loading-spinner {
                border: 6px solid #f3f3f3;
                border-top: 6px solid #2962FF;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin-bottom: 15px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .loading-text {
                font-size: 18px;
                color: #2962FF;
                font-weight: 500;
            }
        </style>
        
        <div class="loading-container">
            <div class="loading-spinner"></div>
            <div class="loading-text">正在启动AutoChoreographer分析...</div>
        </div>
        """, unsafe_allow_html=True)
    
        # Create timestamp directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_dir = f"Qwen_results/{timestamp}"
        
        # Prepare command to run main.py
        cmd = [
            "python", 
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "main.py"),
            "--dataroot", dataset_path,
            "--version", dataset_version,
            "--model", model_name,
            "--max_frames", str(max_frames)
        ]
            
        # Add optional parameters
        if specific_scene:
            cmd.extend(["--scene", specific_scene])
        
        # Run the command
        with st.spinner("正在后台处理数据集，请耐心等待...根据模型的不同和数据集的大小，这可能需要这可能需要几分钟到几小时不等的时间，提前停止请按右上角的stop按钮"):
            try:
                # Create a temporary environment variable to set the API key
                env = os.environ.copy()
                env["QIANWEN_API_KEY"] = api_key
                
                # 显示启动信息
                st.info(f"处理场景: {specific_scene if specific_scene else '所有场景'}, 每个场景最多处理 {max_frames} 帧")
                
                # Execute the command in background mode
                # 使用subprocess.run而不是Popen，避免实时读取输出
                result = subprocess.run(
                    cmd, 
                    env=env,
                    capture_output=True,  # 捕获输出但不实时处理
                    text=True,
                    encoding='utf-8',
                    check=False,  # 不自动检查返回码，避免抛出异常
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                )
                
                # 处理完成后检查结果
                if result.returncode != 0:
                    st.error(f"处理过程中发生错误:\n{result.stderr}")
                else:
                    # 美化成功消息
                    st.markdown(f"""
                    <div style='background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;'>
                        <h3 style='color: #2E7D32; margin-top: 0;'>✅ 分析完成!</h3>
                        <p style='margin-bottom: 0;'>结果已保存到: <code>{result_dir}</code></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 生成结果摘要卡片
                    st.markdown("<div class='stCard'>", unsafe_allow_html=True)
                    st.markdown("<h3 style='color: #1E88E5;'>📊 分析结果摘要</h3>", unsafe_allow_html=True)
                    
                    # 尝试统计和显示结果
                    try:
                        scene_files = glob.glob(f"{result_dir}/*_logs.txt")
                        scenes = sorted(list(set([os.path.basename(f).split('_')[0] for f in scene_files])))
                        
                        if scenes:
                            st.markdown(f"<p>共处理了 <b>{len(scenes)}</b> 个场景，<b>{len(scene_files)}</b> 帧画面</p>", unsafe_allow_html=True)
                            
                            # 计算平均ADE
                            ade_sum = 0
                            ade_count = 0
                            
                            for log_file in scene_files:
                                try:
                                    with open(log_file, 'r', encoding='utf-8') as f:
                                        log_content = f.read()
                                        for line in log_content.split('\n'):
                                            if line.startswith("Average Displacement Error:"):
                                                ade_value = float(line.replace("Average Displacement Error:", "").strip())
                                                ade_sum += ade_value
                                                ade_count += 1
                                                break
                                except:
                                    continue
                            
                            if ade_count > 0:
                                st.markdown(f"<p>平均位移误差(ADE): <b>{ade_sum/ade_count:.4f}</b></p>", unsafe_allow_html=True)
                            
                            # 显示处理时间
                            st.markdown(f"<p>处理时间: <b>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</b></p>", unsafe_allow_html=True)
                            
                            # 生成结果预览
                            st.markdown("<h4 style='color: #0D47A1;'>🔍 场景预览</h4>", unsafe_allow_html=True)
                            
                            scene_cols = st.columns(len(scenes) if len(scenes) < 4 else 4)
                            for i, scene in enumerate(scenes[:4]):  # 最多显示4个场景预览
                                col_index = i % 4
                                with scene_cols[col_index]:
                                    # 尝试获取该场景的第一帧
                                    scene_frames = sorted([int(os.path.basename(f).split('_')[1]) 
                                                        for f in scene_files if os.path.basename(f).startswith(f"{scene}_")])
                                    if scene_frames:
                                        first_frame = scene_frames[0]
                                        front_cam_path = f"{result_dir}/{scene}_{first_frame}_front_cam.jpg"
                                        if os.path.exists(front_cam_path):
                                            st.image(front_cam_path, caption=f"场景: {scene}", use_column_width=True)
                                        st.markdown(f"<p style='text-align:center;'><a href='#' onclick='document.getElementById(\"{scene}_{first_frame}_button\").click();'>查看详情</a></p>", unsafe_allow_html=True)
                                        # 隐藏按钮，用于JS点击
                                        if st.button(f"查看场景 {scene}", key=f"{scene}_{first_frame}_button", visible=False):
                                            st.session_state.selected_result = result_dir
                                            st.session_state.selected_scene = scene
                                            st.session_state.selected_frame = first_frame
                                            st.experimental_rerun()
                    except Exception as e:
                        st.warning(f"生成结果摘要时发生错误: {str(e)}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # 在右侧显示执行日志摘要
                    with st.expander("📝 查看执行日志摘要", expanded=False):
                        # 只显示最后100行的日志
                        output_lines = result.stdout.splitlines()
                        if len(output_lines) > 100:
                            st.code("\n".join(output_lines[-100:]), language="bash")
                        else:
                            st.code(result.stdout, language="bash")
                    
                    # Update the results folder selection
                    result_folders = glob.glob("Qwen_results/*")
                    result_folders.sort(reverse=True)
                    if result_folders:
                        st.session_state.selected_result = result_folders[0]
                        st.experimental_rerun()
                    else:
                        st.warning("未找到结果文件夹")
                    
            except Exception as e:
                st.error(f"发生错误: {str(e)}")

# Visualize the selected frame if available
if load_button and selected_result and selected_scene and selected_frame is not None:  # 修改条件判断
    st.session_state.selected_result = selected_result
    st.session_state.selected_scene = selected_scene
    st.session_state.selected_frame = selected_frame
    safe_visualize_frame(selected_result, selected_scene, selected_frame)
elif 'selected_result' in st.session_state and 'selected_scene' in st.session_state and 'selected_frame' in st.session_state and st.session_state.selected_frame is not None:  # 修改条件判断
    safe_visualize_frame(st.session_state.selected_result, st.session_state.selected_scene, st.session_state.selected_frame)

# Footer
st.markdown("<div class='main-content'></div>", unsafe_allow_html=True)
st.markdown("<div class='footer'>", unsafe_allow_html=True)
footer_cols = st.columns([1, 2, 1])
with footer_cols[1]:
    st.markdown("""
    <div style='text-align: center;'>
        <h4 style='color: #616161; font-size: 1.1em; font-weight: 400;'>AutoChoreographer自动驾驶框架</h4>
        <p style='color: #9E9E9E; font-size: 0.9em;'>基于千问大模型和计算机视觉的端到端自动驾驶决策系统</p>
        <p style='color: #9E9E9E; font-size: 0.8em;'>© 2024-2025 | <a href="https://github.com/obsidian368/AutoChoreographer" target="_blank" style='color: #1E88E5; text-decoration: none;'>GitHub</a> | <a href="https://dashscope.aliyun.com/" target="_blank" style='color: #1E88E5; text-decoration: none;'>千问API</a></p>
    </div>
    """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)