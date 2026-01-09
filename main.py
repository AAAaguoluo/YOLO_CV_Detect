import os
import json
import time
import zipfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse, StreamingResponse
from ultralytics import YOLO
import base64

# ===================== 配置项 =====================
# 本机IP和端口
HOST = "0.0.0.0"  # 监听所有网卡，支持局域网访问
PORT = 8000
BASE_URL = f"https://172.27.119.58:{PORT}"

# 目录配置
STATIC_DIR = Path(__file__).parent / "static"
UPLOAD_DIR = Path(__file__).parent / "uploads"
RESULT_DIR = Path(__file__).parent / "results"
TEMP_DIR = Path(__file__).parent / "temp"

# 创建目录（确保递归创建）
for dir_path in [STATIC_DIR, UPLOAD_DIR, RESULT_DIR, TEMP_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# ===================== 模型加载（增加异常处理） =====================
try:
    # 加载YOLO模型（确保路径正确）
    MODEL_PATH = Path(__file__).parent / "best.pt"
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
    
    MODEL = YOLO(MODEL_PATH)
    print(f"✅ 成功加载模型: {MODEL_PATH}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    MODEL = None

# ===================== 初始化FastAPI =====================
app = FastAPI(title="YOLO水稻害虫检测系统", version="1.0")

# 跨域配置（支持局域网其他设备访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定具体IP
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件（前端页面）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# 挂载结果目录（供前端访问图片/视频）
app.mount("/results", StaticFiles(directory=RESULT_DIR), name="results")
app.mount("/temp", StaticFiles(directory=TEMP_DIR), name="temp")

# ===================== 工具函数 =====================
def save_upload_file(file: UploadFile, target_dir: Path) -> Optional[Path]:
    """保存上传的文件到指定目录（增加异常处理）"""
    try:
        # 生成安全的文件名
        safe_filename = file.filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
        file_path = target_dir / f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{safe_filename}"
        
        # 分块保存大文件
        with open(file_path, "wb") as f:
            while chunk := file.file.read(1024 * 1024):  # 1MB分块
                f.write(chunk)
        
        # 验证文件是否保存成功
        if not file_path.exists() or file_path.stat().st_size == 0:
            raise Exception("文件保存为空")
        
        return file_path
    except Exception as e:
        print(f"保存文件失败: {e}")
        return None

def detect_image(image_path: Path, confidence: float = 0.25) -> Dict[str, Any]:
    """单张图片检测（修复所有潜在错误）"""
    start_time = time.time()
    detections = []
    annotated_image = None
    img_url = ""
    img_name = ""
    
    try:
        # 验证图片文件
        if not image_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {image_path}")
        
        # 检查MODEL是否加载成功
        if MODEL is None:
            raise Exception("YOLO模型未加载，请检查模型文件")
        
        # YOLO检测
        results = MODEL(image_path, conf=confidence)
        
        if results and len(results) > 0:
            result = results[0]
            # 提取检测结果
            for box in result.boxes:
                cls = int(box.cls[0])
                cls_name = MODEL.names.get(cls, f"未知类别_{cls}")  # 防止类别名不存在
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                
                detections.append({
                    "class_id": cls,
                    "class_name": cls_name,
                    "confidence": round(conf, 4),
                    "bbox": [round(x, 2) for x in xyxy]  # 边界框 [x1, y1, x2, y2]
                })
            
            # 生成标注后的图片
            annotated_image = result.plot()
            img_name = f"annotated_{image_path.name}"
            img_path = RESULT_DIR / img_name
            
            # 保存标注图片（处理中文路径问题）
            cv2.imencode('.jpg', annotated_image)[1].tofile(str(img_path))
            
            # 生成可访问的URL
            img_url = f"{BASE_URL}/results/{img_name}"
    
    except Exception as e:
        print(f"图片检测失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "original_filename": image_path.name,
            "detect_time": round(time.time() - start_time, 4),
            "detections": [],
            "total_detections": 0,
            "annotated_image_url": ""
        }
    
    detect_time = round(time.time() - start_time, 4)
    
    return {
        "success": True,
        "original_filename": image_path.name,
        "annotated_image_url": img_url,
        "detect_time": detect_time,
        "detections": detections,
        "total_detections": len(detections)
    }

def create_zip_archive(files: List[Path], zip_name: str) -> Optional[Path]:
    """创建ZIP压缩包（增加异常处理）"""
    try:
        zip_path = TEMP_DIR / f"{zip_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
        
        # 过滤不存在的文件
        valid_files = [f for f in files if f.exists()]
        if not valid_files:
            raise Exception("没有可压缩的有效文件")
        
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in valid_files:
                zipf.write(file, arcname=file.name)
        
        return zip_path
    except Exception as e:
        print(f"创建ZIP失败: {e}")
        return None

# ===================== 路由 - 前端页面 =====================
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """提供前端页面访问"""
    try:
        frontend_path = STATIC_DIR / "UI.html"
        if not frontend_path.exists():
            return HTMLResponse(content="<h1>前端文件未找到，请将UI.html放入static目录</h1>", status_code=404)
        
        with open(frontend_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载前端页面失败: {str(e)}")

# ===================== 路由 - 单张图片检测 =====================
@app.post("/api/predict")
async def predict_single(
    file: UploadFile = File(...),
    confidence: str = Form("0.25")
):
    """单张图片检测接口（完整异常处理）"""
    try:
        # 验证输入
        if not file:
            raise HTTPException(status_code=400, detail="未上传图片文件")
        
        # 转换置信度
        try:
            confidence_val = float(confidence)
            if not (0.0 <= confidence_val <= 1.0):
                raise ValueError("置信度必须在0-1之间")
        except ValueError:
            raise HTTPException(status_code=400, detail="置信度必须是有效的数字（0-1）")
        
        # 保存上传文件
        file_path = save_upload_file(file, UPLOAD_DIR)
        if file_path is None:
            raise HTTPException(status_code=500, detail="文件保存失败")
        
        # 执行检测
        result = detect_image(file_path, confidence_val)
        
        return JSONResponse(content=result)
    
    except HTTPException as e:
        return JSONResponse(content={"success": False, "error": e.detail}, status_code=e.status_code)
    except Exception as e:
        print(f"单张检测接口错误: {e}")
        return JSONResponse(content={
            "success": False,
            "error": f"服务器内部错误: {str(e)}"
        }, status_code=500)

@app.post("/api/save-single-result")
async def save_single_result(
    original_filename: str = Form(...),
    confidence: float = Form(...),
    detect_time: float = Form(...),
    detections: str = Form(...),
    image_data: str = Form(...)
):
    """保存单张检测结果（生成ZIP下载）"""
    try:
        # 解析检测结果
        try:
            detections_data = json.loads(detections)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="检测结果JSON格式错误")
        
        # 生成结果文件
        result_info = {
            "original_filename": original_filename,
            "confidence": confidence,
            "detect_time": detect_time,
            "detections": detections_data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存JSON结果
        json_file = TEMP_DIR / f"single_result_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result_info, f, ensure_ascii=False, indent=2)
        
        # 下载标注图片
        zip_files = [json_file]
        if image_data and image_data.startswith(BASE_URL):
            img_filename = image_data.split("/")[-1]
            img_file = RESULT_DIR / img_filename
            if img_file.exists():
                zip_files.append(img_file)
        
        # 创建ZIP包
        zip_path = create_zip_archive(zip_files, "single_result")
        if zip_path is None:
            raise Exception("创建ZIP压缩包失败")
        
        # 返回下载链接
        download_url = f"{BASE_URL}/download/{zip_path.name}"
        
        return JSONResponse(content={
            "success": True,
            "message": "保存成功",
            "download_url": download_url
        })
    
    except HTTPException as e:
        return JSONResponse(content={"success": False, "error": e.detail}, status_code=e.status_code)
    except Exception as e:
        print(f"保存单张结果错误: {e}")
        return JSONResponse(content={
            "success": False,
            "error": f"服务器内部错误: {str(e)}"
        }, status_code=500)

# ===================== 路由 - 批量图片检测 =====================
@app.post("/api/batch-predict")
async def predict_batch(
    files: List[UploadFile] = File(...),
    confidence: str = Form("0.25")
):
    """批量图片检测接口（完整异常处理）"""
    try:
        # 验证输入
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="未选择任何图片文件")
        
        # 转换置信度
        try:
            confidence_val = float(confidence)
            if not (0.0 <= confidence_val <= 1.0):
                raise ValueError("置信度必须在0-1之间")
        except ValueError:
            raise HTTPException(status_code=400, detail="置信度必须是有效的数字（0-1）")
        
        start_time = time.time()
        
        # 批量检测
        detection_results = []
        success_count = 0
        failed_count = 0
        class_stats = {}
        
        for file in files:
            try:
                file_path = save_upload_file(file, UPLOAD_DIR)
                if file_path is None:
                    failed_count += 1
                    detection_results.append({
                        "success": False,
                        "filename": file.filename,
                        "error": "文件保存失败"
                    })
                    continue
                
                result = detect_image(file_path, confidence_val)
                
                if result["success"]:
                    success_count += 1
                    detection_results.append(result)
                    
                    # 统计类别
                    for det in result["detections"]:
                        cls_name = det["class_name"]
                        class_stats[cls_name] = class_stats.get(cls_name, 0) + 1
                else:
                    failed_count += 1
                    detection_results.append(result)
            except Exception as e:
                failed_count += 1
                detection_results.append({
                    "success": False,
                    "filename": file.filename,
                    "error": str(e)
                })
        
        total_time = round(time.time() - start_time, 4)
        
        # 生成批量结果ZIP
        zip_files = []
        
        # 收集标注图片
        for res in detection_results:
            if res["success"] and res["annotated_image_url"]:
                img_filename = res["annotated_image_url"].split("/")[-1]
                img_file = RESULT_DIR / img_filename
                if img_file.exists():
                    zip_files.append(img_file)
        
        # 生成统计JSON
        stats_file = TEMP_DIR / f"batch_stats_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        stats_data = {
            "total_files": len(files),
            "success_count": success_count,
            "failed_count": failed_count,
            "total_time": total_time,
            "class_stats": class_stats,
            "detection_results": detection_results
        }
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        zip_files.append(stats_file)
        
        # 创建ZIP包
        zip_url = ""
        zip_path = create_zip_archive(zip_files, "batch_result")
        if zip_path is not None:
            zip_url = f"{BASE_URL}/download/{zip_path.name}"
        
        return JSONResponse(content={
            "success": True,
            "total_files": len(files),
            "success_count": success_count,
            "failed_count": failed_count,
            "total_time": total_time,
            "class_stats": class_stats,
            "detection_results": detection_results,
            "zip_url": zip_url
        })
    
    except HTTPException as e:
        return JSONResponse(content={"success": False, "error": e.detail}, status_code=e.status_code)
    except Exception as e:
        print(f"批量检测接口错误: {e}")
        return JSONResponse(content={
            "success": False,
            "error": f"服务器内部错误: {str(e)}"
        }, status_code=500)

@app.post("/api/save-batch-result")
async def save_batch_result(
    total_files: int = Form(...),
    success_count: int = Form(...),
    class_stats: str = Form(...),
    detection_results: str = Form(...)
):
    """保存批量检测结果（返回下载链接）"""
    try:
        # 解析数据
        try:
            class_stats_data = json.loads(class_stats)
            detection_results_data = json.loads(detection_results)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="JSON格式错误")
        
        # 生成统计文件
        result_info = {
            "total_files": total_files,
            "success_count": success_count,
            "failed_count": total_files - success_count,
            "class_stats": class_stats_data,
            "detection_results": detection_results_data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存JSON
        json_file = TEMP_DIR / f"batch_save_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result_info, f, ensure_ascii=False, indent=2)
        
        # 收集所有标注图片
        img_files = []
        for res in detection_results_data:
            if res.get("annotated_image_url") and res.get("annotated_image_url").startswith(BASE_URL):
                img_filename = res["annotated_image_url"].split("/")[-1]
                img_file = RESULT_DIR / img_filename
                if img_file.exists():
                    img_files.append(img_file)
        
        # 创建ZIP包
        zip_files = [json_file] + img_files
        zip_path = create_zip_archive(zip_files, "batch_save_result")
        
        if zip_path is None:
            raise Exception("创建ZIP压缩包失败")
        
        download_url = f"{BASE_URL}/download/{zip_path.name}"
        
        return JSONResponse(content={
            "success": True,
            "message": "批量结果保存成功",
            "download_url": download_url
        })
    
    except HTTPException as e:
        return JSONResponse(content={"success": False, "error": e.detail}, status_code=e.status_code)
    except Exception as e:
        print(f"保存批量结果错误: {e}")
        return JSONResponse(content={
            "success": False,
            "error": f"服务器内部错误: {str(e)}"
        }, status_code=500)

# ===================== 路由 - 视频检测 =====================
@app.post("/api/video-predict")
async def predict_video(
    file: UploadFile = File(...),
    confidence: str = Form("0.25"),
    interval: int = Form(5)
):
    """视频检测接口（完整异常处理）"""
    try:
        # 验证输入
        if not file:
            raise HTTPException(status_code=400, detail="未上传视频文件")
        
        # 转换参数
        try:
            confidence_val = float(confidence)
            if not (0.0 <= confidence_val <= 1.0):
                raise ValueError("置信度必须在0-1之间")
            interval_val = max(1, int(interval))  # 确保抽帧间隔≥1
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"参数错误: {str(e)}")
        
        # 检查MODEL是否加载成功
        if MODEL is None:
            raise HTTPException(status_code=500, detail="YOLO模型未加载，请检查模型文件")
        
        # 保存视频文件
        video_path = save_upload_file(file, UPLOAD_DIR)
        if video_path is None:
            raise HTTPException(status_code=500, detail="视频文件保存失败")
        
        # 处理视频
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception("无法打开视频文件")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 输出视频路径
        output_name = f"processed_{video_path.name}"
        output_path = RESULT_DIR / output_name
        
        # 处理视频编码
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise Exception("无法创建输出视频文件")
        
        # 逐帧检测
        class_stats = {}
        frame_count = 0
        detected_frames = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # 按间隔抽帧检测
            if frame_count % interval_val == 0:
                detected_frames += 1
                # YOLO检测
                results = MODEL(frame, conf=confidence_val)
                annotated_frame = results[0].plot()
                
                # 统计类别
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    cls_name = MODEL.names.get(cls, f"未知类别_{cls}")
                    class_stats[cls_name] = class_stats.get(cls_name, 0) + 1
            else:
                annotated_frame = frame
            
            # 写入输出视频
            out.write(annotated_frame)
        
        # 释放资源
        cap.release()
        out.release()
        
        # 计算耗时
        total_time = round(time.time() - start_time, 4)
        video_url = f"{BASE_URL}/results/{output_name}"
        
        return JSONResponse(content={
            "success": True,
            "original_filename": file.filename,
            "processed_video_url": video_url,
            "total_frames": total_frames,
            "detected_frames": detected_frames,
            "total_time": total_time,
            "class_stats": class_stats,
            "fps": fps,
            "resolution": f"{width}x{height}"
        })
    
    except HTTPException as e:
        return JSONResponse(content={"success": False, "error": e.detail}, status_code=e.status_code)
    except Exception as e:
        print(f"视频检测错误: {e}")
        return JSONResponse(content={
            "success": False,
            "error": f"服务器内部错误: {str(e)}"
        }, status_code=500)
# ===================== 路由 - 视频检测结果保存 =====================
@app.post("/api/save-video-result")
async def save_video_result(
    original_filename: str = Form(...),
    confidence: float = Form(...),
    frame_interval: int = Form(...),
    total_frames: int = Form(...),
    detected_frames: int = Form(...),
    class_stats: str = Form(...),
    detect_time: float = Form(...),
    video_url: str = Form(...)
):
    """保存视频检测结果（生成ZIP下载，包含视频文件+统计JSON）"""
    try:
        # 解析类别统计（JSON字符串转字典）
        try:
            class_stats_data = json.loads(class_stats)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="类别统计数据格式错误（非有效JSON）")
        
        # 1. 生成视频检测结果JSON
        result_info = {
            "original_filename": original_filename,
            "confidence": confidence,
            "frame_interval": frame_interval,
            "total_frames": total_frames,
            "detected_frames": detected_frames,
            "class_stats": class_stats_data,
            "detect_time": detect_time,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 保存JSON文件
        json_file = TEMP_DIR / f"video_result_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result_info, f, ensure_ascii=False, indent=2)
        
        # 2. 收集视频文件（从video_url中提取文件名）
        zip_files = [json_file]
        if video_url and video_url.startswith(BASE_URL):
            video_filename = video_url.split("/")[-1]
            video_file = RESULT_DIR / video_filename
            if video_file.exists():
                zip_files.append(video_file)
        
        # 3. 创建ZIP压缩包
        zip_path = create_zip_archive(zip_files, "video_result")
        if zip_path is None:
            raise Exception("创建视频结果ZIP包失败")
        
        # 4. 返回下载链接
        download_url = f"{BASE_URL}/download/{zip_path.name}"
        
        return JSONResponse(content={
            "success": True,
            "message": "视频检测结果保存成功",
            "download_url": download_url
        })
    
    except HTTPException as e:
        return JSONResponse(content={"success": False, "error": e.detail}, status_code=e.status_code)
    except Exception as e:
        print(f"保存视频结果错误: {e}")
        return JSONResponse(content={
            "success": False,
            "error": f"服务器内部错误: {str(e)}"
        }, status_code=500)



# 新增：下载接口（通用）
@app.get("/download/{filename}")
async def download_file(filename: str):
    """下载生成的ZIP包"""
    try:
        file_path = TEMP_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 返回文件下载响应
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/zip"
        )
    except HTTPException as e:
        return JSONResponse(content={"success": False, "error": e.detail}, status_code=e.status_code)
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": f"下载失败: {str(e)}"
        }, status_code=500)

# ===================== 路由 - WebSocket摄像头 =====================

# ===================== WebSocket - 摄像头实时检测 =====================
@app.websocket("/ws/camera")
async def websocket_camera(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 接收前端发送的Base64帧数据
            data = await websocket.receive_text()
            if not data.startswith('data:image/jpeg;base64,'):
                continue
            
            # 解码Base64为图片
            base64_data = data.split(',')[1]
            img_bytes = base64.b64decode(base64_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # YOLO检测（如果模型加载成功）
            if MODEL is not None:
                results = MODEL(frame, conf=0.25)
                annotated_frame = results[0].plot()
            else:
                annotated_frame = frame
            
            # 编码为JPEG并返回Base64
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_text(f'data:image/jpeg;base64,{img_base64}')
            
    except WebSocketDisconnect:
        print("客户端断开WebSocket连接")
    except Exception as e:
        print(f"WebSocket错误: {e}")
        await websocket.close()


# ===================== 路由 - 文件下载 =====================
@app.get("/download/{filename}")
async def download_file(filename: str):
    """通用文件下载接口（支持ZIP包下载）"""
    try:
        file_path = TEMP_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        
        # 返回文件下载响应
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/zip"
        )
    except HTTPException as e:
        return JSONResponse(content={"success": False, "error": e.detail}, status_code=e.status_code)
    except Exception as e:
        return JSONResponse(content={
            "success": False,
            "error": f"下载失败: {str(e)}"
        }, status_code=500)


# ===================== 健康检查接口 =====================
@app.get("/health")
async def health_check():
    """健康检查接口，用于测试服务器状态"""
    return {
        "status": "healthy",
        "server_url": BASE_URL,
        "model_loaded": MODEL is not None,
        "directories": {
            "static": str(STATIC_DIR),
            "uploads": str(UPLOAD_DIR),
            "results": str(RESULT_DIR),
            "temp": str(TEMP_DIR)
        },
        "timestamp": datetime.now().isoformat()
    }

# ===================== 启动服务 =====================
if __name__ == "__main__":
    import uvicorn
    
    print(f"\n🚀 服务器启动中...")
    print(f"📡 访问地址: {BASE_URL}")
    print(f"🔍 健康检查: {BASE_URL}/health")
    print(f"📂 静态文件: {STATIC_DIR}")
    print(f"🤖 模型路径: {MODEL_PATH if 'MODEL_PATH' in locals() else '未定义'}\n")
    
    """# 启动服务（支持HTTP，HTTPS需配置证书）
    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,  # 开发模式，生产环境关闭
        access_log=True,
        log_level="info"
    )"""



    
    uvicorn.run(
    "main:app",
    host=HOST,
    port=PORT,
    ssl_keyfile=r"C:\Users\mjuGGbond\key.pem",  # 私钥文件
    ssl_certfile=r"C:\Users\mjuGGbond\cert.pem", # 证书文件
    reload=True
)
    