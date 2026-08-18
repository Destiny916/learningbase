import kfc
import cv2
import time
import numpy as np
from act_async_infer_distributed_demo.scripts.utils_distributed import (
    log_info,
    log_error,
    log_warning,
)
def calculate_average_fps(camera, num_frames=200, mode_name="测试"):
    """
    计算相机的平均帧率
    
    Args:
        camera: 相机对象
        num_frames: 采集帧数
        mode_name: 测试模式名称
    
    Returns:
        平均帧率
    """
    log_info(f"\n{mode_name}: 开始采集 {num_frames} 帧...")
    
    start_time = time.time()
    valid_frames = 0
    
    for i in range(num_frames):
        # 捕获图像
        left, right = camera.capture()
        
        # 检查图像是否有效
        frame_valid = left is not None and right is not None
        if frame_valid:
            # 检查图像是否全黑
            if np.max(left) == 0 or np.max(right) == 0:
                frame_valid = False
        
        if frame_valid:
            valid_frames += 1
        
        # 每50帧打印一次进度
        if (i + 1) % 50 == 0:
            log_info(f"  已采集 {i+1}/{num_frames} 帧")
        
        # 检查按键，允许提前退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            log_info(f"用户提前退出，已采集 {i+1} 帧")
            break
    
    elapsed_time = time.time() - start_time
    
    if elapsed_time > 0:
        average_fps = valid_frames / elapsed_time
    else:
        average_fps = 0.0
    
    log_info(f"{mode_name}结果:")
    log_info(f"  总时间: {elapsed_time:.2f}秒")
    log_info(f"  有效帧数: {valid_frames}/{num_frames}")
    log_info(f"  平均帧率: {average_fps:.2f} FPS")
    
    return average_fps

def test_camera_fps_simple():
    """简化的相机帧率测试"""
    log_info("Kingfisher相机帧率测试")
    log_info("="*50)
    
    try:
        # 初始化相机
        camera = kfc.Camera()
        
        # 连接相机
        if not camera.connect("99"):
            log_info("相机连接成功")
        else:
            log_error("相机连接失败")
            return
        
        # 获取分辨率
        w, h = camera.getResolution()
        log_info(f"相机分辨率: {w} x {h}")
        
        # 测试1: 固定曝光模式
        log_info("\n" + "="*50)
        log_info("测试1: 固定曝光模式 (Exposure=1000)")
        log_info("="*50)
        
        ret = camera.setExposure(1000)
        exposure = camera.getExposure()
        log_info(f"设置曝光: {exposure}")
        
        fixed_fps = calculate_average_fps(camera, num_frames=200, mode_name="固定曝光模式")
        
        # 测试2: 自动曝光模式
        log_info("\n" + "="*50)
        log_info("测试2: 自动曝光模式")
        log_info("="*50)
        
        ret = camera.setAutoExposure(True)
        if ret != 0:
            log_warning("设置自动曝光失败")
        
        auto_fps = calculate_average_fps(camera, num_frames=200, mode_name="自动曝光模式")
        
        # 比较结果
        log_info("\n" + "="*50)
        log_info("性能比较:")
        log_info("="*50)
        
        log_info(f"固定曝光平均帧率: {fixed_fps:.2f} FPS")
        log_info(f"自动曝光平均帧率: {auto_fps:.2f} FPS")
        log_info(f"差值: {auto_fps - fixed_fps:+.2f} FPS")
        
        if fixed_fps > auto_fps:
            log_info(f"\n固定曝光模式更快，优势: {fixed_fps - auto_fps:.2f} FPS")
        else:
            log_info(f"\n自动曝光模式更快，优势: {auto_fps - fixed_fps:.2f} FPS")
        
        # 断开相机连接
        camera.disConnect()
        log_info("\n相机已断开连接")
        
        return fixed_fps, auto_fps
        
    except Exception as e:
        log_error(f"测试过程中出现错误: {e}")
        import traceback
        traceback.log_info_exc()
        return None, None

if __name__ == "__main__":
    fixed_fps, auto_fps = test_camera_fps_simple()
    
    if fixed_fps is not None and auto_fps is not None:
        log_info("\n" + "="*50)
        log_info("测试完成!")
        log_info("="*50)
