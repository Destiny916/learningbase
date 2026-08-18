import asyncio
import logging
import os
import subprocess

from dexe_agent import Player
from dexe_agent import Recognizer
from dexe_agent import Recorder
from dexe_agent import Synthesizer
from dexe_agent import setup_logging
from dexe_agent.common.exceptions import ASRError
from dexe_agent.common.exceptions import NetworkError
from dexe_agent.common.exceptions import TTSError


setup_logging(logging.DEBUG)
logger = logging.getLogger("dexe_agent")


def _expand_config_paths(value):
    defaults = {
        "W1_HOME": "/home/dexforce/w1",
        "W1_ACT_HOME": "/home/dexforce/w1/w1_act",
    }
    if isinstance(value, str):
        for name, default in defaults.items():
            value = value.replace(f"${name}", os.environ.get(name, default))
            value = value.replace(f"${{{name}}}", os.environ.get(name, default))
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_config_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_config_paths(item) for key, item in value.items()}
    return value

CONFIG = {
    "player": {
        "OUTPUT_DEVICE": {
            "TYPE": "pyaudio",
            "PARAMS": {
                "FORMAT": "int16",
                "CHANNELS": 1,
                "FRAMES_PER_BUFFER": 1024,
                "NUM_FRAMES": 1024,
                "SAMPLE_RATE": 44100,
                "DEVICE_NAME": "",
                "HOST_API": 0,
            }
        }
    },
    "recorder": {
        "INPUT_DEVICE": {
            "TYPE": "pyaudio",
            "PARAMS": {
                "FORMAT": "int16",
                "CHANNELS": 1,
                "FRAMES_PER_BUFFER": 1024,
                "NUM_FRAMES": 1024,
                "SAMPLE_RATE": 44100,
                "DEVICE_NAME": "",
                "HOST_API": 0,
            }
        },
        "VAD": {
            "TYPE": "rms",
            "PARAMS": {
                "RMS_THRESHOLD": 4000,
            },
        },
        "MIN_SPEECH_DURATION": 0.06,
        "MIN_SILENCE_DURATION": 0.8,
        "PREFIX_PADDING_DURATION": 0.1,
        "MAX_BUFFER_DURATION": 10.0,
    },
    "asr": {
        "TYPE": "volcengine_big_model_no_stream",
        "PARAMS": {
            "APPID": "3079220620",
            "ACCESS_TOKEN": "st1TUCcjY7f3DXG-rMJ6DEX6I1Kljo4C",
            "LANGUAGE":""
        },
    },
    "tts": {
        "TYPE": "volcengine_sentence",
        "PARAMS": {
            "APPID": "3079220620",
            "ACCESS_TOKEN": "st1TUCcjY7f3DXG-rMJ6DEX6I1Kljo4C",
            "VOICE_TYPE": "BV033_streaming",
            "EMOTION": "narrator",
            "SPEED_RATIO": 0.87,
            "SAMPLE_RATE": 44100,
        },
    }
}


class VLAAgent(object):
    def __init__(self, config, task_config):
        self.config = config
        self.task_config = task_config
        self.prompt = task_config["prompt"]
        self.loop = None
        self.speech_started_event = asyncio.Event()  # 用于通知录音开始的事件

        self.recorder = Recorder(
            self.config["recorder"],
            self.on_start_speaking,
            self.on_stop_speaking,
        )
        self.player = Player(self.config["player"])
        self.recognizer = Recognizer(self.config["asr"])
        self.synthesizer = Synthesizer(self.config["tts"])

        self.task_running = False  # 标记当前是否有任务在运行
        self.current_player_task = None  # 当前的播放任务

    def on_start_speaking(self):
        """
        录音开始回调。
        """
        self.player.interrupt()  # 打断当前播放的音频，如果有

        if self.loop and self.current_player_task and not self.current_player_task.done():
            logger.info(">>> 取消播放任务 <<<")
            self.loop.call_soon_threadsafe(self.current_player_task.cancel)

        # 通知主循环，音频开始开始填充，可以获取音频资源进行新一轮处理
        if self.loop and self.speech_started_event:
            self.loop.call_soon_threadsafe(self.speech_started_event.set)

    def on_stop_speaking(self):
        logger.info("停止录制...")

    async def _process_play_task(self, text: str):
        """播放任务"""
        try:
            try:
                audio_segment = await asyncio.to_thread(self.synthesizer.synthesize, text)
            except (TTSError, NetworkError) as e:
                logger.error(f"TTS 合成失败: {e}")
                return

            self.player.play(audio_segment)

        except asyncio.CancelledError:
            logger.warning("播放任务取消 (用户说话打断)")
            # 这里的异常通过 raise 抛出，或者直接 swallow 都可以，Task 状态会变为 Cancelled
            raise
        except Exception as e:
            logger.exception(f"任务处理异常: {e}")

    async def _process_command_task(self, text: str):
        """执行指令的任务"""
        try:            
            # 可能需要对用户指令进行解析
            if text in self.task_config["go_home"]:
                await asyncio.to_thread(self._execute_command_go_home)
                return
            elif text not in self.prompt.keys():
                logger.warning(f"未识别到有效指令")
                text = self.task_config["failed_detect_voice"]
                self.current_player_task = asyncio.create_task(self._process_play_task(text))
                return

            command = self.prompt[text]

            logger.info(f"执行指令: {command}")
            text = f"好的，我正在{command}"
            self.current_player_task = asyncio.create_task(self._process_play_task(text))

            self.task_running = True
            # 模拟执行指令的耗时操作
            await asyncio.to_thread(self._execute_command, command)
            logger.info(f"指令执行完成: {command}")
            self.task_running = False

            text = f"{text}任务已完成"
            self.current_player_task = asyncio.create_task(self._process_play_task(text))

        except asyncio.CancelledError:
            raise


    def _execute_command_go_home(self):
        """
        执行go home
        """
        
        # 使用传入的参数覆盖默认配置
        

        cmd_body = self.task_config["command"]["go_home_body"]

        cmd_hand = self.task_config["command"]["go_home_hand"]
        
        logger.info(f"执行命令: {' '.join(cmd_body)}")
        logger.info(f"执行命令: {' '.join(cmd_hand)}")

        
        # 执行命令
        try:
            result = subprocess.run(
                cmd_hand,
                capture_output=True,
                text=True,
                check=False
            )
            result = subprocess.run(
                cmd_body,
                capture_output=True,
                text=True,
                check=False
            )

            # 记录日志
            self._log_command_execution(cmd_body, result)

            return result
            
        except Exception as e:
            logger.info(f"命令执行异常: {e}")
            return subprocess.CompletedProcess(
                args=cmd_body,
                returncode=1,
                stdout="",
                stderr=str(e)
            )

    def _execute_command(self, command=None, **kwargs):
        """
        执行机器人客户端命令
        
        Args:
            command: 可选的命令参数（保持接口兼容）
            **kwargs: 可覆盖的配置参数
        """
        # 默认配置
        prompt = f'{command}'
        logger.info(prompt)

        
        # 使用传入的参数覆盖默认配置
        self.config.update(kwargs)
        
        # 构建命令
        cmd = ["taskset", "-c", self.task_config["execute_task"]["cpu_core"]]
        cmd.extend([
            "python", "-m", "act_async_infer_distributed_demo.scripts.client.run_robot_client",
            "--server_host", self.task_config["execute_task"]["server_host"],
            "--server_port", self.task_config["execute_task"]["server_port"],
            "--model_config", self.task_config["execute_task"]["model_config"],
            "--control_frequency", self.task_config["execute_task"]["control_frequency"],
            "--collect_frequency", self.task_config["execute_task"]["collect_frequency"],
            "--get_action_frequency", self.task_config["execute_task"]["get_action_frequency"],
            "--chunk_size_threshold", self.task_config["execute_task"]["chunk_size_threshold"],
        ])
        
        if self.task_config["execute_task"]["use_lipo"]:
            cmd.append("--use_lipo")
        if self.task_config["execute_task"]["debug"]:
            cmd.append("--debug")
        
        cmd.extend([
            "--max_steps", self.task_config["execute_task"]["max_steps"],
            "--time_infer", self.task_config["execute_task"]["time_infer"],
            "--sample_factor", self.task_config["execute_task"]["sample_factor"],
            "--prompt", prompt
        ])
        
        logger.info(f"执行命令: {' '.join(cmd)}")
        
        # 执行命令
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            # 记录日志
            self._log_command_execution(cmd, result)

            return result
            
        except Exception as e:
            logger.info(f"命令执行异常: {e}")
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr=str(e)
            )

    def _log_command_execution(self, cmd, result):
        """记录命令执行日志"""
        logger.info(f"命令: {' '.join(cmd)}")
        logger.info(f"返回码: {result.returncode}")
        
        if result.stdout:
            logger.info(f"标准输出:\n{result.stdout}")
        
        if result.stderr:
            logger.info(f"错误输出:\n{result.stderr}")


    async def _main_loop(self):
        self.recorder.start()
        logger.info("请开始说话...")

        try:
            while True:
                await self.speech_started_event.wait()
                self.speech_started_event.clear()
                # 获取当前音频源，此时录音器已经开始录音，音频数据会持续填充直到录音结束
                # audio_source内部是queue，在同步asr接口中，会等待音频数据被填充完整后才识别
                audio_source = self.recorder.get_audio_source()

                try:
                    text_segment = await asyncio.to_thread(self.recognizer.recognize, audio_source)
                    logger.info(f"识别结果: {text_segment.text}")
                except (ASRError, NetworkError) as e:
                    logger.error(f"ASR 识别失败: {e}")
                    continue

                if not text_segment.text:
                    logger.warning("未识别到有效语音，跳过本轮对话")
                    continue

                if self.task_running:
                    # 正在执行任务，播放提示音频
                    logger.info("正在执行任务，播放提示音...")
                    text = "抱歉，我正在忙"
                    self.current_player_task = asyncio.create_task(self._process_play_task(text))
                    continue

                user_text = text_segment.text
                await asyncio.create_task(self._process_command_task(user_text))

        except asyncio.CancelledError:
            pass
        finally:
            self.recorder.stop()

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            self.loop.run_until_complete(self._main_loop())
        except KeyboardInterrupt:
            self.stop()
        finally:
            # 优雅退出：取消所有悬挂任务
            tasks = asyncio.all_tasks(self.loop)
            for t in tasks:
                t.cancel()
            self.loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
            self.loop.close()
            logger.info("流程结束")

    def stop(self):
        # 触发 run 方法中的退出逻辑
        if self.loop:
            for task in asyncio.all_tasks(self.loop):
                task.cancel()


if __name__ == '__main__':
    import argparse
    from act_async_infer_distributed_demo.scripts.utils_distributed import (
        load_json,
        log_error,
    )
    parser = argparse.ArgumentParser(description="Robot Client Voice Control")
    parser.add_argument(
        "--task_config", type=str, required=True, default="vla_control_instruction.json",help="Policy prompt"
    )
    args = parser.parse_args()
    task_config = _expand_config_paths(load_json(args.task_config))
    agent = VLAAgent(CONFIG,task_config)
    agent.run()

