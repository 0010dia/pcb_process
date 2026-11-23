"""이 프로젝트에서 실측으로 확인한 TensorFlow/Metal 그래프 오류를 회피한다."""

import platform

import tensorflow as tf


def configure_runtime() -> dict:
    # 이 장비의 TF 2.18.1/Metal 1.2.0에서 graph/eager 출력 불일치가 재현됐다.
    # meta optimizer를 끄면 동일 입력/가중치 출력이 일치했고 GPU 커널은 계속 사용한다.
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return dict(optimizer_options=tf.config.optimizer.get_experimental_options(),
                gpu_devices=[str(gpu) for gpu in gpus])
