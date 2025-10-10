# 验证步骤与示例输出

在新环境中执行下面的命令以验证 TensorFlow / Keras 导入与我们添加的 shim 行为：

1. 导入 TensorFlow 与 Keras：

```bash
python -c "import tensorflow as tf; import tensorflow.keras as keras; print('tf', tf.__version__, 'keras', getattr(keras,'__version__', 'N/A'))"
```

期望输出示例：

```
tf 2.15.1 keras 2.15.0
```

2. 测试项目入口导入（确保 shim 执行且不抛出错误）：

```bash
python -c "import scripts.train; print('imported scripts.train OK')"
```

期望输出：

```
imported scripts.train OK
```

Observed on this machine (example):

```
tf 2.15.1
keras spec None
```

Note: in some environments `keras` may be a separate top-level package; this project prefers the `tf.keras` shipped with TensorFlow.
如果看到 `DeprecationWarning: The name tf.ragged.RaggedTensorValue is deprecated`：
- 确保你已经激活 conda 环境并安装了 `requirements.lock.txt` 中的依赖
- shim 代码会临时抑制该弃用信息；如果仍然出现，可能是 shim 未在最早入口执行（检查所有入口脚本是否尽早 import `src.bootstrap`）。
