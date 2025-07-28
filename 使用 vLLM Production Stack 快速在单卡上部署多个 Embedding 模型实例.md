# 使用 vLLM Production Stack 快速在单卡上部署多个 Embedding 模型实例

## 背景介绍

[vLLM Production Stack](https://github.com/vllm-project/production-stack) 是一个基于 Kubernetes 的大语言模型部署平台，支持高效的模型服务化部署。本文将演示如何在单张 GPU 上同时运行多个 Embedding 模型实例，通过合理的资源分配实现GPU资源的最大化利用。

## 部署步骤

### 1. 准备模型存储卷

首先创建 PersistentVolume 来存储 Qwen3 Embedding 模型，避免每个实例重复下载模型文件：

```yaml
# pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: test-vllm-pv
  labels:
    model: "Qwen3-Embedding-0.6B"
spec:
  capacity:
    storage: 500Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: standard
  hostPath:
    path: /data/model/qwen3
```

### 2. 配置多实例部署

关键配置说明：

- `replicaCount: 4` - 启动4个模型实例副本
- `requestGPU: 0.2` - 每个实例分配20%的GPU资源
- `gpuMemoryUtilization: 0.15` - 每个实例使用15%的GPU显存

```yaml
# embedding.yaml
servingEngineSpec:
  strategy:
    type: Recreate
  runtimeClassName: ""
  modelSpec:
  - name: "qwen3embed"
    repository: "vllm/vllm-openai"
    tag: "latest"
    modelURL: "Qwen/Qwen3-Embedding-0.6B"
    replicaCount: 4  # 4个实例副本
    requestCPU: 6
    requestMemory: "16Gi"
    requestGPU: 0.2  # 每个实例占用20%GPU

    pvcStorage: "50Gi"
    pvcMatchLabels:
      model: "Qwen3-Embedding-0.6B"

    vllmConfig:
      gpuMemoryUtilization: 0.15  # 显存利用率15%
      extraArgs: ["--disable-log-requests", "--task", "embed", "--served-model-name", "Qwen3-Embedding-0.6B", "--max-model-len", "8192"]

    env:
    # 这里可以配置代理，如果需要代理的话
    - name: CUDA_VISIBLE_DEVICES
      value: "0"  # 指定使用第0号GPU
```

### 3. 执行部署

```bash
# 应用存储卷配置
kubectl apply -f pv.yaml

# 使用 Helm 部署 vLLM Stack
helm install vllm vllm/vllm-stack -f embedding.yaml

# 端口转发以便本地访问
kubectl port-forward svc/vllm-router-service 8888:80
```

### 4. 验证部署

```bash
# 查看可用模型
curl -o- http://localhost:8888/v1/models
```

## 资源分配说明

此配置实现了在单张 GPU 上运行 4 个 Embedding 模型实例：

- **GPU 使用率**：4 × 20% = 80%，留有20%缓冲
- **显存分配**：4 × 15% = 60%，确保不会显存溢出
- **并发处理**：4个实例可同时处理不同的 Embedding 请求

## 参考资料

更多详细信息请参考 vLLM Production Stack 官方教程：[Tutorial: Loading Model Weights from Persistent Volume](https://github.com/vllm-project/production-stack/blob/main/tutorials/03-load-model-from-pv.md)
