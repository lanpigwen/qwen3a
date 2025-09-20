# 项目调优与稳定性总结

## 一、背景与目标
将 Whisper Large V3 Turbo 音频编码结果接入 Qwen3 Causal LM，实现语音到对话文本的指令/多轮生成微调。在单卡（32GB）环境下以 LoRA + 投影器 (AudioProjection) 为主的参数高效微调，解决：显存溢出、loss 恒为 0、数值 NaN、量化加载、标签掩码、训练稳定性。

## 二、主要问题与根因
| 问题 | 现象 | 根因分析 | 解决策略 |
|------|------|----------|----------|
| 初始 OOM | 加载后未训练即显存爆 | 模型初始 FP32 + 未量化；Qwen3+Whisper 同时上 GPU | half/bfloat16 加载 + 可选 quant + whisper_cpu |
| unwrap_model 报错 | accelerate unwrap_model 参数不匹配 | accelerate 与 transformers 版本差异 | 升级 accelerate 并 pin 版本 |
| loss 恒为 0 | 训练除第一步外日志 loss=0 | 实际 logits NaN 导致 trainer 聚合后返回 0 | 手动 loss 路径 + NaN 监控定位 |
| 全部标签被 mask | 有效标签数=0 fallback 出现 | 角色分段 offset 错配或拼接方式全掩码 | 重写 ConversationCollator 角色 span + 回退策略 |
| projector NaN | projector.proj.0.weight 第一次 backward 后 NaN | fp16 + 大初始化 + 上游激活方差 | FP32 前向 + 小尺度 Xavier *0.1 + freeze warmup |
| LoRA 权重 NaN | lora_A q_proj 早期 NaN | 半精度 LoRA + 未缩放初始化 + 梯度峰值大 | LoRA FP32 + init scale 0.1 + 独立 grad clip + early LR decay |
| 梯度爆峰 (上千) | grad_norm 周期性 >1000 | 注意力前层激活尖峰 + 未局部裁剪 | 全局 + LoRA 裁剪、可加激活 clamp（备选） |

## 三、关键实现与新增参数
### 训练脚本 (`finetune_audio.py`)
- --manual_loss: 手动计算 CE 验证遮罩与 logits 数值
- --projector_scale / --projector_fp32 / --freeze_projector_steps
- --lora_fp32 / --lora_init_scale / --lora_grad_clip
- --early_lr_factor / --early_lr_steps 早期学习率缩放
- --max_grad_norm 全局梯度裁剪

### 模型 (`modeling_qwen3_audio.py`)
- Whisper -> FP32 projector -> 转回主 dtype
- Projector Xavier uniform 初始化后整体 *0.1
- 记录 last_logits、手动 loss 分支

### 数据 (`data.py`)
- ConversationCollator 基于角色 token 前缀划 span，掩码用户+系统，保留助手
- Fallback：若全被 mask，保留末尾助手回复
- 提供 valid_label_count 统计

### 回调
- LossDebugCallback: 每步有效标签与 loss
- NaNMonitorCallback: 参数 / last_logits NaN 快照
- ProjectorFreezeHook: 早期冻结 projector 梯度
- LoRAStabilizeCallback: 早期 lr 缩放 + LoRA 独立梯度裁剪 + NaN grad 清零

## 四、数值稳定路径演进
1. 修复 mask/StopIteration/dtype → 发现真实是 NaN 而不是“loss=0”
2. 定位 projector 溢出 → FP32 前向 & 小初始化后消失
3. 暴露 LoRA NaN → LoRA FP32 + init scale + grad clip + early lr 后稳定
4. 后续仅剩梯度峰值偶发（不再生成 NaN）

## 五、推荐默认配置（稳定优先）
```
--freeze_whisper \
--use_lora \
--lora_fp32 --lora_init_scale 0.1 --lora_grad_clip 5.0 \
--projector_fp32 --freeze_projector_steps 20 \
--early_lr_factor 0.2 --early_lr_steps 100 \
--learning_rate 3e-5 \
--max_grad_norm 1.0 \
--gradient_accumulation_steps 4 \
--per_device_train_batch_size 1
```
在验证无 NaN 后可：
- 提升 per_device_train_batch_size（显存富余时）
- 逐步减小 early_lr_steps 或增大学习率微调速度
- 若 loss 曲线平滑，可关掉 --manual_loss

## 六、4bit 量化路线建议
1. 加 `--load_in_4bit --device_map auto`；保留 projector_fp32 + lora_fp32。
2. 学习率初始降 30%（例如 2e-5），观察 200 步梯度峰值。
3. 若出现再度 NaN：
   - 降低 lora_grad_clip (5 → 2.5)
   - 增加 projector_scale < 1（如 0.7）
   - 启用激活 clamp（需后续添加 hook）

## 七、可选进一步优化
| 优化 | 说明 | 价值 |
|------|------|------|
| 激活 clamp hook | 对第一层 q_proj 输入 clamp(-8,8) | 降低梯度峰值 | 
| Grad norm EMA 观测 | 平滑记录 grad_norm 均值方差 | 动态调 lr | 
| 仅保存可训练权重 | projector + LoRA 状态单独保存 | 降低 IO / checkpoint 体积 | 
| 断点恢复校验脚本 | 加载后跑 1 step 验证无 NaN | 提升恢复安全性 | 
| Label 质量过滤 | 丢弃有效 token < 阈值样本 | 减噪 / 加速收敛 | 

## 八、常见故障排查速表
| 现象 | 快速检查 | 快速缓解 |
|------|----------|----------|
| 再次出现 NaN | 看 NaNMonitor 快照 param 名 | 降 lr / 启动 fp32 路径 / 加裁剪 |
| loss 不下降 | valid_label_count 是否过低 | 调整对话拼接 / 放宽掩码 |
| 显存紧张 | --whisper_cpu / 4bit / 减 batch | 增梯度累计 steps |
| grad_norm > 5000 频繁 | 开启激活 clamp (后续) | 降 lr 或增 early steps |

## 九、当前仍未做 / 待扩展
- 4bit 全流程验证 (待办 17)
- 自动激活 clamp 选项（尚未实现）
- 诊断 loss=0 条目用于正式模式（待办 22：默认 loss 分支再验证一次）
- README 参数文档化补充（可在后续补一节）

## 十、结论
通过分阶段定位（mask → projector → LoRA）与精细数值策略（FP32 关键子模块、初始化缩放、早期 LR 缩放、分组梯度裁剪），已将初始单步后 NaN 崩溃的训练稳定为持续下降的 loss 轨迹。后续主要工作集中在：量化场景再验证、简化运行参数、以及自动化数值防护（可选激活 clamp 与动态 LR）。

---
更新时间: 2025-09-20
