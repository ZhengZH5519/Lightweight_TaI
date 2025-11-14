import os
import copy
import pprint
from collections import defaultdict

import torch
import numpy as np
import json

import modelopt.torch.quantization as mtq
import sys


class EngineBuilder:
    def __init__(
        self,
        model,
        calibration_data,
        target_keywords=None,
        std_threshold=0.3,
        valid_param_suffix=("weight", "bias"),
        engine_name="mtq_int8_fp16",
        batch_size=1,
        metadata=None,
        onnx_out_dir=".",
        engine_out_dir=".",
    ):
        if model is None:
            raise RuntimeError("Please provide a loaded torch.nn.Module (model).")
        if calibration_data is None:
            raise RuntimeError("Please provide calibration_data (iterable yielding tensors or (tensor, ...)).")

        self.TARGET_KEYWORDS = target_keywords
        self.STD_THRESHOLD = std_threshold
        self.VALID_PARAM_SUFFIX = valid_param_suffix
        self.engine_name = engine_name
        self.batch_size = batch_size
        self.onnx_out_dir = onnx_out_dir
        self.engine_out_dir = engine_out_dir

        # 用户提供的资源
        self.model = model
        self.calibration_data = calibration_data
        # 从 model 获取 state_dict（备用，用于计算层的 std）
        self.state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        self.example_inputs = None
        self.CUSTOM_CFG = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
        self.filtered_layers = []
        self.metadata = metadata

        # 基本的 smoke test：验证 calibration_data 可迭代并且能做一次前向
        try:
            sample = next(iter(self.calibration_data))
        except Exception as e:
            raise RuntimeError("calibration_data is not iterable or is empty") from e

        inp = sample[0] if isinstance(sample, (list, tuple)) else sample
        if not torch.is_tensor(inp):
            raise RuntimeError("calibration_data must yield tensors or (tensor, ...).")

        # 把 model 移到 sample 的 device 并做一次 no_grad forward 验证
        self.model.eval()
        device = inp.device
        try:
            self.model.to(device)
            with torch.no_grad():
                test_input = inp if inp.dim() > 0 else inp.unsqueeze(0)
                _ = self.model(test_input)
        except Exception as e:
            raise RuntimeError(
                "Model forward failed on a sample from calibration_data. Check shape/device/signature."
            ) from e

    # -----------------------
    # MTQ 量化与配置生成
    # -----------------------
    def prepare_custom_cfg_and_quantize(self, save_quant_summary_path=None):
        """
        基于 model.state_dict() 找到目标层、计算 std，生成 CUSTOM_CFG 并对模型执行 mtq.quantize。
        - save_quant_summary_path: 若提供，将把 mtq.print_quant_summary 的输出写入该文件
        返回量化后的 model
        """
        if self.model is None:
            raise RuntimeError("model is not provided.")

        # 复制默认配置并做必要调整
        CUSTOM_CFG = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
        CUSTOM_CFG["quant_cfg"]["*conT*"] = {"enable": False}

        # 使用 self.state_dict 提取目标层参数
        state_dict = self.state_dict

        target_layers_params = defaultdict(list)
        for param_name, param_tensor in state_dict.items():
            if not param_name.endswith(self.VALID_PARAM_SUFFIX):
                continue
            layer_name = param_name.rsplit(".", 1)[0]

            # 当 self.TARGET_KEYWORDS 为假值（None/[]/''）时，匹配所有层；否则按关键字过滤
            if not self.TARGET_KEYWORDS or any(k.lower() in layer_name.lower() for k in self.TARGET_KEYWORDS):
                target_layers_params[layer_name].append(param_tensor.cpu().numpy())

        # 计算各层标准差并筛选
        layer_std_info = []
        for layer_name, param_list in target_layers_params.items():
            merged = np.concatenate([p.flatten() for p in param_list])
            param_std = float(np.std(merged))
            layer_std_info.append((layer_name, param_std, merged.size))

        filtered_layers = [(ln, s, n) for (ln, s, n) in layer_std_info if s > self.STD_THRESHOLD]

        # 生成 CUSTOM_CFG：以 blocks.x 作为核心路径并用 * 包裹
        processed_keys = set()
        for layer_name, _, _ in filtered_layers:
            cleaned_layer = layer_name.replace("module.", "")
            parts = cleaned_layer.split(".")
            if "blocks" in parts:
                blocks_idx = parts.index("blocks")
                core_path = (
                    ".".join(parts[:blocks_idx + 2])
                    if (blocks_idx + 1) < len(parts)
                    else ".".join(parts[:blocks_idx + 1])
                )
            else:
                core_path = cleaned_layer
            config_key = f"*{core_path}*"
            if config_key not in processed_keys:
                processed_keys.add(config_key)
                CUSTOM_CFG["quant_cfg"][config_key] = {"enable": False}

        # 保存结果到实例，以便后续查看或测试
        self.CUSTOM_CFG = CUSTOM_CFG
        self.filtered_layers = filtered_layers

        # 打印/保存 summary（可选）
        print("=== 标准差 > threshold 的层（示例输出） ===")
        for layer, std, _ in filtered_layers:
            print(f"- {layer}: {std:.6f}")
        print("\n生成的 CUSTOM_CFG 部分示例:")
        pprint.pprint({k: CUSTOM_CFG["quant_cfg"][k] for k in list(CUSTOM_CFG["quant_cfg"].keys())[:10]})

        # 校验 calibration_data 并准备 forward_loop
        def _iter_images(item):
            # item 可以是 tensor 或 (tensor, ...)，返回 tensor
            return item[0] if isinstance(item, (list, tuple)) else item

        # smoke test: ensure one sample can be forwarded on model device
        try:
            sample = next(iter(self.calibration_data))
            sample_imgs = _iter_images(sample)
            device = sample_imgs.device
            self.model.to(device).eval()
        except Exception as e:
            raise RuntimeError("Failed to validate calibration_data sample for forward.") from e

        # 通用 forward loop：逐批次将数据移动到 model device 并调用 forward
        def forward_loop(model):
            model_device = next(model.parameters()).device
            for item in self.calibration_data:
                imgs = _iter_images(item)
                imgs = imgs.to(model_device)
                with torch.no_grad():
                    model(imgs)
                # 显式删除临时变量以降低显存峰值
                del imgs

        # Quantize the model and perform calibration (PTQ)
        self.model = mtq.quantize(self.model, CUSTOM_CFG, forward_loop)

        # 可选：把量化 summary 写入文件
        if save_quant_summary_path:
            original_stdout = sys.stdout
            try:
                with open(save_quant_summary_path, "w") as f:
                    sys.stdout = f
                    mtq.print_quant_summary(self.model)
            finally:
                sys.stdout = original_stdout

        return self.model

    # -----------------------
    # ONNX 导出与 TensorRT engine 构建
    # -----------------------
    def export_onnx_and_build_engine(
        self, engine_name=None, batch_size=None, onnx_filename=None, engine_filename=None, export_dynamic_axes=False, opset_version=11
    ):
        """
        1) 导出 ONNX（如果 self.example_inputs 为空，将尝试从 calibration_data 取第一个样本作为 example_inputs）
        2) 用 trt.OnnxParser 解析并构建 Serialized engine 文件
        返回 engine 保存路径
        """
        engine_name = engine_name or self.engine_name
        batch_size = batch_size or self.batch_size

        if self.model is None:
            raise RuntimeError("model not prepared. Ensure quantize step is done.")

        # 如果未显式设置 example_inputs，从 calibration_data 取第一个样本（更可靠）
        if self.example_inputs is None:
            try:
                sample = next(iter(self.calibration_data))
                inp = sample[0] if isinstance(sample, (list, tuple)) else sample
                # 如果 sample 是 batch size 1 且用户指定 batch_size >1，重复；否则直接用 sample
                if inp.shape[0] == 1 and batch_size > 1:
                    self.example_inputs = inp.repeat(batch_size, 1, 1, 1)
                else:
                    self.example_inputs = inp
            except Exception:
                # fallback to random tensor
                self.example_inputs = torch.randn(batch_size, 3, 256, 256)

        # ONNX 文件路径
        onnx_file = onnx_filename or os.path.join(self.onnx_out_dir, f"{engine_name}_batch{batch_size}.onnx")
        # 导出 ONNX，用户可调整 opset_version 或 dynamic_axes 参数
        torch.onnx.export(
            self.model,
            self.example_inputs,
            onnx_file,
            input_names=["images"],
            dynamic_axes=None if not export_dynamic_axes else {"images": {0: "batch"}},
            opset_version=opset_version,
        )
        print(f"exported onnx to {onnx_file}")

        # TensorRT 构建
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.INFO)
        logger.min_severity = trt.Logger.Severity.VERBOSE

        builder = trt.Builder(logger)
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)

        if not parser.parse_from_file(onnx_file):
            # 收集并打印 parsing 错误
            for i in range(parser.num_errors):
                try:
                    print(parser.get_error(i))
                except Exception:
                    pass
            raise RuntimeError(f"failed to parse ONNX file: {onnx_file}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

        build = builder.build_serialized_network
        engine_file = engine_filename or os.path.join(self.engine_out_dir, f"{engine_name}_batch{batch_size}.engine")
        with build(network, config) as engine, open(engine_file, "wb") as f:
            if self.metadata is not None:# 如果有元数据 yolo依靠这个元数据运行
                meta = json.dumps(self.metadata)
                f.write(len(meta).to_bytes(4, byteorder="little", signed=True))
                f.write(meta.encode())
            f.write(engine)
        print(f"wrote engine to {engine_file}")
        return engine_file

    # -----------------------
    # 运行入口（当用户仍希望一步式调用）
    # -----------------------
    def run(self, save_quant_summary_path=None, **export_kwargs):
        """
        按顺序执行：生成 CUSTOM_CFG 并量化 -> 导出 ONNX 并构建 engine
        export_kwargs 直接传给 export_onnx_and_build_engine
        返回 engine 保存路径
        """
        self.prepare_custom_cfg_and_quantize(save_quant_summary_path=save_quant_summary_path)
        engine_path = self.export_onnx_and_build_engine(**export_kwargs)
        return engine_path
