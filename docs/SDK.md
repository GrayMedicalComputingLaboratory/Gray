# Gray SDK 使用说明

`gray infer` 加载 `--model` 指定的 ONNX 模型。SDK 自带的 `synthetic-identity-2d` 用于验证张量输入输出。单张 2D 皮肤镜图片使用 `gray skin_inference`，返回包含病灶掩膜、分类结果和权重列表的 JSON。

安装步骤、命令示例和 JSON 字段见独立 SDK 仓库的 [Gray SDK 使用说明](https://github.com/GrayMedicalComputingLaboratory/gray_sdk/blob/main/docs/SDK.md)。皮肤镜命令由 [Skin AI 插件](https://github.com/GrayMedicalComputingLaboratory/skin_ai) 提供。
