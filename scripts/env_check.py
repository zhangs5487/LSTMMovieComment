# env_check.py
import torch
import sys
import platform
from packaging import version

print("=" * 60)
print("环境兼容性验证报告")
print("=" * 60)

# 基础信息
print(f"操作系统: {platform.platform()}")
print(f"Python版本: {sys.version.split()[0]}")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"CUDA版本: {torch.version.cuda}")
print(f"GPU型号: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "N/A")

# 兼容性检查
pytorch_version = version.parse(torch.__version__)
cuda_version = version.parse(torch.version.cuda) if torch.cuda.is_available() else None

print("\n" + "=" * 60)
print("兼容性分析")
print("=" * 60)

# 检查PyTorch 2.10.0 + CUDA 13.1兼容性
if pytorch_version >= version.parse("2.10.0") and cuda_version and cuda_version >= version.parse("13.1"):
    print("✅ PyTorch 2.10.0 与 CUDA 13.1 兼容")
else:
    print("⚠️  警告：版本可能不兼容！")
    print("   建议：PyTorch 2.10.0 官方推荐 CUDA 12.1")
    print("   但实测在CUDA 13.1上通常可以运行")

# 检查Windows 11兼容性
if "Windows-11" in platform.platform():
    print("✅ Windows 11 环境检测通过")
else:
    print("⚠️  非Windows 11环境")

# 检查Python 3.12兼容性
python_version = version.parse(sys.version.split()[0])
if python_version >= version.parse("3.12"):
    print("✅ Python 3.12 兼容性良好")
    print("   注意：部分旧库可能需要更新版本")
else:
    print("⚠️  Python版本低于3.12")

print("\n" + "=" * 60)
print("依赖库兼容性检查")
print("=" * 60)

# 检查关键库
required_libs = {
    'transformers': '4.40.0',
    'datasets': '2.19.0',
    'accelerate': '0.30.0',
    'evaluate': '0.4.0'
}

for lib, min_version in required_libs.items():
    try:
        import importlib
        module = importlib.import_module(lib)
        installed_version = version.parse(module.__version__)
        required_version = version.parse(min_version)
        
        if installed_version >= required_version:
            print(f"✅ {lib} {installed_version} >= {min_version}")
        else:
            print(f"⚠️  {lib} {installed_version} < {min_version} (建议升级)")
    except ImportError:
        print(f"❌ {lib} 未安装")

print("\n" + "=" * 60)
print("验证结论")
print("=" * 60)

if all([
    torch.cuda.is_available(),
    pytorch_version >= version.parse("2.10.0"),
    cuda_version and cuda_version >= version.parse("12.1"),
    "Windows-11" in platform.platform(),
    python_version >= version.parse("3.12")
]):
    print("✅ 环境验证通过！可以开始项目")
else:
    print("❌ 环境验证失败！请根据上述警告调整") 