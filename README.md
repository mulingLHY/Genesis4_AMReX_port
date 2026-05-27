## Genesis4_AMReX_port
由AI协助将 GENESIS 1.3 Version 4 (Free-Electron Laser 模拟代码) 渐进式改造为基于 AMReX 的 GPU 加速版本。

目前在RTX A4000, A6000, V100测试通过，小case测试结果完全一致，A4000单卡运行效率大约相当于AMD EPYC 7H12 64-Core Processor的100个CPU核心，但图形卡如RTX 4060Ti, 4070Ts运行效率较低。


### 编译指南
AMReX要求CUDA大于12.2，最好大于12.5

可能要手动指定HDF5(cmake xxx -DHDF5_ROOT=/path/to/hdf5)等库路径
```bash
cmake -S . -B build -DAMReX_GPU_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release

cmake --build build -j 8
```


### `&track/use_cuda`
`use_cuda` 控制本 port 中独立拆分出的 CUDA 路径，默认为true：
- `use_cuda = false`：FieldSolver、BeamSolver、TrackBeam、Control::applySlippage、Diagnostic::calc 使用 Genesis4 上游 CPU 实现。
- `use_cuda = true`：FieldSolver 使用 `FieldSolverADICUDA`/`FieldSolverFFTCUDA`，BeamSolver 使用 `BeamSolverCUDA`，束流 transverse/R56 路径使用 `TrackBeamCUDA`，slippage 使用 `ControlCUDA`，内置诊断计算使用 `DiagnosticCUDA`。


### test
小的计算case，用于测试运行结果一致性。

### VSCode解决代码提示intellisense报错
如果你用 CMake，重新配置时生成 compile_commands.json

```bash
cmake -S . -B build \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DAMReX_GPU_BACKEND=CUDA \
  -DCMAKE_BUILD_TYPE=Release
```
然后在 .vscode/settings.json 里加：
```json
{
  "C_Cpp.default.compileCommands": "${workspaceFolder}/build/compile_commands.json",
  "C_Cpp.default.cppStandard": "c++17"
}
```


### 已知问题
- 束流advance中的，尾场、集体效应等等都没port
- sort功能暂未port，开启可能会出错
- Marker写出光场和束流未实现，而且实现了也会显著影响速度