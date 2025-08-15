# Cuda-Accelerated-Renderer-for-SciVis

## 模块划分方式

### 1. **核心渲染模块（Core Rendering）**

负责核心的 raymarching 体绘制逻辑，是最重要的部分。

| 模块名               | 功能说明                   |
| ----------------- | ---------------------- |
| `volume_renderer` | c++ class for volume renderer，体积渲染核心逻辑 |
| `render_kernal`   | cuda kernel for accelerated renderering |
| `camera`          | 构造视图光线                 |
| `scene`            | 存储渲染结果图像并导出为 PNG       |
| `transfer_function`| 向量、矩阵运算，常用函数           |
| `config`          | 渲染参数、图像尺寸、步长等常量配置      |

> ✅：实现一个最小可运行的 pipeline。

---

### 2. **数据加载模块（Data I/O）**

负责读取 `.bin`、`.raw` 等体数据格式并传给 CUDA。

| 模块名           | 功能说明               |
| ------------- | ------------------ |
| `res_manager` | 加载资源（体数据、配置等）      |
| `bbox`        | 提供 Volume 的边界盒裁剪信息 |

> 🟨 在渲染基本完成后添加，方便更换体数据和测试。

---

### 3. **光照 & 光线模块（Ray & Lighting）**

负责构造光线、采样体素密度、进行光照计算。

| 模块名            | 功能说明              |
| -------------- | ----------------- |
| `ray`          | 光线数据结构            |
| `light`        | 简单方向光、Phong 光照模型等 |
| `interpolator` | 三线性插值体素值          |

> 🟨 可在初版完成后加入照明与插值增强视觉效果。

---

### 4. **高级功能模块（高级特性）**

包括交互、GUI、分类器（TF）、隐式几何等进阶特性。

| 模块名             | 功能说明                   |
| --------------- | ---------------------- |
| `gui`           | ImGui 实时交互             |
| `classifier`    | 灰度值 → 颜色/透明度传输函数       |
| `implicit_geom` | 隐式几何体绘制（e.g. Metaball） |
| `main_scene`    | 管理整体渲染场景               |

---

## 开发顺序

| 阶段      | 模块                                              | 目标                           |
| ------- | ----------------------------------------------- | ---------------------------- |
| 🟢 阶段 1 | `camera` + `volume_renderer` + `film` + `utils` | 实现最小可运行 CUDA 渲染器，输出一张 PNG    |
| 🟡 阶段 2 | `config` + `res_manager`                        | 支持换体数据、配置步长、分辨率等             |
| 🟠 阶段 3 | `interpolator` + `ray` + `light`                | 实现插值采样、光照增强，提升图像质量           |
| 🔵 阶段 4 | `classifier`, `gui`, `bbox`                     | 加入传输函数 GUI、Bounding Box 裁剪优化 |
| 🟣 阶段 5 | `main_scene`, `implicit_geom`                   | 实现复杂场景管理、添加可编程的隐式体绘制         |

---

## Core principles

1. **Two-worlds, one contract**
   Treat **Host** and **Device** as two separate worlds with a **formal contract** between them. Host owns loading, lifetime, and metadata; Device sees a compact, immutable **snapshot** (struct of POD + texture/surface handles) that’s cheap to pass into kernels.

2. **Data‑oriented over OO**
   Kernels want contiguous, cache-friendly, trivially-copyable data. Prefer flat POD structs (`DeviceScene`, `DeviceVolume`, `DeviceTF`, `DeviceLight[]`) and texture objects over pointer-rich graphs.

3. **Immutable snapshots, explicit commits**
   Host objects are editable. Rendering uses a frozen **DeviceScene snapshot** created by `scene.commit()`. No hidden mutations during render. This gives determinism and easy multi-threading.

4. **Descriptors in, handles out**
   Construction uses **descriptors** (dims, spacing, origin, units, ranges). Upload returns **handles** (texture objects, device pointers) managed by RAII wrappers. Avoid exposing raw arrays.

5. **Clear frames & units**
   Be explicit about spaces: **Index (i,j,k)**, **Texture (u,v,w in \[0,1])**, **World (x,y,z)**. Store `origin`, `voxelSize`, `dim`, and value units/range. Conversions must be single-line and consistent.

6. **Consistency beats cleverness**
   One pixel format (`uchar4`), one GL internal format (`GL_RGBA8`), normalized texture coords on by default, linear filtering, border/clamp addressing. Fewer knobs → fewer bugs.

7. **Zero‑copy where it matters**
   Prefer writing directly to GL texture via surface object or PBO when mature. Until then, one device-to-device blit is fine. Don’t micro-opt prematurely—design for the *option*.

8. **Change tracking (dirty bits)**
   Every host-side component sets a dirty flag on mutation. `commit()` rebuilds only what changed: TF table? only TF; camera moved? only camera; new volume? rebuild volume block.

9. **Extensibility without recompiling kernels**
   Leave versioned fields in `DeviceScene` (e.g., `mode`, `opacityScale`, optional `gradTex`), and a `caps`/`flags` bitfield so kernels can branch safely when features exist.

10. **Graceful fallbacks**
    If optional resources are missing (no gradient texture), kernels switch to finite differences. If no lights, do emission-only. No crashing because a feature wasn’t set.

11. **Async everywhere**
    Use CUDA streams for uploads and double-buffering for time steps. `commit(stream)` so big 3D copies don’t stall the render stream.

12. **Observability**
    Build in lightweight stats: GPU mem footprint, array dims, min/max value, step counts, timings. Expose a `SceneDebugInfo`—you’ll need it.

13. **Testability**
    Make CPU reference samplers for 1D/3D textures and TF mapping. Golden tests for ray-march accumulation and TF application with fixed seeds.

# High-level architecture

* **Host layer**

  * `Volume` (RAII): owns CUDA 3D array + `cudaTextureObject_t`, metadata (dim, spacing, origin, valueRange).
  * `TransferFunction` (RAII): 1D `float4` table → 1D texture.
  * `Lights` (RAII): device buffer of `DeviceLight` (small).
  * `Camera` (host math only): exposes position + orthonormal basis + fov.
  * **Scene** (or `SceneBuilder`): references components, tracks dirty state, validates, and produces a **DeviceScene snapshot**.
  * **ResourceManager** (optional): deduplicates identical TFs/volumes, manages lifetimes across scenes.

* **Device layer**

  * `DeviceScene` (POD): `DeviceCamera`, `DeviceVolume`, `DeviceTF`, pointer to `DeviceLight[]`, counts, and render params. No STL, no Eigen, no virtual, no host-only types.

# Scene lifecycle

1. **Build**

   * Set or replace components: `scene.setVolume(vol)`, `scene.setTransferFunction(tf)`, `scene.setLights(lights)`, `scene.setCamera(&cam)`.
   * Set params: `scene.setRenderParams(step, opacityScale, mode, iso)`; `scene.setClipBox(...)`.

2. **Validate**
   On `commit()`, check invariants:

   * Volume exists if `mode` requires it.
   * TF table count > 1, domain valid, finite.
   * `stepSize` > 0 and relative to min(voxelSize).
   * Basis vectors orthonormal within tolerance.

3. **Pack**

   * Convert host camera → `DeviceCamera` (float3 only).
   * Write `DeviceVolume` from `Volume::Desc` + texture handle(s).
   * Write `DeviceTF` from TF handle + domain.
   * Attach `DeviceLight*` and count.
   * Copy to `DeviceScene` **by value** or upload to a device buffer for pointer passing.

4. **Use**

   * Pass `DeviceScene` **by value** to kernels for simplicity and ABI stability.
   * Alternatively pass a const pointer to a device-side `DeviceScene` for multi-kernel passes.

# API sketch (clean and future-proof)

```cpp
// Host-side
class Scene {
public:
    // Configuration
    Scene& setCamera(const Camera* cam);
    Scene& setVolume(std::shared_ptr<Volume> vol);
    Scene& setTransferFunction(std::shared_ptr<TransferFunction> tf);
    Scene& setLights(const Lights* lights);
    Scene& setRenderParams(float step, float opacityScale, int mode, float iso = 0.f);
    Scene& setClipBox(float3 mn, float3 mx);

    // Commit: creates an immutable snapshot for kernels
    // Optionally accepts a CUDA stream for async uploads
    void commit(cudaStream_t stream = 0);

    // Accessors for rendering
    const DeviceScene&   snapshotHost()  const; // by-value pass
    const DeviceScene*   snapshotDevice() const; // pointer pass (optional)

    // Diagnostics
    SceneDebugInfo debug() const;
};
```

Key choices here:

* **Builder-style setters** make composition readable.
* **`commit()`** is explicit and boundaries are clear. You can render older snapshots while building a new one (double buffering).
* **`snapshotHost()`** enables passing the whole struct by value to a kernel—fast and simple.
* **`snapshotDevice()`** lets you store snapshots on GPU for multi-pass pipelines.

# Memory & performance tactics

* **Textures first**: 3D scalar → linear-filtered, normalized coords. TF → linear-filtered 1D `float4`. Gradient is optional 3D `float4`.
* **Pitch and alignment**: unify pixel buffers to `uchar4`. Keep `DeviceScene` size < 4–8 KB to stay register/cache friendly when passed by value.
* **Constant memory**: tiny, truly constant things (e.g., mode enums, small LUT sizes) can live in `__constant__`; but a single by-value struct is often good enough and simpler.
* **Streams**: `commit(streamUpload)`; `render(streamRender)`. For time-varying data, ping-pong volumes/arrays per timestep.

# Extensibility hooks

* **Multi-field volumes**: make `DeviceScene` hold an array of `DeviceVolume` or a small fixed N (e.g., 2–4) and a mapping policy (`combineMode`: emission/absorption fields, bivariate TF, etc.).
* **AMR / bricked volumes**: add a bricking layer above `Volume` that assembles virtual texture coordinates; `DeviceScene` holds current brick atlas handles.
* **Masks & cut planes**: extra 3D masks as optional textures; store plane equations for slice/render modes.
* **Time series**: keep `std::vector<std::shared_ptr<Volume>>` with a `currentTimestep` and prefetch the next into a second snapshot.

# Error handling & UX

* Throw clear exceptions on resource failures with context (dims, bytes, device ID).
* Make all public methods **no-throw** except `commit()`, where allocation/copy happens.
* Provide `SceneDebugInfo` with: mem bytes, dims, value range, step size, mode, TF samples, number of lights, and a “valid” bitset.

# What this buys you

* **Determinism** (render uses frozen snapshots).
* **Simplicity** (flat structs & handles).
* **Speed** (texture sampling, by-value scene).
* **Flexibility** (optional features via flags/handles).
* **Safety** (validation + dirty-bit rebuilds).
* **Scalability** (streams, double-buffering, multi-field ready).

If you want, I can turn this into a minimal but production-ready `Scene`/`commit()` implementation skeleton with the exact method names you prefer—and a tiny kernel showing how the snapshot is consumed.

