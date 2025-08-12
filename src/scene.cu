#include "scene.hpp"
#include "cuda_utils.hpp"
#include <Eigen/Dense>

Scene::~Scene()
{
    if (m_ds_dev)
    {
        CUDA_CHECK(cudaFree(m_ds_dev));
    }
}
Scene &Scene::setCamera(const Camera *cam)
{
    m_cam = cam;
    m_dirtyCam = true;
    return *this;
}
Scene &Scene::setVolume(std::shared_ptr<Volume> vol)
{
    m_volume = std::move(vol);
    m_dirtyVol = true;
    return *this;
}
Scene &Scene::setTransferFunction(std::shared_ptr<TransferFunction> tf)
{
    m_tf = std::move(tf);
    m_dirtyTF = true;
    return *this;
}
Scene &Scene::setLights(const Lights *lights)
{
    m_lights = lights;
    m_dirtyLights = true;
    return *this;
}
Scene &Scene::setRenderParams(float step, float opacityScale, int mode, float iso)
{
    m_stepSize = step;
    m_opacityScale = opacityScale;
    m_mode = mode;
    m_isoValue = iso;
    m_dirtyParams = true;
    return *this;
}
Scene &Scene::setClipBox(float3 clip_min, float3 clip_max)
{
    m_clipMin = clip_min;
    m_clipMax = clip_max;
    m_dirtyParams = true;
    return *this;
}
void Scene::validateOrThrow() const
{
    if (!m_cam)
        throw std::runtime_error("Scene: No camera set");
    if (!m_volume)
        throw std::runtime_error("Scene: No volume set");
    if (!m_tf)
        throw std::runtime_error("Scene: No transfer function set");

    if (m_stepSize <= 0.f)
        throw std::runtime_error("Scene: Invalid step size");

    if (m_volume->getDesc().dim.x <= 0 || m_volume->getDesc().dim.y <= 0 || m_volume->getDesc().dim.z <= 0)
        throw std::runtime_error("Scene: invalid volume dimensions.");
}
void Scene::commit(cudaStream_t stream)
{
    validateOrThrow();

    if (m_dirtyCam)
    {
        DeviceCamera dc{};
        dc.position_ = camera_utils::f3(m_cam->getPosition());
        dc.forward_ = camera_utils::f3(m_cam->getForward().normalized());
        dc.up_ = camera_utils::f3(m_cam->getUp().normalized());
        dc.right_ = camera_utils::f3(m_cam->getRight().normalized());
        dc.vertical_fov_ = m_cam->getVerticalFov();

        m_ds_host.d_camera = dc;
        m_dirtyCam = false;
    }
    if (m_dirtyVol)
    {
        const auto &d = m_volume->getDesc();
        DeviceVolume dv{};
        dv.field_tex = m_volume->getFieldTex();
        dv.grad_tex = m_volume->getGradTex();
        dv.dim = d.dim;
        dv.voxel_size = d.voxelSize;
        dv.origin = d.origin;
        dv.value_range = d.valueRange;
        dv.density_scale = d.densityScale;

        m_ds_host.d_volume = dv;
        m_dirtyVol = false;
    }
    if (m_dirtyTF)
    {
        DeviceTF d_tf{};
        d_tf.tf1D = m_tf->getCudaTex();
        d_tf.domain = m_tf->getDomain();

        m_ds_host.d_tf = d_tf;
        m_dirtyTF = false;
    }

    if (m_dirtyLights)
    {
        m_ds_host.d_lights = m_lights ? m_lights->getDevicePointer() : nullptr;
        m_ds_host.lights_count = m_lights ? m_lights->count() : 0;
        m_dirtyLights = false;
    }

    // Params
    if (m_dirtyParams)
    {
        m_ds_host.step_size = m_stepSize;
        m_ds_host.opacityScale = m_opacityScale;
        m_ds_host.mode = m_mode;
        m_ds_host.isoValue = m_isoValue;
        m_ds_host.clipMin = m_clipMin;
        m_ds_host.clipMax = m_clipMax;
        m_dirtyParams = false;
    }

    if (!m_ds_dev)
    {
        CUDA_CHECK(
            cudaMalloc(&m_ds_dev, sizeof(m_ds_host)));

        CUDA_CHECK(
            cudaMemcpyAsync(m_ds_dev, &m_ds_host, sizeof(m_ds_host), cudaMemcpyHostToDevice));
    }
}

SceneDebugInfo Scene::debug() const
{
    SceneDebugInfo info{};
    if (!m_volume || !m_tf)
        return info;
    info.valid = true;
    info.dim = m_volume->getDesc().dim;
    info.voxelSize = m_volume->getDesc().voxelSize;
    info.valueRange = m_volume->getDesc().valueRange;
    info.tfSize = 0; // 若需要可在 TF 中保存 N
    info.lightCount = m_lights ? m_lights->count() : 0;
    info.stepSize = m_stepSize;
    info.mode = m_mode;
    return info;
}