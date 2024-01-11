# vk-bootstrap-triangle

## 1. triangle with vk-bootstrap + push constants

![push constants](screenshots/Snipaste_2023-12-07_22-00-09.png)

## 2. dynamicRendering + synchronization2

![dynamic rendering](screenshots/Snipaste_2023-12-07_21-58-27.png)

## 3. mesh shader

![mesh shader rdc](screenshots/Snipaste_2023-12-07_21-55-12.png)

## 4. imgui

![imgui](screenshots/Snipaste_2023-12-22_11-55-51.png)

## 5. ozz-animation + VK_KHR_push_descriptor

![ozz](screenshots/Snipaste_2023-12-22_13-05-46.png)

## 6. pbr

![pbr](screenshots/Snipaste_2023-12-24_21-13-41.png)

## 7. ibl

![ibl](screenshots/Snipaste_2023-12-26_18-34-34.png)

## 8. buffer device address

![bda](screenshots/Snipaste_2024-01-03_20-01-14.png)

```glsl
// ...
struct Vertex
{
    vec3 position;
    vec3 color;
};

layout(buffer_reference, std430, buffer_reference_align = 32) buffer vertex_buffer_type {
    Vertex v;
};

layout(push_constant) uniform constants
{
    // ...
    uint64_t vertex_buffer_address;
} PushConstants;

void main ()
{
    vertex_buffer_type vertex_buffer = vertex_buffer_type(PushConstants.vertex_buffer_address);
    vec3 inPosition = vertex_buffer[gl_VertexIndex].v.position;
    vec3 inColor    = vertex_buffer[gl_VertexIndex].v.color;
    // ...
}
```

or

```glsl
layout(push_constant) uniform constants
{
    //...
    // for renderdoc support
    vertex_buffer_type vertex_buffer;
} PushConstants;

void main ()
{
    vec3 inPosition = PushConstants.vertex_buffer[gl_VertexIndex].v.position;
    vec3 inColor    = PushConstants.vertex_buffer[gl_VertexIndex].v.color;
    //...
}
```

## 9. gltf + bindless (descriptor indexing)

![gltf](screenshots/Snipaste_2024-01-08_22-47-55.png)

![gltfrdc](screenshots/Snipaste_2024-01-06_17-26-05.png)

> push descriptor only supporting up to 32 descriptors

## 10. timeline semaphore

![timeline semaphore](screenshots/Snipaste_2024-01-08_21-47-17.png)

## 11. no pipeline (VK_EXT_shader_object)

> `VK_EXT_shader_object` is not supported by RenderDoc or Nsight

## 12. compute shader

![cs](screenshots/Snipaste_2024-01-10_22-21-55.png)

## 13. multi queue

![one queue](screenshots/Snipaste_2024-01-10_22-56-29.png)

> one queue

![multi queue](screenshots/Snipaste_2024-01-10_22-57-28.png)

> multi queue
> 
> use `queueWaitIdle(graphics_queue);` to simplify multi queue synchronous

## 14. async compute

![async compute](screenshots/Snipaste_2024-01-11_00-02-39.png)

> **Main pass** and **CS pass** overlap

## 15. headless

Examples that run one-time tasks and don't make use of visual output (no window system integration).
