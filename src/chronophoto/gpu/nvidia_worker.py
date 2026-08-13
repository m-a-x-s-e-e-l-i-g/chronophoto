from __future__ import annotations

# ruff: noqa: E402, I001 - optional GPU imports require DLL search setup first

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _prepare_dll_search() -> None:
    root = Path(sys.executable).resolve().parent
    paths = (
        root / "Lib/site-packages/PyNvVideoCodec",
        root / "Lib/site-packages/nvidia/cuda_runtime/bin",
        root / "Lib/site-packages/nvidia/cuda_nvrtc/bin",
    )
    os.environ["PATH"] = os.pathsep.join((*map(str, paths), os.environ.get("PATH", "")))
    if hasattr(os, "add_dll_directory"):
        for path in paths:
            if path.is_dir():
                os.add_dll_directory(str(path))


_prepare_dll_search()

import cupy as cp  # noqa: E402
import PyNvVideoCodec as nvc  # noqa: E402


_CUDA_SOURCE = r"""
extern "C" __global__ void nv12_to_rgb(
    const unsigned char* y, const unsigned char* uv, unsigned char* rgb,
    int width, int height, int y_pitch, int uv_pitch) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= width || row >= height) return;
  float yy = fmaxf((float)y[row * y_pitch + x] - 16.0f, 0.0f);
  int uv_index = (row / 2) * uv_pitch + (x / 2) * 2;
  float u = (float)uv[uv_index] - 128.0f;
  float v = (float)uv[uv_index + 1] - 128.0f;
  int out = (row * width + x) * 3;
  rgb[out] = (unsigned char)fminf(fmaxf(roundf(1.164383f*yy + 1.792741f*v),0),255);
  rgb[out+1] = (unsigned char)fminf(fmaxf(roundf(1.164383f*yy-.213249f*u-.532909f*v),0),255);
  rgb[out+2] = (unsigned char)fminf(fmaxf(roundf(1.164383f*yy + 2.112402f*u),0),255);
}
extern "C" __global__ void difference_mask(
    const unsigned char* rgb, const unsigned char* bg, unsigned char* mask,
    int width, int height, int threshold) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= width || y >= height) return;
  int sums[3] = {0,0,0}; int bgs[3] = {0,0,0};
  const int k[5] = {1,4,6,4,1};
  for (int dy=-2; dy<=2; ++dy) for (int dx=-2; dx<=2; ++dx) {
    int xx=min(max(x+dx,0),width-1), yy=min(max(y+dy,0),height-1);
    int weight=k[dx+2]*k[dy+2], p=(yy*width+xx)*3;
    sums[0]+=rgb[p]*weight; sums[1]+=rgb[p+1]*weight; sums[2]+=rgb[p+2]*weight;
    bgs[0]+=bg[p]*weight; bgs[1]+=bg[p+1]*weight; bgs[2]+=bg[p+2]*weight;
  }
  float dr=fabsf((sums[0]-bgs[0])/256.0f), dg=fabsf((sums[1]-bgs[1])/256.0f);
  float db=fabsf((sums[2]-bgs[2])/256.0f);
  mask[y*width+x]=(0.299f*dr+0.587f*dg+0.114f*db)>threshold?255:0;
}
extern "C" __global__ void morphology(
    const unsigned char* input, unsigned char* output, int width, int height, int dilate) {
  int x=blockDim.x*blockIdx.x+threadIdx.x, y=blockDim.y*blockIdx.y+threadIdx.y;
  if(x>=width||y>=height)return; int value=dilate?0:255;
  for(int dy=-2;dy<=2;++dy) for(int dx=-2;dx<=2;++dx) {
    if(abs(dx)+abs(dy)>3)continue;
    int xx=min(max(x+dx,0),width-1), yy=min(max(y+dy,0),height-1);
    int sample=input[yy*width+xx]; value=dilate?max(value,sample):min(value,sample);
  } output[y*width+x]=(unsigned char)value;
}
extern "C" __global__ void feather(
    const unsigned char* input, unsigned char* output, int width, int height, int radius) {
  int x=blockDim.x*blockIdx.x+threadIdx.x, y=blockDim.y*blockIdx.y+threadIdx.y;
  if(x>=width||y>=height)return; int sum=0,count=0;
  for(int dy=-radius;dy<=radius;++dy)for(int dx=-radius;dx<=radius;++dx){
    int xx=min(max(x+dx,0),width-1),yy=min(max(y+dy,0),height-1);
    sum+=input[yy*width+xx];++count;
  } output[y*width+x]=(unsigned char)(sum/count);
}
extern "C" __global__ void init_result(
    const unsigned char* bg, float* result, int pixels) {
  int i=blockDim.x*blockIdx.x+threadIdx.x; if(i<pixels*3)result[i]=(float)bg[i];
}
extern "C" __global__ void blend(
    const unsigned char* rgb, const unsigned char* mask, float* result, int pixels) {
  int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=pixels)return;float a=mask[i]/255.0f;
  int p=i*3; result[p]=rgb[p]*a+result[p]*(1-a);
  result[p+1]=rgb[p+1]*a+result[p+1]*(1-a);
  result[p+2]=rgb[p+2]*a+result[p+2]*(1-a);
}
extern "C" __global__ void rgb_to_y(
    const float* rgb,unsigned char* y,int width,int height,int pitch){
  int x=blockDim.x*blockIdx.x+threadIdx.x,row=blockDim.y*blockIdx.y+threadIdx.y;
  if(x>=width||row>=height)return;int p=(row*width+x)*3;
  float v=16+.182586f*rgb[p]+.614231f*rgb[p+1]+.062007f*rgb[p+2];
  y[row*pitch+x]=(unsigned char)fminf(fmaxf(roundf(v),16),235);
}
extern "C" __global__ void rgb_to_uv(
    const float* rgb,unsigned char* uv,int width,int height,int pitch){
  int x=blockDim.x*blockIdx.x+threadIdx.x,row=blockDim.y*blockIdx.y+threadIdx.y;
  if(x>=width/2||row>=height/2)return;float u=0,v=0;
  for(int dy=0;dy<2;++dy)for(int dx=0;dx<2;++dx){int p=(((row*2+dy)*width)+(x*2+dx))*3;
    u+=128-.100644f*rgb[p]-.338572f*rgb[p+1]+.439216f*rgb[p+2];
    v+=128+.439216f*rgb[p]-.398942f*rgb[p+1]-.040274f*rgb[p+2];}
  uv[row*pitch+x*2]=(unsigned char)fminf(fmaxf(roundf(u/4),16),240);
  uv[row*pitch+x*2+1]=(unsigned char)fminf(fmaxf(roundf(v/4),16),240);
}
"""
_MODULE = cp.RawModule(code=_CUDA_SOURCE)
_KERNELS = {name: _MODULE.get_function(name) for name in (
    "nv12_to_rgb", "difference_mask", "morphology", "feather", "init_result",
    "blend", "rgb_to_y", "rgb_to_uv")}


def _event(kind: str, **values: object) -> None:
    print(json.dumps({"kind": kind, **values}), flush=True)


def _plane_array(view: object, owner: object, *, dtype: object = cp.uint8) -> cp.ndarray:
    shape = tuple(int(value) for value in view.shape)
    strides = tuple(int(value) for value in view.stride)
    size = strides[0] * shape[0]
    memory = cp.cuda.UnownedMemory(int(view.dataptr), size, owner)
    return cp.ndarray(shape, dtype=dtype, memptr=cp.cuda.MemoryPointer(memory, 0), strides=strides)


def _rgb_from_nv12(frame: object) -> cp.ndarray:
    planes = frame.cuda()
    y = _plane_array(planes[0], frame)
    uv = _plane_array(planes[1], frame)
    height, width = y.shape[:2]
    rgb = cp.empty((height, width, 3), dtype=cp.uint8)
    _KERNELS["nv12_to_rgb"](
        ((width + 15) // 16, (height + 15) // 16), (16, 16),
        (y, uv, rgb, width, height, y.strides[0], uv.strides[0]),
    )
    return rgb


def _write_rgb_to_nv12(rgb: cp.ndarray, frame: object) -> None:
    planes = frame.cuda()
    y_plane = _plane_array(planes[0], frame)[..., 0]
    uv_plane = _plane_array(planes[1], frame)
    height, width = y_plane.shape
    grid = ((width + 15) // 16, (height + 15) // 16)
    _KERNELS["rgb_to_y"](grid, (16, 16), (rgb, y_plane, width, height, y_plane.strides[0]))
    _KERNELS["rgb_to_uv"](
        ((width // 2 + 15) // 16, (height // 2 + 15) // 16), (16, 16),
        (rgb, uv_plane, width, height, uv_plane.strides[0]),
    )


def _motion_mask(
    frame: cp.ndarray,
    background: cp.ndarray,
    threshold: int,
    feather: int,
    minimum_ratio: float,
) -> cp.ndarray:
    del minimum_ratio
    height, width = frame.shape[:2]
    grid = ((width + 15) // 16, (height + 15) // 16)
    first = cp.empty((height, width), dtype=cp.uint8)
    second = cp.empty_like(first)
    _KERNELS["difference_mask"](
        grid, (16, 16), (frame, background, first, width, height, threshold)
    )
    for dilate in (0, 1, 1, 1, 0, 0):
        _KERNELS["morphology"](grid, (16, 16), (first, second, width, height, dilate))
        first, second = second, first
    if feather:
        _KERNELS["feather"](grid, (16, 16), (first, second, width, height, feather))
        return second
    return first


def _sample_indices(first: int, last: int, count: int) -> list[int]:
    if last <= first:
        return [first]
    values = cp.linspace(first, last, min(count, last - first + 1))
    return sorted({int(round(value)) for value in cp.asnumpy(values)})


def _clean_plate(decoder: object, indices: list[int], mode: str) -> cp.ndarray:
    if mode == "first":
        return _rgb_from_nv12(decoder[indices[0]]).copy()
    if mode == "last":
        return _rgb_from_nv12(decoder[indices[-1]]).copy()
    samples = [_rgb_from_nv12(decoder[index]).copy() for index in indices]
    height, width = samples[0].shape[:2]
    result = cp.empty((height, width, 3), dtype=cp.uint8)
    for top in range(0, height, 64):
        bottom = min(height, top + 64)
        tile = cp.stack([sample[top:bottom] for sample in samples], axis=0)
        result[top:bottom] = cp.median(tile, axis=0).astype(cp.uint8)
    return result


def _packets_data(packets: list[dict[str, object]]) -> bytes:
    return b"".join(bytes(packet["data"]) for packet in packets)


def render(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    decoder = nvc.SimpleDecoder(
        args.source,
        use_device_memory=True,
        output_color_type=nvc.OutputColorType.NATIVE,
    )
    metadata = decoder.get_stream_metadata()
    fps = float(args.fps or metadata.average_fps)
    first = max(0, round(args.start * fps))
    last = min(len(decoder) - 1, max(first + 1, round(args.end * fps) - 1))
    samples = _sample_indices(first, last, 21)
    background = _clean_plate(decoder, samples, args.background)
    encoder = nvc.CreateEncoder(
        int(metadata.width),
        int(metadata.height),
        "NV12",
        False,
        gpu_id=0,
        codec="h264",
        fps=str(fps),
        rc="vbr",
        bitrate=args.bitrate,
    )
    active: list[tuple[float, cp.ndarray, cp.ndarray]] = []
    pixels_count = int(metadata.width) * int(metadata.height)
    result = cp.empty((int(metadata.height), int(metadata.width), 3), dtype=cp.float32)
    completed = 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as target:
        for index in range(first, last + 1):
            decoded = decoder[index]
            timestamp = index / fps
            rgb = _rgb_from_nv12(decoded)
            mask = _motion_mask(
                rgb,
                background,
                args.threshold,
                args.feather,
                args.minimum_component_ratio,
            )
            active.append((timestamp, rgb.copy(), mask))
            cutoff = timestamp - args.trail_duration
            active = [pose for pose in active if pose[0] >= cutoff]
            _KERNELS["init_result"](
                ((pixels_count * 3 + 255) // 256,), (256,),
                (background, result, pixels_count),
            )
            stack = active if args.overlap == "newest" else reversed(active)
            for _pose_time, pose_rgb, pose_mask in stack:
                _KERNELS["blend"](
                    ((pixels_count + 255) // 256,), (256,),
                    (pose_rgb, pose_mask, result, pixels_count),
                )
            if completed == 0:
                _KERNELS["blend"](
                    ((pixels_count + 255) // 256,), (256,),
                    (active[-1][1], active[-1][2], result, pixels_count),
                )
            _write_rgb_to_nv12(result, decoded)
            target.write(_packets_data(encoder.Encode(decoded)))
            completed += 1
            if index == last and args.reference_frame:
                cp.save(args.reference_frame, cp.clip(cp.rint(result), 0, 255).astype(cp.uint8))
            _event(
                "progress",
                value=round(completed / (last - first + 1) * 95),
                message=f"NVIDIA GPU frame {completed} of {last - first + 1}",
            )
        target.write(_packets_data(encoder.EndEncode()))
    elapsed = max(0.001, time.perf_counter() - started)
    _event(
        "complete",
        frames=completed,
        seconds=elapsed,
        fps=completed / elapsed,
        width=int(metadata.width),
        height=int(metadata.height),
        active_window_peak=max(1, round(args.trail_duration * fps) + 1),
    )


def probe() -> None:
    device_count = int(cp.cuda.runtime.getDeviceCount())
    cp.arange(1).sum().item()
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    _event(
        "probe",
        available=device_count > 0,
        devices=device_count,
        pynvvideocodec=nvc.__version__,
        cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        device_name=name,
        compute_capability=[int(properties["major"]), int(properties["minor"])],
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--probe", action="store_true")
    result.add_argument("--source")
    result.add_argument("--output")
    result.add_argument("--start", type=float, default=0.0)
    result.add_argument("--end", type=float, default=0.0)
    result.add_argument("--fps", type=float, default=0.0)
    result.add_argument("--trail-duration", type=float, default=1.0)
    result.add_argument("--threshold", type=int, default=17)
    result.add_argument("--feather", type=int, default=1)
    result.add_argument("--minimum-component-ratio", type=float, default=0.00035)
    result.add_argument(
        "--background",
        choices=("automatic", "median", "first", "last"),
        default="automatic",
    )
    result.add_argument("--overlap", choices=("newest", "oldest"), default="newest")
    result.add_argument("--bitrate", default="20M")
    result.add_argument("--reference-frame")
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.probe:
        probe()
    else:
        render(arguments)
