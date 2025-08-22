# main.py —— FastAPI + Replicate + HuggingFace/Groq 环境变量统一版（修正版）

import io
import os
import uuid
import shutil
import logging
import time
import json
import datetime
from pathlib import Path
from typing import Tuple

import requests
import replicate
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from typing import Optional, List
from fastapi import Body
import re

# ────────────────────────── 环境变量 ──────────────────────────
load_dotenv()

# 基础配置
ENV = os.getenv("ENV", "development")
PORT = int(os.getenv("PORT", "8000"))

# Replicate（必填 Token）
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_MODEL = os.getenv("REPLICATE_MODEL", "timothybrooks/instruct-pix2pix")
REPLICATE_VERSION = os.getenv("REPLICATE_VERSION", "")  # 建议填写完整 40 位 hash

# 额外模型：放大与抠图（可按需在 .env 替换/指定版本）
UPSCALE_MODEL = os.getenv("UPSCALE_MODEL", "nightmareai/real-esrgan")
UPSCALE_VERSION = os.getenv("UPSCALE_VERSION", "")
REM_BG_MODEL = os.getenv("REM_BG_MODEL", "fofr/transparent-background")
REM_BG_VERSION = os.getenv("REM_BG_VERSION", "")

# HuggingFace（当前未在路由中使用，仅在 /health 展示）
HF_TOKEN = os.getenv("HF_TOKEN")
HF_API_URL = os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models")
HF_MODEL = os.getenv("HF_MODEL", "timothybrooks/instruct-pix2pix")

# Groq（当前未在路由中使用，仅在 /health 展示）
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# 公网地址（前端访问文件）
PUBLIC_BASE_URL = os.getenv("API_PUBLIC_BASE", f"http://127.0.0.1:{PORT}")

if not REPLICATE_API_TOKEN:
    raise RuntimeError("缺少 REPLICATE_API_TOKEN，请在 .env 中配置")

# ────────────────────────── 目录 / 日志 ──────────────────────────
BASE = Path(__file__).parent.resolve()
DATA = BASE / "data"
UPLD = DATA / "uploads"
OUT = DATA / "outputs"
for p in (DATA, UPLD, OUT):
    p.mkdir(parents=True, exist_ok=True)

# 作品集目录
PORTF = DATA / "portfolios"
PORTF.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("uvicorn.error")

# ────────────────────────── FastAPI ──────────────────────────
app = FastAPI(title="PixelMagic API (Replicate/HF/Groq)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 前端本地 file:// 或 127.0.0.1:5500 均可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件映射
app.mount("/files", StaticFiles(directory=DATA), name="files")

# ────────────────────────── Replicate 客户端 ──────────────────────────
rep_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# ────────────────────────── 工具函数 ──────────────────────────
def _abs_url_from_data_path(path_under_data: Path) -> str:
    rel = path_under_data.relative_to(DATA).as_posix()
    return f"{PUBLIC_BASE_URL}/files/{rel}"

def _abs_url_from_data_rel(rel_path_under_data: Path) -> str:
    return f"{PUBLIC_BASE_URL}/files/{rel_path_under_data.as_posix()}"

def _resize_image_file_to_max_side(src_path: Path, max_side: int) -> io.BytesIO:
    """把图片最长边压到 max_side，返回内存文件（PNG/RGB）。"""
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        scale = max(w, h) / float(max_side)
        if scale > 1.0:
            new_w, new_h = int(w / scale), int(h / scale)
            im = im.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        buf.seek(0)
        return buf

def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def _make_portfolio(pid: str) -> Path:
    root = PORTF / pid
    (root / "items").mkdir(parents=True, exist_ok=True)
    manifest = {
        "id": pid,
        "created_at": _now_iso(),
        "items": []
    }
    with (root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 注意：f-string 里的 JS/CSS 花括号要用 {{ }}
    html = f"""<!doctype html><meta charset="utf-8">
<title>Portfolio {pid}</title>
<style>
body{{{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial;padding:24px;background:#111;color:#fff}}}}
.grid{{{{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:16px}}}}
img{{{{width:100%;border-radius:12px;background:#222}}}}
</style>
<h1>Portfolio {pid}</h1>
<p>Generated at {manifest["created_at"]}</p>
<div class="grid" id="g"></div>
<script>
fetch('manifest.json').then(r => r.json()).then(m => {{{{
  const g = document.getElementById('g');
  m.items.forEach(u => {{{{
    const img = document.createElement('img');
    img.src = u;
    g.appendChild(img);
  }}}});
}}}});
</script>"""
    with (root / "index.html").open("w", encoding="utf-8") as f:
        f.write(html)
    return root

def _load_manifest(pid: str) -> Tuple[Path, dict]:
    root = PORTF / pid
    mf = root / "manifest.json"
    if not mf.exists():
        raise HTTPException(404, "portfolio not found")
    with mf.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return root, data

def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "").suffix.lower() or ".png"
    if suffix not in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        suffix = ".png"
    out = UPLD / f"{uuid.uuid4().hex}{suffix}"
    with out.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return out

def _save_bytes_to_outputs(raw: bytes, prefer_png=True) -> Path:
    ext = ".png" if prefer_png else ".jpg"
    out = OUT / f"{uuid.uuid4().hex}{ext}"
    with out.open("wb") as f:
        f.write(raw)
    return out

def _download_to_outputs(url: str) -> Path:
    headers = {"User-Agent": "PixelMagic/1.0"}
    r = requests.get(url, headers=headers, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"download fail {r.status_code}: {r.text[:200]}")
    data = r.content

    try:
        im = Image.open(io.BytesIO(data))
        im.load()  # 读满，避免懒加载
        has_alpha = (
            im.mode in ("RGBA", "LA") or
            (im.mode == "P" and "transparency" in im.info)
        )
        if has_alpha:
            im = im.convert("RGBA")
        else:
            im = im.convert("RGB")

        out = OUT / f"{uuid.uuid4().hex}.png"
        im.save(out, format="PNG")  # PNG 保留透明
        return out
    except UnidentifiedImageError:
        raise RuntimeError("Downloaded file is not an image (got non-image bytes)")

def _normalize_and_save_output(output) -> Path:
    # 先处理“像文件”的对象：有 url / read()
    if hasattr(output, "url") and isinstance(getattr(output, "url"), str):
        return _download_to_outputs(getattr(output, "url"))

    if hasattr(output, "read") and callable(getattr(output, "read")):
        data = output.read()
        if isinstance(data, (bytes, bytearray)):
            return _save_bytes_to_outputs(bytes(data))

    # 纯 URL
    if isinstance(output, str) and output.startswith("http"):
        return _download_to_outputs(output)

    # 纯字节
    if isinstance(output, (bytes, bytearray)):
        return _save_bytes_to_outputs(bytes(output))

    # 列表：依次尝试
    if isinstance(output, list):
        last_err = None
        for item in output:
            try:
                return _normalize_and_save_output(item)
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"all list items failed: {last_err}")

    # 字典：尝试常见键；不行就遍历所有值
    if isinstance(output, dict):
        for key in ["image", "images", "url", "output", "result"]:
            if key in output and output[key]:
                try:
                    return _normalize_and_save_output(output[key])
                except Exception:
                    pass
        last_err = None
        for v in output.values():
            if v:
                try:
                    return _normalize_and_save_output(v)
                except Exception as e:
                    last_err = e
                    continue
        raise RuntimeError(f"unexpected dict, tried keys failed: {list(output.keys())}; last_err={last_err}")

    # 兜底：再试常见属性
    for attr in ("url", "path", "file", "content"):
        if hasattr(output, attr):
            try:
                return _normalize_and_save_output(getattr(output, attr))
            except Exception:
                pass

    raise RuntimeError(f"unknown output type: {type(output)}")

def _ensure_seekable(v):
    try:
        v.seek(0)
    except Exception:
        pass
    return v

def _prep_payload(payload: dict) -> dict:
    """统一把可能的二进制流归零，避免因为上游处理过导致传空。"""
    out = {}
    for k, v in payload.items():
        out[k] = _ensure_seekable(v) if hasattr(v, "read") else v
    return out

def _model_id(model: str, version: str = "") -> str:
    model = (model or "").strip()
    version = (version or "").strip()
    return f"{model}:{version}" if version else model

def _run_replicate(model: str, input_payload: dict, version: str = "") -> Path:
    mid = _model_id(model, version)
    try:
        log.info("replicate.run -> %s", mid)
        input_payload = _prep_payload(input_payload)
        out = rep_client.run(mid, input=input_payload)
        return _normalize_and_save_output(out)

    except Exception as e:
        msg = str(e).lower()

        # 410: 版本弃用；422: 版本无效/无权限 —— 自动退回 latest 再试一次
        if any(x in msg for x in ["410", "gone", "invalid version", "not permitted", "does not exist"]) and version:
            log.warning("replicate %s failed due to version issue (%s). Fallback to latest...", mid, e)
            try:
                mid_latest = _model_id(model, "")
                log.info("replicate.run (fallback latest) -> %s", mid_latest)
                out = rep_client.run(mid_latest, input=_prep_payload(input_payload))
                return _normalize_and_save_output(out)
            except Exception as e2:
                log.exception("replicate latest fallback error: %s", e2)
                raise HTTPException(status_code=502, detail=f"replicate error (fallback latest failed): {e2}")

        # 422 输入校验失败时，给更清晰的提示
        if "422" in msg or "validation" in msg or "required" in msg:
            raise HTTPException(status_code=422, detail=f"replicate input validation failed: {e}")

        log.exception("replicate run error: %s", e)
        raise HTTPException(status_code=502, detail=f"replicate error: {e}")

def _share_url_for(pid: str) -> str:
    # /files/portfolios/<pid>/index.html
    share_rel = Path("portfolios") / pid / "index.html"
    return _abs_url_from_data_rel(share_rel)

def _safe_pid(pid: str) -> str:
    """只允许 hex 目录名，防止路径穿越。"""
    if not re.fullmatch(r"[a-f0-9]{8,64}", pid):
        raise HTTPException(400, "invalid portfolio id")
    return pid
# ────────────────────────── 路由 ──────────────────────────
@app.get("/health")
def health():
    def _check(model: str) -> str:
        try:
            rep_client.models.get(model)
            return "ok"
        except Exception as e:
            return f"error: {e}"

    return {
        "ok": True,
        "env": ENV,
        "public_base": PUBLIC_BASE_URL,
        "models": {
            "replicate": {
                "name": REPLICATE_MODEL,
                "version": REPLICATE_VERSION or "(none)",
                "status": _check(REPLICATE_MODEL),
            },
            "upscale": {
                "name": UPSCALE_MODEL,
                "version": UPSCALE_VERSION or "(none)",
                "status": _check(UPSCALE_MODEL),
            },
            "remove_bg": {
                "name": REM_BG_MODEL,
                "version": REM_BG_VERSION or "(none)",
                "status": _check(REM_BG_MODEL),
            },
            "huggingface": {"name": HF_MODEL, "token": "ok" if HF_TOKEN else "missing"},
            "groq": {"name": GROQ_MODEL, "token": "ok" if GROQ_API_KEY else "missing"},
        },
    }

# —— 图像编辑（风格化） ——
@app.post("/api/edit")
async def edit_image(prompt: str = Form(...), image: UploadFile = File(...)):
    img_path = _save_upload(image)
    # 逐级降维的档位（max_side, steps, g_scale, img_g_scale）
    tiers = [
        (768, 20, 7.0, 1.2),
        (640, 18, 6.5, 1.1),
        (512, 16, 6.0, 1.0),
        (384, 14, 5.5, 0.9),
        (256, 12, 5.0, 0.8),
    ]
    last_err = None
    try:
        for max_side, steps, g_scale, img_g_scale in tiers:
            try:
                buf = _resize_image_file_to_max_side(img_path, max_side)
                payload = {
                    "image": buf,
                    "prompt": prompt,
                    "num_inference_steps": steps,
                    "guidance_scale": g_scale,
                    "image_guidance_scale": img_g_scale,
                    "num_outputs": 1
                }
                out_path = _run_replicate(REPLICATE_MODEL, payload, REPLICATE_VERSION)
                return {"url": _abs_url_from_data_path(out_path)}
            except HTTPException as he:
                last_err = he
                msg = str(he.detail).lower()
                if ("out of memory" in msg) or ("cuda" in msg):
                    log.warning("OOM at %spx, retrying smaller tier...", max_side)
                    continue
                raise

        # —— 所有档位都 OOM：切到 SDXL-Turbo(img2img) 兜底 ——
        try:
            fallback_model = os.getenv("FALLBACK_MODEL", "stability-ai/sdxl-turbo")
            fallback_ver = os.getenv("FALLBACK_VERSION", "")
            buf = _resize_image_file_to_max_side(img_path, 512)
            payload = {"image": buf, "prompt": prompt, "strength": 0.5, "num_outputs": 1}
            out_path = _run_replicate(fallback_model, payload, fallback_ver)
            return {"url": _abs_url_from_data_path(out_path)}
        except Exception as fe:
            log.exception("fallback sdxl-turbo failed: %s", fe)
            raise HTTPException(502, f"replicate error: OOM after retries and fallback: {fe}")

    except Exception as e:
        log.exception("edit_image failed: %s", e)
        if isinstance(last_err, HTTPException):
            raise HTTPException(502, f"replicate error: {last_err.detail}")
        raise HTTPException(500, f"edit failed: {e}")

# —— 放大增强（2x/3x/4x） ——
@app.post("/api/upscale")
async def upscale_image(image: UploadFile = File(...), scale: str = Form("2")):
    img_path = _save_upload(image)
    img_url = _abs_url_from_data_path(img_path)

    # 候选模型：.env 指定 + 同模型 latest
    candidates = [
        (UPSCALE_MODEL, UPSCALE_VERSION),
        (UPSCALE_MODEL, ""),  # latest
    ]

    # 显存友好：逐步降尺寸
    caps = {"2": 1024, "3": 1024, "4": 768}
    sides = (caps.get(str(scale), 1024), 768, 640)

    def payloads(buf: io.BytesIO | None):
        """不同键名 + 文件/URL 两种形态 + 可选 scale"""
        bases = []
        if buf is not None:
            bases += [{"img": buf}, {"image": buf}, {"input_image": buf}]
        bases += [{"img": img_url}, {"image": img_url}, {"image_url": img_url}, {"input": img_url}]
        out = []
        for b in bases:
            out.append(b.copy())
            try:
                s = int(scale)
                c = b.copy()
                c["scale"] = s
                out.append(c)
            except Exception:
                pass
        return out

    last_err = None

    for model, version in candidates:
        for side in sides:
            # 生成缩放后内存图（失败就只用 URL）
            try:
                buf = _resize_image_file_to_max_side(img_path, side)
            except Exception:
                buf = None

            for p in payloads(buf):
                try:
                    out_path = _run_replicate(model, p, version)
                    url = _abs_url_from_data_path(out_path) + f"?t={int(time.time())}"
                    log.info(
                        "UPSCALE ok -> %s (model=%s, ver=%s, side=%s, keys=%s)",
                        url, model, version or "latest", side, list(p.keys()),
                    )
                    return {"url": url}

                except HTTPException as he:
                    last_err = he
                    msg = str(he.detail).lower()

                    # 显存：降尺寸
                    if "out of memory" in msg or "cuda" in msg:
                        log.warning("UPSCALE OOM at %spx on %s; retry smaller...", side, model)
                        break  # 去更小尺寸

                    # 版本/权限/校验：换下一种 payload 或模型
                    if ("not found" in msg or "404" in msg or
                        "invalid version" in msg or "not permitted" in msg or
                        "422" in msg or "validation" in msg or "required" in msg):
                        log.warning("UPSCALE failed on %s:%s with %s -> %s",
                                    model, version or "latest", list(p.keys()), msg)
                        continue

                    # 其它错误：换模型
                    log.warning("UPSCALE error on %s:%s -> %s; try next model",
                                model, version or "latest", msg)
                    break  # 换下一个模型

    raise HTTPException(502, f"upscale failed: {getattr(last_err, 'detail', last_err)}")

# —— 抠图（去背景） ——
@app.post("/api/remove_bg")
async def remove_bg(image: UploadFile = File(...)):
    img_path = _save_upload(image)
    img_url = _abs_url_from_data_path(img_path)  # 供只接受 URL 的模型使用

    try:
        # 压到 1024，OOM 再降
        for side in (1024, 768, 640):
            buf = _resize_image_file_to_max_side(img_path, side)

            # 依次尝试常见字段名与载荷形态
            trials = [
                {"image": buf},          # 大部分模型
                {"input_image": buf},    # 有些用 input_image
                {"image_url": img_url},  # 只支持 URL 的
                {"input": img_url},      # 有些把输入叫 input
            ]
            for payload in trials:
                try:
                    out_path = _run_replicate(REM_BG_MODEL, payload, REM_BG_VERSION)
                    url = _abs_url_from_data_path(out_path) + f"?t={int(time.time())}"
                    return {"url": url}
                except HTTPException as he:
                    msg = str(he.detail).lower()
                    if ("out of memory" in msg) or ("cuda" in msg):
                        log.warning("REMOVE_BG OOM at side=%s, retry smaller", side)
                        break  # 换更小尺寸
                    # 其它错误：继续试下一个 payload（字段不匹配/版本不对等）
                    continue

        raise HTTPException(502, "remove_bg failed after payload & resize retries")
    except Exception as e:
        log.exception("remove_bg failed: %s", e)
        raise HTTPException(500, f"remove_bg failed: {e}")

@app.post("/api/portfolio/create")
def portfolio_create(name: Optional[str] = Body(None)):
    pid = uuid.uuid4().hex[:8]
    root = _make_portfolio(pid)
    share_path = Path("portfolios") / pid / "index.html"  # 相对于 DATA
    page_url = _abs_url_from_data_rel(share_path)         # 对外可访问地址
    return {"id": pid, "page": page_url}

@app.post("/api/portfolio/{pid}/add")
async def portfolio_add(pid: str, files: List[UploadFile] = File(...)):
    root, manifest = _load_manifest(pid)
    added_urls = []
    for f in files:
        # 保存到 DATA/portfolios/<pid>/items/ 下
        ext = (Path(f.filename).suffix or ".png").lower()
        if ext not in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            ext = ".png"
        dst = root / "items" / f"{uuid.uuid4().hex}{ext}"
        with dst.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        # 转成 /files/... 访问 URL
        rel = dst.relative_to(DATA)
        url = _abs_url_from_data_rel(rel)
        manifest["items"].append(url)
        added_urls.append(url)

    # 回写清单
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "count": len(added_urls), "items": added_urls}

@app.get("/api/portfolio/{pid}/share")
def portfolio_share(pid: str):
    root, _ = _load_manifest(pid)
    share_path = Path("portfolios") / pid / "index.html"
    return {"page": _abs_url_from_data_rel(share_path)}

@app.get("/api/portfolio/{pid}/download")
def portfolio_download(pid: str):
    root, _ = _load_manifest(pid)
    zip_path = root / f"{pid}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        # 打包清单和所有 item
        z.write(root / "manifest.json", arcname="manifest.json")
        for p in (root / "items").glob("*"):
            z.write(p, arcname=f"items/{p.name}")
    return FileResponse(zip_path, media_type="application/zip", filename=f"portfolio_{pid}.zip")

# —— 放在 DATA/UPLD/OUT 定义之后 —— #
PORTF = DATA / "portfolios"
PORTF.mkdir(parents=True, exist_ok=True)

@app.get("/api/portfolios")
def portfolios_list():
    """返回现有作品集列表。任何单个作品集出错都跳过，不影响整体。"""
    items = []
    try:
        for d in sorted(PORTF.glob("*")):
            if not d.is_dir():
                continue
            mf = d / "manifest.json"
            if not mf.exists():
                # 没有 manifest 的目录直接跳过
                continue
            try:
                with mf.open("r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                # 某个 JSON 损坏：跳过并记录
                log.warning("skip bad manifest: %s (%s)", mf, e)
                continue

            pid = data.get("id") or d.name
            urls = data.get("items") or []
            cover_url = urls[0] if urls else ""
            share_url = f"{PUBLIC_BASE_URL}/files/portfolios/{pid}/index.html"

            items.append({
                "id": pid,
                "count": len(urls),
                "cover": cover_url,
                "share_url": share_url,
            })

        return {"items": items}
    except Exception as e:
        # 不要把异常抛给前端，返回空列表并打日志
        log.exception("list portfolios error: %s", e)
        return {"items": []}

@app.get("/api/portfolio/{pid}")
def portfolio_detail(pid: str):
    """返回某个作品集的 manifest（含分享页地址）"""
    _, data = _load_manifest(pid)
    data["share_url"] = _share_url_for(pid)
    return data

@app.delete("/api/portfolio/{pid}")
def portfolio_delete(pid: str):
    pid = _safe_pid(pid)
    root = PORTF / pid
    if not root.exists():
        raise HTTPException(404, "portfolio not found")
    try:
        shutil.rmtree(root)
        return {"ok": True, "id": pid}
    except Exception as e:
        log.exception("delete portfolio failed: %s", e)
        raise HTTPException(500, f"delete portfolio failed: {e}")

# ────────────────────────── 入口（可选） ──────────────────────────
# 生产环境一般用： uvicorn main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=True)