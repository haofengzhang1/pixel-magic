1.	项目简介：PixelMagic 是一个结合 FastAPI、Replicate API 以及 HuggingFace 模型的 AI 照片处理平台，用户可以上传自己的照片并一键完成风格化转换、自动增强和背景去除，前端界面简洁直观，同时支持生成和管理个人作品集页面，方便展示和分享。
	
    
    
2.	后端功能：核心路由包括 /api/edit 实现基于提示词的风格化编辑，/api/upscale 提供分辨率放大和细节增强，/api/remove_bg 完成自动抠图去背景，同时还提供 /api/portfolio 系列接口支持创建作品集、批量添加图片、生成静态页面、压缩打包下载以及删除功能，保证用户能够完整管理生成的内容。
	
    
    
3.	前端交互：前端 HTML 页面提供上传框、风格选择面板、实时前后对比区域和作品集管理模块，支持用户自由切换多种风格（如 Anime、Oil Painting、Pixar、Cyberpunk 等），处理完成后自动展示效果图并记录到最近历史，还能一键保存到作品集，生成可分享的链接或直接下载全部成果，整个过程无需命令行操作。
	
    

4.	运行方式：首先通过 pip install -r requirements.txt 安装依赖，然后在 .env 文件中配置必要的 API Key（如 REPLICATE_API_TOKEN），接着运行 uvicorn main:app --reload --port 8000 启动服务，最后在浏览器中打开 pixel_magic.html 前端文件即可体验完整功能，整个部署流程轻量快捷，适合本地测试和展示。