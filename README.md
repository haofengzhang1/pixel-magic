Project Overview:
PixelMagic is an AI-powered photo processing platform that integrates FastAPI, the Replicate API, and Hugging Face models. It allows users to upload their own images and perform one-click operations such as style transformation, automatic enhancement, and background removal. The front-end interface is clean and intuitive, and it also supports generating and managing personal portfolio pages for easy display and sharing.

Backend Features:
The core routes include /api/edit for prompt-based stylized editing, /api/upscale for resolution enhancement and detail refinement, and /api/remove_bg for automatic background removal. In addition, the /api/portfolio series of endpoints supports creating portfolios, batch image uploads, static page generation, compressed downloads, and deletion. These features ensure that users can fully manage their generated content.

Frontend Interaction:
The front-end HTML page provides an upload panel, a style selection interface, a real-time before-and-after comparison area, and a portfolio management module. Users can freely switch between multiple styles (such as Anime, Oil Painting, Pixar, Cyberpunk, etc.). After processing, the results are automatically displayed and saved in recent history. Users can also save outputs to their portfolio with one click, generate shareable links, or download all results directly. The entire process requires no command-line operations.

How to Run:
First, install dependencies using pip install -r requirements.txt. Then configure the required API keys (such as REPLICATE_API_TOKEN) in the .env file. Next, start the server with uvicorn main:app --reload --port 8000. Finally, open the pixel_magic.html file in your browser to experience the full functionality. The deployment process is lightweight and efficient, making it suitable for local testing and demonstrations.
