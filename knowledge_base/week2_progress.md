# Week 2 Progress: LLM Benchmarking, RAG Pipeline, and Hardware Verification

## Team Alpha (Robot & Telepresence)

- Completed ROS2 fundamentals practice: workspace/package creation, nodes, topic
  publishers/subscribers, velocity command publishing, service server/client, parameter nodes,
  action server/client, and ROS bag record/replay.
- Verified the RPLIDAR S2M1R2 hardware: connected via the SLAMTEC USB adapter, detected as
  `/dev/ttyUSB0`, launched the SLAMTEC ROS2 driver, confirmed the real `/scan` topic and its
  publish frequency, and visualized the LIDAR data in RViz2.
- Built a lightweight Python-based local scan visualization from real LIDAR data (not full SLAM):
  measured minimum front distance of 0.197 meters and average front distance of 0.221 meters.
- Inspected the AION Robotics UGV platform through the Jetson Nano: identified Ubuntu 18.04,
  ROS1 Melodic, a Cube/Pixhawk autopilot running ArduRover firmware, and a HERE3 GPS/Compass
  module connected to the Cube's CAN port. Existing UGV/Jetson files were copied for study.
- Remaining/pending: connecting the RPLIDAR to the Jetson itself, verifying `/odom` and the
  `/tf` frame tree (map, odom, base_link, laser), real SLAM/mapping, and a real UGV movement
  test.

## Team Beta (AI Stack & Backend)

- Benchmarked four candidate LLMs via Ollama (Llama 3.1 8B, Mistral 7B, Phi-3 Mini 3.8B, Qwen2
  7B) using Ollama's own native timing metrics. On a CPU-only baseline machine, Phi-3 Mini was
  fastest at 3.91 tokens/sec, and Mistral was slowest at 1.89 tokens/sec.
- Built the initial Retrieval-Augmented Generation (RAG) pipeline: document ingestion and
  chunking, embeddings via the `nomic-embed-text` model, a persistent ChromaDB vector store, and
  retrieval connected to Ollama for grounded answer generation, exposed via `POST /rag/ask`.
- Added a **router**: a distance-threshold check (tuned to 0.40 using real evaluation data) that
  decides whether a question needs the RAG/knowledge-base path or can be answered as normal
  conversation, without requiring an extra LLM call.
- Added a **query rewriter**: reformulates casual or speech-transcribed questions into clearer
  search terms before retrieval, using the lightweight Phi-3 model.
- Added structured evaluation logging: every RAG query is logged with its routing decision,
  retrieved sources, distance scores, and per-stage latency breakdown.
- Defined two deployment profiles: the Jetson Nano (edge, no GPU) runs Phi-3 Mini with the query
  rewriter disabled to minimize latency, while the GPU workstation (16GB VRAM) runs Llama 3.1 as
  the primary model with the query rewriter enabled, since the GPU removes the latency penalty.

## Hardware Used for AI Inference

The AI backend was tested on a workstation with an NVIDIA Quadro P5000 GPU (16GB GDDR5X VRAM),
which hosts Ollama and serves model inference requests from the Jetson-based robot over
Tailscale.
